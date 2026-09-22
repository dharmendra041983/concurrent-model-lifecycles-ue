"""
Shared channel and measurement layer.

This implements Sec. IV-A of the paper: a single channel realization H_t that
drives all three pipelines, plus the measurement operator

    m_t = M(H_t, B_t) + n_t

where B_t is the set of beam resources currently configured -- itself an
output of the beam-management pipeline, which is the first coupling channel.

Design commitments (do not relax these later without rerunning everything):

  1. Realizations are seed-addressed and cached. The eight forced-subset arms
     of the interaction experiment must see byte-identical H trajectories, so
     the channel is generated once per seed and replayed, never regenerated
     inside an arm.

  2. The beam codebook is built analytically from the array geometry rather
     than read from Sionna internals, so it survives Sionna version changes.

VERIFY ON FIRST RUN (marked #CHECK below):
  - the shape/units convention of `ut_orientation` / `bs_orientation`
  - the element ordering of Sionna's AntennaArray vs. the codebook's assumed
    row-major (elevation-major) ordering. `sanity_check_codebook()` will fail
    loudly if these disagree.
"""

from __future__ import annotations

import os
import hashlib
import numpy as np
import torch

import sionna.phy
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

from config import ChannelConfig, ExperimentConfig


# ---------------------------------------------------------------------------
# Beam codebook
# ---------------------------------------------------------------------------

def dft_codebook(num_rows: int, num_cols: int,
                 oversample_el: int = 1, oversample_az: int = 4) -> torch.Tensor:
    """Oversampled 2D-DFT codebook for a uniform rectangular array.

    Assumes 0.5-lambda spacing and row-major (elevation-major) element
    ordering: element index = row * num_cols + col.

    Returns
    -------
    W : [num_beams, num_rows*num_cols] complex64, unit-norm rows.
    """
    n_el = num_rows * oversample_el
    n_az = num_cols * oversample_az

    rows = np.arange(num_rows)[:, None]          # [R,1]
    cols = np.arange(num_cols)[:, None]          # [C,1]

    el_idx = np.arange(n_el)[None, :]            # [1,n_el]
    az_idx = np.arange(n_az)[None, :]            # [1,n_az]

    a_el = np.exp(2j * np.pi * rows * el_idx / n_el) / np.sqrt(num_rows)
    a_az = np.exp(2j * np.pi * cols * az_idx / n_az) / np.sqrt(num_cols)

    beams = []
    for e in range(n_el):
        for a in range(n_az):
            # kron gives element index = row*num_cols + col
            beams.append(np.kron(a_el[:, e], a_az[:, a]))
    W = np.stack(beams, axis=0)
    return torch.tensor(W, dtype=torch.complex64)


def set_b_indices(num_beams: int, fraction: float, seed: int) -> np.ndarray:
    """Sparse measured subset (Set B) drawn from the full codebook (Set A).

    Fixed per configuration, not per realization: Set B is an RRC-configured
    measurement set, not something that is redrawn every slot.
    """
    rng = np.random.default_rng(seed)
    k = max(2, int(round(fraction * num_beams)))
    return np.sort(rng.choice(num_beams, size=k, replace=False))


# ---------------------------------------------------------------------------
# Shared channel
# ---------------------------------------------------------------------------

class SharedChannel:
    """Generates and caches the single channel realization all pipelines share."""

    def __init__(self, cfg: ChannelConfig, exp: ExperimentConfig,
                 device: str = "cpu"):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        sionna.phy.config.device = device

        bs_pol = cfg.bs_polarization
        ut_pol = cfg.ut_polarization

        self.bs_array = AntennaArray(
            num_rows=cfg.bs_num_rows,
            num_cols=cfg.bs_num_cols,
            polarization=bs_pol,
            polarization_type="V" if bs_pol == "single" else "cross",
            antenna_pattern="38.901",
            carrier_frequency=cfg.carrier_frequency,
        )
        self.ut_array = AntennaArray(
            num_rows=cfg.ut_num_rows,
            num_cols=cfg.ut_num_cols,
            polarization=ut_pol,
            polarization_type="V" if ut_pol == "single" else "cross",
            antenna_pattern="omni",
            carrier_frequency=cfg.carrier_frequency,
        )

        self.frequencies = subcarrier_frequencies(
            cfg.num_subcarriers, cfg.subcarrier_spacing
        )
        if torch.is_tensor(self.frequencies):
            self.frequencies = self.frequencies.to(device)

        self.codebook = dft_codebook(cfg.bs_num_rows, cfg.bs_num_cols).to(device)
        self.num_beams = self.codebook.shape[0]
        self.set_b = set_b_indices(self.num_beams, exp.setB_fraction,
                                   exp.base_seed)

    # -- caching ------------------------------------------------------------

    def _cache_path(self, seed: int) -> str:
        """Cache key covering the FULL channel configuration.

        An earlier version keyed only on name/model/seed/length, so changing
        speed, delay spread, carrier frequency or array geometry silently
        served stale channels. Hash every field that affects H, so a config
        change simply misses the cache instead of quietly reusing it.
        """
        from dataclasses import asdict
        cfg_key = repr(sorted(asdict(self.cfg).items()))
        key = f"{cfg_key}|{seed}|{self.exp.episode_slots}"
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return os.path.join(self.exp.cache_dir, f"{self.cfg.name}_{h}.pt")

    def cache_size_gb(self) -> float:
        """Total size of the realization cache on disk, in GB."""
        d = self.exp.cache_dir
        if not os.path.isdir(d):
            return 0.0
        tot = sum(os.path.getsize(os.path.join(d, f))
                  for f in os.listdir(d) if f.endswith(".pt"))
        return tot / 1e9

    def realize(self, seed: int, use_cache: bool = True) -> torch.Tensor:
        """Return H for one realization.

        Returns
        -------
        H : [T, num_rx_ant, num_tx_ant, num_subcarriers] complex64
        """
        path = self._cache_path(seed)
        if use_cache and os.path.exists(path):
            return torch.load(path, map_location=self.device)

        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))

        cdl = CDL(
            model=self.cfg.cdl_model,
            delay_spread=self.cfg.delay_spread,
            carrier_frequency=self.cfg.carrier_frequency,
            ut_array=self.ut_array,
            bs_array=self.bs_array,
            direction="downlink",
            min_speed=self.cfg.min_speed,
            max_speed=self.cfg.max_speed,
            **self._orientation_kwargs(seed),
        )

        a, tau = cdl(
            batch_size=1,
            num_time_steps=self.exp.episode_slots,
            sampling_frequency=self.cfg.sampling_frequency,
        )
        # a: [1, 1, num_rx_ant, 1, num_tx_ant, num_paths, T]
        H = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
        # H: [1, 1, num_rx_ant, 1, num_tx_ant, T, num_subcarriers]
        H = H[0, 0, :, 0]                       # [num_rx_ant, num_tx_ant, T, Nf]
        H = H.permute(2, 0, 1, 3).contiguous()  # [T, num_rx_ant, num_tx_ant, Nf]

        if use_cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(H, path)
        return H

    def _orientation_kwargs(self, seed: int) -> dict:
        """Per-realization random orientation.

        Without this the CDL cluster angles are fixed and the best beam is
        near-constant across realizations, which makes beam prediction
        degenerate and removes the headroom the B pipeline needs.
        """
        if not self.cfg.randomize_orientation:
            return {}
        rng = np.random.default_rng(seed)
        # #CHECK: confirm expected shape is [batch_size, 3] and units are
        # radians (alpha=bearing, beta=downtilt, gamma=slant) on first run.
        ut_or = torch.tensor(
            [[rng.uniform(-np.pi, np.pi), 0.0, 0.0]], dtype=torch.float32
        )
        return {"ut_orientation": ut_or}

    # -- measurement operator ----------------------------------------------

    def beamformed_gain(self, H: torch.Tensor,
                        beam_idx: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Post-beamforming received power per slot, in dB.

        Implements the measurement side of m_t = M(H_t, B_t) + n_t for an
        RSRP-style observation: apply TX beam, combine coherently across the
        (small) UE array, average over subcarriers.

        Parameters
        ----------
        H        : [T, Nr, Nt, Nf]
        beam_idx : indices into the codebook (the configured set B_t)

        Returns
        -------
        rsrp_db : [T, len(beam_idx)]
        """
        if isinstance(beam_idx, np.ndarray):
            beam_idx = torch.tensor(beam_idx, device=H.device)
        beam_idx = beam_idx.to(H.device)
        W = self.codebook[beam_idx].to(H.device)        # [B, Nt]

        # [T, Nr, Nf, Nt] @ [Nt, B] -> [T, Nr, Nf, B]
        Hp = H.permute(0, 1, 3, 2)
        y = torch.matmul(Hp, W.conj().transpose(0, 1))

        power = (y.abs() ** 2).sum(dim=1).mean(dim=1)   # sum RX ant, mean subc
        return 10.0 * torch.log10(power + 1e-12)        # [T, B]

    def measure(self, H: torch.Tensor, beam_idx, seed: int) -> torch.Tensor:
        """Noisy measurement m_t over the configured resource set B_t."""
        clean = self.beamformed_gain(H, beam_idx)
        g = torch.Generator(device=clean.device).manual_seed(seed)
        n = torch.randn(clean.shape, generator=g, device=clean.device)
        return clean + self.exp.meas_noise_std_db * n

    def best_beam(self, H: torch.Tensor) -> torch.Tensor:
        """Ground-truth best beam over the full codebook (Set A), per slot."""
        all_idx = torch.arange(self.num_beams, device=H.device)
        return self.beamformed_gain(H, all_idx).argmax(dim=1)


# ---------------------------------------------------------------------------
# Sanity checks -- run these before trusting anything downstream
# ---------------------------------------------------------------------------

def sanity_check_codebook(chan: SharedChannel, seed: int = 0) -> dict:
    """Fails loudly if the codebook and Sionna's element ordering disagree.

    If the assumed row-major ordering is wrong, the best DFT beam will not
    meaningfully beat a random beam and the gap below will be near zero.
    A healthy FR2 setup should show many dB of separation.
    """
    H = chan.realize(seed, use_cache=False)
    all_idx = torch.arange(chan.num_beams)
    g = chan.beamformed_gain(H, all_idx)            # [T, num_beams]

    best = g.max(dim=1).values.mean().item()
    mean = g.mean().item()
    worst = g.min(dim=1).values.mean().item()

    best_idx = g.argmax(dim=1)
    switch_rate = (best_idx[1:] != best_idx[:-1]).float().mean().item()

    out = {
        "best_minus_mean_db": best - mean,
        "best_minus_worst_db": best - worst,
        "num_distinct_best_beams": int(best_idx.unique().numel()),
        "best_beam_switch_rate": switch_rate,
    }
    if out["best_minus_mean_db"] < 3.0:
        out["WARNING"] = (
            "Beam selection gives <3 dB over the codebook mean. Element "
            "ordering is probably mismatched, or the array is too small."
        )
    return out


def sanity_check_diversity(chan: SharedChannel, n: int = 32) -> dict:
    """Confirms randomized orientation actually produces beam diversity.

    If the best beam is the same index across realizations, beam prediction
    is degenerate and the B pipeline has no headroom to degrade.
    """
    bests = []
    for s in range(n):
        H = chan.realize(chan.exp.base_seed + s, use_cache=False)
        bests.append(chan.best_beam(H)[0].item())
    uniq = len(set(bests))
    out = {"distinct_best_beams_across_realizations": uniq,
           "num_realizations": n,
           "num_beams": chan.num_beams}
    if uniq < max(4, n // 8):
        out["WARNING"] = (
            "Best beam is nearly constant across realizations. Orientation "
            "randomization is not taking effect -- check the "
            "ut_orientation shape/units."
        )
    return out


if __name__ == "__main__":
    from config import FR2, DEFAULT_EXPERIMENT

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    chan = SharedChannel(FR2, DEFAULT_EXPERIMENT, device=dev)
    print(f"device: {dev} | codebook beams: {chan.num_beams} "
          f"| set B size: {len(chan.set_b)}")
    print(sanity_check_codebook(chan))
    print(sanity_check_diversity(chan, n=16))
