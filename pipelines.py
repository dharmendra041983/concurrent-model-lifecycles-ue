"""
The three pipelines (Sec. IV-B) and the utility model (Sec. IV-D).

Signal chain, which *is* the coupling structure:

    H_t  --[B: pick beam b_t]-->  h_t = H_t w_{b_t}          (Nr x Nsb)
         --[P: predict t+k]---->  h_hat_{t+k}
         --[C: compress]------->  bits -> h_rec at the gNB
         --[link adaptation]--->  MCS -> throughput

  B -> P   because the CSI the predictor sees is the CSI on the selected beam.
  P -> C   because the encoder input is the predicted CSI when P is active
           and the measured CSI otherwise.

Utility is effective throughput under link adaptation with outage: the gNB
picks an MCS from the *reported* SNR and the transmission fails if the *true*
SNR falls short. This is deliberately one primary metric. All three pipelines
degrade it through different routes -- B through the SNR level itself, P
through staleness under Doppler, C through quantization error -- which is what
makes the interaction term meaningful rather than an artifact of aggregation.

Note the asymmetry in the beam fallback: exhaustive sweep is MORE accurate
than the model, but costs measurement overhead. Its cost is therefore an
overhead cost, not an accuracy cost, and it makes the downstream pipelines'
jobs EASIER. This is the concrete mechanism by which Delta can come out
negative, and it is modelled explicitly rather than assumed away.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import ChannelConfig, ExperimentConfig, PipelineConfig


# ---------------------------------------------------------------------------
# 5G-like MCS table: (required post-equalization SNR [dB], spectral efficiency)
# ---------------------------------------------------------------------------
MCS_TABLE = [
    (-6.0, 0.1523), (-4.0, 0.2344), (-2.0, 0.3770), (0.0, 0.6016),
    (2.0, 0.8770), (4.0, 1.1758), (6.0, 1.4766), (8.0, 1.9141),
    (10.0, 2.4063), (12.0, 2.7305), (14.0, 3.3223), (16.0, 3.9023),
    (18.0, 4.5234), (20.0, 5.1152), (22.0, 5.5547),
]
MCS_SNR = torch.tensor([m[0] for m in MCS_TABLE])
MCS_SE = torch.tensor([m[1] for m in MCS_TABLE])


def select_mcs(snr_db: torch.Tensor, margin_db: float) -> torch.Tensor:
    """Highest MCS whose required SNR is met by the reported SNR minus margin.

    Returns the index; -1 means no MCS is supportable (outage by design).
    """
    eff = snr_db.unsqueeze(-1) - margin_db
    ok = eff >= MCS_SNR.to(snr_db.device)
    idx = ok.to(torch.int64).sum(dim=-1) - 1
    return idx


def throughput(reported_snr_db: torch.Tensor,
               true_snr_db: torch.Tensor,
               margin_db: float) -> torch.Tensor:
    """Per-slot delivered spectral efficiency [bit/s/Hz].

    The gNB chooses an MCS from the report; the transmission succeeds only if
    the true SNR supports it. Mis-prediction is punished asymmetrically --
    over-reporting causes outage, under-reporting causes lost rate -- which is
    the correct behaviour and the reason a symmetric NMSE metric would have
    hidden the effect.
    """
    idx = select_mcs(reported_snr_db, margin_db)
    valid = idx >= 0
    idx_c = idx.clamp(min=0)
    req = MCS_SNR.to(idx.device)[idx_c]
    se = MCS_SE.to(idx.device)[idx_c]
    success = (true_snr_db >= req) & valid
    return torch.where(success, se, torch.zeros_like(se))



def throughput_olla(reported_snr_db, true_snr_db, target_bler: float,
                    step_db: float, init_margin_db: float):
    """Delivered spectral efficiency under outer-loop link adaptation.

    The margin is adapted sequentially: raised after a failure, lowered
    after a success, with the ratio of the two steps set so the loop
    converges to `target_bler`. Every arm therefore operates at the same
    error rate, and an arm can only win by reporting SNR more accurately --
    not by reporting it optimistically.

    Returns (tput [N], achieved_bler, mean_margin_db).
    """
    rep = reported_snr_db.detach().cpu().numpy()
    tru = true_snr_db.detach().cpu().numpy()
    snr_tab = MCS_SNR.numpy()
    se_tab = MCS_SE.numpy()

    d_up = step_db
    d_dn = step_db * target_bler / (1.0 - target_bler)

    n = rep.shape[0]
    out = np.zeros(n, dtype=float)
    margin = float(init_margin_db)
    fails = 0
    msum = 0.0
    for i in range(n):
        eff = rep[i] - margin
        idx = int((snr_tab <= eff).sum()) - 1
        msum += margin
        if idx < 0:
            out[i] = 0.0
            margin = max(-10.0, margin - d_dn)
            continue
        if tru[i] >= snr_tab[idx]:
            out[i] = se_tab[idx]
            margin = max(-10.0, margin - d_dn)
        else:
            out[i] = 0.0
            fails += 1
            margin = min(20.0, margin + d_up)
    return (torch.tensor(out, dtype=torch.float32,
                         device=reported_snr_db.device),
            fails / max(1, n), msum / max(1, n))


# ---------------------------------------------------------------------------
# CSI representation helpers
# ---------------------------------------------------------------------------

def to_subbands(h: torch.Tensor, num_subbands: int) -> torch.Tensor:
    """[..., Nr, Nf] complex -> [..., Nr, S] complex, by subband averaging."""
    *lead, nr, nf = h.shape
    per = nf // num_subbands
    h = h[..., : per * num_subbands]
    return h.reshape(*lead, nr, num_subbands, per).mean(dim=-1)


def complex_to_real(x: torch.Tensor) -> torch.Tensor:
    """[..., D] complex -> [..., 2D] real, real parts then imaginary."""
    return torch.cat([x.real, x.imag], dim=-1)


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.complex(x[..., :d], x[..., d:])


def snr_from_csi(h: torch.Tensor, snr_ref_db: float) -> torch.Tensor:
    """Post-MRC SNR [dB] from an effective channel [..., Nr, S]."""
    g = (h.abs() ** 2).sum(dim=-2).mean(dim=-1)     # sum RX ant, mean subband
    return snr_ref_db + 10.0 * torch.log10(g + 1e-12)


def hold_beam(sel: torch.Tensor, period: int) -> torch.Tensor:
    """Hold the selected beam across a sweep period.

    Reselecting every slot makes the effective channel h_eff = H w_b
    discontinuous wherever the best beam changes, which destroys the temporal
    correlation the CSI predictor depends on. Real systems update the beam on
    a sweep periodicity; holding here is both more faithful and what makes the
    P pipeline a well-posed problem at all.
    """
    T = sel.shape[0]
    idx = (torch.arange(T, device=sel.device) // period) * period
    return sel[idx]


# ---------------------------------------------------------------------------
# Pipeline B: beam management
# ---------------------------------------------------------------------------

class BeamModel(nn.Module):
    """Set B measurements -> logits over Set A (Rel-18 BM-Case1, spatial)."""

    def __init__(self, num_measured: int, num_beams: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_measured, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_beams),
        )

    def forward(self, m_db: torch.Tensor) -> torch.Tensor:
        # Normalize per sample: the model should key on the shape of the
        # measured profile, not on absolute pathloss.
        x = m_db - m_db.mean(dim=-1, keepdim=True)
        return self.net(x)


class BeamPipeline:
    """Active = learned prediction from Set B. Fallback = exhaustive sweep."""

    def __init__(self, model: BeamModel, set_b: np.ndarray, num_beams: int,
                 pcfg: PipelineConfig):
        self.model = model
        self.set_b = set_b
        self.num_beams = num_beams
        self.pcfg = pcfg

    def select(self, meas_setB_db: torch.Tensor, true_best: torch.Tensor,
               fallback: bool) -> torch.Tensor:
        """Return the selected beam index per slot. [T]"""
        if fallback:
            # Exhaustive sweep measures everything, so it is correct. Its cost
            # appears as overhead, not error.
            return true_best
        with torch.no_grad():
            return self.model(meas_setB_db).argmax(dim=-1)

    def overhead_fraction(self, fallback: bool) -> float:
        n = self.num_beams if fallback else len(self.set_b)
        cost = n * self.pcfg.slots_per_beam_measurement
        return min(0.95, cost / self.pcfg.sweep_period_slots)


# ---------------------------------------------------------------------------
# Pipeline P: CSI prediction
# ---------------------------------------------------------------------------

class CSIPredictor(nn.Module):
    """GRU over a window of past effective-channel reports; predict t+k."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq)
        return self.head(out[:, -1])


class PredictionPipeline:
    """Active = learned extrapolation. Fallback = most recent estimate."""

    def __init__(self, model: CSIPredictor, pcfg: PipelineConfig):
        self.model = model
        self.pcfg = pcfg

    def predict(self, hist: torch.Tensor, fallback: bool) -> torch.Tensor:
        """hist: [N, W, 2D] real. Returns [N, 2D]."""
        if fallback:
            return hist[:, -1]          # hold last measurement (stale)
        with torch.no_grad():
            return self.model(hist)


# ---------------------------------------------------------------------------
# Pipeline C: CSI compression (two-sided)
# ---------------------------------------------------------------------------

class STEQuantizer(torch.autograd.Function):
    """Uniform quantizer with a straight-through gradient estimator."""

    @staticmethod
    def forward(ctx, x, levels):
        return torch.round(x * (levels - 1)) / (levels - 1)

    @staticmethod
    def backward(ctx, g):
        return g, None


class CSIAutoencoder(nn.Module):
    """UE-side encoder + gNB-side decoder with a quantized latent."""

    def __init__(self, dim: int, latent: int, bits: int, hidden: int = 256):
        super().__init__()
        self.levels = 2 ** bits
        self.enc = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.Sigmoid(),   # bounded for quantizer
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        z = self.enc(x / scale)
        zq = STEQuantizer.apply(z, self.levels)
        return self.dec(zq) * scale


class CompressionPipeline:
    """Active = learned autoencoder. Fallback = wideband codebook report.

    The fallback must be a genuinely LOWER-RATE report, or it is not a
    fallback at all. An earlier version quantized every subband at 2 bits
    (48 dims x 2 = 96 bits) against the autoencoder's 16 latents x 4 bits
    (64 bits) -- the "degraded" path spent 50% more feedback than the model
    it replaced, and unsurprisingly cost nothing.

    A Type-I-style report is WIDEBAND: one coarse quantized value per
    antenna across the whole band, broadcast back over subbands. That is
    2*Nr real dims at `fallback_bits` each -- a small fraction of the
    learned report, which is what makes it a real degradation.
    """

    def __init__(self, model: CSIAutoencoder, pcfg: PipelineConfig,
                 num_rx_ant: int, num_subbands: int,
                 fallback_bits: int = 3):
        self.model = model
        self.pcfg = pcfg
        self.nr = num_rx_ant
        self.ns = num_subbands
        self.fallback_bits = fallback_bits

    @property
    def learned_bits(self) -> int:
        return self.pcfg.latent_dim * self.pcfg.bits_per_latent

    @property
    def fallback_report_bits(self) -> int:
        return 2 * self.nr * self.fallback_bits

    def report(self, x: torch.Tensor, fallback: bool) -> torch.Tensor:
        if not fallback:
            with torch.no_grad():
                return self.model(x)

        # x is [N, 2D] with D = nr*ns: real parts then imaginary parts.
        N = x.shape[0]
        D = self.nr * self.ns
        re = x[:, :D].reshape(N, self.nr, self.ns)
        im = x[:, D:].reshape(N, self.nr, self.ns)

        # Wideband: collapse the subband axis before quantizing.
        re_w = re.mean(dim=2, keepdim=True)
        im_w = im.mean(dim=2, keepdim=True)

        wb = torch.cat([re_w, im_w], dim=1)                    # [N, 2nr, 1]
        scale = wb.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        lv = 2 ** self.fallback_bits
        q = torch.round(((wb / scale + 1.0) / 2.0) * (lv - 1)) / (lv - 1)
        wb = (q * 2.0 - 1.0) * scale

        re_q = wb[:, :self.nr].expand(N, self.nr, self.ns)
        im_q = wb[:, self.nr:].expand(N, self.nr, self.ns)
        return torch.cat([re_q.reshape(N, D), im_q.reshape(N, D)], dim=-1)


# ---------------------------------------------------------------------------
# Joint rollout
# ---------------------------------------------------------------------------

class ConcurrentSystem:
    """Runs all three pipelines over one realization with a forced fallback set."""

    def __init__(self, chan, beam: BeamPipeline, pred: PredictionPipeline,
                 comp: CompressionPipeline, ccfg: ChannelConfig,
                 ecfg: ExperimentConfig, pcfg: PipelineConfig):
        self.chan = chan
        self.beam, self.pred, self.comp = beam, pred, comp
        self.ccfg, self.ecfg, self.pcfg = ccfg, ecfg, pcfg

    def rollout(self, H: torch.Tensor, seed: int,
                fallback_set: set[str]) -> dict:
        """
        Parameters
        ----------
        H            : [T, Nr, Nt, Nf] complex
        fallback_set : subset of {"B","P","C"} forced into fallback

        Returns
        -------
        dict with per-slot throughput and the diagnostics the LCM monitors read.
        """
        pcfg = self.pcfg
        T = H.shape[0]

        fb_B = "B" in fallback_set
        fb_P = "P" in fallback_set
        fb_C = "C" in fallback_set

        # --- B: beam selection -------------------------------------------
        meas = self.chan.measure(H, self.chan.set_b, seed)          # [T, |B|]
        true_best = self.chan.best_beam(H)                           # [T]
        sel = self.beam.select(meas, true_best, fb_B)                # [T]
        sel = hold_beam(sel, pcfg.sweep_period_slots)
        overhead = self.beam.overhead_fraction(fb_B)

        # Beamforming gain loss is the quantity that actually reaches utility.
        # Top-1 exact match punishes a model that picks a near-optimal
        # neighbouring beam, which is not a real degradation.
        all_idx = torch.arange(self.chan.num_beams, device=H.device)
        g_all = self.chan.beamformed_gain(H, all_idx)                # [T, B]
        g_sel = g_all[torch.arange(T, device=H.device), sel]
        g_best = g_all.max(dim=1).values
        beam_gain_loss_db = (g_best - g_sel).mean().item()
        beam_acc = (sel == true_best).float().mean().item()

        # --- effective channel on the SELECTED beam (coupling B -> P) -----
        W = self.chan.codebook[sel]                                  # [T, Nt]
        h_eff = torch.einsum("tnmf,tm->tnf", H, W.conj())            # [T, Nr, Nf]
        h_sb = to_subbands(h_eff, pcfg.csi_subbands)                 # [T, Nr, S]

        D = h_sb.shape[1] * h_sb.shape[2]
        flat = complex_to_real(h_sb.reshape(T, D))                   # [T, 2D]

        # --- P: prediction (coupling P -> C) -------------------------------
        W_win, k = pcfg.pred_window, pcfg.feedback_delay
        starts = torch.arange(0, T - W_win - k)
        hist = torch.stack([flat[s: s + W_win] for s in starts])     # [N,W,2D]
        target_idx = starts + W_win + k
        pred_in = self.pred.predict(hist, fb_P)                      # [N, 2D]

        # --- C: compression ------------------------------------------------
        rec = self.comp.report(pred_in, fb_C)                        # [N, 2D]

        # --- utility -------------------------------------------------------
        h_true = real_to_complex(flat[target_idx]).reshape(-1, h_sb.shape[1],
                                                           h_sb.shape[2])
        h_rec = real_to_complex(rec).reshape(-1, h_sb.shape[1], h_sb.shape[2])

        true_snr = snr_from_csi(h_true, pcfg.snr_ref_db)
        rep_snr = snr_from_csi(h_rec, pcfg.snr_ref_db)

        tput_raw, bler, mean_margin = throughput_olla(
            rep_snr, true_snr, pcfg.target_bler, pcfg.olla_step_db,
            pcfg.la_margin_db)
        tput = tput_raw * (1.0 - overhead)

        nmse = ((h_rec - h_true).abs() ** 2).sum() / (h_true.abs() ** 2).sum()

        # ---- per-slot monitoring statistics (Sec. V) ---------------------
        # Each must be computable AT THE UE. Noted where that is a stretch.
        #
        # s_B: measured-vs-selected RSRP gap over the configured set. The UE
        #      measures Set B and the serving beam, so this is observable.
        sB = (meas.max(dim=1).values - g_sel)[target_idx]
        #
        # s_P: running prediction error against the CSI subsequently
        #      measured. Observable at the UE one feedback delay later.
        tgt = flat[target_idx]
        sP = (((pred_in - tgt) ** 2).sum(dim=1) /
              (tgt ** 2).sum(dim=1).clamp(min=1e-9))
        #
        # s_C: encoder-side reconstruction error against a LOCAL decoder
        #      replica. This is the weakest observability claim in the paper
        #      -- a UE holding a copy of the paired decoder can compute it,
        #      but under Rel-20 pairing that copy is not guaranteed current.
        #      State this limitation explicitly rather than gloss it.
        sC = (((rec - pred_in) ** 2).sum(dim=1) /
              (pred_in ** 2).sum(dim=1).clamp(min=1e-9))

        return {
            "throughput": tput,                       # [N] per-slot SE
            "mean_throughput": tput.mean().item(),
            # Pre-overhead throughput. Overhead composes MULTIPLICATIVELY
            # with every other cost source, so a subset that changes overhead
            # produces a nonzero Delta by arithmetic alone, with no lifecycle
            # coupling involved. Delta computed on this quantity isolates the
            # coupling; the overhead term is reported separately.
            "mean_throughput_raw": tput_raw.mean().item(),
            "tput_slot": tput,
            "tput_slot_raw": tput_raw,
            "s_B": sB.detach(),
            "s_P": sP.detach(),
            "s_C": sC.detach(),
            # Encoder input, so alternative quantizers can be evaluated on
            # exactly the distribution the learned encoder actually sees.
            "x_enc_in": pred_in.detach(),
            "x_target": flat[target_idx].detach(),
            "beam_accuracy": beam_acc,
            "beam_gain_loss_db": beam_gain_loss_db,
            "overhead": overhead,
            "csi_nmse": nmse.item(),
            "true_snr_db": true_snr,
            "reported_snr_db": rep_snr,
            "outage_rate": (tput == 0).float().mean().item(),
            "achieved_bler": bler,
            "olla_margin_db": mean_margin,
        }
