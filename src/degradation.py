"""
Degradation drivers (Sec. VII-B).

The paper's premise is SHARED CAUSATION: one physical event perturbs all
three pipelines at once. A driver that only degrades one pipeline would test
nothing, and uniform attenuation only scales SNR without changing any input
distribution, so it is nearly useless here.

Blockage is the right driver. It removes energy from a contiguous ANGULAR
SECTOR, which:
  - forces the beam-management pipeline off its serving beam (B),
  - abruptly changes the statistics of the effective channel (P),
  - shifts the encoder's input distribution (C),
all from a single cause, which is exactly the correlated-monitor premise.

Applied post-hoc to a cached realization via a unitary beam-domain
transform, so no channel regeneration is needed and the underlying
realization is preserved for matched comparison.
"""

from __future__ import annotations

import numpy as np
import torch


def orthogonal_dft(num_rows: int, num_cols: int,
                   device="cpu") -> torch.Tensor:
    """Unitary [Nt, Nt] DFT over the array, row-major element ordering.

    Unitary (unlike the oversampled steering codebook), so attenuating in
    this basis and transforming back is exact.
    """
    r = np.arange(num_rows)[:, None]
    c = np.arange(num_cols)[:, None]
    fe = np.exp(2j * np.pi * r * np.arange(num_rows)[None, :] / num_rows)
    fa = np.exp(2j * np.pi * c * np.arange(num_cols)[None, :] / num_cols)
    fe /= np.sqrt(num_rows)
    fa /= np.sqrt(num_cols)
    beams = [np.kron(fe[:, e], fa[:, a])
             for e in range(num_rows) for a in range(num_cols)]
    return torch.tensor(np.stack(beams, 0), dtype=torch.complex64,
                        device=device)


def apply_blockage(H: torch.Tensor, num_rows: int, num_cols: int,
                   start_slot: int, ramp_slots: int, sector_width: int,
                   atten_db: float, sector_center: int | None = None,
                   ) -> tuple[torch.Tensor, dict]:
    """Attenuate an angular sector from `start_slot` onward.

    Parameters
    ----------
    H            : [T, Nr, Nt, Nf] complex
    sector_width : number of contiguous orthogonal beams blocked
    atten_db     : final attenuation (positive dB of loss)
    sector_center: blocked sector centre; if None, centred on the beam
                   carrying most energy before the blockage starts, so the
                   event actually blocks the serving direction.

    Returns (H_blocked, info).
    """
    T, Nr, Nt, Nf = H.shape
    dev = H.device
    F = orthogonal_dft(num_rows, num_cols, device=dev)          # [Nt, Nt]

    Hb = torch.einsum("tnmf,bm->tnbf", H, F.conj())             # beam domain

    if sector_center is None:
        pre = Hb[:max(1, start_slot)]
        power = (pre.abs() ** 2).sum(dim=(0, 1, 3))             # [Nt]
        sector_center = int(power.argmax().item())

    idx = [(sector_center + d - sector_width // 2) % Nt
           for d in range(sector_width)]

    prof = torch.zeros(T, device=dev)
    if ramp_slots > 0:
        end = min(T, start_slot + ramp_slots)
        if end > start_slot:
            prof[start_slot:end] = torch.linspace(
                0.0, atten_db, end - start_slot, device=dev)
        prof[end:] = atten_db
    else:
        prof[start_slot:] = atten_db

    gain = (10.0 ** (-prof / 20.0)).view(T, 1, 1, 1)
    mask = torch.ones(Nt, device=dev)
    mask[idx] = 0.0
    mask = mask.view(1, 1, Nt, 1)
    # Blocked beams scale with the ramp; others untouched.
    Hb = Hb * (mask + (1.0 - mask) * gain)

    Hout = torch.einsum("tnbf,bm->tnmf", Hb, F)
    return Hout, {"sector_center": sector_center, "sector": idx,
                  "start_slot": start_slot, "atten_db": atten_db}
