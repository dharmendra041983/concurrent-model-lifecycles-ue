"""
Encoder out-of-distribution check (gate for the arbiter redesign).

The arbiter failed because C's fallback is worse than staying active in every
regime, so no trigger policy can make falling back correct. The suspected
cause is an omission in the utility model: CSI feedback overhead is never
charged, so a 12-bit wideband report and a 64-bit learned encoder cost the
same -- nothing. Under that accounting the fallback is strictly dominated by
construction.

A real non-AI CSI report is HIGH rate and distribution-free: expensive in
uplink, but robust when the encoder goes out of distribution. That is the
configuration in which falling back can be the right action, and in which
discrimination can beat blind suppression.

This script tests the premise BEFORE any of that is built. Two questions:

  Q1. Does the learned encoder actually degrade under blockage relative to
      nominal? If it does not, no fallback can win on accuracy and the
      redesign is pointless.

  Q2. At what rate does a distribution-free scalar codebook beat the learned
      encoder under blockage? That rate sets the fallback's overhead cost,
      and therefore whether the tradeoff is interesting or trivial.

If Q1 shows little degradation, STOP: report the negative arbiter result as
robust rather than configuration-specific.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from config import FR1, FR2, DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
from degradation import apply_blockage
from experiment import load_system


def uniform_codebook(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-dimension uniform scalar quantization -- distribution-free."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    lv = 2 ** bits
    q = torch.round(((x / scale + 1.0) / 2.0) * (lv - 1)) / (lv - 1)
    return (q * 2.0 - 1.0) * scale


def wideband_codebook(x: torch.Tensor, nr: int, ns: int,
                      bits: int) -> torch.Tensor:
    """The current fallback: one coarse value per antenna across the band."""
    N, D = x.shape[0], x.shape[1] // 2
    re = x[:, :D].reshape(N, nr, ns).mean(dim=2, keepdim=True)
    im = x[:, D:].reshape(N, nr, ns).mean(dim=2, keepdim=True)
    wb = torch.cat([re, im], dim=1)
    scale = wb.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    lv = 2 ** bits
    q = torch.round(((wb / scale + 1.0) / 2.0) * (lv - 1)) / (lv - 1)
    wb = (q * 2.0 - 1.0) * scale
    re_q = wb[:, :nr].expand(N, nr, ns).reshape(N, D)
    im_q = wb[:, nr:].expand(N, nr, ns).reshape(N, D)
    return torch.cat([re_q, im_q], dim=-1)


def nmse_db(rec: torch.Tensor, ref: torch.Tensor) -> float:
    return float(10 * np.log10(
        (((rec - ref) ** 2).sum() / (ref ** 2).sum()).item() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["FR2", "FR1"], default="FR2")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--atten-db", type=float, default=20.0)
    ap.add_argument("--sector-width", type=int, default=6)
    args = ap.parse_args()

    ccfg = FR2 if args.band == "FR2" else FR1
    ecfg, pcfg = DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    nr, ns = ccfg.num_ut_ant, pcfg.csi_subbands

    chan, sys, _ = load_system(ccfg, ecfg, pcfg,
                               f"checkpoints/models_{ccfg.name}.pt", dev)
    base = ecfg.base_seed + pcfg.train_realizations + pcfg.val_realizations
    seeds = [base + 30000 + i for i in range(args.n)]
    start = ecfg.episode_slots // 3
    ramp = 20
    off = pcfg.pred_window + pcfg.feedback_delay

    learned_bits = pcfg.latent_dim * pcfg.bits_per_latent
    rates = [2, 3, 4, 5]           # bits per real dimension
    dims = 2 * nr * ns

    acc = {"nominal": {}, "post": {}}
    for phase in acc:
        acc[phase] = {"learned": [], "wideband12": [],
                      **{f"uni{b}": [] for b in rates}}

    print(f"band {ccfg.name} | learned report {learned_bits} bits | "
          f"{dims} real dims")
    print(f"running {len(seeds)} realizations, nominal and blocked...")

    for i, s in enumerate(seeds):
        H = chan.realize(s)
        Hb, _ = apply_blockage(H, ccfg.bs_num_rows, ccfg.bs_num_cols,
                               start_slot=start, ramp_slots=ramp,
                               sector_width=args.sector_width,
                               atten_db=args.atten_db)

        for phase, Hx in (("nominal", H), ("post", Hb)):
            out = sys.rollout(Hx, s, set())
            x = out["x_enc_in"]
            n_slots = x.shape[0]
            if phase == "post":
                m = (np.arange(n_slots) + off) >= (start + ramp)
                if not m.any():
                    continue
                x = x[torch.tensor(m, device=x.device)]

            with torch.no_grad():
                rec = sys.comp.model(x)
            acc[phase]["learned"].append(nmse_db(rec, x))
            acc[phase]["wideband12"].append(
                nmse_db(wideband_codebook(x, nr, ns, 3), x))
            for b in rates:
                acc[phase][f"uni{b}"].append(
                    nmse_db(uniform_codebook(x, b), x))

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(seeds)}")

    print(f"\nreconstruction NMSE [dB] of the encoder input\n")
    print(f"{'scheme':>12} {'bits':>6} {'nominal':>9} {'blocked':>9} "
          f"{'degrade':>9}")
    rows = {}
    for name, bits in [("learned", learned_bits), ("wideband12", 2 * nr * 3)] \
            + [(f"uni{b}", dims * b) for b in rates]:
        nom = float(np.mean(acc["nominal"][name]))
        pos = float(np.mean(acc["post"][name]))
        print(f"{name:>12} {bits:>6} {nom:9.2f} {pos:9.2f} {pos-nom:+9.2f}")
        rows[name] = (bits, nom, pos)

    ln_bits, ln_nom, ln_post = rows["learned"]
    print()

    # Q1 -- is the encoder fragile out of distribution?
    degrade = ln_post - ln_nom
    if degrade < 1.0:
        print(f"Q1 FAILS: the learned encoder degrades by only "
              f"{degrade:+.2f} dB under\nblockage. It is not going out of "
              f"distribution, so no fallback can win on\naccuracy and the "
              f"redesign will not rescue the arbiter. Report the negative\n"
              f"result as robust rather than configuration-specific.")
        return
    print(f"Q1 PASSES: the encoder degrades {degrade:+.2f} dB under blockage.")

    # Q2 -- what rate does a distribution-free codebook need to beat it?
    winner = None
    for b in rates:
        if rows[f"uni{b}"][2] < ln_post:
            winner = b
            break
    if winner is None:
        print(f"Q2 FAILS: no tested rate (up to {dims*rates[-1]} bits) beats "
              f"the learned\nencoder under blockage. The encoder is strong "
              f"enough that its fallback\ncannot be worth its overhead.")
        return
    wb_bits, _, wb_post = rows[f"uni{winner}"]
    print(f"Q2 PASSES: {winner} bits/dim ({wb_bits} bits) reaches "
          f"{wb_post:.2f} dB under\nblockage, beating the learned encoder's "
          f"{ln_post:.2f} dB at {ln_bits} bits.")
    print(f"\nThe fallback costs {wb_bits/ln_bits:.1f}x the uplink of the "
          f"learned report and\nbuys {ln_post-wb_post:+.2f} dB when the "
          f"encoder is out of distribution. That is a\nreal tradeoff: "
          f"proceed with charging feedback overhead in the utility.")


if __name__ == "__main__":
    main()
