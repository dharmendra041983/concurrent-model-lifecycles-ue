"""
Arbitration under pairing mismatch (Sec. VII positive control, part 2).

The blockage regime refuted the arbiter: blind delay beat dependency-aware
arbitration because C's fallback is harmful whenever taken, so suppressing it
indiscriminately was optimal. pairing.py established the missing regime --
under encoder/decoder mismatch the compression fallback GAINS 0.389 bit/s/Hz
when its monitor fires, so falling back is the right action and blind
suppression should now be costly.

PRE-REGISTERED PREDICTION (written before this was run): a mismatched
decoder is not preceded by any upstream state change, so the chain arbiter
has no reason to hold C's request while blind delay holds it regardless.
Chain should beat blind here, reversing the blockage result. If it does not,
the arbiter is refuted in both regimes and that is the final answer.

TWO DESIGN POINTS:

  1. The mismatch is introduced MID-EPISODE, at the same slot as blockage.
     A mismatch present from slot zero saturates the monitor and leaves
     nothing to discriminate; the arbiter's behaviour at the transition is
     the quantity of interest.

  2. Severity is SWEPT, not chosen. Decoder weights are interpolated between
     the matched decoder A and the mismatched decoder B, so alpha=0 is a
     healthy pair and alpha=1 is total mismatch. Real pairing failures are
     often partial -- a stale decoder version rather than a random one -- and
     a sweep shows where the crossover sits instead of asserting a point.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import numpy as np
import torch

from config import FR1, FR2, DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
from experiment import load_system, SUBSETS
from induced import nominal_statistics, KEYS, _key
from lcm import LCMConfig, calibrate_threshold
from arbiter import run_arbitrated
from pipelines import (CSIAutoencoder, STEQuantizer, snr_from_csi,
                       real_to_complex, throughput_olla)
from stats import paired_bootstrap_ci
from check_encoder_ood import wideband_codebook


def blend_decoder(model_a: CSIAutoencoder, model_b: CSIAutoencoder,
                  alpha: float) -> CSIAutoencoder:
    """Decoder weights interpolated A->B. alpha=0 matched, alpha=1 mismatched."""
    blended = copy.deepcopy(model_a)
    sa, sb = model_a.dec.state_dict(), model_b.dec.state_dict()
    blended.dec.load_state_dict(
        {k: (1.0 - alpha) * sa[k] + alpha * sb[k] for k in sa})
    blended.eval()
    return blended


def rollout_with_mismatch(sys, comp_blend, H, seed, S, nr, ns, pcfg,
                          onset_idx):
    """Rollout for one state vector with the decoder swapped from onset_idx.

    Only the compression path changes, so beam and prediction outputs are
    taken unmodified from the base rollout.
    """
    out = sys.rollout(H, seed, S)
    x = out["x_enc_in"]
    x_true = out["x_target"]
    n = x.shape[0]

    fb = "C" in S
    with torch.no_grad():
        if fb:
            rec = wideband_codebook(x, nr, ns, 3)
        else:
            rec_pre = sys.comp.model(x)
            scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
            z = sys.comp.model.enc(x / scale)
            zq = STEQuantizer.apply(z, sys.comp.model.levels)
            rec_post = comp_blend.dec(zq) * scale
            m = (torch.arange(n, device=x.device) >= onset_idx).unsqueeze(1)
            rec = torch.where(m, rec_post, rec_pre)

    h_rec = real_to_complex(rec).reshape(-1, nr, ns)
    h_true = real_to_complex(x_true).reshape(-1, nr, ns)
    rep = snr_from_csi(h_rec, pcfg.snr_ref_db)
    tru = snr_from_csi(h_true, pcfg.snr_ref_db)
    t_raw, _, _ = throughput_olla(rep, tru, pcfg.target_bler,
                                  pcfg.olla_step_db, pcfg.la_margin_db)

    sC = (((rec - x) ** 2).sum(dim=1)
          / (x ** 2).sum(dim=1).clamp(min=1e-9))
    o = dict(out)
    o["tput_slot"] = t_raw * (1.0 - out["overhead"])
    o["s_C"] = sC.detach()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["FR2", "FR1"], default="FR2")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--n-nominal", type=int, default=64)
    ap.add_argument("--target-fa", type=float, default=0.05)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--holds", type=int, nargs="+", default=[0, 10, 20])
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ccfg = FR2 if args.band == "FR2" else FR1
    ecfg, pcfg = DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    nr, ns = ccfg.num_ut_ant, pcfg.csi_subbands
    dim = 2 * nr * ns

    chan, sys, _ = load_system(ccfg, ecfg, pcfg,
                               f"checkpoints/models_{ccfg.name}.pt", dev)
    ckb = torch.load(f"checkpoints/models_{ccfg.name}_pairB.pt",
                     map_location=dev, weights_only=False)
    comp_b = CSIAutoencoder(dim, pcfg.latent_dim,
                            pcfg.bits_per_latent).to(dev)
    comp_b.load_state_dict(ckb["comp_b"]); comp_b.eval()

    cal = (ecfg.base_seed + pcfg.train_realizations
           + pcfg.val_realizations + 10000)
    nom = nominal_statistics(chan, sys,
                             [cal + i for i in range(args.n_nominal)])
    taus = {k: calibrate_threshold(nom[k], args.target_fa) for k in KEYS}

    base = (ecfg.base_seed + pcfg.train_realizations
            + pcfg.val_realizations + 20000)
    seeds = [base + i for i in range(args.n)]
    onset = ecfg.episode_slots // 3 - (pcfg.pred_window + pcfg.feedback_delay)

    free_cfg = {k: LCMConfig(tau=taus[k]) for k in KEYS}
    blends = {a: blend_decoder(sys.comp.model, comp_b, a) for a in args.alphas}

    res = {"band": ccfg.name, "n": args.n, "window_Wa": args.window,
           "onset_slot": onset, "by_alpha": {}}

    print(f"band {ccfg.name} | mismatch onset at report index {onset} | "
          f"n={args.n}")
    print(f"\n{'alpha':>6} {'H':>4} {'chain':>9} {'blind':>9} "
          f"{'chain-blind':>12} {'CI95':>22} {'C_fb':>6}")

    for a in args.alphas:
        acc = {h: {m: {"t": [], "e": []} for m in ("chain", "blind")}
               for h in args.holds}
        for s in seeds:
            H = chan.realize(s)
            arms = {_key(S): rollout_with_mismatch(
                sys, blends[a], H, s, S, nr, ns, pcfg, onset)
                for S in SUBSETS}
            n_slots = int(arms["none"]["tput_slot"].shape[0])
            for h in args.holds:
                for mode in ("chain", "blind"):
                    r = run_arbitrated(arms, free_cfg, n_slots, h,
                                       args.window, mode)
                    acc[h][mode]["t"].append(r["throughput"])
                    acc[h][mode]["e"].append(r["fallback_entries"]["C"])

        res["by_alpha"][a] = {}
        for h in args.holds:
            tc = np.asarray(acc[h]["chain"]["t"], dtype=float)
            tb = np.asarray(acc[h]["blind"]["t"], dtype=float)
            t0 = np.asarray(acc[0]["chain"]["t"], dtype=float)
            lo, hi, p = paired_bootstrap_ci(tc - tb)
            ci = f"[{lo:+.4f},{hi:+.4f}]"
            print(f"{a:6.2f} {h:>4} {(tc-t0).mean():+9.4f} "
                  f"{(tb-t0).mean():+9.4f} {(tc-tb).mean():+12.4f} {ci:>22} "
                  f"{np.mean(acc[h]['chain']['e']):6.2f}")
            res["by_alpha"][a][h] = {
                "chain_delta": float((tc - t0).mean()),
                "blind_delta": float((tb - t0).mean()),
                "chain_minus_blind": float((tc - tb).mean()),
                "ci95": [lo, hi], "p": p}
        print()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"arbiter_mismatch_{ccfg.name}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=float)

    h = max(args.holds)
    full = res["by_alpha"][max(args.alphas)][h]
    if full["ci95"][0] > 0:
        print(f"PREDICTION CONFIRMED: at full mismatch, chain beats blind by "
              f"{full['chain_minus_blind']:+.4f}\n(CI excludes zero). "
              f"Arbitration helps where the downstream fallback has positive\n"
              f"conditional value, and hurts where it does not. Report both "
              f"regimes together.")
    elif full["ci95"][1] < 0:
        print(f"PREDICTION REFUTED: blind still beats chain "
              f"({full['chain_minus_blind']:+.4f}) even where\nthe fallback "
              f"is clearly the right action. The arbiter fails in both "
              f"regimes;\nreport that as the final answer.")
    else:
        print("No separation between chain and blind at full mismatch. "
              "Raise --n\nbefore concluding.")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
