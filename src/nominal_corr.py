"""
Nominal correlated-firing statistics at larger sample size.

Table~II reports nominal figures from 64 realizations against 256 for the
blocked condition. A 5 percent tail estimated from 64 episodes is the
lightest sample in the paper, and the correlated-firing result is the one it
carries.

The thresholds are NOT recalibrated. They are recomputed from the same 64
calibration realizations used everywhere else, so they are identical to the
ones behind the induced-fallback, conditional and arbiter results; only the
evaluation sample grows. Recalibrating on a larger set would shift every
downstream number in the paper and require rerunning all of it.

The evaluation realizations are disjoint from the calibration set, so this
also removes the mild optimism of estimating a tail on the same episodes
that defined it.
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np
import torch

from config import FR1, FR2, DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
from experiment import load_system
from induced import nominal_statistics, KEYS
from lcm import calibrate_threshold, monitor_correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["FR2", "FR1"], default="FR2")
    ap.add_argument("--n-calib", type=int, default=64,
                    help="must match the value used elsewhere, or the "
                         "thresholds will not be the paper's")
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--target-fa", type=float, default=0.05)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ccfg = FR2 if args.band == "FR2" else FR1
    ecfg, pcfg = DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    chan, sys, _ = load_system(ccfg, ecfg, pcfg,
                               f"checkpoints/models_{ccfg.name}.pt", dev)
    base = ecfg.base_seed + pcfg.train_realizations + pcfg.val_realizations

    cal_seeds = [base + 10000 + i for i in range(args.n_calib)]
    print(f"band {ccfg.name} | calibrating on {len(cal_seeds)} realizations "
          f"(unchanged)...")
    cal = nominal_statistics(chan, sys, cal_seeds)
    taus = {k: calibrate_threshold(cal[k], args.target_fa) for k in KEYS}
    for k in KEYS:
        print(f"  tau_{k} = {taus[k]:.6f}")

    # Disjoint evaluation block, well clear of the blocked-experiment seeds.
    ev_seeds = [base + 40000 + i for i in range(args.n_eval)]
    print(f"\nevaluating firing statistics on {len(ev_seeds)} disjoint "
          f"nominal realizations...")
    ev = nominal_statistics(chan, sys, ev_seeds)

    cal_corr = monitor_correlation(cal, taus)
    ev_corr = monitor_correlation(ev, taus)

    print(f"\n{'quantity':>22} {'n=' + str(args.n_calib):>12} "
          f"{'n=' + str(args.n_eval):>12}")
    for k in KEYS:
        print(f"{'marginal FA ' + k:>22} "
              f"{cal_corr['marginal_fa'][k]:12.4f} "
              f"{ev_corr['marginal_fa'][k]:12.4f}")
    for a, b in (("B", "P"), ("B", "C"), ("P", "C")):
        print(f"{'corr ' + a + b:>22} {cal_corr[f'corr_{a}{b}']:+12.3f} "
              f"{ev_corr[f'corr_{a}{b}']:+12.3f}")
    print(f"{'joint (measured)':>22} {cal_corr['joint_measured']:12.5f} "
          f"{ev_corr['joint_measured']:12.5f}")
    print(f"{'joint (independent)':>22} "
          f"{cal_corr['joint_under_independence']:12.5f} "
          f"{ev_corr['joint_under_independence']:12.5f}")
    print(f"{'triple ratio':>22} {cal_corr['ratio']:11.2f}x "
          f"{ev_corr['ratio']:11.2f}x")

    res = {"band": ccfg.name, "taus": taus,
           "n_calib": args.n_calib, "n_eval": args.n_eval,
           "calibration_set": cal_corr, "evaluation_set": ev_corr}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"nominal_corr_{ccfg.name}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=float)

    drift = abs(ev_corr["ratio"] - cal_corr["ratio"])
    print()
    if drift / max(1e-9, cal_corr["ratio"]) < 0.25:
        print(f"The nominal ratio is stable: {cal_corr['ratio']:.2f}x at "
              f"n={args.n_calib} against\n{ev_corr['ratio']:.2f}x at "
              f"n={args.n_eval}. Report the larger-sample figure in Table~II "
              f"and\nnote that thresholds remain calibrated on "
              f"{args.n_calib} realizations.")
    else:
        print(f"The nominal ratio moves materially with sample size "
              f"({cal_corr['ratio']:.2f}x to\n{ev_corr['ratio']:.2f}x). The "
              f"64-realization figure was not stable; use the larger sample "
              f"and\nsay so in the methodology.")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
