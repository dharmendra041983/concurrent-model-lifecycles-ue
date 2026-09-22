"""
Correlated monitor firing and induced fallback (Sec. VI-A and VI-C).

Two results, in order:

  1. CORRELATED FIRING. Thresholds are calibrated per pipeline in isolation
     at a common nominal false-alarm rate. Because the three statistics are
     functions of a common channel realization, their firings coincide far
     more often than independent calibration assumes. Reported as the ratio
     of the measured joint rate to the product of the marginals.

  2. INDUCED FALLBACK. Matched pairs of realizations differing ONLY in
     whether pipeline i is permitted to fall back. A fallback in j that
     occurs in the permissive arm and not in the restrictive one was induced
     by i's corrective action, not by j's own degradation.

The second is the result that motivates arbitration. The interaction term
from experiment.py says independent accounting mis-estimates cost; induced
fallback says one lifecycle's corrective action actively triggers another's,
which is a stronger and more operational claim.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import numpy as np
import torch

from config import FR1, FR2, DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
from channel import SharedChannel
from degradation import apply_blockage
from experiment import load_system, SUBSETS
from lcm import (LCMConfig, calibrate_threshold, monitor_correlation,
                 run_state_machines, FALLBACK)
from stats import paired_bootstrap_ci

KEYS = ("B", "P", "C")


def _key(S) -> str:
    return "".join(sorted(S)) or "none"


def collect_arms(sys, H, seed) -> dict:
    """Per-slot statistics and throughput for all eight state vectors."""
    return {_key(S): sys.rollout(H, seed, S) for S in SUBSETS}


def nominal_statistics(chan, sys, seeds) -> dict:
    """Monitor statistics under nominal conditions, all pipelines active."""
    acc = {k: [] for k in KEYS}
    for s in seeds:
        H = chan.realize(s)
        out = sys.rollout(H, s, set())
        for k in KEYS:
            acc[k].append(out["s_" + k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["FR2", "FR1"], default="FR2")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--n-nominal", type=int, default=64)
    ap.add_argument("--target-fa", type=float, default=0.05)
    ap.add_argument("--atten-db", type=float, default=20.0)
    ap.add_argument("--sector-width", type=int, default=6)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ccfg = FR2 if args.band == "FR2" else FR1
    ecfg, pcfg = DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt = f"checkpoints/models_{ccfg.name}.pt"

    print(f"device: {device}  band: {ccfg.name}")
    chan, sys, _ = load_system(ccfg, ecfg, pcfg, ckpt, device)

    base = ecfg.base_seed + pcfg.train_realizations + pcfg.val_realizations

    # ---- 1. calibrate thresholds on NOMINAL realizations -----------------
    nom_seeds = [base + 10000 + i for i in range(args.n_nominal)]
    print(f"calibrating thresholds on {len(nom_seeds)} nominal "
          f"realizations at FA={args.target_fa}...")
    nom = nominal_statistics(chan, sys, nom_seeds)
    taus = {k: calibrate_threshold(nom[k], args.target_fa) for k in KEYS}
    corr = monitor_correlation(nom, taus)

    print("\ncorrelated monitor firing under nominal conditions:")
    for k in KEYS:
        print(f"  marginal FA  {k}: {corr['marginal_fa'][k]:.4f}   "
              f"tau = {taus[k]:.5f}")
    for a, b in itertools.combinations(KEYS, 2):
        print(f"  pair {a}{b}: corr {corr[f'corr_{a}{b}']:+.3f}  "
              f"joint {corr[f'joint_{a}{b}']:.4f} vs "
              f"{corr[f'indep_{a}{b}']:.4f} indep  "
              f"(x{corr[f'ratio_{a}{b}']:.2f})")
    print(f"  triple: {corr['joint_measured']:.5f} measured vs "
          f"{corr['joint_under_independence']:.5f} under independence "
          f"(x{corr['ratio']:.2f})")

    # ---- 2. induced fallback under blockage ------------------------------
    seeds = [base + 20000 + i for i in range(args.n)]
    start = ecfg.episode_slots // 3
    ramp = 20

    free_cfg = {k: LCMConfig(tau=taus[k]) for k in KEYS}

    induced = {}
    for src in KEYS:
        induced[src] = {j: {"free": [], "pinned": [], "delta": []}
                        for j in KEYS if j != src}
    osc = {k: [] for k in KEYS}
    dwell = {k: [] for k in KEYS}
    blocked_stats = {k: [] for k in KEYS}
    tput_free, tput_pin = [], []

    print(f"\nrunning {len(seeds)} blocked realizations "
          f"(sector {args.sector_width} beams, {args.atten_db} dB)...")
    for i, s in enumerate(seeds):
        H = chan.realize(s)
        Hb, _info = apply_blockage(
            H, ccfg.bs_num_rows, ccfg.bs_num_cols,
            start_slot=start, ramp_slots=ramp,
            sector_width=args.sector_width, atten_db=args.atten_db)

        arms = collect_arms(sys, Hb, s)
        n_slots = int(arms["none"]["tput_slot"].shape[0])

        # Correlation under DEGRADATION, not just nominal. Common causation
        # is a claim about what happens when something goes wrong; nominal
        # correlation can be near zero while degraded correlation is strong.
        for k in KEYS:
            blocked_stats[k].append(arms["none"]["s_" + k].cpu().numpy())

        res_free = run_state_machines(arms, free_cfg, n_slots)
        tput_free.append(res_free["throughput"])
        for k in KEYS:
            osc[k].append(res_free["oscillation"][k])
            dwell[k].append(res_free["dwell"][k]["FALLBACK"])

        # Matched restrictive arms: pin one pipeline ACTIVE, everything else
        # identical -- same realization, same blockage, same thresholds.
        for src in KEYS:
            cfgs = {k: LCMConfig(tau=taus[k], pinned_active=(k == src))
                    for k in KEYS}
            res_pin = run_state_machines(arms, cfgs, n_slots)
            if src == "P":
                tput_pin.append(res_pin["throughput"])
            for j in KEYS:
                if j == src:
                    continue
                f = res_free["fallback_entries"][j]
                p = res_pin["fallback_entries"][j]
                induced[src][j]["free"].append(f)
                induced[src][j]["pinned"].append(p)
                induced[src][j]["delta"].append(f - p)

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(seeds)}")

    out = {"band": ccfg.name, "n": args.n, "taus": taus,
           "target_fa": args.target_fa,
           "blockage": {"atten_db": args.atten_db,
                        "sector_width": args.sector_width,
                        "start_slot": start, "ramp_slots": ramp},
           "correlation": corr, "induced": {}, "oscillation": {}}

    print("\ninduced fallback (entries per realization):")
    print(f"{'src':>4} {'tgt':>4} {'free':>8} {'pinned':>8} {'induced':>9} "
          f"{'CI95':>22} {'p':>8}")
    for src in KEYS:
        for j in KEYS:
            if j == src:
                continue
            d = np.asarray(induced[src][j]["delta"], dtype=float)
            lo, hi, p = paired_bootstrap_ci(d)
            f = float(np.mean(induced[src][j]["free"]))
            pn = float(np.mean(induced[src][j]["pinned"]))
            ci = f"[{lo:+.3f},{hi:+.3f}]"
            print(f"{src:>4} {j:>4} {f:8.3f} {pn:8.3f} {d.mean():+9.3f} "
                  f"{ci:>22} {p:8.4f}")
            out["induced"][f"{src}->{j}"] = {
                "free": f, "pinned": pn, "induced": float(d.mean()),
                "ci95": [lo, hi], "p": p}

    blocked = {k: np.concatenate(v) for k, v in blocked_stats.items()}
    corr_b = monitor_correlation(blocked, taus)
    out["correlation_blocked"] = corr_b
    print("\ncorrelated monitor firing UNDER BLOCKAGE:")
    for a, b in itertools.combinations(KEYS, 2):
        print(f"  pair {a}{b}: corr {corr_b[f'corr_{a}{b}']:+.3f}  "
              f"joint {corr_b[f'joint_{a}{b}']:.4f} vs "
              f"{corr_b[f'indep_{a}{b}']:.4f} indep  "
              f"(x{corr_b[f'ratio_{a}{b}']:.2f})")
    print(f"  triple: {corr_b['joint_measured']:.5f} vs "
          f"{corr_b['joint_under_independence']:.5f} "
          f"(x{corr_b['ratio']:.2f})")

    print("\noscillation (share of disjoint 40-slot windows with >=4 "
          "transitions)\nand mean FALLBACK dwell (slots; short dwell = "
          "chattering):")
    out["dwell_fallback"] = {}
    for k in KEYS:
        v = float(np.mean(osc[k]))
        d = float(np.mean(dwell[k]))
        out["oscillation"][k] = v
        out["dwell_fallback"][k] = d
        print(f"  {k}: oscillation {v:.3f}   dwell {d:6.1f} slots")

    out["throughput_free"] = float(np.mean(tput_free))
    out["throughput_P_pinned"] = float(np.mean(tput_pin))
    print(f"\nthroughput, all lifecycles free : "
          f"{out['throughput_free']:.4f}")
    print(f"throughput, P pinned ACTIVE      : "
          f"{out['throughput_P_pinned']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"induced_{ccfg.name}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwritten -> {path}")

    sig = [k for k, v in out["induced"].items()
           if v["p"] < 0.05 and v["induced"] > 0]
    if sig:
        print(f"\nINDUCED FALLBACK FOUND: {', '.join(sig)}")
        print("One lifecycle's corrective action triggers another's. This is "
              "the\nargument for arbitration -- stronger than the interaction "
              "term alone.")
    else:
        print("\nNo induced fallback distinguishable from zero. The "
              "interaction result\nstands on its own, but the arbiter has no "
              "motivating failure to prevent.\nCheck first that the blockage "
              "actually drives fallbacks: if 'free'\ncounts are near zero, "
              "raise --atten-db or --sector-width.")


if __name__ == "__main__":
    main()
