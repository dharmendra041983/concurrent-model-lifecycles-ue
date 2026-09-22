"""
Why induced C fallbacks are false corrective actions (Sec. VI-E support).

The paper establishes that a prediction fallback raises compression fallback
entries by 53 percent, and separately that the compression fallback loses
throughput whenever it fires. What it does not yet show is the mechanism
joining those two facts.

The hypothesis: a P fallback shifts the encoder's input distribution enough
to push the compression monitor over its threshold, but NOT enough to make
the learned encoder worse than the low-rate report that replaces it. The
induced firings are therefore real detections of a real change that has no
useful corrective action available.

This measures both halves on the same realizations:

  * the monitor side -- s_C under P active vs P fallback, mean and 95th
    percentile, and the fraction exceeding tau_C (which is the induced
    firing itself);

  * the value side -- the learned encoder's reconstruction of its own input
    against the 12-bit wideband fallback's, under both P states. If the
    learned encoder stays well ahead under P fallback, the extra firings
    cannot justify falling back.

A tail statistic is reported because a monitor fires on excursions, not on
means: a modest mean shift with a heavy tail is what trips a threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np
import torch

from config import FR1, FR2, DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
from degradation import apply_blockage
from experiment import load_system
from induced import nominal_statistics, KEYS
from lcm import calibrate_threshold
from stats import paired_bootstrap_ci
from check_encoder_ood import wideband_codebook, nmse_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["FR2", "FR1"], default="FR2")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--n-nominal", type=int, default=64)
    ap.add_argument("--target-fa", type=float, default=0.05)
    ap.add_argument("--atten-db", type=float, default=20.0)
    ap.add_argument("--sector-width", type=int, default=6)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ccfg = FR2 if args.band == "FR2" else FR1
    ecfg, pcfg = DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    nr, ns = ccfg.num_ut_ant, pcfg.csi_subbands

    chan, sys, _ = load_system(ccfg, ecfg, pcfg,
                               f"checkpoints/models_{ccfg.name}.pt", dev)
    base = ecfg.base_seed + pcfg.train_realizations + pcfg.val_realizations

    nom = nominal_statistics(
        chan, sys, [base + 10000 + i for i in range(args.n_nominal)])
    tau_C = calibrate_threshold(nom["C"], args.target_fa)
    print(f"band {ccfg.name} | tau_C = {tau_C:.6f} at FA={args.target_fa}")

    seeds = [base + 20000 + i for i in range(args.n)]
    start = ecfg.episode_slots // 3

    cols = ["sC_mean", "sC_p95", "fire_frac", "learned_db", "fallback_db"]
    acc = {cond: {arm: {c: [] for c in cols} for arm in ("Pactive", "Pfb")}
           for cond in ("nominal", "blocked")}

    print(f"running {len(seeds)} realizations...")
    for i, s in enumerate(seeds):
        H = chan.realize(s)
        Hb, _ = apply_blockage(H, ccfg.bs_num_rows, ccfg.bs_num_cols,
                               start_slot=start, ramp_slots=20,
                               sector_width=args.sector_width,
                               atten_db=args.atten_db)

        for cond, Hx in (("nominal", H), ("blocked", Hb)):
            for arm, S in (("Pactive", set()), ("Pfb", {"P"})):
                out = sys.rollout(Hx, s, S)
                sC = out["s_C"].cpu().numpy()
                x = out["x_enc_in"]
                with torch.no_grad():
                    rec = sys.comp.model(x)
                a = acc[cond][arm]
                a["sC_mean"].append(float(sC.mean()))
                a["sC_p95"].append(float(np.percentile(sC, 95)))
                a["fire_frac"].append(float((sC > tau_C).mean()))
                a["learned_db"].append(nmse_db(rec, x))
                a["fallback_db"].append(
                    nmse_db(wideband_codebook(x, nr, ns, 3), x))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(seeds)}")

    out_json = {"band": ccfg.name, "n": args.n, "tau_C": tau_C, "rows": {}}

    print(f"\n{'cond':>8} {'arm':>8} {'sC mean':>9} {'sC p95':>9} "
          f"{'fire':>7} {'learned':>9} {'12-bit fb':>10} {'margin':>8}")
    for cond in ("nominal", "blocked"):
        for arm in ("Pactive", "Pfb"):
            a = {c: float(np.mean(acc[cond][arm][c])) for c in cols}
            margin = a["learned_db"] - a["fallback_db"]
            print(f"{cond:>8} {arm:>8} {a['sC_mean']:9.4f} {a['sC_p95']:9.4f} "
                  f"{a['fire_frac']:7.3f} {a['learned_db']:9.2f} "
                  f"{a['fallback_db']:10.2f} {margin:8.2f}")
            out_json["rows"][f"{cond}_{arm}"] = {**a, "margin_db": margin}
        print()

    # Paired tests on the P-fallback effect, per condition.
    print("effect of the P fallback (Pfb minus Pactive), paired over "
          "realizations:")
    for cond in ("nominal", "blocked"):
        for c, label in (("sC_p95", "s_C 95th pct"),
                         ("fire_frac", "C firing frac"),
                         ("learned_db", "learned NMSE [dB]")):
            d = (np.asarray(acc[cond]["Pfb"][c])
                 - np.asarray(acc[cond]["Pactive"][c]))
            lo, hi, p = paired_bootstrap_ci(d)
            print(f"  {cond:>8} {label:>20}: {d.mean():+9.4f}  "
                  f"[{lo:+.4f},{hi:+.4f}]  p={p:.4f}")
            out_json["rows"].setdefault("effects", {})[f"{cond}_{c}"] = {
                "delta": float(d.mean()), "ci95": [lo, hi], "p": p}

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"mechanism_{ccfg.name}.json")
    with open(path, "w") as f:
        json.dump(out_json, f, indent=2, default=float)

    mb = out_json["rows"]["blocked_Pfb"]["margin_db"]
    fa = out_json["rows"]["blocked_Pactive"]["fire_frac"]
    fb = out_json["rows"]["blocked_Pfb"]["fire_frac"]
    print()
    if fb > fa and mb < -3.0:
        print(f"MECHANISM CONFIRMED: the P fallback raises C firing from "
              f"{fa:.3f} to {fb:.3f},\nyet the learned encoder still beats "
              f"the 12-bit report by {-mb:.1f} dB on the\nshifted input. The "
              f"induced firings detect a real distribution change for\nwhich "
              f"no useful corrective action exists.")
    elif fb <= fa:
        print("The P fallback does not raise C firing here. The induced "
              "effect must\ncome from the state machine dynamics rather than "
              "the monitor statistic;\nre-examine before claiming a "
              "distribution-shift mechanism.")
    else:
        print(f"The learned margin narrows to {-mb:.1f} dB under P fallback. "
              f"If it closes\nfurther in some regime, that is where the "
              f"compression fallback could\nbecome worthwhile.")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
