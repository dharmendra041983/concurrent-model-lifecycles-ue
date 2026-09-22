"""
Conditional fallback value (Sec. VI-D).

The arbiter result rests on a claim that has so far only been measured
INDIRECTLY: that C's monitor fires on degradations its fallback cannot
repair. What has been shown is that C_raw({C}) > 0 -- the fallback costs
throughput averaged over the whole episode. That is a weaker statement, and
the obvious objection is that a fallback is not meant to pay off on slots
where nothing is wrong.

This measures the thing directly: restricted to slots where the monitor
would actually have triggered, is the fallback better or worse than staying
active?

TWO DESIGN POINTS THAT MATTER:

  1. Conditioning uses s_C from the ACTIVE arm -- the monitor state that
     would have driven the decision. Conditioning on the fallback arm's
     statistic would select on an outcome the decision itself produced,
     making the comparison circular.

  2. Results are split by degradation severity (pre-blockage, ramp,
     post-blockage). If the fallback loses under mild degradation but wins
     under severe blockage, the finding is that the THRESHOLD is
     miscalibrated, not that the fallback is useless -- a different and
     more actionable conclusion.

The same check is run for P, since P's fallback is the upstream half of the
induced-fallback result and the same objection applies to it.
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
from induced import collect_arms, nominal_statistics, KEYS
from lcm import calibrate_threshold
from stats import paired_bootstrap_ci


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
    print(f"device: {device}  band: {ccfg.name}")

    chan, sys, _ = load_system(ccfg, ecfg, pcfg,
                               f"checkpoints/models_{ccfg.name}.pt", device)
    base = ecfg.base_seed + pcfg.train_realizations + pcfg.val_realizations

    nom = nominal_statistics(
        chan, sys, [base + 10000 + i for i in range(args.n_nominal)])
    taus = {k: calibrate_threshold(nom[k], args.target_fa) for k in KEYS}
    print(f"thresholds at FA={args.target_fa}: "
          + ", ".join(f"{k}={taus[k]:.5f}" for k in KEYS))

    seeds = [base + 20000 + i for i in range(args.n)]
    start = ecfg.episode_slots // 3
    ramp = 20

    # Per-realization mean throughput difference (fallback - active),
    # restricted to the conditioning mask. Paired at realization level.
    rows = {p: {r: [] for r in ("all", "fired", "pre", "ramp", "post")}
            for p in ("P", "C")}
    fired_frac = {p: [] for p in ("P", "C")}

    print(f"running {len(seeds)} blocked realizations...")
    for i, s in enumerate(seeds):
        H_ = chan.realize(s)
        Hb, _ = apply_blockage(H_, ccfg.bs_num_rows, ccfg.bs_num_cols,
                               start_slot=start, ramp_slots=ramp,
                               sector_width=args.sector_width,
                               atten_db=args.atten_db)
        arms = collect_arms(sys, Hb, s)
        act = arms["none"]
        n_slots = int(act["tput_slot"].shape[0])

        # Slot index within the episode, offset by the prediction window so
        # the blockage phase labels line up with the reported slots.
        off = pcfg.pred_window + pcfg.feedback_delay
        idx = np.arange(n_slots) + off
        phase = {
            "pre": idx < start,
            "ramp": (idx >= start) & (idx < start + ramp),
            "post": idx >= start + ramp,
        }

        t_act = act["tput_slot"].cpu().numpy()
        for p in ("P", "C"):
            t_fb = arms[p]["tput_slot"].cpu().numpy()
            d = t_fb - t_act                      # positive => fallback helps

            # Condition on the ACTIVE arm's statistic: the monitor state that
            # would have driven the decision.
            fired = act["s_" + p].cpu().numpy() > taus[p]
            fired_frac[p].append(float(fired.mean()))

            rows[p]["all"].append(float(d.mean()))
            rows[p]["fired"].append(float(d[fired].mean())
                                    if fired.any() else np.nan)
            for ph, m in phase.items():
                mm = m & fired
                rows[p][ph].append(float(d[mm].mean()) if mm.any() else np.nan)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(seeds)}")

    out = {"band": ccfg.name, "n": args.n, "taus": taus, "results": {}}
    print(f"\nthroughput change from falling back (positive = fallback "
          f"HELPS), bit/s/Hz")
    print(f"{'pipe':>5} {'regime':>7} {'n_real':>7} {'delta':>9} "
          f"{'CI95':>22} {'p':>8}")
    for p in ("P", "C"):
        out["results"][p] = {"fired_fraction": float(np.mean(fired_frac[p]))}
        for r in ("all", "fired", "pre", "ramp", "post"):
            v = np.asarray(rows[p][r], dtype=float)
            v = v[~np.isnan(v)]
            if v.size < 8:
                print(f"{p:>5} {r:>7} {v.size:>7}   (too few realizations)")
                continue
            lo, hi, pv = paired_bootstrap_ci(v)
            ci = f"[{lo:+.4f},{hi:+.4f}]"
            print(f"{p:>5} {r:>7} {v.size:>7} {v.mean():+9.4f} {ci:>22} "
                  f"{pv:8.4f}")
            out["results"][p][r] = {"delta": float(v.mean()),
                                    "ci95": [lo, hi], "p": pv,
                                    "n": int(v.size)}
        print(f"{'':>5} monitor fired on "
              f"{100*np.mean(fired_frac[p]):.1f}% of slots")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"conditional_{ccfg.name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)

    print()
    for p in ("P", "C"):
        r = out["results"][p].get("fired")
        if r is None:
            continue
        post = out["results"][p].get("post", {})
        if r["ci95"][1] < 0:
            msg = (f"{p}: fallback is WORSE even when the monitor fires "
                   f"({r['delta']:+.4f}).")
            if post and post["ci95"][0] > 0:
                msg += ("\n   BUT it helps post-blockage -- the threshold is "
                        "miscalibrated,\n   not the fallback useless. Report "
                        "it that way.")
            else:
                msg += ("\n   The fallback is the wrong action at the moment "
                        "it is triggered.\n   This is the direct evidence the "
                        "arbiter result needs.")
        elif r["ci95"][0] > 0:
            msg = (f"{p}: fallback HELPS when the monitor fires "
                   f"({r['delta']:+.4f}).\n   The negative arbiter result "
                   f"needs rewriting -- episode-averaged\n   accounting "
                   f"buried a fallback that works in its own regime.")
        else:
            msg = (f"{p}: fallback is indistinguishable from staying active "
                   f"when the\n   monitor fires. The monitor is not "
                   f"selecting a regime where the\n   fallback matters "
                   f"either way.")
        print(msg)
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
