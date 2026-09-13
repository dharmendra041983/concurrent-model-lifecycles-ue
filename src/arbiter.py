"""
UE-level fallback arbitration (Sec. VII).

MECHANISM. When a pipeline's fallback becomes due and an UPSTREAM pipeline
in the dependency chain B -> P -> C changed state within the last W_a
evaluations, the arbiter withholds the request for at most H evaluations.
The rationale is exactly the induced-fallback result: a downstream monitor
firing immediately after an upstream state change is often observing a
transient it did not cause. If the statistic recovers during the hold, the
request lapses and an induced fallback has been avoided. If it persists,
the fallback proceeds.

The arbiter observes ONLY lifecycle-visible outputs -- the monitor
statistics, the detector outputs, and the states. It does not read any
model's internals or the controlled process.

TWO SAFETY PROPERTIES, both structural rather than tuned:
  * The hold is bounded by H. Fallback to the non-AI state is delayed, never
    suppressible indefinitely, and remains locally executable throughout.
  * Only DOWNSTREAM requests are ever held. The most upstream pipeline (B)
    is never arbitrated, so the chain cannot deadlock.

HONEST ACCOUNTING. An arbiter that only helps is not credible. Suppressing
an induced fallback is a win; suppressing one that would have fired anyway
is a loss. The matched upstream-pinned arm separates the two: fallbacks
occurring with the upstream pinned ACTIVE were not induced, so any of those
the arbiter removes are legitimate fallbacks denied.
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np
import torch

from config import FR1, FR2, DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
from degradation import apply_blockage
from experiment import load_system, SUBSETS
from induced import collect_arms, nominal_statistics, KEYS, _key
from lcm import (LCMConfig, LCMStateMachine, calibrate_threshold,
                 ACTIVE, FALLBACK)
from stats import paired_bootstrap_ci

# Dependency chain: each pipeline's upstream neighbours.
UPSTREAM = {"B": [], "P": ["B"], "C": ["B", "P"]}


def run_arbitrated(arms: dict, cfgs: dict, n_slots: int,
                   hold_H: int = 0, window_Wa: int = 10,
                   mode: str = "chain") -> dict:
    """Step the three machines with optional arbitration.

    hold_H = 0 disables arbitration and reproduces the unarbitrated system
    exactly, so the comparison is like-for-like.

    mode = "chain": hold a downstream request only when an UPSTREAM pipeline
                    changed state recently. This is the proposed mechanism.
    mode = "blind": hold every downstream request for H evaluations
                    regardless of upstream state. This is the CONTROL, and
                    it is the one that matters. A fallback that costs
                    throughput whenever it fires can be suppressed for
                    benefit by any delay at all, so if "blind" matches
                    "chain", the dependency-chain condition is decorative
                    and the mechanism claim is unsupported.
    """
    m = {k: LCMStateMachine(cfgs[k]) for k in KEYS}
    state = {k: ACTIVE for k in KEYS}
    last_change = {k: -10 ** 6 for k in KEYS}
    holding = {k: 0 for k in KEYS}
    hold_runs = []

    tput = np.zeros(n_slots)
    for t in range(n_slots):
        a = arms[_key([k for k in KEYS if state[k] == FALLBACK])]
        tput[t] = float(a["tput_slot"][t])

        prev = dict(state)
        for k in KEYS:
            allow = True
            if hold_H > 0 and UPSTREAM[k]:
                if mode == "blind":
                    recent = True
                else:
                    recent = any(t - last_change[u] <= window_Wa
                                 for u in UPSTREAM[k])
                if recent and holding[k] < hold_H:
                    allow = False
            before = m[k].fallback_entries
            state[k] = m[k].step(float(a["s_" + k][t]), allow=allow)
            if not allow and m[k].state == 1:      # SUSPECT, request withheld
                holding[k] += 1
            elif m[k].fallback_entries > before or m[k].state == ACTIVE:
                if holding[k] > 0:
                    hold_runs.append(holding[k])
                holding[k] = 0

        for k in KEYS:
            if state[k] != prev[k]:
                last_change[k] = t

    return {
        "throughput": tput.mean(),
        "throughput_post": tput[n_slots // 3:].mean(),
        "fallback_entries": {k: m[k].fallback_entries for k in KEYS},
        "requests": {k: m[k].requests for k in KEYS},
        "held": {k: m[k].held_evaluations for k in KEYS},
        "mean_hold": float(np.mean(hold_runs)) if hold_runs else 0.0,
        "max_hold": int(np.max(hold_runs)) if hold_runs else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["FR2", "FR1"], default="FR2")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--n-nominal", type=int, default=64)
    ap.add_argument("--target-fa", type=float, default=0.05)
    ap.add_argument("--atten-db", type=float, default=20.0)
    ap.add_argument("--sector-width", type=int, default=6)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--holds", type=int, nargs="+",
                    default=[0, 5, 10, 20, 40])
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

    seeds = [base + 20000 + i for i in range(args.n)]
    start = ecfg.episode_slots // 3

    free_cfg = {k: LCMConfig(tau=taus[k]) for k in KEYS}
    # Upstream-pinned reference: fallbacks that still occur here were NOT
    # induced by P, so they are the legitimate ones.
    pin_cfg = {k: LCMConfig(tau=taus[k], pinned_active=(k == "P"))
               for k in KEYS}

    acc = {h: {m: {"tput": [], "entries_C": [], "mean_hold": []}
               for m in ("chain", "blind")} for h in args.holds}
    legit_C = []

    print(f"running {len(seeds)} blocked realizations, "
          f"H in {args.holds}, W_a={args.window}...")
    for i, s in enumerate(seeds):
        H_ = chan.realize(s)
        Hb, _ = apply_blockage(H_, ccfg.bs_num_rows, ccfg.bs_num_cols,
                               start_slot=start, ramp_slots=20,
                               sector_width=args.sector_width,
                               atten_db=args.atten_db)
        arms = collect_arms(sys, Hb, s)
        n_slots = int(arms["none"]["tput_slot"].shape[0])

        legit_C.append(run_arbitrated(arms, pin_cfg, n_slots, 0
                                      )["fallback_entries"]["C"])
        for h in args.holds:
            for mode in ("chain", "blind"):
                r = run_arbitrated(arms, free_cfg, n_slots, h,
                                   args.window, mode)
                acc[h][mode]["tput"].append(r["throughput"])
                acc[h][mode]["entries_C"].append(r["fallback_entries"]["C"])
                acc[h][mode]["mean_hold"].append(r["mean_hold"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(seeds)}")

    legit = np.asarray(legit_C, dtype=float)
    base_entries = np.asarray(acc[0]["chain"]["entries_C"], dtype=float)
    base_t = np.asarray(acc[0]["chain"]["tput"], dtype=float)
    induced_n = base_entries - legit

    print(f"\nC fallbacks with no arbitration : {base_entries.mean():.3f} "
          f"per realization")
    print(f"  of which induced by P          : {induced_n.mean():.3f}")
    print(f"  legitimate (P pinned)          : {legit.mean():.3f}")

    print(f"\n{'H':>4} {'mode':>6} {'d_tput':>8} {'CI95':>20} "
          f"{'C_fb':>7} {'good':>7} {'BAD':>7} {'hold':>6}")
    print("  (good = induced fallbacks avoided; BAD = legitimate ones denied)")
    print("  chain vs blind: if these match, the upstream condition does "
          "nothing.")
    rows = {}
    for h in args.holds:
        rows[h] = {}
        for mode in ("chain", "blind"):
            t = np.asarray(acc[h][mode]["tput"], dtype=float)
            e = np.asarray(acc[h][mode]["entries_C"], dtype=float)
            d = t - base_t
            lo, hi, p = paired_bootstrap_ci(d)
            removed = base_entries - e
            good = np.minimum(removed, np.maximum(induced_n, 0))
            bad = np.maximum(removed - np.maximum(induced_n, 0), 0)
            ci = f"[{lo:+.4f},{hi:+.4f}]"
            print(f"{h:>4} {mode:>6} {d.mean():+8.4f} {ci:>20} "
                  f"{e.mean():7.3f} {good.mean():7.3f} {bad.mean():7.3f} "
                  f"{np.mean(acc[h][mode]['mean_hold']):6.2f}")
            rows[h][mode] = {"delta_throughput": float(d.mean()),
                             "ci95": [lo, hi], "p": p,
                             "C_entries": float(e.mean()),
                             "induced_avoided": float(good.mean()),
                             "legitimate_denied": float(bad.mean())}
        # Does conditioning on the dependency chain add anything?
        dc = np.asarray(acc[h]["chain"]["tput"], dtype=float)
        db = np.asarray(acc[h]["blind"]["tput"], dtype=float)
        lo, hi, p = paired_bootstrap_ci(dc - db)
        rows[h]["chain_minus_blind"] = {"delta": float((dc - db).mean()),
                                        "ci95": [lo, hi], "p": p}
        if h > 0:
            print(f"{'':>4} {'diff':>6} {(dc-db).mean():+8.4f} "
                  f"{f'[{lo:+.4f},{hi:+.4f}]':>20}   <-- chain minus blind")

    out = {"band": ccfg.name, "n": args.n, "window_Wa": args.window,
           "taus": taus, "by_hold": rows,
           "C_entries_unarbitrated": float(base_entries.mean()),
           "C_induced": float(induced_n.mean()),
           "C_legitimate": float(legit.mean())}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"arbiter_{ccfg.name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)

    best = max((h for h in args.holds if h > 0),
               key=lambda h: rows[h]["chain"]["delta_throughput"])
    r = rows[best]["chain"]
    cb = rows[best]["chain_minus_blind"]
    print(f"\nbest H = {best}: {r['delta_throughput']:+.4f} bit/s/Hz, "
          f"{r['induced_avoided']:.2f} induced avoided, "
          f"{r['legitimate_denied']:.2f} legitimate denied")
    if r["ci95"][0] <= 0:
        print("No significant benefit. Report as a negative result: the "
              "induced\nfallbacks are real but too cheap for arbitration to "
              "pay for itself.")
    elif cb["ci95"][1] < 0:
        print("MECHANISM CLAIM REFUTED. Blind delay BEATS the "
              "chain-conditioned arbiter\n(chain - blind is significantly "
              "negative), so the best policy here is\nsimply never to fall "
              "back downstream. Coordination adds nothing when the\n"
              "downstream fallback costs throughput whenever it fires. "
              "Report this;\ndo not present the raw chain benefit as "
              "evidence for arbitration.")
    elif cb["ci95"][0] <= 0 <= cb["ci95"][1]:
        print("BENEFIT IS NOT FROM ARBITRATION. Blind delay matches the "
              "chain-conditioned\narbiter, so the upstream-state condition "
              "is doing no work: any delay\nsuppresses a fallback that costs "
              "throughput whenever it fires. Either\nreport the mechanism "
              "claim as unsupported, or find a regime where the\ndownstream "
              "fallback is genuinely beneficial when NOT induced.")
    else:
        print("Chain conditioning BEATS blind delay "
              f"({cb['delta']:+.4f}, CI excludes zero).\nThe mechanism claim "
              "holds. Report the conservatism column in the SAME table.")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
