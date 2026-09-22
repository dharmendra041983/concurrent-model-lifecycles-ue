"""
The gating experiment (Sec. VI-B).

Measures

    Delta(S) = C(S) - sum_{i in S} C({i})

across all eight forced-fallback subsets on MATCHED channel realizations.

Three design points that are not negotiable if the result is to survive review:

  1. Subsets are FORCED, not waited for. Every C(S) is measured on the same
     realizations with the same seeds, so the arms differ only in which
     pipelines are in fallback.

  2. The unit of analysis is the REALIZATION, not the slot. Per-slot samples
     within an episode are strongly dependent; treating them as independent
     draws inflates n by ~200x and is the single most likely thing to get this
     result thrown out. Everything below aggregates to one number per
     realization before any statistics are computed.

  3. The hypothesis is |Delta| distinguishable from zero, NOT Delta > 0.
     Sub-additive interaction is a real finding with the opposite operational
     implication -- per-functionality accounting is over-conservative and
     independent calibration leaves margin unclaimed. The beam fallback is
     accurate-but-costly, which makes sub-additivity a live possibility, so
     pre-register the two-sided test and report whichever sign appears.
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
from pipelines import (BeamModel, CSIPredictor, CSIAutoencoder,
                       BeamPipeline, PredictionPipeline, CompressionPipeline,
                       ConcurrentSystem)
from stats import paired_bootstrap_ci, holm_adjust

SUBSETS = [set(s) for r in range(4)
           for s in itertools.combinations(["B", "P", "C"], r)]


def load_system(ccfg, ecfg, pcfg, ckpt_path, device):
    chan = SharedChannel(ccfg, ecfg, device=device)
    ck = torch.load(ckpt_path, map_location=device)

    beam_m = BeamModel(len(chan.set_b), chan.num_beams).to(device)
    beam_m.load_state_dict(ck["beam"]); beam_m.eval()

    dim = 2 * ccfg.num_ut_ant * pcfg.csi_subbands
    pred_m = CSIPredictor(dim).to(device)
    pred_m.load_state_dict(ck["pred"]); pred_m.eval()

    comp_m = CSIAutoencoder(dim, pcfg.latent_dim,
                            pcfg.bits_per_latent).to(device)
    comp_m.load_state_dict(ck["comp"]); comp_m.eval()

    sys = ConcurrentSystem(
        chan,
        BeamPipeline(beam_m, chan.set_b, chan.num_beams, pcfg),
        PredictionPipeline(pred_m, pcfg),
        CompressionPipeline(comp_m, pcfg,
                            num_rx_ant=ccfg.num_ut_ant,
                            num_subbands=pcfg.csi_subbands),
        ccfg, ecfg, pcfg,
    )
    return chan, sys, ck.get("metrics", {})


def run(ccfg, ecfg, pcfg, ckpt, device, n_real):
    chan, sys, train_metrics = load_system(ccfg, ecfg, pcfg, ckpt, device)

    # Evaluation seeds are disjoint from training and validation.
    base = ecfg.base_seed + pcfg.train_realizations + pcfg.val_realizations
    seeds = [base + i for i in range(n_real)]

    key = lambda S: "".join(sorted(S)) or "none"
    util = {key(S): np.zeros(n_real) for S in SUBSETS}
    util_raw = {key(S): np.zeros(n_real) for S in SUBSETS}
    diag = {key(S): [] for S in SUBSETS}

    for r, s in enumerate(seeds):
        H = chan.realize(s)                       # generated once, replayed
        for S in SUBSETS:
            out = sys.rollout(H, s, S)
            util[key(S)][r] = out["mean_throughput"]
            util_raw[key(S)][r] = out["mean_throughput_raw"]
            diag[key(S)].append({
                "beam_accuracy": out["beam_accuracy"],
                "overhead": out["overhead"],
                "csi_nmse": out["csi_nmse"],
                "outage_rate": out["outage_rate"],
                "achieved_bler": out["achieved_bler"],
                "olla_margin_db": out["olla_margin_db"],
            })
        if (r + 1) % 25 == 0:
            print(f"  {r+1}/{n_real} realizations")

    # Per-realization cost relative to the all-active baseline.
    base_u = util["none"]
    cost = {k: base_u - v for k, v in util.items() if k != "none"}
    base_r = util_raw["none"]
    cost_raw = {k: base_r - v for k, v in util_raw.items() if k != "none"}

    results, pvals, labels = {}, [], []
    for S in SUBSETS:
        if len(S) < 2:
            continue
        k = key(S)
        singles = sum(cost["".join(sorted({i}))] for i in S)
        delta = cost[k] - singles                 # paired, per realization
        ci_lo, ci_hi, p = paired_bootstrap_ci(delta)

        singles_r = sum(cost_raw["".join(sorted({i}))] for i in S)
        delta_r = cost_raw[k] - singles_r
        rlo, rhi, rp = paired_bootstrap_ci(delta_r)

        results[k] = {
            "C_S_raw": float(cost_raw[k].mean()),
            "sum_C_raw_singletons": float(singles_r.mean()),
            # Delta relative to what per-functionality accounting predicts.
            # Absolute Delta is not comparable across operating points:
            # sub-additivity is largely saturation, and saturation is near
            # guaranteed wherever singleton costs are large. The relative
            # term is what says how wrong independent calibration is.
            "delta_raw_rel": float(delta_r.mean() /
                                   max(1e-9, abs(singles_r.mean()))),
            "delta_raw": float(delta_r.mean()),
            "delta_raw_ci95": [rlo, rhi],
            "delta_raw_p": rp,
            "C_S": float(cost[k].mean()),
            "sum_C_singletons": float(singles.mean()),
            "delta": float(delta.mean()),
            "ci95": [ci_lo, ci_hi],
            "p_raw": p,
            "sign": "super-additive" if delta.mean() > 0 else "sub-additive",
        }
        pvals.append(p)
        labels.append(k)

    for lab, p_adj in zip(labels, holm_adjust(pvals)):
        results[lab]["p_holm"] = p_adj
        results[lab]["significant_total"] = bool(p_adj < 0.05)

    # The gate must be judged on Delta_raw. A subset containing B changes
    # measurement overhead, and overhead composes multiplicatively with every
    # other cost source, so Delta computed on delivered throughput is nonzero
    # by arithmetic alone -- no lifecycle coupling required. Holm is applied
    # separately to the raw p-values.
    raw_p = [results[l]["delta_raw_p"] for l in labels]
    for lab, p_adj in zip(labels, holm_adjust(raw_p)):
        results[lab]["delta_raw_p_holm"] = p_adj
        results[lab]["significant"] = bool(p_adj < 0.05)

    for S in SUBSETS:
        if len(S) == 1:
            results[key(S)] = {"C_S": float(cost[key(S)].mean()),
                               "C_S_raw": float(cost_raw[key(S)].mean())}

    return {
        "band": ccfg.name,
        "n_realizations": n_real,
        "baseline_throughput": float(base_u.mean()),
        "train_metrics": train_metrics,
        "interaction": results,
        "diagnostics": {k: {
            m: float(np.mean([d[m] for d in v]))
            for m in ("beam_accuracy", "overhead", "csi_nmse",
                      "outage_rate", "achieved_bler", "olla_margin_db")
        } for k, v in diag.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["FR2", "FR1"], default="FR2")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ccfg = FR2 if args.band == "FR2" else FR1
    ecfg, pcfg = DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or f"checkpoints/models_{ccfg.name}.pt"
    n = args.n or ecfg.num_realizations

    print(f"device: {device}  band: {ccfg.name}  n={n}")
    _c = CompressionPipeline(None, pcfg, ccfg.num_ut_ant, pcfg.csi_subbands)
    print(f"CSI report: learned {_c.learned_bits} bits, "
          f"fallback {_c.fallback_report_bits} bits")
    res = run(ccfg, ecfg, pcfg, ckpt, device, n)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"interaction_{ccfg.name}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\nbaseline throughput: {res['baseline_throughput']:.4f} bit/s/Hz\n")
    print(f"{'S':>6} {'C_raw':>9} {'sum_raw':>9} {'Delta_raw':>10} "
          f"{'rel%':>8} {'CI95_raw':>20} {'p':>8}")
    print("  (raw = pre-overhead, isolating pipeline coupling; "
          "rel% = Delta/sum)")
    for k, v in res["interaction"].items():
        if "delta_raw" not in v:
            print(f"{k:>6} {v['C_S_raw']:9.4f}")
            continue
        cir = f"[{v['delta_raw_ci95'][0]:+.4f},{v['delta_raw_ci95'][1]:+.4f}]"
        print(f"{k:>6} {v['C_S_raw']:9.4f} {v['sum_C_raw_singletons']:9.4f} "
              f"{v['delta_raw']:+10.4f} {100*v['delta_raw_rel']:+8.1f} "
              f"{cir:>20} {v['delta_raw_p_holm']:8.4f}")

    sig = [k for k, v in res["interaction"].items() if v.get("significant")]
    print()
    if sig:
        signs = {("sub-additive" if res["interaction"][k]["delta_raw"] < 0
                  else "super-additive") for k in sig}
        print(f"GATE PASSED on Delta_raw: {len(sig)} subset(s) "
              f"({', '.join(sorted(signs))}): {', '.join(sig)}")
        if len(signs) > 1:
            print("Both signs present -- per-functionality calibration errs "
                  "in opposite\ndirections depending on which pipelines "
                  "coincide. This is the stronger result.")
        print("Proceed to the induced-fallback experiment, then the arbiter.")
    else:
        print("GATE NOT PASSED: no subset shows Delta distinguishable from "
              "zero.\nBefore reframing, check in order: (1) do the trained "
              "models beat\ntheir fallbacks by a clear margin, (2) is the "
              "beam sweep overhead\nmodel dominating, (3) is n large enough "
              "for the observed variance.")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
