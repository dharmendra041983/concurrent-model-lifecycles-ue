"""
Pairing-mismatch regime: gate check (Sec. VII positive control, part 1).

The blockage regime showed that C's fallback is harmful whenever taken, so
no trigger policy can help. The objection is that arbitration was never given
a fair test. This builds the fair test: a degradation under which the
compression fallback genuinely IS the better action.

Encoder A into decoder B. The latent space of an independently initialized
autoencoder is arbitrary, so a mismatched decoder should fail badly -- unlike
blockage, which merely rescaled the channel.

Two gates, in order:

  Q3. Does mismatch actually break the learned path? Compare matched,
      mismatched, and the 12-bit fallback on the same inputs.

  Q4. Conditional on C's own monitor firing, is the fallback now BETTER than
      staying active? This is the property the blockage regime lacked
      (-0.269 bit/s/Hz there). Without it, the positive control is not a
      positive control and the arbiter experiment should not be run.

If Q4 fails, report that the fallback has negative conditional value in both
regimes studied, and stop. Do not search for a third mechanism.
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
from lcm import calibrate_threshold
from pipelines import (CSIAutoencoder, STEQuantizer, snr_from_csi,
                       real_to_complex, throughput_olla, to_subbands,
                       complex_to_real, hold_beam)
from stats import paired_bootstrap_ci
from check_encoder_ood import wideband_codebook, nmse_db


def mismatched_reconstruct(enc_model: CSIAutoencoder,
                           dec_model: CSIAutoencoder,
                           x: torch.Tensor) -> torch.Tensor:
    """Encoder A's latent, decoded by decoder B.

    Reproduces CSIAutoencoder.forward but splits the two halves across
    models, including the per-sample scaling, which is part of the pairing
    contract: the decoder assumes the scaling convention its own encoder
    used.
    """
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    z = enc_model.enc(x / scale)
    zq = STEQuantizer.apply(z, enc_model.levels)
    return dec_model.dec(zq) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["FR2", "FR1"], default="FR2")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--n-nominal", type=int, default=64)
    ap.add_argument("--target-fa", type=float, default=0.05)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ccfg = FR2 if args.band == "FR2" else FR1
    ecfg, pcfg = DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    nr, ns = ccfg.num_ut_ant, pcfg.csi_subbands
    dim = 2 * nr * ns

    chan, sys, _ = load_system(ccfg, ecfg, pcfg,
                               f"checkpoints/models_{ccfg.name}.pt", dev)

    ckb_path = f"checkpoints/models_{ccfg.name}_pairB.pt"
    if not os.path.exists(ckb_path):
        print(f"missing {ckb_path} -- run train_pair.py first.")
        return
    ckb = torch.load(ckb_path, map_location=dev, weights_only=False)
    comp_b = CSIAutoencoder(dim, pcfg.latent_dim,
                            pcfg.bits_per_latent).to(dev)
    comp_b.load_state_dict(ckb["comp_b"])
    comp_b.eval()

    cal_base = (ecfg.base_seed + pcfg.train_realizations
                + pcfg.val_realizations + 10000)
    nom = nominal_statistics(
        chan, sys, [cal_base + i for i in range(args.n_nominal)])
    tau_C = calibrate_threshold(nom["C"], args.target_fa)
    print(f"band {ccfg.name} | tau_C = {tau_C:.6f}")

    base = (ecfg.base_seed + pcfg.train_realizations
            + pcfg.val_realizations + 20000)
    seeds = [base + i for i in range(args.n)]

    rec_rows = {"matched": [], "mismatched": [], "fallback12": []}
    sC_mis, fire_mis = [], []
    d_all, d_fired = [], []

    print(f"running {len(seeds)} realizations (no blockage; "
          f"mismatch only)...")
    for i, s in enumerate(seeds):
        H = chan.realize(s)
        out = sys.rollout(H, s, set())
        x = out["x_enc_in"]
        x_true = out["x_target"]

        with torch.no_grad():
            rec_m = sys.comp.model(x)
            rec_x = mismatched_reconstruct(sys.comp.model, comp_b, x)
        rec_f = wideband_codebook(x, nr, ns, 3)

        rec_rows["matched"].append(nmse_db(rec_m, x))
        rec_rows["mismatched"].append(nmse_db(rec_x, x))
        rec_rows["fallback12"].append(nmse_db(rec_f, x))

        # The monitor sees encoder-vs-decoded error, which under mismatch is
        # exactly what blows up.
        sC = (((rec_x - x) ** 2).sum(dim=1)
              / (x ** 2).sum(dim=1).clamp(min=1e-9)).cpu().numpy()
        sC_mis.append(float(np.mean(sC)))
        fire_mis.append(float((sC > tau_C).mean()))

        # Utility: staying active (mismatched) vs falling back, per slot.
        def tput(rec):
            h_rec = real_to_complex(rec).reshape(-1, nr, ns)
            h_true = real_to_complex(x_true).reshape(-1, nr, ns)
            rep = snr_from_csi(h_rec, pcfg.snr_ref_db)
            tru = snr_from_csi(h_true, pcfg.snr_ref_db)
            t, _, _ = throughput_olla(rep, tru, pcfg.target_bler,
                                      pcfg.olla_step_db, pcfg.la_margin_db)
            return t.cpu().numpy()

        t_act, t_fb = tput(rec_x), tput(rec_f)
        d = t_fb - t_act
        d_all.append(float(d.mean()))
        m = sC > tau_C
        d_fired.append(float(d[m].mean()) if m.any() else np.nan)

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(seeds)}")

    m_db = float(np.mean(rec_rows["matched"]))
    x_db = float(np.mean(rec_rows["mismatched"]))
    f_db = float(np.mean(rec_rows["fallback12"]))

    print(f"\nreconstruction NMSE of the encoder input [dB]")
    print(f"  matched pair (A->A)      {m_db:8.2f}")
    print(f"  mismatched pair (A->B)   {x_db:8.2f}")
    print(f"  12-bit wideband fallback {f_db:8.2f}")
    print(f"\nC monitor under mismatch: mean s_C = {np.mean(sC_mis):.4f} "
          f"(tau = {tau_C:.4f}), firing on "
          f"{100*np.mean(fire_mis):.1f}% of slots")

    da = np.asarray(d_all, dtype=float)
    df = np.asarray(d_fired, dtype=float)
    df = df[~np.isnan(df)]
    lo_a, hi_a, p_a = paired_bootstrap_ci(da)
    print(f"\nthroughput change from falling back "
          f"(positive = fallback helps):")
    print(f"  all slots      {da.mean():+8.4f}  [{lo_a:+.4f},{hi_a:+.4f}]  "
          f"p={p_a:.4f}")
    if df.size >= 8:
        lo_f, hi_f, p_f = paired_bootstrap_ci(df)
        print(f"  monitor fired  {df.mean():+8.4f}  "
              f"[{lo_f:+.4f},{hi_f:+.4f}]  p={p_f:.4f}")
    else:
        lo_f = hi_f = p_f = float("nan")
        print("  monitor fired  (too few realizations with firings)")

    res = {"band": ccfg.name, "n": args.n, "tau_C": tau_C,
           "matched_db": m_db, "mismatched_db": x_db, "fallback_db": f_db,
           "sC_mean": float(np.mean(sC_mis)),
           "fire_frac": float(np.mean(fire_mis)),
           "d_all": {"delta": float(da.mean()), "ci95": [lo_a, hi_a],
                     "p": p_a},
           "d_fired": {"delta": float(df.mean()) if df.size else None,
                       "ci95": [lo_f, hi_f], "p": p_f}}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"pairing_{ccfg.name}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, default=float)

    print()
    if x_db - m_db < 3.0:
        print(f"Q3 FAILS: mismatch costs only {x_db - m_db:+.2f} dB. The "
              f"decoders are\ninterchangeable, which is itself worth "
              f"reporting, but this is not a\npositive control. Stop.")
    elif df.size >= 8 and lo_f > 0:
        print(f"Q3 and Q4 PASS: mismatch degrades the learned path to "
              f"{x_db:.2f} dB,\nbelow the {f_db:.2f} dB fallback, and the "
              f"fallback gains {df.mean():+.4f} bit/s/Hz\nwhen the monitor "
              f"fires. This is a regime where falling back is the RIGHT\n"
              f"action, so arbitration has something real to discriminate.\n"
              f"Proceed to the chain-vs-blind comparison in this regime.")
    elif df.size >= 8 and hi_f < 0:
        print(f"Q4 FAILS: even under pairing mismatch the fallback loses "
              f"({df.mean():+.4f})\nwhen the monitor fires. The compression "
              f"fallback has negative conditional\nvalue in BOTH regimes. "
              f"Report that and stop -- do not look for a third\nmechanism.")
    else:
        print("Q4 INCONCLUSIVE: the conditional effect is not "
              "distinguishable from\nzero. Raise --n before drawing any "
              "conclusion.")
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
