"""
Train the three pipelines (Sec. VII-A).

Each model must pass an explicit acceptance criterion before it is allowed to
serve as the "healthy" baseline. If a model is weak, its fallback looks cheap,
and the interaction term you measure describes a badly trained model rather
than lifecycle coupling.

Two design points that were learned the hard way and should stay:

  * The beam is HELD across a sweep period, not reselected every slot.
    Reselecting per slot makes h_eff = H w_b discontinuous wherever the best
    beam changes, which destroys the temporal correlation CSI prediction
    depends on and makes even hold-last useless.

  * Beam health is judged by BEAMFORMING GAIN LOSS, not top-1 accuracy.
    Top-1 penalises picking a near-optimal neighbouring beam, which costs
    almost nothing in utility. Gain loss is what propagates to throughput.

Training and evaluation realizations are disjoint by construction.
"""

from __future__ import annotations

import argparse
import copy
import os
import numpy as np
import torch
import torch.nn as nn

from config import FR1, FR2, DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
from channel import SharedChannel
from pipelines import (BeamModel, CSIPredictor, CSIAutoencoder,
                       to_subbands, complex_to_real, hold_beam)

# Acceptance criteria. Tune once, then freeze and report in the paper.
ACCEPT = {
    # The right reference for beam health is the NON-AI FALLBACK the model
    # replaces -- exhaustive sweep, selected per slot and held across the
    # period. The period-optimal ceiling is computed with hindsight over the
    # whole period and is unattainable from a single-slot measurement
    # snapshot (which carries no information about UE motion direction);
    # the fallback does not reach it either, so grading against it was wrong.
    # What matters is that the model tracks the fallback closely while
    # measuring a fraction of the codebook.
    "beam_excess_over_fallback_db": 0.35,
    "beam_gain_loss_db": 2.5,   # loose sanity bound only
    "pred_gain_over_hold_db": 3.0,  # predictor must beat hold-last by this
    "pred_nmse_db": -5.0,
    "comp_nmse_db": -10.0,
}


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"



def forward_period_optimal(gain: torch.Tensor, period: int) -> torch.Tensor:
    """For each slot t, the beam maximising MEAN gain over [t, t+period).

    Under a held beam regime the instantaneously-best beam is the wrong
    target: the selection made at t governs the whole period, so the model
    should be trained to pick what is best across it. Training on the
    per-slot argmax produces a model that matches a perfect instantaneous
    selector and still loses ~1 dB to the period-optimal ceiling.

    Averaging is done in linear power; averaging dB understates deep fades.
    """
    T, B = gain.shape
    lin = 10.0 ** (gain / 10.0)
    cs = torch.cat([torch.zeros(1, B, device=gain.device, dtype=lin.dtype),
                    lin.cumsum(dim=0)], dim=0)                  # [T+1, B]
    idx = torch.arange(T, device=gain.device)
    end = torch.clamp(idx + period, max=T)
    window = cs[end] - cs[idx]                                  # [T, B]
    count = (end - idx).unsqueeze(1).to(window.dtype)
    return (window / count).argmax(dim=1)


def period_optimal_ceiling(data, pcfg) -> float:
    """Gain loss of the best possible HELD selector, in dB. The floor set by
    the sweep periodicity, below which no beam model can go."""
    from pipelines import hold_beam as _hb
    out = []
    for gain in data["gain"]:
        tgt = forward_period_optimal(gain, pcfg.sweep_period_slots)
        held = _hb(tgt, pcfg.sweep_period_slots)
        T = gain.shape[0]
        g_sel = gain[torch.arange(T, device=gain.device), held]
        out.append((gain.max(dim=1).values - g_sel).mean().item())
    return float(np.mean(out))



def fallback_reference(data, pcfg) -> float:
    """Gain loss of the exhaustive-sweep fallback: per-slot best, held.

    This is what the non-AI beam path actually delivers, at 4x the
    measurement overhead. It is the reference the AI model must track.
    """
    from pipelines import hold_beam as _hb
    out = []
    for gain, best in zip(data["gain"], data["best"]):
        held = _hb(best, pcfg.sweep_period_slots)
        T = gain.shape[0]
        g_sel = gain[torch.arange(T, device=gain.device), held]
        out.append((gain.max(dim=1).values - g_sel).mean().item())
    return float(np.mean(out))


def build_dataset(chan: SharedChannel, seeds, pcfg):
    """Collect per-realization tensors."""
    out = {"meas": [], "best": [], "csi": [], "gain": [], "target": []}
    for s in seeds:
        H = chan.realize(s)
        meas = chan.measure(H, chan.set_b, s)
        all_idx = torch.arange(chan.num_beams, device=H.device)
        gain = chan.beamformed_gain(H, all_idx)         # [T, num_beams]
        best = gain.argmax(dim=1)

        # Hold the beam across the sweep period, then build the CSI sequence
        # on the HELD beam. P and C are trained on the genuine best beam, not
        # the beam model's output, so that one particular beam-model error
        # pattern is not baked into the downstream models.
        held = hold_beam(best, pcfg.sweep_period_slots)
        Wb = chan.codebook[held]
        h_eff = torch.einsum("tnmf,tm->tnf", H, Wb.conj())
        h_sb = to_subbands(h_eff, pcfg.csi_subbands)
        flat = complex_to_real(h_sb.reshape(h_sb.shape[0], -1))

        out["meas"].append(meas)
        out["best"].append(best)
        out["target"].append(
            forward_period_optimal(gain, pcfg.sweep_period_slots))
        out["csi"].append(flat)
        out["gain"].append(gain)
    return out


def _beam_gain_loss(model, data, pcfg, device) -> tuple[float, float]:
    """(mean gain loss in dB, top-1 accuracy) on held beams."""
    losses, accs = [], []
    with torch.no_grad():
        for meas, gain, best in zip(data["meas"], data["gain"], data["best"]):
            pred = model(meas.to(device)).argmax(dim=-1)
            pred = hold_beam(pred, pcfg.sweep_period_slots)
            gain = gain.to(device)
            T = gain.shape[0]
            g_sel = gain[torch.arange(T, device=device), pred]
            g_best = gain.max(dim=1).values
            losses.append((g_best - g_sel).mean().item())
            accs.append((pred == best.to(device)).float().mean().item())
    return float(np.mean(losses)), float(np.mean(accs))


def train_beam(chan, tr, va, pcfg, device):
    X = torch.cat(tr["meas"]).to(device)
    y = torch.cat(tr["target"]).to(device)   # period-optimal, not per-slot
    model = BeamModel(len(chan.set_b), chan.num_beams).to(device)
    # Weight decay: the beam model overfits badly without it -- training loss
    # keeps falling while validation accuracy degrades.
    opt = torch.optim.Adam(model.parameters(), lr=pcfg.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n = X.shape[0]

    best_loss, best_state, best_ep = float("inf"), None, -1
    for ep in range(pcfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, pcfg.batch_size):
            idx = perm[i: i + pcfg.batch_size]
            opt.zero_grad()
            loss = lossf(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == pcfg.epochs - 1:
            model.eval()
            gl, acc = _beam_gain_loss(model, va, pcfg, device)
            print(f"  [beam] epoch {ep:3d}  loss {tot/n:.4f}  "
                  f"val gain-loss {gl:.3f} dB  top1 {acc:.3f}")
            # Select on validation gain loss, not training loss.
            if gl < best_loss:
                best_loss, best_ep = gl, ep
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [beam] restored epoch {best_ep} "
              f"(val gain-loss {best_loss:.3f} dB)")
    model.eval()
    return model


def _pred_tensors(csi, pcfg, device):
    W, k = pcfg.pred_window, pcfg.feedback_delay
    xs, ys = [], []
    for flat in csi:
        T = flat.shape[0]
        for s in range(0, T - W - k):
            xs.append(flat[s: s + W])
            ys.append(flat[s + W + k])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def train_predictor(tr, va, pcfg, device):
    X, Y = _pred_tensors(tr["csi"], pcfg, device)
    Xv, Yv = _pred_tensors(va["csi"], pcfg, device)
    model = CSIPredictor(X.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=pcfg.lr, weight_decay=1e-5)
    n = X.shape[0]
    best, best_state, best_ep = float("inf"), None, -1
    for ep in range(pcfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, pcfg.batch_size):
            idx = perm[i: i + pcfg.batch_size]
            opt.zero_grad()
            loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == pcfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                v = (((model(Xv) - Yv) ** 2).sum() / (Yv ** 2).sum()).item()
            print(f"  [pred] epoch {ep:3d}  mse {tot/n:.6f}  "
                  f"val nmse {10*np.log10(v+1e-12):.2f} dB")
            if v < best:
                best, best_ep = v, ep
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [pred] restored epoch {best_ep}")
    model.eval()
    return model


def train_compressor(tr, pcfg, device):
    X = torch.cat(tr["csi"]).to(device)
    model = CSIAutoencoder(X.shape[-1], pcfg.latent_dim,
                           pcfg.bits_per_latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=pcfg.lr)
    n = X.shape[0]
    for ep in range(pcfg.epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, pcfg.batch_size):
            idx = perm[i: i + pcfg.batch_size]
            opt.zero_grad()
            loss = ((model(X[idx]) - X[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 20 == 0:
            print(f"  [comp] epoch {ep:3d}  mse {tot/n:.6f}")
    model.eval()
    return model


def evaluate(models, va, pcfg, device) -> dict:
    beam, pred, comp = models
    gl, acc = _beam_gain_loss(beam, va, pcfg, device)
    Xs, Ys = _pred_tensors(va["csi"], pcfg, device)
    with torch.no_grad():
        p = pred(Xs)
        pred_nmse = (((p - Ys) ** 2).sum() / (Ys ** 2).sum()).item()
        hold_nmse = (((Xs[:, -1] - Ys) ** 2).sum() / (Ys ** 2).sum()).item()
        Xc = torch.cat(va["csi"]).to(device)
        r = comp(Xc)
        comp_nmse = (((r - Xc) ** 2).sum() / (Xc ** 2).sum()).item()
    pn = 10 * np.log10(pred_nmse + 1e-12)
    hn = 10 * np.log10(hold_nmse + 1e-12)
    ceiling = period_optimal_ceiling(va, pcfg)
    fb = fallback_reference(va, pcfg)
    return {
        "beam_gain_loss_db": gl,
        "beam_fallback_db": fb,
        "beam_excess_over_fallback_db": gl - fb,
        "beam_ceiling_db": ceiling,
        "beam_excess_over_ceiling_db": gl - ceiling,
        "beam_top1": acc,
        "pred_nmse_db": pn,
        "hold_last_nmse_db": hn,
        "pred_gain_over_hold_db": hn - pn,
        "comp_nmse_db": 10 * np.log10(comp_nmse + 1e-12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["FR2", "FR1"], default="FR2")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    ccfg = FR2 if args.band == "FR2" else FR1
    ecfg, pcfg = DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
    device = _device()
    print(f"device: {device}  band: {ccfg.name}")

    chan = SharedChannel(ccfg, ecfg, device=device)
    base = ecfg.base_seed
    train_seeds = range(base, base + pcfg.train_realizations)
    val_seeds = range(base + pcfg.train_realizations,
                      base + pcfg.train_realizations + pcfg.val_realizations)

    print("building training set...")
    tr = build_dataset(chan, train_seeds, pcfg)
    print("building validation set...")
    va = build_dataset(chan, val_seeds, pcfg)

    beam = train_beam(chan, tr, va, pcfg, device)
    pred = train_predictor(tr, va, pcfg, device)
    comp = train_compressor(tr, pcfg, device)

    m = evaluate((beam, pred, comp), va, pcfg, device)
    print("\nvalidation:")
    for k_, v in m.items():
        print(f"  {k_:26s} {v:8.4f}")

    fails = []
    if m["beam_excess_over_fallback_db"] > ACCEPT["beam_excess_over_fallback_db"]:
        fails.append("beam model does not track the exhaustive-sweep fallback")
    if m["beam_gain_loss_db"] > ACCEPT["beam_gain_loss_db"]:
        fails.append("beam gain loss above the sanity bound")
    if m["pred_gain_over_hold_db"] < ACCEPT["pred_gain_over_hold_db"]:
        fails.append("predictor does not beat hold-last by enough -- P's "
                     "fallback would look nearly free")
    if m["pred_nmse_db"] > ACCEPT["pred_nmse_db"]:
        fails.append("prediction NMSE above acceptance")
    if m["comp_nmse_db"] > ACCEPT["comp_nmse_db"]:
        fails.append("compression NMSE above acceptance")

    if fails:
        print("\nACCEPTANCE FAILED:")
        for f in fails:
            print("  -", f)
        print("Do not run the interaction experiment on these models.")
    else:
        print("\nall models accepted.")

    os.makedirs(args.out, exist_ok=True)
    torch.save({"beam": beam.state_dict(), "pred": pred.state_dict(),
                "comp": comp.state_dict(), "metrics": m, "band": ccfg.name},
               os.path.join(args.out, f"models_{ccfg.name}.pt"))
    print(f"saved -> {args.out}/models_{ccfg.name}.pt")


if __name__ == "__main__":
    main()
