"""
Beam gain-loss decomposition.

The evaluated gain loss of the beam pipeline mixes two things:

  1. HOLDING LOSS -- the beam is held across a sweep period while the channel
     keeps evolving. Even a perfect selector pays this. It is a property of
     the sweep periodicity and the Doppler, not of the model.

  2. MODEL LOSS -- the excess above the best achievable held selection.

Only (2) is something training can improve. If (1) dominates, the acceptance
threshold is measuring the configuration rather than the model, and the fix is
a shorter sweep period or a period-aware training target, not more epochs.

Reference points computed here, all in dB below the per-slot optimum:

  per-slot oracle       0 by definition (reselects every slot, unattainable)
  held-from-start       oracle picks the best beam at the period start, holds
  period-optimal held   oracle picks the beam maximising MEAN gain over the
                        period, holds -- this is the true ceiling for any
                        held selector
  model                 what the trained beam model actually achieves
"""

from __future__ import annotations

import sys
import numpy as np
import torch

sys.path.insert(0, ".")

from config import FR2, FR1, DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
from channel import SharedChannel
from pipelines import BeamModel, hold_beam


def period_optimal_beam(gain: torch.Tensor, period: int) -> torch.Tensor:
    """Beam maximising mean gain within each period, held across it. [T]"""
    T, B = gain.shape
    n_full = (T // period) * period
    g = gain[:n_full].reshape(-1, period, B)          # [P, period, B]
    # Average in linear power, not dB -- averaging dB understates deep fades.
    lin = 10.0 ** (g / 10.0)
    choice = lin.mean(dim=1).argmax(dim=1)            # [P]
    held = choice.repeat_interleave(period)
    if n_full < T:
        held = torch.cat([held, choice[-1].repeat(T - n_full)])
    return held


def gain_loss(gain: torch.Tensor, sel: torch.Tensor) -> float:
    T = gain.shape[0]
    g_sel = gain[torch.arange(T, device=gain.device), sel]
    g_best = gain.max(dim=1).values
    return (g_best - g_sel).mean().item()


def main():
    band = sys.argv[1] if len(sys.argv) > 1 else "FR2"
    ccfg = FR2 if band == "FR2" else FR1
    ecfg, pcfg = DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    P = pcfg.sweep_period_slots

    chan = SharedChannel(ccfg, ecfg, device=dev)

    ck = torch.load(f"checkpoints/models_{ccfg.name}.pt", map_location=dev,
                    weights_only=False)
    model = BeamModel(len(chan.set_b), chan.num_beams).to(dev)
    model.load_state_dict(ck["beam"])
    model.eval()

    base = ecfg.base_seed + pcfg.train_realizations
    seeds = [base + i for i in range(pcfg.val_realizations)]

    rows = {"held_from_start": [], "period_optimal": [], "model": [],
            "model_held_from_start_pred": [], "switch_rate": []}

    all_idx = torch.arange(chan.num_beams, device=dev)
    for s in seeds:
        H = chan.realize(s)
        gain = chan.beamformed_gain(H, all_idx)
        best = gain.argmax(dim=1)

        rows["switch_rate"].append(
            (best[1:] != best[:-1]).float().mean().item())
        rows["held_from_start"].append(gain_loss(gain, hold_beam(best, P)))
        rows["period_optimal"].append(
            gain_loss(gain, period_optimal_beam(gain, P)))

        meas = chan.measure(H, chan.set_b, s)
        with torch.no_grad():
            pred = model(meas).argmax(dim=-1)
        rows["model"].append(gain_loss(gain, hold_beam(pred, P)))
        rows["model_held_from_start_pred"].append(gain_loss(gain, pred))

    m = {k: float(np.mean(v)) for k, v in rows.items()}

    print(f"\nband {ccfg.name} | sweep period {P} slots "
          f"({P * ccfg.slot_duration * 1e3:.2f} ms) | "
          f"speed {ccfg.min_speed}-{ccfg.max_speed} m/s")
    print(f"best-beam switch rate: {m['switch_rate']:.3f} per slot\n")

    print("mean gain loss vs per-slot optimum [dB]:")
    print(f"  oracle, held from period start   {m['held_from_start']:7.3f}")
    print(f"  oracle, period-optimal held      {m['period_optimal']:7.3f}"
          "   <-- ceiling for any held selector")
    print(f"  trained model, held              {m['model']:7.3f}")
    print(f"  trained model, NOT held          "
          f"{m['model_held_from_start_pred']:7.3f}"
          "   <-- per-slot skill of the model")

    excess = m["model"] - m["period_optimal"]
    print(f"\n  model excess over held ceiling   {excess:7.3f} dB")

    print()
    if m["period_optimal"] > 1.0:
        print("VERDICT: the holding floor dominates. The acceptance threshold "
              "is\nmeasuring the sweep periodicity, not the model. Shorten "
              "sweep_period_slots\nor train against the period-optimal target "
              "-- more epochs will not help.")
    elif excess > 0.5:
        print("VERDICT: real model error. The held ceiling is cheap, so the "
              "gap is\nthe model's. Worth more capacity, a larger Set B, or "
              "more realizations.")
    else:
        print("VERDICT: the model is close to the held ceiling. Raise the "
              "acceptance\nthreshold to just above the ceiling and proceed.")


if __name__ == "__main__":
    main()
