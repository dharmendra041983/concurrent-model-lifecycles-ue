"""
Train a second CSI autoencoder for the pairing-mismatch positive control.

Release-20 two-sided CSI compression pairs a UE encoder with a network
decoder. Pairing can go wrong: the network may switch to a decoder the UE's
encoder was not trained against, or a transition may leave the pair
inconsistent. That is a real, standards-motivated failure mode, and unlike
blockage it should genuinely take the encoder out of distribution -- the
latent space of an independently initialized autoencoder is arbitrary, so a
mismatched decoder has no reason to interpret it correctly.

This trains decoder B with a different initialization on the same data. The
mismatched pair is then encoder A into decoder B.

PRE-REGISTERED, before any of this was run:
  * The mismatch is chosen on standards grounds, not selected because it
    produces a convenient result.
  * Both regimes -- blockage and mismatch -- will be reported side by side
    whichever way the second one lands.
  * Prediction: a mismatched decoder is NOT preceded by an upstream state
    change, so the dependency-chain arbiter has no reason to hold C's
    request while blind delay holds it anyway. Chain should therefore beat
    blind here, reversing the blockage result. If it does not, the arbiter
    is refuted in both regimes and the negative result stands as final.
"""

from __future__ import annotations

import argparse
import os
import torch

from config import FR1, FR2, DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
from channel import SharedChannel
from train import build_dataset, train_compressor, _device
from pipelines import CSIAutoencoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["FR2", "FR1"], default="FR2")
    ap.add_argument("--seed-offset", type=int, default=777,
                    help="changes initialization only; training data is "
                         "identical to decoder A's")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    ccfg = FR2 if args.band == "FR2" else FR1
    ecfg, pcfg = DEFAULT_EXPERIMENT, DEFAULT_PIPELINE
    device = _device()
    print(f"device: {device}  band: {ccfg.name}")

    chan = SharedChannel(ccfg, ecfg, device=device)
    base = ecfg.base_seed
    train_seeds = range(base, base + pcfg.train_realizations)

    print("building training set (same data as decoder A)...")
    tr = build_dataset(chan, train_seeds, pcfg)

    # Different initialization, identical data and hyperparameters. The point
    # is a legitimately trained decoder that simply is not the UE's pair.
    torch.manual_seed(ecfg.base_seed + args.seed_offset)
    comp_b = train_compressor(tr, pcfg, device)

    X = torch.cat(tr["csi"]).to(device)
    with torch.no_grad():
        own = (((comp_b(X) - X) ** 2).sum() / (X ** 2).sum()).item()
    print(f"\ndecoder B, matched with its own encoder: "
          f"{10 * torch.log10(torch.tensor(own)).item():.2f} dB")

    path = os.path.join(args.out, f"models_{ccfg.name}_pairB.pt")
    os.makedirs(args.out, exist_ok=True)
    torch.save({"comp_b": comp_b.state_dict(), "band": ccfg.name,
                "seed_offset": args.seed_offset}, path)
    print(f"saved -> {path}")
    print("\nNext: python pairing.py --band " + ccfg.name)


if __name__ == "__main__":
    main()
