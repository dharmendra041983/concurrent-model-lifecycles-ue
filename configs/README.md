# Final training configuration

The manuscript and repository should use the following confirmed settings.

| Parameter | Beam | Predictor | Compressor |
|---|---:|---:|---:|
| Optimizer | Adam | Adam | Adam |
| Learning rate | 1e-3 | 1e-3 | 1e-3 |
| Weight decay | 1e-4 | 1e-5 | 0 |
| Batch size | 64 | 64 | 64 |
| Epochs | 200 | 200 | 200 |
| Validation metric | gain loss | NMSE | none |
| Checkpoint | best evaluated validation checkpoint | best evaluated validation checkpoint | final epoch |
| Validation cadence | every 10 epochs | every 10 epochs | n/a |

Shared data/configuration details:

- Training realizations: 1024
- Validation realizations: 128, disjoint from training
- Predictor history window W: 8 slots
- Predictor horizon k: 2 slots
- Compressor latent dimension: 16
- Quantization: 4 bits per latent
- Learned compression report: 64 bits
- LCM q_exit: 25 consecutive clean evaluations
- Main statistical realization count in config: 512

The predictor and compressor are trained on the genuine best beam rather than
the beam model output, matching the manuscript methodology.
