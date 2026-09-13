# Reproducibility Notes

## Paper configuration captured in the manuscript

### Channel / simulation
- Sionna PHY: **2.0.1**
- FR2: CDL-B, 28 GHz, 120 kHz SCS, 30 ns delay spread, 4×8 single-polarized BS array, 1×2 UE array
- FR1: CDL-C, 3.5 GHz, 30 kHz SCS, 300 ns delay spread, 2×8 BS array
- UE speed: uniform on [1, 15] m/s
- Episode length: 200 slots
- UE orientation randomized per realization
- DFT beam codebook oversampled 4× in azimuth: 128 beams FR2, 64 beams FR1
- Set B: fixed 25% subset

### Learned pipelines
- Beam model: two-hidden-layer MLP, 256 units, mean-normalized Set-B RSRP → Set-A beam logits
- Predictor: single-layer GRU, 128 units, history W=8, prediction horizon k=2
- Compressor: MLP autoencoder, 16-dimensional latent, 4-bit quantization with straight-through estimator, 64-bit report
- Compression fallback: wideband 3-bit-per-antenna report, 12 bits total

### Training
- 1024 realizations for training
- 128 held-out validation realizations
- Adam learning rate: 1e-3
- Batch size: 512
- Epochs: 200
- Best validation checkpoint retained
- Predictor and compressor trained on the genuine best beam rather than the beam model output

### Lifecycle
- States: ACTIVE → SUSPECT → FALLBACK
- q_enter = 3 consecutive exceedances
- q_exit = 25 consecutive clean evaluations in reported results
- Per-pipeline thresholds calibrated independently to 5% nominal false-alarm rate on 64 nominal realizations
- Nominal monitor statistics evaluated on a disjoint 512-realization set

### Statistics
- Unit of analysis: realization, not slot
- Paired matched-seed comparisons
- 95% percentile bootstrap
- 10,000 bootstrap resamples
- Holm correction across the four planned subset comparisons
- Interaction experiment: 512 realizations
- Lifecycle experiments: 256 realizations unless otherwise stated

## Exact software environment still required

Before public release, export the exact environment used for the final runs. At minimum record:

- Python version
- Sionna version (paper states 2.0.1)
- numerical/ML backend versions
- NumPy/SciPy/pandas/matplotlib versions if used
- GPU/CPU package variants if they affect deterministic execution
- operating-system information if relevant

Do not guess these versions from a different project environment.

## Hardware

The manuscript makes no runtime, energy, or hardware-efficiency claim. Hardware is therefore not required to interpret the scientific claims, but recording the final execution platform in the repository is useful for reproducibility.
