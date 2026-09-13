# Concurrent Model Lifecycles at the UE

Reproducibility repository for the manuscript:

**Concurrent Model Lifecycles at the UE: Interaction Between Beam Management, CSI Prediction, and CSI Compression**  
Dharmendra Kumar, Member, IEEE

## Scope

This repository is intended to reproduce the link-level experiments studying concurrent AI/ML lifecycle state machines for beam management (B), CSI prediction (P), and CSI compression (C). The manuscript evaluates:

- independently calibrated lifecycle monitors and correlated firing;
- non-additive fallback interaction along the B → P → C dependency chain;
- induced downstream fallback;
- conditional fallback value;
- dependency-aware arbitration versus blind delay;
- encoder rate–robustness under blockage; and
- a controlled encoder–decoder pairing-mismatch severity sweep.

## Repository status

This package contains the **public-repository structure and paper-to-artifact manifest**. The exact author-supplied `src/arbiter.py` is now included. The remaining experiment source files, imported project modules, checkpoints, and raw/processed result files still need to be copied from the author's final local experiment directory before the repository is made public. No missing scientific scripts have been reconstructed from the paper text.

Run:

```bash
python scripts/verify_repo.py
```

before creating the public GitHub release. The check will report any required reproducibility artifacts that are still missing.

## Expected structure

```text
concurrent-model-lifecycles-ue/
├── README.md
├── CITATION.cff
├── .zenodo.json
├── .gitignore
├── REPRODUCIBILITY.md
├── RELEASE_CHECKLIST.md
├── artifact_manifest.yaml
├── src/                  # experiment source code
├── configs/              # paper configuration files
├── results/              # compact result files used by tables/figures
├── environment/          # exact Python/package environment
├── paper/                # optional final manuscript/preprint pointer
└── scripts/
    └── verify_repo.py
```

## Headline paper results

The final public repository should reproduce, within the paper's stated statistical procedure:

- maximum joint-monitor excess over independence: **25.3×**;
- P→C joint fallback interaction: **−26.2%** relative interaction at FR2;
- prediction fallback increases compression-fallback entries by **53%** at FR2;
- prediction fallback is beneficial conditional on its monitor firing, while the 12-bit compression fallback is harmful under blockage;
- the first tested fixed scalar quantizer matching the learned encoder uses **240 bits**, or **3.75×** the 64-bit learned feedback rate;
- blind delay outperforms dependency-aware arbitration under blockage;
- under pairing mismatch, dependency-aware arbitration beats blind delay only over an intermediate FR2 severity range and never beats no arbitration.

See `results/PAPER_VALUE_CHECKS.csv` and `artifact_manifest.yaml` for the full artifact mapping.

## Data policy

Channel realizations do not need to be distributed if they are deterministically regenerable from released seeds and configuration. Do **not** upload large caches if the repository includes all information necessary to regenerate them.

## GitHub → Zenodo release workflow

1. Complete and audit the repository.
2. Replace all `<...>` placeholders in `CITATION.cff` and `.zenodo.json`.
3. Choose and add a software license.
4. Make the GitHub repository public.
5. Enable the repository in Zenodo's GitHub integration.
6. Create a GitHub release, recommended tag: `v1.0-twc-submission`.
7. Let Zenodo archive the release and mint the DOI.
8. Add the Zenodo DOI to the manuscript's reproducibility statement.

## Citation

Use the citation metadata in `CITATION.cff` after replacing the GitHub URL placeholder and, once available, the Zenodo DOI.

## License

No software license has been selected in this scaffold. Add the license you intend to grant **before making the repository public**.

## Confirmed training and execution environment

The final training settings have been checked against the author-used `train.py` and `config.py`: Adam at `1e-3`, batch size 64, and 200 epochs for all three pipelines. Beam/predictor use weight decay `1e-4`/`1e-5` and validation-based checkpoint selection; the compressor uses no weight decay and the final-epoch model. See `configs/README.md` for the full table.

The confirmed execution environment is Python 3.12.14, PyTorch 2.14.0+cu130, Sionna 2.0.1, and an NVIDIA GeForce RTX 5070 Laptop GPU (12 GB). See `environment/README.md`. No training-time or inference-latency benchmark is claimed because this project did not log a dedicated timing measurement.

## Repository completeness status

This archive is **not yet a complete reproducibility release**. The exact author-used `arbiter.py` is included, but the remaining scientific source files and result artifacts must still be copied from the final experiment directory before publication. `scripts/verify_repo.py` is intentionally expected to report NOT READY until those required files are present.

