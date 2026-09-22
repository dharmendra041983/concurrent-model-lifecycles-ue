# Concurrent Model Lifecycles at the UE

Reproducibility artifact for the manuscript:

**Concurrent Model Lifecycles at the UE: Interaction Between Beam Management, CSI Prediction, and CSI Compression**  
Dharmendra Kumar, Member, IEEE

## What is included

This release contains the complete author-supplied experiment source set needed
for the reported beam-management, CSI-prediction, CSI-compression, lifecycle,
arbitration, encoder-robustness, and pairing-mismatch experiments. It also
contains the final manuscript-level configuration, deterministic seed plan,
environment record, run scripts, and compact paper-reported aggregate checks.

The channel realizations and raw result JSON files are not distributed. They are
regenerable from the released code, seeds, and configuration. The CSV files under
`results/paper_reported/` are transcriptions of the manuscript tables for
post-run sanity checking; they are not raw simulation outputs.

## Important synchronization note

The source snapshot supplied for packaging contained older defaults in
`config.py` and `lcm.py`. This release synchronizes those defaults to the final
experiment settings reported in the submitted manuscript: prediction horizon
`k=2`, 1024 training realizations, 128 validation realizations, 200 epochs,
batch size 64, and `q_exit=25`. Details are in `checks/CONFIG_SYNC.md`.

## Quick audit

```bash
python scripts/verify_repo.py
```

This checks repository completeness, Python syntax, and the key configuration
values. It does not replace the full scientific rerun.

## Full rerun

Linux/macOS:

```bash
./scripts/run_all.sh
```

Windows PowerShell:

```powershell
./scripts/run_all.ps1
```

The full experiment is computationally expensive because it retrains both FR1
and FR2 models and executes hundreds of Sionna channel realizations per study.

## Repository layout

```text
concurrent-model-lifecycles-ue/
├── src/                  # complete experiment source code
├── configs/              # final configuration + seed plan
├── results/
│   ├── paper_reported/   # aggregate manuscript values (sanity checks)
│   └── raw/              # regenerated JSON/text outputs go here
├── environment/          # execution environment
├── paper/                # paper-to-artifact notes
├── scripts/              # audit + full rerun scripts
├── checks/               # source/config audit notes + hashes
├── artifact_manifest.yaml
├── CITATION.cff
└── .zenodo.json
```

## GitHub / Zenodo

Repository: https://github.com/dharmendra041983/concurrent-model-lifecycles-ue

Recommended corrected release tag after uploading this package:
`v1.0.1-twc-submission`. Let Zenodo archive that release and use the resulting
version-specific DOI in the TWC manuscript.

## License

A software license has not yet been selected. `LICENSE_REQUIRED.txt` is retained
intentionally; replace it with the chosen `LICENSE` before treating the release
as the final public software distribution.
