# Scientific source files

Use only the **exact final experiment scripts** used for the manuscript. Do not replace them with reconstructed code.

## Included exact source

- `arbiter.py` — author-supplied dependency-aware/blind-delay arbitration experiment used for Table VIII. The file is preserved without scientific edits.

Manuscript provenance currently identifies these filenames:

- `train.py`
- `induced.py`
- `experiment.py`
- `conditional.py`
- `check_encoder_ood.py`

Two final-paper artifacts still need their exact source filenames confirmed:

- Table VI mechanism diagnostic
- Table IX pairing-mismatch severity sweep

Also include imported project modules required by these scripts and any deterministic model/checkpoint generation code.

## Imports required by `arbiter.py`

The supplied arbitration script imports project-local modules that must also be added before the repository is reproducible:

- `config.py`
- `degradation.py`
- `experiment.py`
- `induced.py`
- `lcm.py`
- `stats.py`

It also expects trained checkpoints under `checkpoints/models_<band>.pt` unless the final project layout is changed. Keep the original relative layout if possible rather than editing the scientific script.
