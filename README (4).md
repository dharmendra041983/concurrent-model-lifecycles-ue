# Results

`paper_reported/` contains compact aggregate values transcribed from the TWC
submission manuscript. They are provided as sanity checks and are **not raw
simulation outputs**.

`raw/` is intentionally empty in this archive. The experiment scripts write
JSON outputs there (or to any directory passed via `--out`) when rerun. The
channel realizations are also not distributed because they are deterministically
regenerable from the released seeds and configuration.

Expected generated files include:

- `interaction_FR1.json`, `interaction_FR2.json`
- `induced_FR1.json`, `induced_FR2.json`
- `conditional_FR1.json`, `conditional_FR2.json`
- `mechanism_FR1.json`, `mechanism_FR2.json`
- `nominal_corr_FR1.json`, `nominal_corr_FR2.json`
- `pairing_FR1.json`, `pairing_FR2.json`
- `arbiter_FR1.json`, `arbiter_FR2.json`
- `arbiter_mismatch_FR1.json`, `arbiter_mismatch_FR2.json`

`check_encoder_ood.py` prints the rate-robustness table; the provided run script
captures that output to text.
