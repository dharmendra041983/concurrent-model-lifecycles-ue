# `arbiter.py` audit

Status: **exact author-supplied file included**

SHA-256: `dd5b1a23d48b413e0ff11c6d66271fa6deec53979d5a0658a2f435fb40249708`

Observed project-local imports:

- `config`
- `degradation`
- `experiment`
- `induced`
- `lcm`
- `stats`

External imports: `numpy`, `torch`.

The script selects CUDA when available and otherwise CPU, loads `checkpoints/models_<band>.pt`, calibrates per-pipeline thresholds from nominal realizations, evaluates blocked realizations, and compares dependency-aware (`chain`) versus unconditional (`blind`) downstream hold policies.

No credentials, user-specific absolute paths, or obvious secrets were found in this file. The file passes Python syntax compilation.
