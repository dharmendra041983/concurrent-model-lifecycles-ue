# Configuration

`final_config.json` records the manuscript-level configuration. The executable
configuration is `src/config.py`. The release checker verifies the key values
that must match the submitted manuscript: prediction horizon `k=2`, 1024/128
train/validation realizations, 200 epochs, batch size 64, and `q_exit=25`.

The uploaded source snapshot contained older defaults (`k=4`, 256/64
train/validation realizations, 40 epochs, `q_exit=10`). Those defaults were
synchronized here to the final settings independently confirmed from the
author's final experiment environment and reported in the manuscript. See
`checks/CONFIG_SYNC.md`.
