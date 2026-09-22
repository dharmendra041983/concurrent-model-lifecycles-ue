# Configuration synchronization audit

The uploaded source snapshot contained older experimental defaults:

- prediction horizon `feedback_delay = 4`;
- training realizations `256`;
- validation realizations `64`;
- epochs `40`;
- lifecycle `q_exit = 10`.

Those values do not match the final TWC manuscript or the author's final local
run configuration previously verified from the experiment environment. This
package therefore changes only those defaults to the final reported values:

- `feedback_delay = 2`;
- `train_realizations = 1024`;
- `val_realizations = 128`;
- `epochs = 200`;
- `batch_size = 64` (already correct);
- `q_exit = 25`.

No scientific algorithm, model architecture, statistic, or arbitration rule was
otherwise rewritten during packaging.
