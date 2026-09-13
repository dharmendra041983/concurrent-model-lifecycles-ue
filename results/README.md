# Results directory

Place only the compact artifacts needed to verify the paper:

- per-realization outputs used for paired statistics;
- bootstrap/summary outputs used by the tables;
- CSV/JSON files used to generate figures;
- final model-validation metrics;
- mismatch-sweep outputs;
- seed lists / split manifests.

Avoid uploading large regenerable channel caches. The paper states that channel realizations are seed-addressed and regenerable.

`PAPER_VALUE_CHECKS.csv` contains a compact set of headline values for a post-run sanity check. It is not a substitute for the underlying per-realization result files.
