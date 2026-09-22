# Seed plan

The code is seed-addressed through `ExperimentConfig.base_seed = 20260907`.
With the final training split (1024 train + 128 validation):

- training: `base_seed ... base_seed+1023`;
- validation: next 128 seeds;
- nominal monitor calibration: `base + train + val + 10000 + i`;
- blocked lifecycle experiments: `base + train + val + 20000 + i`;
- encoder robustness: `base + train + val + 30000 + i`;
- disjoint large nominal correlation evaluation: `base + train + val + 40000 + i`.

Individual scripts are authoritative for the exact sample count and offset used.
