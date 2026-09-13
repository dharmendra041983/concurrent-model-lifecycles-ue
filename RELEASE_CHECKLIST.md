# Public Release Checklist

## Scientific completeness
- [ ] Exact `train.py` added
- [ ] Exact `induced.py` added
- [ ] Exact `experiment.py` added
- [ ] Exact `conditional.py` added
- [ ] Exact `arbiter.py` added
- [ ] Exact `check_encoder_ood.py` added
- [ ] Exact Table VI diagnostic script added
- [ ] Exact Table IX mismatch-sweep script added
- [ ] Imported source modules added
- [ ] Final config files added
- [ ] Train/validation/evaluation seed lists added
- [ ] Required model checkpoints or deterministic checkpoint-generation procedure added
- [ ] Compact per-realization / summary results added
- [ ] Figure-generation scripts added

## Reproducibility
- [ ] Exact environment exported
- [ ] Clean-machine/sandbox smoke test passes
- [ ] `python scripts/verify_repo.py` passes
- [ ] Headline values match `results/PAPER_VALUE_CHECKS.csv`
- [ ] No dependence on absolute local paths

## Public-release hygiene
- [ ] No credentials, tokens, keys, cookies, or `.env` files
- [ ] No employer/proprietary code or data
- [ ] No unrelated unpublished-paper artifacts
- [ ] No peer-review correspondence or submission-system files
- [ ] No large regenerable channel caches
- [ ] Git history checked for accidentally committed secrets
- [ ] Software license selected and added
- [ ] `<YOUR_GITHUB_USERNAME>` placeholder replaced

## GitHub / Zenodo
- [ ] GitHub repository made public
- [ ] Zenodo GitHub integration enabled
- [ ] Release tag created: `v1.0-twc-submission`
- [ ] Zenodo DOI minted
- [ ] DOI added to README / CITATION metadata
- [ ] Manuscript reproducibility statement updated with stable DOI
