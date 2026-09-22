# Release checklist for v1.0.1-twc-submission

- [x] Complete experiment source set included
- [x] `config.py` synchronized to k=2, 1024/128, 200 epochs, batch 64
- [x] `lcm.py` synchronized to q_exit=25
- [x] Table/figure-to-script mapping included
- [x] Paper-reported aggregate checks included
- [x] Environment recorded
- [x] Linux and PowerShell rerun scripts included
- [x] No raw channel cache included
- [x] GitHub and Zenodo metadata updated to exact paper title
- [ ] Choose software license and replace `LICENSE_REQUIRED.txt` with `LICENSE`
- [ ] Run `python scripts/verify_repo.py` after upload
- [ ] Ideally rerun the scientific workflow or at least selected smoke tests
- [ ] Create GitHub release `v1.0.1-twc-submission`
- [ ] Confirm Zenodo archives v1.0.1 and copy the new version DOI
- [ ] Replace old DOI in the TWC manuscript with the new version DOI
