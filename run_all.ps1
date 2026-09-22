$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PWD\src"
New-Item -ItemType Directory -Force checkpoints, results\raw | Out-Null
python src/channel.py
python src/train.py --band FR2 --out checkpoints
python src/train.py --band FR1 --out checkpoints
python src/experiment.py --band FR2 --n 512 --out results/raw
python src/experiment.py --band FR1 --n 512 --out results/raw
python src/nominal_corr.py --band FR2 --n-eval 512 --out results/raw
python src/nominal_corr.py --band FR1 --n-eval 512 --out results/raw
python src/induced.py --band FR2 --n 256 --out results/raw
python src/induced.py --band FR1 --n 256 --out results/raw
python src/conditional.py --band FR2 --n 256 --out results/raw
python src/conditional.py --band FR1 --n 256 --out results/raw
python src/mechanism.py --band FR2 --n 64 --out results/raw
python src/check_encoder_ood.py --band FR2 --n 64 | Tee-Object -FilePath results/raw/encoder_rate_robustness_FR2.txt
python src/arbiter.py --band FR2 --n 256 --out results/raw
python src/arbiter.py --band FR1 --n 256 --out results/raw
python src/train_pair.py --band FR2 --out checkpoints
python src/train_pair.py --band FR1 --out checkpoints
python src/pairing.py --band FR2 --n 64 --out results/raw
python src/pairing.py --band FR1 --n 64 --out results/raw
python src/arbiter_mismatch.py --band FR2 --n 512 --out results/raw
python src/arbiter_mismatch.py --band FR1 --n 128 --out results/raw
