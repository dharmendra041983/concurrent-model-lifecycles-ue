#!/usr/bin/env python3
"""Lightweight repository-completeness checker.

This does not validate scientific correctness. It checks that the files the
paper says are needed for reproduction have been supplied before public release.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

required = [
    "README.md",
    "CITATION.cff",
    ".zenodo.json",
    "artifact_manifest.yaml",
    "REPRODUCIBILITY.md",
    "results/PAPER_VALUE_CHECKS.csv",
    "src/train.py",
    "src/induced.py",
    "src/experiment.py",
    "src/conditional.py",
    "src/arbiter.py",
    "src/check_encoder_ood.py",
]

missing = [p for p in required if not (ROOT / p).exists()]

placeholders = []
for rel in ["README.md", "CITATION.cff", ".zenodo.json", "RELEASE_CHECKLIST.md"]:
    p = ROOT / rel
    if p.exists() and re.search(r"<[^>]+>", p.read_text(encoding="utf-8", errors="ignore")):
        placeholders.append(rel)

# Basic secret-like filename scan only. Still inspect Git history manually.
secret_names = []
for p in ROOT.rglob("*"):
    if p.is_file() and p.name.lower() in {".env", "id_rsa", "credentials.json", "secrets.json"}:
        secret_names.append(str(p.relative_to(ROOT)))

print("Repository:", ROOT)
if missing:
    print("\nMISSING REQUIRED FILES:")
    for p in missing:
        print("  -", p)
else:
    print("\nRequired core files: PASS")

if placeholders:
    print("\nPLACEHOLDERS STILL PRESENT:")
    for p in placeholders:
        print("  -", p)
else:
    print("\nMetadata placeholders: PASS")

if secret_names:
    print("\nPOTENTIAL SECRET FILES:")
    for p in secret_names:
        print("  -", p)
else:
    print("\nObvious secret filenames: PASS")

if missing or placeholders or secret_names:
    print("\nSTATUS: NOT READY FOR PUBLIC RELEASE")
    sys.exit(1)

print("\nSTATUS: STRUCTURE READY. Run the scientific reproduction checks next.")
