#!/usr/bin/env python3
from pathlib import Path
import re, sys, py_compile
ROOT = Path(__file__).resolve().parents[1]
required = [
 'README.md','CITATION.cff','.zenodo.json','artifact_manifest.yaml','REPRODUCIBILITY.md',
 'configs/final_config.json','environment/README.md','results/README.md',
 'src/config.py','src/channel.py','src/degradation.py','src/pipelines.py','src/lcm.py','src/stats.py',
 'src/train.py','src/train_pair.py','src/experiment.py','src/nominal_corr.py','src/induced.py',
 'src/conditional.py','src/mechanism.py','src/check_encoder_ood.py','src/pairing.py',
 'src/arbiter.py','src/arbiter_mismatch.py','src/diagnose_beam.py'
]
missing=[p for p in required if not (ROOT/p).exists()]
text=(ROOT/'src/config.py').read_text()
checks={
 'feedback_delay=2': bool(re.search(r'feedback_delay:\s*int\s*=\s*2\b',text)),
 'train_realizations=1024': bool(re.search(r'train_realizations:\s*int\s*=\s*1024\b',text)),
 'val_realizations=128': bool(re.search(r'val_realizations:\s*int\s*=\s*128\b',text)),
 'epochs=200': bool(re.search(r'epochs:\s*int\s*=\s*200\b',text)),
 'batch_size=64': bool(re.search(r'batch_size:\s*int\s*=\s*64\b',text)),
 'q_exit=25': bool(re.search(r'q_exit:\s*int\s*=\s*25\b',(ROOT/'src/lcm.py').read_text())),
}
compile_errors=[]
for p in (ROOT/'src').glob('*.py'):
    try: py_compile.compile(str(p), doraise=True)
    except Exception as e: compile_errors.append(f'{p.name}: {e}')
print('Repository:',ROOT)
print('Core files:', 'PASS' if not missing else 'FAIL')
for p in missing: print('  missing:',p)
print('Paper-config synchronization:')
for k,v in checks.items(): print(f'  {k}:', 'PASS' if v else 'FAIL')
print('Python syntax:', 'PASS' if not compile_errors else 'FAIL')
for e in compile_errors: print(' ',e)
if (ROOT/'LICENSE_REQUIRED.txt').exists():
    print('License: PENDING (replace LICENSE_REQUIRED.txt before final public release)')
failed = bool(missing or compile_errors or not all(checks.values()))
print('STATUS:', 'FAIL' if failed else 'STRUCTURE READY; scientific rerun still required for full verification')
sys.exit(1 if failed else 0)
