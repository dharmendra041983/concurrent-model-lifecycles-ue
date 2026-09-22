# Concurrent Model Lifecycles at the UE — experiment code

Build order. Do not skip ahead; step 3 is a gate.

```
pip install "sionna-no-rt>=2.0"     # pulls torch; needs CUDA 12.8+ for sm_120
```

### 0. Sanity checks — before anything else

```
python channel.py
```

Two checks must pass:

- `sanity_check_codebook` — best beam should beat the codebook mean by well
  over 3 dB. If not, Sionna's antenna element ordering disagrees with the
  row-major assumption in `dft_codebook()` and everything downstream trains
  against a scrambled codebook while still producing plausible numbers.
- `sanity_check_diversity` — the best beam index must vary across
  realizations. If it is near-constant, orientation randomization is not
  taking effect, beam prediction is trivial, and the B pipeline has no
  headroom to degrade. That failure would surface at the end as Δ ≈ 0 and
  look like a negative result rather than a harness bug.

Also verify the `#CHECK` items in `channel.py`: the shape and units of
`ut_orientation`.

### 1. Train

```
python train.py --band FR2
```

Prints validation metrics against fixed acceptance criteria. If acceptance
fails — in particular if the predictor does not beat hold-last — stop. A weak
model has a cheap fallback, and the interaction term then describes your
training run rather than lifecycle coupling.

### 2. Pilot for power

Run the gate with `--n 50` first and read the printed variance. `stats.py`
has `min_detectable_effect()`; if the MDE exceeds any plausible Δ, raise `n`
before spending GPU hours. A null you were never powered to avoid is
uninformative, not negative.

### 3. The gate

```
python experiment.py --band FR2 --n 512
```

Measures Δ(S) across all eight forced-fallback subsets on matched
realizations. Arms differ only in which pipelines are forced to fallback.

**The test is two-sided.** Sub-additive Δ is a real result: it says
per-functionality accounting is over-conservative and independent calibration
leaves margin unclaimed. Given that the beam fallback is accurate-but-costly
(exhaustive sweep is correct, it just costs overhead) and makes the downstream
pipelines' jobs easier, sub-additivity is a live possibility here. Report
whichever sign appears.

### 4. FR1 as mechanism control

```
python train.py --band FR1 && python experiment.py --band FR1 --n 512
```

Not a generalization check. In FR1 the B→P coupling largely dissolves, so a
Δ that shrinks toward zero is evidence the effect comes from the beam–CSI
dependency rather than from the harness.

### 5. Only if the gate passes

Induced fallback (matched pairs differing only in whether pipeline *i*'s
fallback is permitted), oscillation counting, then the arbiter.

## Open decisions left in the code

- `beam_mode` defaults to `"spatial"` (Rel-18 BM-Case1). `"temporal"`
  strengthens the coupling but invites the objection that it was engineered.
  The temporal branch is not yet implemented in `BeamPipeline.select()`.
- Blockage is not implemented. Currently the degradation drivers are speed and
  orientation only. Blockage onset is the driver that hits all three pipelines
  hardest from a common cause and should be added before the final runs.
- The LCM state machines (Sec. V) are not needed for the gate and are not yet
  written. The gate forces subsets directly.
