"""
Per-pipeline lifecycle state machines (Sec. V).

Each pipeline carries a state machine over

    ACTIVE -> SUSPECT -> FALLBACK -> (recovery) -> ACTIVE

driven by a monitoring statistic s_t and threshold tau. Mapping onto the
Release-19 primitives, which is what makes this standards-aligned rather
than generic:

  ACTIVE            functionality activated, inference running
  ACTIVE->SUSPECT   performance monitoring reports degradation
  SUSPECT->FALLBACK deactivation / fallback to the non-AI operation
  FALLBACK->ACTIVE  re-activation after sustained recovery

Thresholds are calibrated PER PIPELINE IN ISOLATION at a common nominal
false-alarm rate, matching the per-functionality organization of the
specifications. That calibration choice is the object of study, not an
oversight: the whole point is that detectors sharing a common cause are
jointly miscalibrated even when each is individually correct.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
try:
    import torch  # noqa: F401  (unused here; kept for API parity)
except Exception:  # pragma: no cover - lcm.py needs only numpy
    torch = None

ACTIVE, SUSPECT, FALLBACK = 0, 1, 2
STATE_NAMES = {ACTIVE: "ACTIVE", SUSPECT: "SUSPECT", FALLBACK: "FALLBACK"}


@dataclass
class LCMConfig:
    tau: float                 # detection threshold on the monitor statistic
    q_enter: int = 3           # consecutive exceedances to enter FALLBACK
    q_exit: int = 10           # consecutive clean evaluations to re-activate
    pinned_active: bool = False   # suppress fallback entirely (control arm)


class LCMStateMachine:
    """Hysteretic three-state machine over one pipeline's monitor."""

    def __init__(self, cfg: LCMConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.state = ACTIVE
        self._over = 0
        self._clean = 0
        self.transitions = 0
        self.fallback_entries = 0
        self.requests = 0            # evaluations at which fallback was due
        self.held_evaluations = 0    # of those, ones an arbiter withheld
        self.history: list[int] = []

    def step(self, s: float, allow: bool = True) -> int:
        """Advance one evaluation.

        `allow=False` means an external arbiter is holding this pipeline's
        fallback request. The machine still RECOGNISES the request -- it
        stays in SUSPECT and the request is recorded -- it simply may not
        act on it this evaluation. The arbiter must bound how long it does
        this; fallback to the non-AI state has to remain locally executable.
        """
        c = self.cfg
        prev = self.state
        over = s > c.tau

        if self.state in (ACTIVE, SUSPECT):
            if over:
                self._over += 1
                self.state = SUSPECT
                if self._over >= c.q_enter and not c.pinned_active:
                    self.requests += 1
                    if allow:
                        self.state = FALLBACK
                        self.fallback_entries += 1
                        self._clean = 0
                    else:
                        self.held_evaluations += 1
            else:
                self._over = 0
                self.state = ACTIVE
        else:  # FALLBACK
            if over:
                self._clean = 0
            else:
                self._clean += 1
                if self._clean >= c.q_exit:
                    self.state = ACTIVE
                    self._over = 0

        if self.state != prev:
            self.transitions += 1
        self.history.append(self.state)
        return self.state

    def oscillation_rate(self, window: int = 40, n_trans: int = 4) -> float:
        """Fraction of NON-OVERLAPPING windows containing >= n_trans changes.

        An earlier version counted overlapping windows, which inflates the
        figure by roughly the window length: a single oscillatory episode
        appears in ~40 consecutive windows and is counted 40 times. Reported
        as a rate over disjoint windows, the number means what it says --
        the share of the episode spent thrashing.
        """
        h = np.asarray(self.history)
        n_win = h.size // window
        if n_win == 0:
            return 0.0
        ch = (h[1:] != h[:-1]).astype(int)
        ch = np.concatenate([ch, [0]])[: n_win * window]
        per = ch.reshape(n_win, window).sum(axis=1)
        return float((per >= n_trans).mean())

    def dwell_times(self) -> dict:
        """Mean consecutive-slot dwell in each state.

        Short dwell in FALLBACK is the signature of chattering: the machine
        is re-entering as soon as it leaves, which means q_exit is too short
        relative to how fast the statistic recrosses the threshold.
        """
        h = np.asarray(self.history)
        if h.size == 0:
            return {n: 0.0 for n in STATE_NAMES.values()}
        out = {}
        edges = np.flatnonzero(np.diff(h)) + 1
        segs = np.split(h, edges)
        for st, name in STATE_NAMES.items():
            lens = [len(s) for s in segs if len(s) and s[0] == st]
            out[name] = float(np.mean(lens)) if lens else 0.0
        return out


def calibrate_threshold(nominal_stats: np.ndarray,
                        target_fa: float = 0.05) -> float:
    """Threshold giving `target_fa` exceedance rate under nominal conditions.

    Calibrated on nominal realizations only, per pipeline, with no reference
    to the other pipelines -- the specification-faithful procedure.
    """
    s = np.asarray(nominal_stats, dtype=float).ravel()
    return float(np.quantile(s, 1.0 - target_fa))


def monitor_correlation(stats: dict, taus: dict) -> dict:
    """Correlated-firing table (Sec. VI-A, Table I).

    Compares the measured joint firing rate against the product of the
    marginals that independent calibration implicitly assumes.
    """
    keys = ["B", "P", "C"]
    fire = {k: (np.asarray(stats[k]) > taus[k]) for k in keys}
    marg = {k: float(fire[k].mean()) for k in keys}

    joint = float(np.logical_and.reduce([fire[k] for k in keys]).mean())
    indep = float(np.prod([marg[k] for k in keys]))

    pair = {}
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = keys[i], keys[j]
            pair[f"corr_{a}{b}"] = float(
                np.corrcoef(np.asarray(stats[a]).ravel(),
                            np.asarray(stats[b]).ravel())[0, 1])
            pj = float(np.logical_and(fire[a], fire[b]).mean())
            pair[f"joint_{a}{b}"] = pj
            pair[f"indep_{a}{b}"] = marg[a] * marg[b]
            pair[f"ratio_{a}{b}"] = pj / max(1e-12, marg[a] * marg[b])

    return {
        "marginal_fa": marg,
        "joint_measured": joint,
        "joint_under_independence": indep,
        "ratio": joint / max(1e-12, indep),
        **pair,
    }


def run_state_machines(arms: dict, cfgs: dict, n_slots: int) -> dict:
    """Step the three machines jointly, selecting from precomputed arms.

    `arms` maps a subset key ("none", "B", "BP", ...) to that arm's per-slot
    statistics and throughput. At each slot the current state vector selects
    which arm's values are in force.

    This is exact provided each pipeline's per-slot output depends only on
    the CURRENT state vector and not on state history. That holds here with
    one exception worth stating in the paper: the CSI prediction window can
    straddle a state change, so a slot immediately after a transition uses a
    window partly generated under the previous state.
    """
    m = {k: LCMStateMachine(cfgs[k]) for k in ("B", "P", "C")}
    key = lambda st: "".join(sorted(k for k in ("B", "P", "C")
                                    if st[k] == FALLBACK)) or "none"

    state = {k: ACTIVE for k in ("B", "P", "C")}
    tput = np.zeros(n_slots)

    for t in range(n_slots):
        a = arms[key(state)]
        tput[t] = float(a["tput_slot"][t])
        for k in ("B", "P", "C"):
            state[k] = m[k].step(float(a["s_" + k][t]))

    return {
        "throughput": tput.mean(),
        "machines": m,
        "fallback_entries": {k: m[k].fallback_entries for k in m},
        "requests": {k: m[k].requests for k in m},
        "held": {k: m[k].held_evaluations for k in m},
        "transitions": {k: m[k].transitions for k in m},
        "oscillation": {k: m[k].oscillation_rate() for k in m},
        "dwell": {k: m[k].dwell_times() for k in m},
        "fallback_slots": {
            k: float(np.mean(np.asarray(m[k].history) == FALLBACK))
            for k in m},
        "history": {k: np.asarray(m[k].history) for k in m},
    }
