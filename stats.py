"""
Statistics for the interaction experiment (Sec. VII-D).

Paired at the realization level with matched seeds, Holm-corrected across the
four planned multi-pipeline comparisons, reporting confidence intervals rather
than p-values alone.

The paired structure is what makes this work with a few hundred realizations:
because every arm sees the same channels, the per-realization difference
cancels the (large) between-realization variance, and the remaining variance
is the quantity of interest.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 10000,
                        alpha: float = 0.05, seed: int = 0):
    """Two-sided BCa-free percentile bootstrap on paired differences.

    Parameters
    ----------
    diff : [n] per-realization paired difference (e.g. Delta).

    Returns
    -------
    (ci_lo, ci_hi, p_two_sided)

    The p-value is the bootstrap two-sided achieved significance level for
    H0: mean(diff) == 0. Note this is deliberately two-sided: sub-additive
    interaction is as much a result as super-additive.
    """
    diff = np.asarray(diff, dtype=float)
    n = diff.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)

    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))

    centred = means - means.mean()
    p = float((np.abs(centred) >= abs(diff.mean())).mean())
    p = min(1.0, max(p, 1.0 / n_boot))
    return lo, hi, p


def holm_adjust(pvals) -> list[float]:
    """Holm-Bonferroni step-down adjustment, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


def min_detectable_effect(diff: np.ndarray, power: float = 0.8,
                          alpha: float = 0.05) -> float:
    """Rough MDE for the observed paired variance, to sanity-check n.

    Run this on a pilot of ~50 realizations before committing to a full sweep.
    If the MDE is larger than any plausible Delta, increase n before spending
    GPU hours -- a null result you were never powered to avoid is not a
    negative result, it is an uninformative one.
    """
    from math import sqrt
    diff = np.asarray(diff, dtype=float)
    s = diff.std(ddof=1)
    z_a, z_b = 1.959963985, 0.8416212336 if power == 0.8 else 1.2815515655
    return float((z_a + z_b) * s / sqrt(len(diff)))
