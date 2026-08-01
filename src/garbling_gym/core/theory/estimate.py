# ABOUTME: Empirical channel estimation from game history
# ABOUTME: L_hat(s|theta) via Laplace (Dirichlet) smoothing with a credible interval

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import mutual_information

__all__ = ["ChannelEstimate", "estimate_channel"]


# Canonical ordering used throughout the theory package.
_QUALITIES = ("LOW", "MEDIUM", "HIGH")
_SIGNALS = ("BAD", "NEUTRAL", "GOOD")


@dataclass
class ChannelEstimate:
    """An empirical channel estimate with uncertainty.

    Attributes:
        L: estimated channel ``L[s | theta]``, shape ``(n_states, n_signals)``,
            row-stochastic (rows index states).  Computed by Laplace smoothing:
            ``L[s|theta] = (n_{theta,s} + 1) / (n_theta + n_signals)``.
        ci_width: per-(state, signal) half-width of the credible interval on
            ``L[s|theta]``, derived from the Dirichlet posterior.  Shape matches
            ``L``.  Shrinks as the sample size grows.
        counts: raw co-occurrence counts ``n_{theta,s}``, shape ``(n_states, n_signals)``.
        mutual_information: ``I(theta; s)`` of the estimated channel under the
            prior used for estimation (nats); ``None`` if no prior was supplied.
    """

    L: np.ndarray
    ci_width: np.ndarray
    counts: np.ndarray
    mutual_information: Optional[float] = None


def _credible_half_width(alpha_row: np.ndarray, credible_mass: float = 0.9) -> np.ndarray:
    """Per-coordinate credible half-width for a Dirichlet posterior row.

    Approximates the equal-tailed ``credible_mass`` interval half-width on each
    component ``L[s|theta]`` using the Dirichlet posterior with concentration
    ``alpha_row``.  A cheap, robust surrogate (normal approximation on each
    margin): ``half-width = z * sqrt(p (1-p) / (alpha0 + 1))`` where
    ``p = alpha/alpha0``, ``alpha0 = sum(alpha)``, and ``z`` is the standard
    normal quantile for ``credible_mass`` (1.645 for 0.9).

    This deliberately errs on the side of reporting *some* uncertainty; the
    point estimate ``L`` is what consumers compare, and ``ci_width`` exists to
    prevent over-claiming a Blackwell ordering the data does not support.
    """
    alpha0 = alpha_row.sum()
    if alpha0 <= 0:
        return np.full_like(alpha_row, 1.0)
    p = alpha_row / alpha0
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(round(credible_mass, 2), 1.645)
    sd = np.sqrt(p * (1.0 - p) / (alpha0 + 1.0))
    return z * sd


def estimate_channel(
    history: Sequence[Dict],
    mu0: Optional[np.ndarray] = None,
    credible_mass: float = 0.9,
) -> ChannelEstimate:
    """Estimate the realized channel ``L_hat(s|theta)`` from a game history.

    Each history entry must carry ``'quality'`` and ``'signal'`` name strings
    (as produced by ``Game.play_round``).  Co-occurrence counts are accumulated
    per ``(quality, signal)`` and converted to a row-stochastic estimate via
    Laplace (Dirichlet(1,...,1)) smoothing, which keeps the estimate well
    defined even for states seen few times or never.

    Args:
        history: sequence of round-record dicts with ``quality`` and ``signal``.
        mu0: optional prior over states; if supplied, the estimate's
            ``mutual_information`` is computed under it.
        credible_mass: mass for the per-coordinate credible interval (0.9, 0.95,
            or 0.99).

    Returns:
        A :class:`ChannelEstimate`.
    """
    counts = np.zeros((3, 3), dtype=float)
    for rec in history:
        q = rec.get("quality")
        s = rec.get("signal")
        if q is None or s is None:
            continue
        try:
            qi = _QUALITIES.index(q)
            si = _SIGNALS.index(s)
        except ValueError:
            continue
        counts[qi, si] += 1.0

    n_signals = 3
    # Laplace smoothing: posterior mean of Dirichlet(counts + 1).
    row_totals = counts.sum(axis=1, keepdims=True)  # (3,1)
    L = (counts + 1.0) / (row_totals + n_signals)

    # Credible half-widths from the Dirichlet posterior concentration = counts + 1.
    ci_width = np.zeros_like(L)
    for theta in range(3):
        alpha_row = counts[theta] + 1.0
        ci_width[theta] = _credible_half_width(alpha_row, credible_mass)

    mi = None
    if mu0 is not None:
        mi = mutual_information(L, np.asarray(mu0, dtype=float))

    return ChannelEstimate(L=L, ci_width=ci_width, counts=counts, mutual_information=mi)
