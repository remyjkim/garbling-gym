# ABOUTME: LRS weak-institution value curve v*_chi (capped concavification)
# ABOUTME: Lipnowski-Ravid-Shishkin (2022): interpolates qcav (chi=0) and cav (chi=1).

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .game_spec import GameSpec
from .simplex2d import SimplexGrid, EnvelopeFunction, cav_simplex, qcav_simplex

__all__ = ["weak_inst_curve", "weak_inst_value", "WeakInstCurve"]


@dataclass
class WeakInstCurve:
    """The weak-institution sender value as a function of credibility chi.

    Constructed once for a game; call ``.evaluate(chi)`` for the sender's value
    at credibility ``chi in [0, 1]``.  ``evaluate(0)`` is the cheap-talk value
    (qcav); ``evaluate(1)`` is the full-commitment value (cav).
    """

    evaluate: Callable[[float], float]


def _credibility_feasible_one_minus_k(mu0: np.ndarray, gamma: np.ndarray, chi: float) -> Optional[float]:
    """Smallest feasible (1-k) for given (mu0, gamma, chi).

    The credibility-feasibility constraint is
        (1-k) * gamma(theta) >= (1-chi) * mu0(theta)   for all theta,
    i.e. (1-k) >= (1-chi) * mu0(theta) / gamma(theta) for all theta.
    Returns the maximum lower bound on (1-k), or None if infeasible
    (i.e. if it exceeds 1, meaning no k in [0,1] satisfies the constraint).
    """
    # ratio per state; guard against gamma(theta)=0 (would need mu0(theta)=0 too)
    one_minus_chi = 1.0 - chi
    if one_minus_chi <= 0.0:  # chi >= 1: no influenced mass required
        return 0.0
    ratios = []
    for theta in range(len(mu0)):
        if mu0[theta] <= 1e-12:
            ratios.append(0.0)
            continue
        if gamma[theta] <= 1e-12:
            return None  # mu0(theta)>0 but gamma(theta)=0 -> infeasible
        ratios.append(mu0[theta] / gamma[theta])
    lb = one_minus_chi * max(ratios)
    if lb > 1.0 + 1e-9:
        return None
    return min(lb, 1.0)


def weak_inst_curve(
    spec: GameSpec,
    grid: SimplexGrid,
    values: np.ndarray,
    cf: Optional[EnvelopeFunction] = None,
    qf: Optional[EnvelopeFunction] = None,
    n_gamma: int = 21,
    n_k: int = 21,
) -> WeakInstCurve:
    """Construct the weak-institution value curve ``v*_chi`` at the prior.

    Solves the LRS program

        v*_chi(mu0) = max_{beta, gamma, k}
                        k * cav(v^/gamma)(beta) + (1-k) * v^CT(gamma)
        s.t.  k*beta + (1-k)*gamma = mu0,
              (1-k)*gamma(theta) >= (1-chi)*mu0(theta)  for all theta,

    where ``v^CT = qcav`` and ``v^/gamma(mu) = min(v(mu), v^CT(gamma))``.

    The maximization grids ``gamma`` over the simplex; for each ``gamma``,
    ``v^CT(gamma)`` and the cap are fixed, the credibility constraint pins the
    feasible range of ``k``, and Bayes plausibility pins ``beta`` given ``k``.

    Args:
        spec: the game (provides ``mu0``).
        grid: the simplex grid on which ``values`` are sampled.
        values: sender value at each grid point.
        cf, qf: precomputed cav / qcav envelope functions (recomputed if None).
        n_gamma: grid resolution for ``gamma`` (per simplex edge).
        n_k: grid resolution for ``k`` within its feasible range.

    Returns:
        A :class:`WeakInstCurve` whose ``.evaluate(chi)`` returns the sender
        value at credibility ``chi``.
    """
    if cf is None:
        cf = cav_simplex(grid, values)
    if qf is None:
        qf = qcav_simplex(grid, values)

    mu0 = spec.mu0
    n_states = spec.n_states

    # Precompute the cap-value cav for every gamma grid point. The capped
    # function v^/gamma = min(v, vCT(gamma)) depends on gamma only through the
    # scalar cap vCT(gamma); we evaluate cav of the capped function lazily.
    # Build a gamma grid over the simplex (reuse SimplexGrid).
    gamma_grid = SimplexGrid(n=n_gamma)
    gammas = gamma_grid.mu  # (Gg, n_states)

    def v_star_at_chi(chi: float) -> float:
        best = -np.inf
        for gamma in gammas:
            vct_g = qf.evaluate(gamma)
            one_minus_k_min = _credibility_feasible_one_minus_k(mu0, gamma, chi)
            if one_minus_k_min is None:
                continue
            k_max = 1.0 - one_minus_k_min
            if k_max <= 1e-9:
                # k=0: value is just vCT(gamma)
                if vct_g > best:
                    best = vct_g
                continue
            # Capped function at this gamma and its cav.
            capped = np.minimum(values, vct_g)
            cav_capped = cav_simplex(grid, capped)
            # Sweep k in (0, k_max]; beta pinned by Bayes plausibility.
            for k in np.linspace(max(1e-6, 0.0), k_max, n_k):
                if k <= 1e-9:
                    continue
                beta = (mu0 - (1.0 - k) * gamma) / k
                # beta must be a valid belief
                if np.any(beta < -1e-7) or np.any(beta > 1.0 + 1e-7):
                    continue
                beta = np.clip(beta, 0.0, None)
                beta = beta / beta.sum()
                val = k * cav_capped.evaluate(beta) + (1.0 - k) * vct_g
                if val > best:
                    best = val
        # Fallback: if nothing was feasible (shouldn't happen for chi in [0,1]),
        # return the qcav floor.
        if best == -np.inf:
            return qf.evaluate(mu0)
        return float(best)

    return WeakInstCurve(evaluate=v_star_at_chi)


def weak_inst_value(
    spec: GameSpec,
    grid: SimplexGrid,
    values: np.ndarray,
    chi: float,
    cf: Optional[EnvelopeFunction] = None,
    qf: Optional[EnvelopeFunction] = None,
) -> float:
    """Convenience: the weak-institution sender value at a single ``chi``."""
    curve = weak_inst_curve(spec, grid, values, cf=cf, qf=qf)
    return curve.evaluate(chi)
