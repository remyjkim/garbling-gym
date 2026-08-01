# ABOUTME: Information metrics — mutual information and the Blackwell garbling LP test
# ABOUTME: Replaces the Frobenius-distance informativeness proxy with decision-relevant quantities.

import numpy as np
from scipy.optimize import linprog

__all__ = ["mutual_information", "is_garbling_of"]


def mutual_information(L: np.ndarray, mu0: np.ndarray) -> float:
    """Mutual information ``I(theta; s)`` between the state and the signal (nats).

    For a channel ``L[s | theta]`` (rows index states, summing to 1) and a
    prior ``mu0(theta)``,

        I(theta; s) = sum_theta mu0(theta)
                        sum_s L[s|theta] log( L[s|theta] / p(s) ),

    where ``p(s) = sum_theta mu0(theta) L[s|theta]`` is the marginal signal
    distribution.  The 0 log 0 terms are treated as 0.

    This is a prior-dependent, model-free measure of how much the channel
    reveals about the state — comparable across channel tiers (matrix,
    continuous, natural language), unlike the prior-free Frobenius proxy.

    Args:
        L: channel matrix, shape ``(n_states, n_signals)``, row-stochastic.
        mu0: prior over states, shape ``(n_states,)``, sums to 1.

    Returns:
        ``I(theta; s)`` in nats; non-negative, bounded above by ``H(theta)``.
    """
    L = np.asarray(L, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    # marginal signal distribution p(s) = sum_theta mu0(theta) L[s|theta]
    p_s = mu0 @ L  # shape (n_signals,)
    total = 0.0
    for theta in range(L.shape[0]):
        for s in range(L.shape[1]):
            l = L[theta, s]
            if l <= 0.0:
                continue
            if p_s[s] <= 0.0:
                continue
            total += mu0[theta] * l * np.log(l / p_s[s])
    return float(total)


def is_garbling_of(L_prime: np.ndarray, L: np.ndarray) -> bool:
    """Blackwell garbling test: is ``L_prime`` a garbling of ``L``?

    Channel ``L_prime`` is a garbling of ``L`` iff there exists a
    state-independent (signal-only) row-stochastic kernel ``K`` with

        L_prime[theta, s'] = sum_s L[theta, s] K[s, s']   for all theta, s',

    i.e. ``L_prime = L @ K``.  Equivalently, ``L`` Blackwell-dominates
    ``L_prime``: ``L`` is at least as informative for every decision problem.

    Solved as a linear-program feasibility problem (no objective): find a
    non-negative ``K`` of shape ``(n_signals, n_signals)`` with each row
    summing to 1, satisfying the equality constraints ``L @ K = L_prime``.

    Args:
        L_prime: the candidate (less-informative) channel, shape
            ``(n_states, n_signals)``, row-stochastic.
        L: the dominant (more-informative) channel, same shape, row-stochastic.

    Returns:
        True iff such a kernel ``K`` exists (``L_prime`` is a garbling of ``L``).
    """
    L = np.asarray(L, dtype=float)
    L_prime = np.asarray(L_prime, dtype=float)
    if L.shape != L_prime.shape:
        raise ValueError(f"channel shape mismatch: {L_prime.shape} vs {L.shape}")
    n_states, n_signals = L.shape

    # Variables: the n_signals*n_signals entries of K, flattened in row-major
    # order (K[s, s'] lives at index s*n_signals + s').
    n_vars = n_signals * n_signals

    # Equality constraints.
    # (1) L @ K = L_prime: for each (theta, s'), sum_s L[theta,s] K[s,s'] = L_prime[theta,s']
    A_eq, b_eq = [], []
    for theta in range(n_states):
        for sp in range(n_signals):
            row = np.zeros(n_vars)
            for s in range(n_signals):
                row[s * n_signals + sp] = L[theta, s]
            A_eq.append(row)
            b_eq.append(L_prime[theta, sp])
    # (2) K is row-stochastic: for each s, sum_sp K[s,sp] = 1
    for s in range(n_signals):
        row = np.zeros(n_vars)
        for sp in range(n_signals):
            row[s * n_signals + sp] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # Feasibility problem: zero objective, K >= 0.
    res = linprog(
        c=np.zeros(n_vars),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * n_vars,
        method="highs",
    )
    return bool(res.success)
