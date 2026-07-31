# ABOUTME: Assemble the per-config benchmark bundle (the solver's public API)
# ABOUTME: babbling, cav, qcav, weak_inst curve, floor, ceiling, value of commitment.

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from ..config import GameConfig
from ..types import AssetQuality
from .game_spec import GameSpec, game_spec
from .simplex2d import SimplexGrid, EnvelopeFunction, cav_simplex, qcav_simplex
from .weak_inst import weak_inst_curve

__all__ = ["BenchmarkBundle", "compute_benchmarks"]


@dataclass
class BenchmarkBundle:
    """The full benchmark ladder for one game configuration.

    All values are the SENDER's payoff (the object the experiments measure
    against), except ``floor``/``ceiling`` which characterize the receiver.

    Attributes:
        babbling: sender value when the receiver gets no information
            (posterior == prior): the sender's payoff if the receiver acts on
            the prior alone.
        cav: full-commitment (Kamenica-Gentzkow) sender value at the prior =
            the concave envelope of the indirect value function.
        qcav: transparent-motive cheap-talk (Lipnowski-Ravid) sender value =
            the quasiconcave envelope. The gym's native one-shot world.
        weak_inst: the LRS weak-institution curve, ``{chi: value}``.
        floor: receiver's no-information payoff (always PASS).
        ceiling: receiver's perfect-information payoff under the prior.
        value_of_commitment: ``cav - qcav``, the entire point of Proposals 09-11.
    """

    babbling: float
    cav: float
    qcav: float
    weak_inst: Dict[float, float]
    floor: float
    ceiling: float
    value_of_commitment: float = 0.0

    def __post_init__(self) -> None:
        self.value_of_commitment = self.cav - self.qcav

    def as_dict(self) -> Dict[str, object]:
        """JSON-serializable view (chi keys rendered as plain floats)."""
        return {
            "babbling": float(self.babbling),
            "cav": float(self.cav),
            "qcav": float(self.qcav),
            "weak_inst": {float(k): float(v) for k, v in self.weak_inst.items()},
            "floor": float(self.floor),
            "ceiling": float(self.ceiling),
            "value_of_commitment": float(self.value_of_commitment),
        }


def _sender_value_on_grid(grid_mu: np.ndarray, spec: GameSpec) -> np.ndarray:
    """Sender indirect value v(mu) sampled on the simplex grid."""
    return np.array([spec.indirect_sender_value(mu) for mu in grid_mu])


def compute_benchmarks(
    config: GameConfig,
    chi_grid: Optional[np.ndarray] = None,
    grid_n: int = 25,
) -> BenchmarkBundle:
    """Compute the benchmark bundle for a game configuration.

    Args:
        config: the game configuration.
        chi_grid: credibility values at which to evaluate the weak-institution
            curve. Defaults to a 0..1 grid in steps of 0.05.
        grid_n: resolution of the simplex grid used for the envelope routines.

    Returns:
        A :class:`BenchmarkBundle`.
    """
    if chi_grid is None:
        chi_grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 4)

    spec = game_spec(config)
    grid = SimplexGrid(n=grid_n)
    values = _sender_value_on_grid(grid.mu, spec)

    cf = cav_simplex(grid, values)
    qf = qcav_simplex(grid, values)

    cav_val = float(cf.evaluate(spec.mu0))
    qcav_val = float(qf.evaluate(spec.mu0))
    babbling_val = float(spec.indirect_sender_value(spec.mu0))

    # Weak-institution curve, reusing the precomputed cav/qcav envelopes.
    curve = weak_inst_curve(spec, grid, values, cf=cf, qf=qf)
    weak_inst = {float(chi): float(curve.evaluate(chi)) for chi in chi_grid}

    # Receiver floor/ceiling derived from config payoffs and prior.
    prior = spec.mu0
    # Floor: receiver gets no information and ( optimally ) always PASSes => 0.
    floor = 0.0
    # Ceiling: receiver has perfect information, buys iff u_R(BUY, theta) > 0.
    # Under the gym's payoffs that means buy on MEDIUM/HIGH, pass on LOW.
    receiver_buy = spec.receiver_buy
    buys_states = receiver_buy > spec.buy_threshold
    ceiling = float(np.sum(prior[buys_states] * receiver_buy[buys_states]))

    return BenchmarkBundle(
        babbling=babbling_val,
        cav=cav_val,
        qcav=qcav_val,
        weak_inst=weak_inst,
        floor=floor,
        ceiling=ceiling,
    )
