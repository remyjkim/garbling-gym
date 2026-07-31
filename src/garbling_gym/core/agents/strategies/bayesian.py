# ABOUTME: Bayesian receiver strategies using Dirichlet conjugate priors
# ABOUTME: Infers the sender's garbling matrix from observed (signal, quality) pairs

from typing import Any, Dict

import numpy as np

from . import ReceiverStrategy
from ...types import Action, AssetQuality, Signal


_QUALITIES = ("LOW", "MEDIUM", "HIGH")
_SIGNALS = ("BAD", "NEUTRAL", "GOOD")

# Defaults — preserved exactly by DirichletBayesianStrategy.__init__ unless
# configure() injects values sourced from GameConfig.
_DEFAULT_PRIOR = {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}
_DEFAULT_PAYOFFS = {"LOW": -15.0, "MEDIUM": 5.0, "HIGH": 20.0}


class DirichletBayesianStrategy(ReceiverStrategy):
    """
    Bayesian inference over the sender's garbling matrix.

    Maintains one Dirichlet distribution per quality level, tracking how often
    each quality produces each signal.  Uses the posterior mean to estimate the
    garbling probabilities, then applies Bayes' rule to compute a posterior over
    quality and takes the action with positive expected value.

    Args:
        prior_strength: Dirichlet concentration parameter for the uniform prior.
            Higher values pull estimates toward 1/3 more strongly.
        forgetting_factor: Before each update, multiply all alpha values by this
            factor (1.0 = full memory, 0.95 = exponential decay).
    """

    def __init__(self, prior_strength: float = 1.0, forgetting_factor: float = 1.0) -> None:
        self._prior_strength = prior_strength
        self._forgetting = forgetting_factor
        self._prior: Dict[str, float] = dict(_DEFAULT_PRIOR)
        self._payoffs: Dict[str, float] = dict(_DEFAULT_PAYOFFS)
        self._alpha: Dict[str, Dict[str, float]] = {}
        self.reset()

    def configure(self, prior, receiver_payoffs) -> None:
        """Inject prior and per-quality BUY payoffs from GameConfig."""
        self._prior = dict(prior)
        self._payoffs = {q: receiver_payoffs[("BUY", q)] for q in _QUALITIES}

    # ------------------------------------------------------------------
    # ReceiverStrategy interface
    # ------------------------------------------------------------------

    def choose_action(self, signal: Any, round_num: int, total_rounds: int) -> Action:
        signal_name = signal.name if isinstance(signal, Signal) else str(signal)

        # Posterior mean estimates P(signal | quality)
        p_signal_given_quality = self._posterior_mean()

        # Bayes: P(quality | signal) ∝ P(signal | quality) × prior(quality)
        unnorm = {
            q: p_signal_given_quality[q].get(signal_name, 1e-9) * self._prior[q]
            for q in _QUALITIES
        }
        total = sum(unnorm.values()) or 1.0
        posterior = {q: unnorm[q] / total for q in _QUALITIES}

        expected_buy = sum(posterior[q] * self._payoffs[q] for q in _QUALITIES)
        return Action.BUY if expected_buy > 0 else Action.PASS

    def update(
        self,
        signal: Any,
        action: Action,
        true_quality: AssetQuality,
        sender_payoff: float,
        receiver_payoff: float,
    ) -> None:
        signal_name = signal.name if isinstance(signal, Signal) else str(signal)
        quality_name = true_quality.name if isinstance(true_quality, AssetQuality) else str(true_quality)

        # Apply forgetting before incorporating new evidence
        if self._forgetting != 1.0:
            for q in _QUALITIES:
                for s in _SIGNALS:
                    self._alpha[q][s] *= self._forgetting

        self._alpha[quality_name][signal_name] += 1.0

    def reset(self) -> None:
        self._alpha = {
            q: {s: self._prior_strength for s in _SIGNALS}
            for q in _QUALITIES
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        pm = self._posterior_mean()
        ess = {q: sum(self._alpha[q].values()) for q in _QUALITIES}
        return {
            "alpha": {q: dict(self._alpha[q]) for q in _QUALITIES},
            "posterior_mean": {q: dict(pm[q]) for q in _QUALITIES},
            "effective_sample_size": ess,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _posterior_mean(self) -> Dict[str, Dict[str, float]]:
        """Compute P_hat(signal | quality) = alpha[q][s] / sum_s(alpha[q][s])."""
        result = {}
        for q in _QUALITIES:
            total = sum(self._alpha[q].values()) or 1.0
            result[q] = {s: self._alpha[q][s] / total for s in _SIGNALS}
        return result


class ThompsonSamplingStrategy(DirichletBayesianStrategy):
    """
    Bayesian exploration via posterior sampling.

    At decision time, samples a garbling matrix from the Dirichlet posterior
    rather than using the posterior mean.  This provides natural exploration
    under uncertainty while converging to the same decisions as
    DirichletBayesianStrategy as evidence accumulates.

    Args:
        prior_strength: Dirichlet concentration parameter.
        forgetting_factor: Exponential decay of prior alpha values on update.
        sample_count: Number of samples to average (1 = pure Thompson).
    """

    def __init__(
        self,
        prior_strength: float = 1.0,
        forgetting_factor: float = 1.0,
        sample_count: int = 1,
    ) -> None:
        super().__init__(prior_strength=prior_strength, forgetting_factor=forgetting_factor)
        self._sample_count = sample_count

    def choose_action(self, signal: Any, round_num: int, total_rounds: int) -> Action:
        signal_name = signal.name if isinstance(signal, Signal) else str(signal)

        # Average over sampled garbling matrices
        total_ev = 0.0
        for _ in range(self._sample_count):
            # Sample one garbling matrix row per quality from the Dirichlet posterior
            sampled = {}
            for q in _QUALITIES:
                alpha_vec = np.array([self._alpha[q][s] for s in _SIGNALS])
                sample = np.random.dirichlet(alpha_vec)
                sampled[q] = {s: float(sample[i]) for i, s in enumerate(_SIGNALS)}

            # Bayes: P(quality|signal) ∝ sampled_p(signal|quality) × prior(quality)
            unnorm = {
                q: sampled[q].get(signal_name, 1e-9) * self._prior[q]
                for q in _QUALITIES
            }
            total = sum(unnorm.values()) or 1.0
            posterior = {q: unnorm[q] / total for q in _QUALITIES}
            total_ev += sum(posterior[q] * self._payoffs[q] for q in _QUALITIES)

        avg_ev = total_ev / self._sample_count
        return Action.BUY if avg_ev > 0 else Action.PASS
