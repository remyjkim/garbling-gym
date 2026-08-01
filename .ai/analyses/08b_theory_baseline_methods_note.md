# Theoretical Baseline — Methods Note

**Proposal:** 08 — The Theory Baseline: An Equilibrium Solver and Information-Metric Layer for the Garbling Gym
**Purpose:** Reusable "Theoretical baseline" section for any paper produced by Proposals 09–11. Paste-able verbatim with the experiment's specific `(prior, payoffs)` substituted.

---

## 1. The benchmark ladder

Every outcome the garbling gym produces is measured against a computed analytical ground truth. For a game with states $\theta \in \{\text{LOW}, \text{MEDIUM}, \text{HIGH}\}$, prior $\mu_0$, and the state-independent sender payoff $u_S = 10 \cdot \mathbf{1}[\text{BUY}]$ (transparent motives, Lipnowski & Ravid 2020), the solver computes four benchmarks at $\mu_0$:

1. **Babbling value** — the sender's payoff when the receiver gets no information (posterior $= \mu_0$). With a binary receiver action this is $10$ if the receiver buys at the prior, else $0$.
2. **Cheap-talk value $V_{qcav}$** — the quasiconcave envelope $\operatorname{qcav} v(\mu_0)$ (Lipnowski & Ravid 2020). This is the gym's *native* one-shot world: transparent-motive cheap talk with no commitment.
3. **Commitment value $V_{cav}$** — the concave envelope $\operatorname{cav} \hat v(\mu_0)$ (Kamenica & Gentzkow 2011). Full Bayesian persuasion.
4. **Weak-institution curve $v^*_\chi$** — the capped-concavification program of Lipnowski, Ravid & Shishkin (2022), interpolating $V_{qcav}$ (at $\chi=0$) and $V_{cav}$ (at $\chi=1$) as a function of institutional credibility $\chi$.

The **value of commitment** is $V_{cav} - V_{qcav}$. This gap is the entire object of study for the downstream proposals: Proposal 09 asks whether reputation manufactures an effective $\chi_{\text{eff}}$ inside this interval; Proposal 10 asks whether learning senders converge to $V_{qcav}$; Proposal 11 recovers an effective perceived commitment $\rho_{\text{perceived}}$.

## 2. The solver and its validation

The envelopes are computed exactly: the concave envelope via the upper convex hull of lifted points $\{(\mu_x, \mu_y, v(\mu))\}$ over a triangular simplex grid (monotone-chain hull, `scipy.spatial.ConvexHull`); the quasiconcave envelope via the superlevel-set convex-hull formulation $\operatorname{qcav} v(\mu) = \sup\{s : \mu \in \operatorname{conv}\{\mu' : v(\mu') \ge s\}\}$. The weak-institution program is solved by gridding $\gamma$ over the simplex, with $\beta, k$ pinned by Bayes plausibility and the credibility-feasibility constraint.

The solver is validated against four closed-form oracles, each reproduced to $\le 10^{-3}$:

- **Kamenica–Gentzkow prosecutor–judge:** $V_{cav}(0.3) = 0.60$ at prior $0.3$, threshold $0.5$.
- **LR ≤ KG:** $V_{qcav}(\mu_0) \le V_{cav}(\mu_0)$ pointwise, verified across the prior simplex.
- **Crawford–Sobel partition:** $N(b)$ collapses $7 \to 5 \to 3 \to 2 \to 1$ over $b \in \{.01, .02, .05, .10, .25\}$.
- **2-D interior consistency:** on the full simplex, $\operatorname{cav}$ of a vertex-indicator equals the analytic $\mu_{\text{HIGH}}$, and the 2-D routines collapse exactly to the validated 1-D routines on every simplex edge.

## 3. Information metrics

Realized channels $\hat L(s|\theta)$ are estimated from each run's history by Laplace (Dirichlet) smoothing, with a per-coordinate credible interval. Informativeness is reported as **mutual information** $I(\theta; s)$ (nats) — comparable across the matrix, continuous, and natural-language channel tiers — and channels are compared via the **Blackwell garbling test** (an LP feasibility problem: $\hat L'$ is a garbling of $\hat L$ iff $\exists$ state-independent stochastic $K$ with $\hat L' = \hat L \, K$). These replace the prior Frobenius-distance-from-identity proxy, which was neither Blackwell-monotone nor prior-dependent.

## 4. Experimental configuration note

The value-of-commitment gap is **zero** at priors where the receiver buys under babbling (e.g. the default $0.3/0.4/0.3$, where $E[\text{BUY}|\mu_0] = 3.5 > 0$): the sender already earns the maximum with no information. Experiments measuring commitment or credibility therefore use a **LOW-heavy prior** (we use $\mu_0 = (0.6, 0.3, 0.1)$, giving $V_{qcav} = 0$, $V_{cav} \approx 6.2$, a gap of $\approx 6.2$), so that realized sender value has a meaningful interval to land in. This is a prior override, not a change to the action space or payoff table.

## 5. Provenance

- Implementation: `src/garbling_gym/core/theory/` (`game_spec`, `envelopes`, `simplex2d`, `weak_inst`, `cs_partition`, `metrics`, `estimate`, `benchmarks`).
- Validation: `tests/core/theory/` (88 tests) and `docs/notes/theory_baseline_validation.md`.
- CLI: `gg solve` prints the benchmark bundle for any `(prior, payoffs)`; `gg run` attaches benchmarks and the realized channel to every stored result.
