# Theory Baseline Validation & Gap Diagnostic

**Proposal:** 08 — The Theory Baseline
**Status:** Validated. All four oracles pass; the cav-vs-qcav gap is characterized across priors and a diagnostic preset is named for Proposals 09–11.
**Date:** 2026-07-31

This note records the validation of the `core/theory/` solver against its four closed-form oracles, and the empirical investigation of the **value-of-commitment gap** (`cav − qcav`) — the quantity Proposals 09–11 measure against — across the prior/payoff space.

---

## 1. Oracle validation (all four pass)

The solver ships with four correctness oracles drawn from the literature review (Proposal 08 §4). Each is encoded as a test in `tests/core/theory/` and must pass to ≤ 1e-3.

| Oracle | Expected | Result | Test |
|---|---|---|---|
| **KG prosecutor–judge** | `V_cav(0.3) = 0.60` (prior 0.3, threshold 0.5) | **PASS** (0.5994) | `test_oracles.py::TestOracleKGProsecutorJudge` |
| **LR ≤ KG** | `V_qcav(μ₀) ≤ V_cav(μ₀)` pointwise, across priors | **PASS** | `test_benchmarks.py::TestOracleLRLeCav` |
| **Crawford–Sobel N(b)** | collapse 7→5→3→2→1 over `b ∈ {.01,.02,.05,.10,.25}` | **PASS** (exact) | `test_cs_partition.py::TestMaxPartitionCount` |
| **2-D interior consistency** | cav of a vertex-indicator = `μ_HIGH` analytically | **PASS** | `test_envelopes_2d.py::TestSimplexOracles` |

The **LRS central-bank curve** `(3/2, 2χ, 1)` is validated structurally rather than against that specific example's (undocumented) payoff structure: the program reproduces the cheap-talk floor at χ=0, the commitment ceiling at χ=1, weak monotonicity in χ, and the `[qcav, cav]` bounds — the four LRS theorems that hold for *any* game. See `test_weak_inst.py`.

The full theory suite is **88 tests**, all passing; the full repository suite is **336 tests**, all passing (no regressions to the pre-existing 238).

## 2. Boundary-collapse agreement (1-D ↔ 2-D)

The 2-D simplex routines agree with the 1-D routines (which pass the KG oracle) to within grid tolerance on every simplex edge — i.e. when one state has probability zero, the 2-D solver reduces exactly to the validated 1-D solver. This is the proposal's §9 mitigation for 2-D boundary risk, and it holds.

## 3. The gap diagnostic — the decisive finding

The entire empirical value of Proposals 09–11 is "measure where the realized sender value lands between `qcav` and `cav`." That requires a **nonzero gap**. The solver reveals that the gap depends entirely on **whether the receiver buys at the prior**:

| Prior (L, M, H) | babbling | qcav | cav | **gap** | receiver at prior |
|---|---|---|---|---|---|
| (0.3, 0.4, 0.3) — *default* | 10.00 | 10.00 | 10.00 | **+0.00** | BUYS |
| (0.2, 0.3, 0.5) | 10.00 | 10.00 | 10.00 | **+0.00** | BUYS |
| (0.6, 0.3, 0.1) | 0.00 | 0.00 | 6.23 | **+6.23** | PASS |
| (0.5, 0.3, 0.2) | 0.00 | 0.00 | 8.52 | **+8.52** | PASS |
| (0.4, 0.4, 0.2) | 0.00 | 0.00 | 9.84 | **+9.84** | PASS |
| (0.7, 0.2, 0.1) | 0.00 | 0.00 | 4.92 | **+4.92** | PASS |

**The mechanism.** The sender payoff is state-independent (+10 on BUY). When `E[BUY | prior] > 0`, the receiver buys under babbling, so the sender earns +10 with *no* information — commitment and cheap talk have nothing to add (gap = 0). When `E[BUY | prior] ≤ 0`, the receiver passes under babbling (sender earns 0); cheap talk still can't help (qcav = 0, because the sender can't credibly claim good news without commitment), but **full commitment lets the sender design signals that move the receiver's posterior above threshold on some states** — producing a gap of 5–10.

This is exactly the 2-action-receiver consideration flagged in the implementation plan (§3.3): the gap is real and diagnostic, but only in the right prior regime.

## 4. Recommendation: the diagnostic preset for Proposals 09–11

**Proposals 09–11 should run against a prior where the receiver passes at the prior.** The clean choice is:

> **Diagnostic prior: (LOW=0.6, MEDIUM=0.3, HIGH=0.1)** — gap ≈ 6.2, `qcav=0`, `cav≈6.2`.

This gives a wide, measurable interval for χ_eff (Proposal 09), the convergence target (Proposal 10), and ρ_perceived (Proposal 11) to land in. The default prior (0.3/0.4/0.3) yields a zero gap and is **not** usable for benchmark-anchored claims about commitment or credibility.

This is a configurable prior override (the whole point of Task 0.2's config consolidation) — **not** a change to the action space or payoff table. The default game remains unchanged; experiments that need the gap simply set a LOW-heavy prior.

## 5. Performance characteristics

- `gg solve` on the default grid (11 χ-points, `grid_n=30`): ~2s.
- The auto-attached benchmark bundle per `play_game` call: ~1s (cached on the Game instance).
- Full test suite: ~83s (was 7s pre-solver; the cost is the benchmark computation across ~30 games in the integration tests, acceptable for a research harness).

For analysis figures needing a fine χ-grid, pass `--chi-grid 0,0.05,0.10,...,1` to `gg solve`; speed is not a concern there.

## 6. Limitations carried forward

- **Empirical-channel noise at 20–50 rounds:** the realized channel estimate carries a Dirichlet credible interval; consumers must not claim a Blackwell ordering the CI does not support (encoded in `estimate.py`).
- **LRS curve resolution:** gridding introduces small numerical error; the central-bank-style discontinuity is reproduced qualitatively (the jump near χ≈2/3 in the 1-D prototype) but its exact location is grid-dependent. For publication-grade LRS figures, refine `n_gamma` in `weak_inst.py`.
- **2-action receiver:** the gap exists only in the LOW-heavy prior regime (§3). This is a feature of the binary-action game, not a bug; the recommendation in §4 addresses it.
