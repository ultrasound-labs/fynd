# Routing Research: the convex flow attempt

Record of an investigation into routing algorithms that optimise over the whole pool network
rather than over a set of pre-enumerated paths. Covers what was surveyed, what was built, what was
measured, and why it does not work — so the dead ends are not walked twice.

Status as of 10 October 2026: **the implementation does not work and should not be shipped.** The
diagnosis is specific and is in [Why it fails](#why-it-fails). The branch is
`convex-routing`.

## The question

Every shipped splitting algorithm — `water_fill`, `path_frank_wolfe` — discovers a handful of
paths and then divides the order between them, capped at four. The research question was whether a
method that optimises over every pool at once does better.

## What was surveyed

| Family | Verdict |
|---|---|
| Convex optimisation over net trade vectors (Angeris, Chitra, Evans, Diamandis 2022) | Global optimum; seconds with general solvers. Too slow for a 100 ms budget. |
| Dual decomposition (Diamandis, Resnick, Chitra, Angeris, FC 2023) | Parallelises over edges; superseded below. |
| **Convex flow** (Diamandis, Angeris, Edelman, arXiv 2404.00765) | **Chosen.** See below. |
| Marginal Price Optimization (Loesch, Richardson, arXiv 2502.08258) | Built for many-curves/few-tokens (Bancor's CarbonDeFi); its own paper reports the implementation degrading past a handful of tokens, and it needs a Jacobian. |
| PRIME / ASGM (Xu, Zhu, Huang, Tang, ICDE 2026) | Keeps the path cap, so it does not answer the question. Its allocation step is still worth stealing for `water_fill`; see [Still worth doing](#still-worth-doing). |
| Batch auctions / CoW | Different product; needs a batch window and settlement control. |

Convex flow was chosen for three reasons that all survived scrutiny:

1. **Its formulation is the route format.** `Route::validate` groups swaps into branch collections
   keyed by `token_in` — conservation at token nodes — which is exactly the paper's net flow `y`.
2. **Its per-edge oracle is `ProtocolSim`.** The subproblem needs a gain function and its one-sided
   derivatives: `get_amount_out` is the first, and `spot_price` on the state it returns is the
   second. Most routers cannot supply the derivative; Tycho can.
3. **Hyperedges are native**, so multi-token pools are first-class.

None of those three turned out to be wrong. The method still fails.

## What was built

`fynd-core/src/algorithm/convex_flow/`, registered as the built-in `convex_flow`, roughly 1,500
lines with 40 tests.

| Module | Contents |
|---|---|
| `edge.rs` | The per-edge arbitrage subproblem: no-flow test, bisection on `η₂h⁺(w⋆) ≤ η₁ ≤ η₂h⁻(w⋆)` |
| `dual.rs` | The reduced dual (one variable per token), gradient, spectral projected gradient coordinator |
| `mod.rs` | Network construction, warm start from `TokenGasPrices`, flow decomposition, `Algorithm` impl |
| `config.rs` | Tuning constants, each with the reasoning |

Reuses `split_primitives` for route construction, the graph's bidirectional search for
reachability, and `water_fill`'s `Deadline` and gas conversion. No new dependencies — the
coordinator is implemented rather than pulled in.

## Measured results

Deterministic runs (`--jobs 1`), recorded Ethereum fixture, `aggregator_trades_10k.json`.

| Config | Solved | Mean bps | Median bps | p50 |
|---|---|---|---|---|
| `convex_flow` (600 orders) | **149 / 600** | −384.3 | −4.5 | — |
| `water_fill` (600 orders) | 591 / 600 | +319.1 | +0.0 | — |
| `convex_flow` (60 orders) | 15 / 60 | −105.7 | +0.0 | 64 ms |
| `water_fill` (60 orders) | 60 / 60 | +41.8 | +0.0 | 15.6 ms |
| `bellman_ford` (60 orders) | 58 / 60 | +40.2 | +0.0 | 11.2 ms |

Roughly a 25% solve rate, and worse than the baseline where it does solve (82 orders worse against
1 better, over 600). **61% of the routes it does return are a single leg** — the algorithm whose
entire purpose is whole-network optimisation mostly returns what Bellman-Ford already gives.

## Why it fails

Four experiments, each eliminating a candidate.

### It is stuck, not slow

Raising the iteration cap 10x (552 median iterations) and the simulation budget 100x left the
net-flow residual at **4.90e+01, identical to three significant figures**, and the active edge
count at 4. More compute buys nothing.

### It only works when it is not needed

| | n | median `u_in` | prices driven to 0 | tokens | edges |
|---|---|---|---|---|---|
| routed | 86 | **1.000** | 0% | 3 | 4 |
| failed | 202 | **141.5** | **25%** | 10 | 68 |

`u_in` is the input token's price as a multiple of its warm-start reference. Solves that produce a
route end at **1.000** — the warm start, untouched: the coordinator contributed nothing and the
reference prices already implied the route. Solves that fail diverge, with the input token at 141x
its reference and a quarter of the remaining prices driven to zero.

### The gradient is correct

Checked numerically by central finite differences against the analytic gradient, off the no-flow
kink: relative error below 1e-3 on every free coordinate
(`test_gradient_matches_finite_difference`). The formulation, the dual, the gradient and the
subproblem were each verified against the source. None of them is wrong.

### The method is wrong for the function

Correct gradient, correct formulation, and it diverges. The dual is **non-smooth by construction**:
each edge contributes a term that is identically zero across its no-flow interval — the pool's
bid-ask spread — and then kinks. The solver is a spectral projected gradient method, Barzilai-Borwein
steps with an Armijo line search, which assumes smoothness. A correct *sub*gradient carries no
descent guarantee on a non-smooth function. With roughly 70 kink surfaces in roughly 10 dimensions
the iterates diverge; on 4-edge instances there are too few kinks for it to matter.

Two structural facts compound it:

- **The warm start is the worst point available.** Duals converge to no-arbitrage prices, which was
  the original argument for warm starting — and no-arbitrage prices are precisely where every edge
  sits exactly on its no-flow boundary. The warm start is a simultaneous intersection of *m* kink
  surfaces.
- **Nothing is strictly concave.** §5.2 of the paper needs `U` or some `Vᵢ` strictly concave for
  subproblem solutions to be primal-feasible. CFMM routing is the `Vᵢ = 0` case and `U(y) = y_out −
  I_S(y)` is linear, so neither holds.

### The difference from the paper's own experiment

The paper's §6.2 DEX experiment converges unregularised in about 20 iterations — on

```text
U(y) = cᵀy − I(y ≥ 0)      "finding arbitrage in the network of markets"
```

That is **arbitrage**: find any profitable trade. A user order pins the input amount, so the dual
must find prices at which precisely `amount` leaves `token_in` — a far tighter condition, and the
thing that forces the coordinator off the warm start at all. This is open question **Q4** in the
requirements analysis, written down at the start and never answered. It is where the difference
lives, and it is the first thing a future attempt should settle:

> **Q4.** A user order fixes the input amount exactly. The convex flow dual parameterises the
> search by prices and restricts it to the boundary of the optimal space — a formulation whose
> natural question is "what is the best trade", not "allocate exactly this much". The equality is
> expressible in the underlying convex problem; whether it survives the price parameterisation,
> and at what cost, is unanswered.

## Hypotheses tested and eliminated

Recorded because each cost real time and none should be retried.

| Hypothesis | Verdict |
|---|---|
| Numerical scale (raw 1e18 amounts) | Eliminated — passes 1e3 to 1e24 |
| Decimal mismatch (18 vs 6) | Real bug, fixed, not the cause — failures are on 18/18 pairs |
| Simulation budget | Eliminated — 20x budget changes nothing |
| Bisection resolution | Eliminated — 40 steps changes nothing |
| Both pool directions active at once | Eliminated — fees make them mutually exclusive, measured 0 occurrences |
| Price degeneracy (identical pools) | Eliminated — distinct prices fail identically |
| Address ordering | Eliminated |
| Token count alone | Partly — breaks between 18 and 34 tokens, but as a symptom of kink count |
| Gradient error | Eliminated — numerically verified |
| Larger trades should fare better | **Falsified** — solve rate falls from 32.5% to 17.5% across size quintiles |

Two fixes were real and are kept: the line search could not backtrack far enough
(`MAX_BACKTRACKS` 12 to 40), and flow decomposition emitted leg fractions summing past one because
nothing bounded them against the order.

## What a future attempt should do

1. **Settle Q4 above first.** Does the exact-input constraint survive the price parameterisation,
   and what does it cost? Everything else is downstream.
2. **Run `ConvexFlows.jl` against one recorded fixture.** The authors' own Julia implementation
   exists (`github.com/tjdiamandis/ConvexFlows.jl`, experiment code in its `paper` directory).
   This separates "the method does not fit" from "our code is wrong" and was never done.
3. **If continuing, change the solver, not the tuning.** A non-smooth dual wants a bundle method
   or a proper subgradient scheme with diminishing steps — not a line search. Regularising instead
   makes the dual smooth but costs the reduced dual and, per the paper's own figures, about 45x the
   iterations, which does not fit 100 ms.

## Still worth doing, independent of all this

**PRIME's ASGM allocation step** (arXiv 2603.08337), applied to `water_fill`'s exchange-refinement
pass. `water_fill` currently searches every ordered pair of paths to find an improving move —
O(k²) simulations, capped at 400 — then halves the step when none helps. ASGM uses the sign of the
marginal-price gradient to pick the pair directly, O(k), and the marginal prices are already free
from `new_state.spot_price()`. It keeps the path cap, so it does not answer the research question,
but it is a cheap, contained improvement to the algorithm that actually wins today.

## Methodology notes

- **Benchmark with `--jobs 1`.** At the default concurrency, contention against the per-solve
  timeout produced run-to-run swings large enough to swamp the effects being measured: nominally
  identical configurations returned 39, 48 and 53 solved. At `--jobs 1` repeated runs are
  byte-identical. Several intermediate readings taken before this was noticed were unreliable.
- **The synthetic harness earns its keep.** `convex_flow`'s tests reproduce behaviour in 0.01s
  where the benchmark needs a 40-second rebuild plus a run, and they eliminated four hypotheses
  that the benchmark could not have separated. They also caught two bugs in their own helpers,
  which is the trap they exist to avoid.
- **The synthetic harness and the real market can disagree.** Flooring prices away from zero fixed
  the 66-token synthetic case outright and halved the real-market solve rate. A synthetic fix is
  not a fix until the benchmark agrees.

## Reproducing

```bash
# the recorded market fixture lives in Git LFS, under the upstream remote
git lfs install && git lfs fetch upstream && git lfs checkout

# the trade dataset is a GitHub release asset
curl -sL -o aggregator_trades_10k.json \
  https://github.com/propeller-heads/fynd/releases/download/benchmark-data-v1/aggregator_trades_10k.json

cargo test -p fynd-core --all-features --lib convex_flow

# the benchmark reads one config file per algorithm; these are not committed
printf 'algorithm = "convex_flow"\nmax_hops = 2\n' > bench-harness/configs/CF_d2.toml
printf 'algorithm = "convex_flow"\nmax_hops = 3\n' > bench-harness/configs/CF_d3.toml

./scripts/bench.sh --name cf --orders 600 --jobs 1 \
  --trades aggregator_trades_10k.json --configs CF_d2,WF_d3,BF_d3
```

## References

| Ref | Paper |
|---|---|
| [2404.00765](https://arxiv.org/abs/2404.00765) | Convex Network Flows — Diamandis, Angeris, Edelman |
| [2408.12761](https://arxiv.org/abs/2408.12761) | The Convex Geometry of Network Flows — fixed costs on edges |
| [2502.08258](https://arxiv.org/abs/2502.08258) | Marginal Price Optimization — Loesch, Richardson |
| [2603.08337](https://arxiv.org/html/2603.08337) | PRIME — Xu, Zhu, Huang, Tang (ICDE 2026) |
| [2603.02844](https://arxiv.org/abs/2603.02844) | Optimal Routing across CFMMs with Gas Fees |
| [cfmm-routing](https://web.stanford.edu/~boyd/papers/pdf/cfmm-routing.pdf) | Optimal Routing for CFMMs — Angeris, Chitra, Evans, Diamandis |
