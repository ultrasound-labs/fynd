# Routing Research: the convex flow attempt

Record of an investigation into routing algorithms that optimise over the whole pool network
rather than over a set of pre-enumerated paths. Covers what was surveyed, what was built, what was
measured, what turned out to be wrong with it, and what it measures now — so the dead ends are not
walked twice, and so the two wrong diagnoses on the way here are not reached for a third time.

## Verdict

**Do not ship `convex_flow`, and do not replace the allocation step of anything that is
shipped.** Measured against a true single-pair gas-aware optimum, built for this purpose and run
over the recorded Ethereum market:

| config | orders compared | mean shortfall | median | p90 | worst |
|---|---|---|---|---|---|
| **`WF_d3`** | 66 | **−0.00 bps** | 0.00 | 0.00 | 0.01 |
| `ML_d3` | 68 | 0.01 bps | 0.00 | 0.00 | 0.36 |
| `BF_d3` | 68 | 0.47 bps | 0.00 | 0.00 | 31.79 |
| `PFW_d3` | 68 | 0.47 bps | 0.00 | 0.00 | 31.79 |
| `CF_d2` | 44 | 20.79 bps | 0.00 | 43.32 | 261.94 |
| `CF_d3` | 30 | 8.73 bps | 0.00 | 0.00 | 261.94 |

Shortfall is net of gas, against the best net output obtainable on that pair with gas charged per
pool used, restricted to orders where the optimum solved *and* the config's own route stayed on
the one pair — so the comparison isolates allocation from everything else. `water_fill` is at the
optimum: zero to two decimal places, a worst case of one hundredth of a basis point, and two
orders where it edges the reference by using a pool outside the six the reference considers.

So there is no allocation problem to solve here. The splitting step of the algorithm that wins
today is already optimal to within a hundredth of a basis point of a gas-aware optimum, which
independently reproduces the only published audit of the question — Xi and Moallemi measure a
0.055 bps mean shortfall for allocation across 2.98 million swaps — on our own market, our own
liquidity and our own gas price. Convex optimisation has nothing left to win in that step, and
`convex_flow` is the only config that is materially off it.

Three further things the reference measured, each worth more than the verdict on convex flow:

- **The optimum splits across more than one pool on 8 of the 148 orders it solved.** On a dataset
  of roughly thousand-dollar orders, charging gas per pool, the best thing to do is usually to use
  one pool. Splitting is a large-order instrument and this dataset is not large orders.
- **Every shipped algorithm beats the single-pair optimum, by up to 180 bps**, and the first
  mechanism is not what one would guess: `BF_d2`'s two-leg answer on `WETH → USDC` is
  `WETH → ETH` through the native wrapper and then `ETH → USDC`. The gain is reaching pools
  denominated in another token, not splitting better. Which liquidity is *reachable* dominates
  what is done with it, exactly as [ROUTING_ALGORITHMS.md](ROUTING_ALGORITHMS.md) found in the
  literature.
- **`convex_flow`'s subproblem is now exact and it changed nothing structural.** It asks the pool
  for the flow that moves its marginal price to the dual's price ratio — one call, the protocol's
  own math, the protocol's own fee convention — and the coordinator still reaches a primal
  feasible point on 2 of 57 solves. What was wrong with the subproblem was worth fixing and was
  never what was wrong with the method's fit here.

## History of the diagnosis

Three passes, two of them wrong, all recorded below so none is repeated.

1. **The method is wrong for the function** — the dual is non-smooth, a spectral projected
   gradient cannot descend it. [Superseded](#why-it-looked-like-the-method-superseded): the dual
   is C¹ where it was believed to kink, and the paper's own solver is L-BFGS-B.
2. **The coordinator has four defects** — point selection, subproblem resolution, a worthless
   conserving flow, step scaling. [All four were real](#what-was-actually-wrong) and fixing them
   converges synthetic networks of 130 tokens and 257 edges to a residual of 4e-9. The recorded
   market moved from 26 solves in 200 to 73, still losing on every measure.
3. **The oracle was wrong, and then the oracle was right and it still failed.**
   `ProtocolSim::spot_price` is not the derivative of `get_amount_out`
   ([the derivative was never the derivative](#the-derivative-was-never-the-derivative)), and
   `SwapConstraint::PoolTargetPrice` answers the subproblem directly. Both corrected; see the
   verdict above.

The branch is `convex-routing`.

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
| PRIME / ASGM (Xu, Zhu, Huang, Tang, ICDE 2026) | Keeps the path cap, so it does not answer the question. Its allocation step is still worth stealing for `water_fill`; see [Still worth doing](#still-worth-doing-independent-of-all-this). |
| Batch auctions / CoW | Different product; needs a batch window and settlement control. |

Convex flow was chosen for three reasons that all survived scrutiny:

1. **Its formulation is the route format.** `Route::validate` groups swaps into branch collections
   keyed by `token_in` — conservation at token nodes — which is exactly the paper's net flow `y`.
2. **Its per-edge oracle is `ProtocolSim`.** The subproblem needs a gain function and its one-sided
   derivatives: `get_amount_out` is the first, and `spot_price` on the state it returns is the
   second. Most routers cannot supply the derivative; Tycho can.
3. **Hyperedges are native**, so multi-token pools are first-class.

The second of those is false, and it took the benchmark to find out:
[`spot_price` is not that derivative](#the-derivative-was-never-the-derivative). The first and
third hold. The method still fails.

## What was built

`fynd-core/src/algorithm/convex_flow/`, registered as the built-in `convex_flow`, roughly 1,600
lines with 53 tests.

| Module | Contents |
|---|---|
| `edge.rs` | The per-edge arbitrage subproblem: no-flow test, regula falsi on `1/√h⁺` for `η₂h⁺(w⋆) ≤ η₁ ≤ η₂h⁻(w⋆)`, and the check that a flow beats not trading |
| `dual.rs` | The reduced dual (one variable per token), gradient, spectral projected gradient coordinator, feasibility-first selection of the point it returns |
| `mod.rs` | Network construction, warm start from `TokenGasPrices`, flow decomposition, `Algorithm` impl |
| `config.rs` | Tuning constants, each with the reasoning |

Reuses `split_primitives` for route construction, the graph's bidirectional search for
reachability, and `water_fill`'s `Deadline` and gas conversion. No new dependencies — the
coordinator is implemented rather than pulled in.

## Measured results

Deterministic runs (`--jobs 1`), recorded Ethereum fixture, `aggregator_trades_10k.json`, 200
orders, `--timeout-ms 60000` for the reference's benefit. Bps are against the `BF_d2` baseline the
harness measures everything against, over the orders a config and the baseline both solved.

| config | solved | mean bps | median bps | better | worse | tie | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| `BF_d2` (baseline) | 193/200 | — | — | — | — | — | 0.7ms | 2.1ms |
| `OPT` (reference) | 148/200 | −106.3 | +0.0 | 5 | 66 | 76 | 4.5ms | 77.1ms |
| **`CF_d2`** | **73/200** | **−125.1** | −41.6 | 1 | 44 | 27 | 109ms | **2.75s** |
| **`CF_d3`** | **31/200** | **−22.7** | +0.0 | 1 | 4 | 25 | 49ms | **0.86s** |
| `WF_d3` | 198/200 | +147.4 | +0.0 | 69 | 0 | 124 | 9.5ms | 13.5ms |
| `BF_d3` | 194/200 | +146.8 | +0.0 | 58 | 0 | 135 | 3.7ms | 7.9ms |
| `ML_d3` | 198/200 | +146.3 | +0.0 | 59 | 0 | 134 | 1.8ms | 3.7ms |
| `PFW_d3` | 194/200 | +146.8 | +0.0 | 58 | 0 | 135 | 4.6ms | 8.4ms |

`OPT` is the single-pair gas-aware optimum (`bench-harness/src/optimum.rs`), not a router: it is
direct-pool only, which is why it misses 52 orders and loses to the baseline on 66 — the baseline
routes through other denominations and it cannot. Its purpose is the shortfall table in the
[verdict](#verdict), which is the measurement this whole record was missing.

For convex flow the three readings are unchanged from the pass before, and no longer close:

**It does not answer.** 15 to 37% against 97 to 99%. Every failure is the same one: the
coordinator returns a flow that cannot be walked from `token_in` to `token_out`.

**Where it answers it does not win.** Two orders out of roughly a hundred compared, across both
configs, beat Bellman-Ford. Most of the rest are ties — the same route, to the wei.

**It does not fit the budget.** A p95 of 0.86 to 2.75 seconds against the 100 ms a quote has.

The progression across the second and third passes, same 200 orders, `CF_d2`:

| | solved | mean bps |
|---|---|---|
| as first written | 26/200 | −251.9 |
| coordinator defects fixed | 50/200 | −88.5 |
| subproblem asked of the pool | 73/200 | −125.1 |

Each fix was correct and each raised the solve rate, and the quality moves with how much the
coordinator is *trusted*: a correct subproblem makes an infeasible dual point propose flows
confidently, `decompose` clamps them into a route, and the route is worse than not answering. The
fixes made the parts right; the whole was never limited by the parts.

### Why the synthetic fixes barely moved it

Per-solve instrumentation over 60 orders, `CF_d2`, counting what the coordinator actually
returned:

| | with `spot_price` as the derivative | with a difference quotient |
|---|---|---|
| usable flow (residual ≤ 1e-8) | 1/47 | 1/39 |
| nothing flows (residual = 1.0) | 34/47 | 25/39 |
| over-trading (residual > 1.0) | 15/47 | 12/39 |
| simulation budget exhausted | 19/47 | 17/39 |
| median edges in the network | 27 | 13 |

One solve in forty reaches a feasible point. The rest alternate between idling and saturating,
which is the same behaviour the first pass recorded as prices diverging — and it survives every
fix that converges the synthetic networks. Those fixes were not wrong, and the synthetic
instances were not the problem: whatever the recorded market has that a market of
`ConstantProductSim` pools does not is still there. The known candidates, in order of suspicion:
pool depths spread over orders of magnitude within one pair, which is the conditioning the
Barzilai-Borwein step length has to track; genuine arbitrage gaps between pools of the same pair,
so the warm start is not a no-arbitrage point; and tick boundaries, where `h⁺` jumps and the
subproblem's root does not exist, which is a real non-smoothness in the dual where the first
pass's no-flow kinks were not.

### The derivative was never the derivative

The second of the three reasons convex flow was chosen is false, and it is false for every pool
in the market.

`ProtocolSim::spot_price` is documented as "the amount of `quote` required to buy exactly 1 unit
of `base`, **accounting for the protocol fee (i.e. `price = pre_fee_price / (1.0 - fee)`)**". It
is a buy price marked *up* by the fee. The derivative of `get_amount_out` on the sell side is the
same pre-fee price marked *down* by it. Thirteen protocol families in tycho-simulation go through
one `add_fee_markup` helper, so across the market

```text
spot_price = h⁺(0) / (1 − fee)²
```

For a 30 bp pool that is 60 bps of error in a quantity whose signal — the price impact of an
ordinary order against pool depth — is one basis point. Solving `spot_price(w) = η₁/η₂` does not
find the optimum of anything; it drives every edge to about `fee × reserves` of flow whatever the
order is worth. That is the over-trading, and it is why the first pass saw routes that were
clamped junk.

Correcting it by `(1 − fee)²` is not available: `ProtocolSim::fee` is `todo!()` for VM pools and
panics the worker thread, which is what a benchmark run with the correction did on its first
Uniswap V4 pool. So the subproblem now takes a difference quotient of `get_amount_out` over a
step a few thousandths of the trial point — the derivative of the function being optimised,
measured on that function. It costs a second simulation per trial, and it is worth it: a
quotient over `[w, w+δ]` is `h⁺(w + δ/2)` to second order, so it displaces the answer by a step
size this module chooses, where a quoted price displaces it by a fee the market chooses,
amplified by the pool's depth.

It moved the mean from −141.5 to −53.1 bps on 60 orders and the solve rate not at all. A real
defect, found by the benchmark, that was not the binding one.

Two notes for anyone reading the subproblem:

- **The secant from zero is not a substitute.** `h(w)/w > η₁/η₂` says the objective is still
  positive at `w`, not that it is still rising. Its root is at exactly twice the optimum for a
  constant-product curve, which is what the first attempt at this measured.
- **A difference quotient needs resolvable amounts.** Outputs are integers, and at toy
  magnitudes — thousand-unit reserves — one raw unit is a large fraction of the quotient and the
  slope is noise. The subproblem falls back to the average rate from zero below 64 raw units of
  output difference, and the tests that need a real derivative were moved to wei magnitudes.

## Why it looked like the method (superseded)

Kept because the measurements are real and the reasoning is where the first pass went wrong. The
conclusion it reaches is contradicted by [the next section](#what-was-actually-wrong); read them
together.

Four experiments, each eliminating a candidate.

### It is stuck, not slow

Raising the iteration cap 10x (552 median iterations) and the simulation budget 100x left the
net-flow residual at **4.90e+01, identical to three significant figures**, and the active edge
count at 4. More compute buys nothing.

> This experiment cannot support that reading. `Solution::residual` was the residual of the
> point with the lowest *dual value*, and the dual value is flat to around 1e-10 over the whole
> region the iterates visit, so the recorded number is insensitive to extra iterations by
> construction. The iterates were not stuck; the figure reporting on them could not move.

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

**This is the conclusion that does not hold.** See the next section. Correct gradient, correct
formulation, and it diverges. The dual is **non-smooth by construction**:
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

## What was actually wrong

Four defects, in the coordinator and the edge subproblem. Each was found by re-implementing
`dual.rs` and `edge.rs` as a standalone model — same constants, same integer bisection, same SPG
loop — and instrumenting it on zero-fee constant-product networks. The failures reproduce there
with no fees, no decimals, no `get_limits` and no market data, which is what says they are not in
the market plumbing.

### The returned point was chosen by the wrong quantity

`minimise` kept the flows from the point with the lowest dual value. The dual value is a bound on
the order's output, so over the region the iterates visit it varies by the slippage the order pays
— basis points — while the net-flow residual over that same region varies by eight orders of
magnitude. The route is built from the flows, and flows that do not conserve cannot be walked into
legs, so selection has to see that difference.

Changing only that comparison, on an 18-token network with the order at a thousandth of pool
depth: residual **2.80e-01 → 6.38e-07**, input allocated **57% → 100%**, active edges **5 → 8**.
The iterates were passing through primal-feasible eight-way splits all along and the bookkeeping
threw them away. Line-search trials are now candidates too — the backtracking is what crosses the
region where edges first take flow, so that is where the feasible points are.

Selection is feasibility first, then dual value once a point conserves to within the stopping
tolerance, which is the condition under which the value is the right discriminator.

### The subproblem could not resolve an ordinary leg

Twelve bisection steps over a bracket of `EDGE_VALUE_SLACK` times the order's value put the
resolution floor at `4/2^12 ≈ 9.8e-04` of the order. Any edge whose optimum was finer reported
**exactly zero**: measured directly, targets at 1e-1, 1e-2 and 1e-3 of the order resolved, and
1e-4 — which `MIN_LEG_FRACTION` admits as a leg — came back as an unused edge. The fallback meant
to catch that case (`!best.is_active() && !low.is_zero()`) was unreachable: `low` only ever
advanced on the same branch that set `best`.

So the computed gradient was a staircase, and the non-smoothness the first pass attributed to the
method was manufactured here. Two constants downstream were inconsistent with it: `MIN_LEG_FRACTION`
at `1e-4` was ten times below what a subproblem could report, and `GAP_TOLERANCE` at `1e-7` four
orders below, so `stationarity() <= GAP_TOLERANCE` could only fire if quantised flows happened to
cancel — which is why runs ended on the iteration cap with hundreds of iterations spent after the
answer was in hand.

The subproblem is now regula falsi on `1/√h⁺`, which is *exactly linear in w* for a
constant-product curve, so one interpolation lands on the root; a tick curve is that shape
piecewise and anything monotone still converges, with the Illinois safeguard and the bracket
underneath. Same step cap, resolution limited by the pool's own arithmetic rather than by the
bracket. `EDGE_VALUE_SLACK` was then measured again and **kept at 4.0**: the slack cost resolution
only while the bracket width set it, and narrowing it to the order exactly loses the widest
instances, which need the room an infeasible intermediate iterate takes.

### A flow that conserves perfectly can be worthless

The fix above surfaced this one. A pool asked for one raw unit pays zero by integer division while
still quoting a marginal price above the price ratio, so the subproblem would return "takes the
whole order, pays nothing". That flow conserves at every node, so it is perfectly feasible, and a
coordinator ranking on conservation prefers it to everything else it will ever see; it also drives
the dual value below the optimum it is supposed to bound, which is what makes the value untrustworthy
as a tie-break. `Edge::worth_trading` now drops any flow worth less than not trading —
`η₂·h(w) > η₁·w`, the subproblem's own objective against its value at zero.

### The step schedule was four to ten orders of magnitude too long

`INITIAL_STEP` was `1.0`, against its own doc comment arguing for a small value. The first trial
projects `ν_in` to exactly zero while the answer sits within O(amount/depth) — about `1e-4` — of
the warm start, so iteration one spent ten to fourteen full dual evaluations backtracking, which on
a 65-edge network is most of the simulation budget before the first accepted step. It is now `1e-6`:
overshooting is paid for one evaluation at a time by the line search, undershooting is corrected for
free by the Barzilai-Borwein length from the second iteration on.

Worth keeping in view, though it is now second-order: the warm-start gradient is set by a tuning
constant, not by the market. Every intermediate token reports exactly `−EDGE_VALUE_SLACK`, because
each edge draining it proposes its whole cap against one unit of unspent order — measured at
`−3.999` on every intermediate coordinate of a ten-token network, against a useful signal of `1.0`
at `token_in`.

### Measured, before and after

Identical synthetic instances, the direct-plus-`k`-two-hop markets from
`test_solve_scales_with_token_count`, order at a ten-thousandth of pool depth, instrumented in
place:

| tokens | edges | iterations | simulations | residual |
|---|---|---|---|---|
| 3 | 3 | 54 → **16** | 11,148 → **136** | 9.6e-07 → **6.3e-09** |
| 6 | 9 | 60 → **20** | 12,324 → **169** | 9.6e-07 → **3.6e-11** |
| 10 | 17 | 56 → **30** | 14,928 → **448** | 7.7e-07 → **5.1e-11** |
| 18 | 33 | 53 → **36** | 15,060 → **974** | 7.7e-07 → **1.6e-15** |
| 34 | 65 | 6 → **8** | 11,808 → **1,659** | **1.0e-01** → **7.7e-09** |
| 66 | 129 | 2 → **9** | 14,820 → **3,486** | **6.4e-02** → **3.7e-10** |
| 130 | 257 | no route at all | — → **7,326** | — → **3.8e-09** |

Twenty to eighty times fewer simulations, three to seven orders of magnitude less infeasibility,
and simulation cost now roughly linear in edge count (257 edges, 7,326 solves — about 28 an edge).
`test_solve_scales_past_thirty_tokens` was `#[ignore]`d as a known limit; it passes and the ignore
is gone. The sweep stops at 128 middles because `mk_token` runs out of single-byte addresses, not
because the solve does.

### The fee convention, verified and wrong

The second pass flagged this as an unverified risk and modelled it: on a two-pool instance that
is exact at zero fee, a 5 bp fee the marginal price does not account for made the solver report
convergence having allocated 4.98 times the order. The benchmark then confirmed it on the real
market and the mechanism turned out to be worse than modelled — not a fee missing from the
marginal, but a fee applied to it in the wrong direction.
[The derivative was never the derivative](#the-derivative-was-never-the-derivative) has it.

`FeeConstantProductSim` (`algorithm/test_utils.rs`) now covers the fee-bearing case with the
correct convention, and four `convex_flow` tests use it. It was the right thing to add and it
could not have found this on its own: a double written to the contract the algorithm needs
cannot reveal that the market does not meet it.

The fix in the end was not to correct the quoted price but to stop quoting. Two things in this
codebase already had the answer, and both are now used:

- **`SwapConstraint::PoolTargetPrice` is the subproblem.** "The swap required to move the pool's
  marginal price down to a target" is the subproblem's own sentence. Around twenty protocols
  implement `query_pool_swap`; Uniswap V2 inverts its reserve formula, V3 and V4 use the pool's
  native `sqrt_price_limit` swap, and the rest fall back to tycho's generic search, which reports
  its probes in `price_points` so the budget can be billed honestly. One call, the protocol's own
  fee convention, and exactness where a tick makes the price discrete.
  `ComponentDepthComputation` has been calling it every block the whole time.
- **The fee-net marginal price is free.** `min(spot(in→out), 1/spot(out→in))` is `h⁺(0)` whether
  a protocol marks the fee up or quotes it net, costs two lookups in a table the derived pass
  refreshes every block, and needs no `ProtocolSim::fee` — which is `todo!()` for VM pools and
  panics the worker. `ComponentDepthComputation::net_marginal_price` already did this.

## Q4, which the first pass left open

Settled, and it is not where the difference lives. The exact-input constraint survives the price
parameterisation at no cost, and the code already implements it: `U(y) = y_out − I_S(y)` over
`y_in ≥ −amount` gives `Ū(ν) = ν_in·amount`, finite on `ν ≥ 0`, and exactness falls out of
complementary slackness — `ν_in > 0` forces `y_in = −amount`. Nothing is lost.

What actually separates this from the paper's §6.2 arbitrage experiment is threefold, and none of it
is the equality:

- **Where the answer sits.** For arbitrage, prices move O(1) from the start. For a user order the
  dual optimum is within O(amount/pool depth) of the no-arbitrage warm start, and *below* it:
  `ν⋆_in = h⁺(amount) ≈ (1 − 2·amount/R)·h⁺(0)` for a constant-product pool. The whole answer lives
  in a ball of radius ~1e-4 around the starting point, which is why the step schedule mattered as
  much as it did.
- **Feasibility is free there.** §6.2 reconstructs `y = Σ Aᵢxᵢ`, so conservation holds by
  construction and the reported residual measures `y ≥ 0` instead. Here the dual has to deliver
  conservation itself, which is why the selection rule was the dominant defect.
- **Their subproblems are unique.** §6.2 adds a strictly concave edge penalty
  `Vᵢ(xᵢ) = −½‖(xᵢ)₋‖²`. That is the same end `Edge::worth_trading` and the feasibility-first
  selection reach by other means, and far cheaper than the regularisation the first pass priced at
  45x the iterations.

Two further corrections to the first pass's reading of the paper:

- **§5.2 does not require what it was read to require.** "If the set `Tᵢ` is strictly convex, then
  the set `∂fᵢ(ηᵢ⋆)` consists of a single point and `x⋆ᵢ = x̃⋆ᵢ`" — primal feasibility at the dual
  optimum, with no condition on `U`. The strict concavity of `U` is a condition for the *recovery
  LP* in the non-strict case, and the non-strict edges are the linear ones (fixed-price pools,
  limit orders), not constant-product pools.
- **The dual is not non-smooth at the no-flow boundary.** For a strictly convex trade set `w⋆(η) → 0`
  continuously as `η₁/η₂ ↑ h⁺(0)`, so `∇fᵢ = x⋆ᵢ → 0`: `fᵢ` is C¹ there, with a jump only in
  curvature. And the paper's own implementation is off-the-shelf **L-BFGS-B**, a smooth quasi-Newton
  method, converging linearly on 100 assets and 2,500 markets. Kinks that broke smooth methods would
  have broken that first.

## Hypotheses tested and eliminated

Recorded because each cost real time and none should be retried.

| Hypothesis | Verdict |
|---|---|
| Numerical scale (raw 1e18 amounts) | Eliminated — passes 1e3 to 1e24 |
| Decimal mismatch (18 vs 6) | Real bug, fixed, not the cause — failures are on 18/18 pairs |
| Simulation budget | Eliminated — 20x budget changes nothing |
| Bisection resolution | **Wrongly eliminated.** Raising the step count alone does not help, because the returned point was selected on dual value and that choice is insensitive to resolution. With selection fixed, resolution is one of the four real defects |
| Both pool directions active at once | Eliminated — fees make them mutually exclusive, measured 0 occurrences |
| Price degeneracy (identical pools) | Eliminated — distinct prices fail identically |
| Address ordering | Eliminated |
| Token count alone | Partly — broke between 18 and 34 tokens, and the cause was cost per evaluation and the selection rule, not kink count. 130 tokens and 257 edges solve now |
| Gradient error | Eliminated — numerically verified |
| Larger trades should fare better | **Falsified** — solve rate falls from 32.5% to 17.5% across size quintiles. Consistent with cost, not with divergence: a larger order activates more edges per evaluation, and at twelve bisection steps an evaluation over a few hundred edges was already thousands of simulations |
| The dual value identifies the best flow | **Falsified, and this was the main defect.** It is flat to ~1e-10 across the region the iterates visit |
| A conserving flow is a useful flow | **Falsified** — taking the whole order and paying nothing conserves at every node |
| `spot_price` is the derivative of `get_amount_out` | **Falsified**, and it was one of the three reasons the method was chosen. It is that derivative divided by `(1 − fee)²`, uniformly across the market |
| Correcting the fee with `ProtocolSim::fee` | Unavailable — `todo!()` for VM pools, panics the worker on the first Uniswap V4 pool |
| The fee convention is what breaks the real market | Partly — it was real, and worth 88 bps of mean quality, and the solve rate did not move |
| A bigger simulation budget would help | **Falsified, twice.** The first pass measured no change; the second found the budget acting as an accidental regulariser, so freeing it made quality worse |
| `MAX_STEP` on a bad secant pair is too long | Eliminated, against expectation. Throwing the step to `1e6` when `sy ≤ 0` looks indefensible next to useful steps of `1e-6` to `1e-2`, and it is the best of five options measured over a 20-instance grid: 18 instances clean, against 17 for a `1e2` ceiling, 17 for `1.0`, 16 for doubling the previous step and 12 for keeping it. A long step with deep backtracking is how the method explores a scale it has no curvature for. Left alone |

Two fixes were real and are kept: the line search could not backtrack far enough
(`MAX_BACKTRACKS` 12 to 40), and flow decomposition emitted leg fractions summing past one because
nothing bounded them against the order.

## What to do next

The benchmark has run, so the question is no longer "does the implementation work" but "is this
worth more time". Read in order:

1. **Nothing here is shippable, and the gap is not incremental.** A 15 to 25% solve rate and a
   2.6 second p95 are not two fixes away from 98% and 13 ms. The three shipped algorithms all
   sit within a basis point of each other at the median, and `water_fill` beats the baseline on
   69 of 193 orders; convex flow beats it on 1 of 49.
2. **If anyone picks it up, start with the idle-or-saturated behaviour**, not with tuning. One
   solve in forty finds a feasible point on the recorded market while synthetic networks six
   times larger converge to 4e-9. That gap is the whole remaining question, and the three
   candidates are listed under
   [why the synthetic fixes barely moved it](#why-the-synthetic-fixes-barely-moved-it). Pool
   depths spread over orders of magnitude is the one to measure first, because it is the
   cheapest to test: take one real pair, keep only the pools within a decade of the deepest, and
   see whether the solve rate moves.
3. **The latency tail needs an edge count bound**, and that is awkward: the networks are 281
   edges at `max_hops = 2` on a busy pair, each needing roughly 0.8 simulations per coordinator
   iteration, and bounding the edge set is the path cap the method exists to avoid. A bound by
   dominance — drop a pool that is both worse priced and shallower than another on the same pair
   — would be legitimate and is unexplored.
4. **Gas is still not in the objective.** `MAX_LEGS` stands in for the fixed-cost term of
   arXiv 2408.12761. Any serious attempt needs it, since the routes this returns are mostly
   single-leg anyway and the ones that are not have to pay for their legs.
5. **`ConvexFlows.jl` against one recorded fixture** remains the clean separator between "the
   method does not fit" and "our code is wrong". It is now worth less than it was: the code has
   been shown to converge on synthetic networks at scale, so the question it would answer is
   narrower — whether the authors' implementation finds feasible points on *this* market.

Three things not to do, each because it has been tried and measured. Q4 is
[settled](#q4-which-the-first-pass-left-open) and was never the problem. A bundle or subgradient
method does not follow from the dual being non-smooth at the no-flow boundary, because it is not.
And raising the simulation budget does not help, for a reason that took the benchmark to see: it
is currently acting as an accidental regulariser. Making the unpriced-input branch of the
subproblem cheap — one simulation instead of twenty-four for the same answer — raised the solve
rate by half and tripled the orders lost, because the budget it freed was spent on further
excursions to a zero price.

## Where to look instead

[ROUTING_ALGORITHMS.md](ROUTING_ALGORITHMS.md) surveys the methods that could replace this one
and concludes that the premise is wrong: across 2.98 million recorded swaps, optimal allocation
given a fixed pool set is worth **0.055 bps**, while one stale block costs **1.3 to 1.8 bps** and
failing to activate the right pools dominates both. It also records two things found in this
codebase while benchmarking, either of which would have saved most of the work above:
`SwapConstraint::PoolTargetPrice` is the per-edge subproblem as a single analytic oracle call,
and the fee-net marginal price is free from the derived spot-price table.

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
- **A zero-fee double cannot find a fee bug.** Every pool in the harness was `ConstantProductSim`,
  which charges nothing, and the subproblem's whole correctness rests on a fee-consistency property
  between two pool methods. The harness could not have failed on it. `FeeConstantProductSim` exists
  now; the lesson generalises to any property of the double that is load-bearing for the algorithm.
- **Instrument the metric before trusting the experiment.** "It is stuck, not slow" was measured
  with `Solution::residual`, which reported the residual of the lowest-dual-value point — a figure
  that cannot respond to more iterations when the dual value is flat. The experiment was sound and
  the instrument was not, and four of the eliminated hypotheses were eliminated through it.
- **Synthetic convergence is not evidence about the market.** Four defects were found and fixed
  against a standalone model and a synthetic harness, taking networks of 130 tokens and 257 edges
  from no answer to a residual of 4e-9, and the recorded market moved from 26 solves in 200 to
  50, still losing. The harness and the model agreed with each other and both were measuring a
  market made of one pool type with one fee convention and depths within a decade of each other.
  The benchmark was the only thing that could have found the convention error, and it was
  available the whole time.
- **Run the cheap end-to-end measurement first.** The order that actually happened — model, fix,
  synthetic test, benchmark — spent most of its effort before the measurement that reframed the
  problem. A 20-order benchmark run takes a minute and would have shown the 1-in-20 solve rate
  and the 1.5 second p95 on the first day.
- **The model is not the solver, in both directions.** It reproduced the real failures faithfully
  enough to localise four defects, and it also reports a stall the Rust code does not have: 34
  tokens and 65 edges against an order a millionth of pool depth cycles between zero flow and
  over-trading two edges, where `test_solve_small_order_against_many_edges` clears its floor on
  the same shape. Differences in `warm_start`'s reference propagation and the single-leg fallback
  in `find_best_route` are enough to account for it. A model finding is a lead, not a result.
- **Re-implementing the solver as a throwaway model paid for itself.** A faithful model of
  `dual.rs` and `edge.rs` — same constants, same integer bisection, same loop — reproduced the
  reported residuals and price divergence on zero-fee networks with no market data at all, which
  is what localised the defects to the coordinator. It also made each candidate fix a one-line
  experiment across an instance grid, where the Rust path costs a rebuild per attempt. The two
  disagreed once, on `EDGE_VALUE_SLACK`, and the rule from the note above applied: the model
  proposed narrowing it, the instance grid said it loses the widest networks, and it was kept.

## Reproducing

```bash
# the recorded market fixture lives in Git LFS; `git lfs pull` is enough on a clone of this
# remote, and the file is ~4.9 MB when it is really there rather than a 132-byte pointer
brew install git-lfs && git lfs install --local && git lfs pull

# the trade dataset is a GitHub release asset
curl -sL -o aggregator_trades_10k.json \
  https://github.com/propeller-heads/fynd/releases/download/benchmark-data-v1/aggregator_trades_10k.json

cargo test -p fynd-core --all-features --lib convex_flow

# the benchmark reads one config file per algorithm; these are not committed
printf 'algorithm = "convex_flow"\nmax_hops = 2\n' > bench-harness/configs/CF_d2.toml
printf 'algorithm = "convex_flow"\nmax_hops = 3\n' > bench-harness/configs/CF_d3.toml

# the comparison in Measured results, about twelve minutes at --jobs 1
./scripts/bench.sh --name cf --orders 200 --jobs 1 --trades aggregator_trades_10k.json \
  --configs CF_d2,CF_d3,WF_d3,BF_d3,ML_d3,PFW_d3

# per-solve residuals, edge counts and simulation counts, which is where the diagnosis is
RUST_LOG="fynd_core::algorithm::convex_flow=debug" ./scripts/bench.sh --name dbg --orders 60 \
  --jobs 1 --logs --trades aggregator_trades_10k.json --configs CF_d2 2>&1 |
  grep "convex flow solved"
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
