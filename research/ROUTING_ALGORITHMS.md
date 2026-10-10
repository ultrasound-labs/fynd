# Routing algorithms: what is worth building next

A survey of convex optimisation methods for order routing, written after
[the convex flow attempt](ROUTING_RESEARCH.md) was benchmarked and lost. It asks one question:
given what that benchmark measured, which method would do better, and by how much.

The short answer is that the question is misdirected, and there are now two independent numbers
that say so — one from the literature and one from this market.

Across 2.98 million WETH–USDC swaps, Xi and Moallemi measure what optimal *allocation* is worth
when the pool set is held fixed: **0.055 bps**. Over the same sample, losing a single block of
pool state costs **1.3 to 1.8 bps**, and failing to activate the right pools is "the overwhelming
driver of suboptimality".

Experiment 1 below has since been built and run, twice: once on realised flow and once on a
dataset built to span order sizes and liquidity tiers, because the first one turned out to be
answering an easier question than it looked. On both, `water_fill`'s allocation is at the
optimum — **0.00 bps mean** on realised flow, **0.01 bps at the p90** on the wide set — so a
better convex optimiser is chasing the smallest of the three terms while the two larger ones are
not optimisation problems.

What the wide set does change is how much the *other* algorithms lose by not splitting: 10 to
15 bps at the p90 and thousands at the p99, against 0.01 to 0.47 bps on realised flow. So "the
four algorithms are within 1.1 bps of each other" was a fact about thousand-dollar orders, not
about the algorithms, and the same caveat applies to the 0.055 bps above — it too is measured on
realised flow. Both tables are in [ROUTING_RESEARCH.md](ROUTING_RESEARCH.md#verdict).

That does not make the methods below uninteresting — one of them is a clear improvement to the
algorithm that wins today, for a reason that has nothing to do with allocation quality. It does
mean none of them should be sold as a quality win.

## What our own benchmark already said

200 orders, recorded Ethereum market, `--jobs 1` (full table in
[ROUTING_RESEARCH.md](ROUTING_RESEARCH.md#measured-results)):

| config | solved | mean bps | better | worse | tie |
|---|---|---|---|---|---|
| `WF_d3` | 198/200 | +147.4 | 69 | 0 | 124 |
| `BF_d3` | 194/200 | +146.8 | 58 | 0 | 135 |
| `ML_d3` | 198/200 | +146.3 | 59 | 0 | 134 |
| `PFW_d3` | 194/200 | +146.8 | 58 | 0 | 135 |

Four algorithms with four different allocation strategies — greedy chunks, shortest path, most
liquid, Frank-Wolfe — land within **1.1 bps of each other** and return the identical route, to
the wei, on 124 to 135 of the 193 orders they share. Where `water_fill` wins its 69 orders it is
not because it allocates better than Bellman-Ford; it is because it finds a split across pools
Bellman-Ford never considered.

That near-agreement is a property of the dataset. On the wide scenario set the same four separate
cleanly — `water_fill` wins 244 orders and loses none, where `bellman_ford` wins 52 — and the
reason is the same one: it splits and they mostly do not. Which liquidity reaches the candidate
set, and whether the order is big enough for that to matter, is the differentiator. The
allocation arithmetic is not.

## Two things in the codebase worth more than any algorithm choice

Both were found while benchmarking convex flow. Either would have saved most of that work.

**`SwapConstraint::PoolTargetPrice` is the per-edge subproblem, as one oracle call.** Every
marginal-price method needs the same primitive: *given a price, how much flow moves this pool's
marginal price to it*. `ProtocolSim::query_pool_swap` answers exactly that, around twenty
protocols implement it, Uniswap V3 and V4 answer it **analytically** through the pool's own
`sqrt_price_limit` swap (`clmm_swap_to_price`), and tycho's generic 30-step search covers the
rest. It also returns `price_points`: the amounts and prices the swap crossed. `convex_flow`
reimplemented this as a 12-step bisection over `get_amount_out`, got the price convention wrong,
and spent its whole simulation budget on it. `ComponentDepthComputation` has been calling the
real thing every block the whole time.

**The fee-net marginal price is free, and `spot_price` is not it.** `ProtocolSim::spot_price` is
contractually `pre_fee_price / (1 - fee)` — a buy price marked *up* by the fee — where the
derivative of `get_amount_out` is the same price marked *down* by it. `ProtocolSim::fee` cannot
be used to correct it (`todo!()` for VM pools; it panics the worker). The codebase already has
the robust answer, in `derived/computations/component_depth.rs`:

```text
net marginal price = min( spot(in→out), 1 / spot(out→in) )
```

which equals `h⁺(0)` whether a protocol adds the fee or not, costs two lookups in a table the
derived pass refreshes every block, and needs no simulation at all. Any method that reasons about
marginal prices should start here.

## The candidates

Ranked by what they would be worth to fynd, which is not the same as how good they are.

### 1. The transfer algorithm — one-hop price equalisation

Jiang, Li and Wen (FC '26), with the convergence rate in
[2603.27172](https://arxiv.org/abs/2603.27172) and an extension to negative allocations in
[2604.02909](https://arxiv.org/abs/2604.02909). Xi and Moallemi arrive at essentially the same
algorithm independently.

Allocate the order across parallel pools on one pair by repeatedly moving flow from the pool with
the highest marginal price to the one with the lowest, halving the transfer until it does not
overshoot. Terminates in `O(Nκ log 1/ε)` rounds, `κ` the liquidity heterogeneity — and the
authors' own experiment reports the `κ` factor is loose: as `κ` goes from 8 to 1003 the round
count stays between 70 and 90. Liquidity spread across orders of magnitude, which is what a real
pair looks like and what we suspect hurts convex flow, costs it almost nothing.

It is a strict generalisation of what `water_fill` does. `water_fill` fills in 20 then 256
chunks, giving each chunk to the best path — a discretised price equalisation — and then refines
by searching every ordered pair of paths for an improving exchange, `O(k²)` simulations capped at
400. The transfer algorithm picks the pair by `argmax`/`argmin` of marginal price, which is
`O(1)` per round with prices that are now free, and converges to a tolerance rather than to a
chunk grid. The paper's own framing is that it is computable "via Uniswap V3 type API", which is
`PoolTargetPrice`.

**What it is worth.** Not allocation quality: `water_fill` already refines to one part in 16,384
of the order, far past the 0.055 bps that perfect allocation is worth. It is worth the
*simulation budget it frees*. Those 400 trial simulations and the chunk grid are what currently
bound `DEFAULT_MAX_PATHS`; spending them on more candidate paths instead buys activation, which
is the term that dominates. That is the case for building it, and it should be measured that way
— candidate count held up, allocation cost down — not as a bps win.

### 2. Gas-aware pool activation — where the money actually is

The dominant share of the measured 2.02 bps, and the one term fynd's algorithms address only
crudely: `water_fill` gates gas per path at 20-chunk granularity, and `convex_flow` ignores gas
entirely, with `MAX_LEGS = 8` as a stand-in.

Three sources, none of them a drop-in:

- **Xi and Moallemi's G-FVO benchmark** ([2607.20762](https://arxiv.org/abs/2607.20762)) is a
  bisection that selects pools accounting for discrete per-pool gas and finds the exact
  maximum-output allocation. Built, as the `OPT` config — and it answered its question against
  expectation: the allocation it was meant to expose a gap in has no gap. What it did expose is
  that the optimum uses one pool on 140 of 148 orders at this dataset's sizes, so gas-aware
  *activation* on a single pair is mostly the trivial decision. The interesting activation
  question is across pairs, which is the next two items.
- **Escudero, Lara and Sama** ([2603.02844](https://arxiv.org/abs/2603.02844)) give the theory:
  gas as fixed activation costs makes a mixed-integer model, with a relaxation, a KKT system
  linking prices, fees and activation, and explicit bounds on the utility gap the relaxation
  admits. Theory only — no algorithm, no experiments — but the no-trade conditions are the
  precise statement of when a leg should not be used at all.
- **Zhavoronkov** ([2607.22540](https://arxiv.org/abs/2607.22540)) is the closest thing to a
  production architecture in the literature, and it is `water_fill`'s architecture: gas-aware
  `k`-shortest-path enumeration with the gas cost folded into the edge weight
  (`w_a = -log π_a + γ g_a`), then concave allocation over `k = 20` candidate paths with a
  per-pool price-impact cap. Against five production routers on WETH→USDT it reports median
  shortfall below 1 bps at small sizes and 3.3 to 4.6 bps at 1 to 2 WETH. Two ideas are directly
  portable: **gas in the edge weight during enumeration** rather than as a gate after it, and
  **k = 20** against our current four.

### 3. Marginal Price Optimization — Newton in price space

Loesch and Richardson ([2502.08258](https://arxiv.org/abs/2502.08258)), the paper already in
`research/`. Parameterise by one price per token, root-find on net flow with Newton-Raphson, 20
to 100× faster than CVXPY with Clarabel.

The first pass dismissed it for needing a Jacobian. That objection is weaker than it looked: its
Proposition 22 shows the Jacobian is `O(N)` to assemble when each curve touches a bounded number
of tokens, which is true here, and the authors compute it by perturbation — which with
`PoolTargetPrice` is one cheap call per entry, parallel across edges. Newton's convergence is
also the right answer to what broke convex flow: our spectral projected gradient oscillated
between idle and saturated on real instances, which is a conditioning failure, and Newton with
the log-price change of variable the paper adopts is the standard remedy.

**What it is worth.** It answers the same question convex flow answered — whole-network
optimisation — and our benchmark says that question is not where the value is. Its own authors
report singular Jacobians needing a fallback, and degradation past a handful of tokens at 1,000
curves. Worth knowing about; not worth building before the two above.

### 4. Unexplored: the piecewise-linear structure, exactly

Concentrated liquidity is piecewise-constant in liquidity per tick, so a CLMM pool's gain
function is piecewise and its derivative jumps at tick boundaries. Every smooth method treats
that as an obstacle — it is our leading remaining suspect for convex flow's real-market
behaviour. The convex flow paper itself notes the linear case "is better solved with
special-purpose algorithms such as the network simplex method".

Nobody in this literature exploits it, and fynd has the hook: `PoolSwap::price_points` returns
the amounts, outputs and prices a swap crossed — the pool's own linearisation, from the protocol's
own math. A min-cost flow with convex piecewise-linear arc costs is solvable exactly by network
simplex, with tick boundaries as vertices rather than kinks. This is speculative and the
unknown is how many price points a real swap crosses, which `price_points` can answer in an
afternoon. It is the only direction here that could plausibly beat the path-enumeration
architecture rather than refine it.

### 5. Not recommended: whole-network convex optimisation, in any formulation

The original research question — does optimising over every pool at once beat a path cap of four
— now has a literature-backed answer, and it is no.

The decisive point is structural rather than empirical. Whole-network routing is convex **only
because it ignores gas**. Add gas as a fixed cost per pool used and the problem becomes a minimum
concave-cost network flow, which is NP-hard; Zhavoronkov states the consequence plainly, that
"the path-separable candidate-set reduction is not a mere convenience but a necessary
approximation". The path cap that convex flow exists to avoid is not a weakness of the shipped
algorithms. It is the principled response to gas, and convex flow's inability to price gas is
the same fact seen from the other side.

If anyone does return to it, the one portable lesson from this pass is to use **L-BFGS-B**, not a
spectral projected gradient. That is what the convex flow authors use in both their experiments
and in `CFMMRouter.jl`, and it is what converges linearly on 100 assets and 2,500 markets.

## Experiments, cheapest first

Each one decides something. None takes more than a day.

1. ~~**Build the single-pair gas-aware optimum as a benchmark oracle**~~ — **done**, in
   `bench-harness/src/optimum.rs`, registered as the `OPT` config. It enumerates every subset of
   the six best-paying direct pools, allocates each by bisecting a common marginal price through
   `PoolTargetPrice`, and prices each by the same re-execution every other config goes through.
   Result: `water_fill` is at the optimum (−0.00 bps mean, 0.01 bps worst), the optimum splits
   across more than one pool on 8 of 148 orders, and every shipped algorithm beats it by reaching
   other denominations. Run it with `--configs OPT` and a raised `--timeout-ms`.
2. **Count the price points a real swap crosses.** One pass over the recorded fixture calling
   `query_pool_swap` with `price_points`. Decides whether experiment 5 is worth anything, and
   tells us how non-smooth the real curves are, which is the open question left by the convex
   flow benchmark.
3. **Raise `water_fill`'s candidate count and measure.** The literature says `k = 20`; we run
   four. If the curve of bps against `k` is still rising at 20, activation is where the headroom
   is and the transfer algorithm's value is established without building it. This is now the
   **first** experiment to run: with allocation measured at the optimum, candidate coverage is
   the only term left that fynd controls.
4. **Put gas in the enumeration weight**, `-log π + γ·g`, instead of gating per path afterwards.
   Small, local to candidate discovery, and directly targets the dominant term.
5. **Transfer algorithm as `water_fill`'s allocation step**, using `PoolTargetPrice` and the
   net marginal price. Measure simulation count at fixed candidate count, then spend the saving
   on candidates. Note what experiment 1 did to the case for this: the allocation it would
   replace is already optimal, so the only thing it can buy is the simulation budget, and that
   case has to be made on the budget rather than on bps.

## References

| Ref | Paper | Why it matters here |
|---|---|---|
| [2607.20762](https://arxiv.org/abs/2607.20762) | Quantifying Sub-Optimality in Routing for AMMs — Xi, Moallemi | The decomposition: 0.055 bps allocation, 1.3–1.8 bps per stale block, activation dominant |
| [2607.22540](https://arxiv.org/abs/2607.22540) | Multi-Path Routing in AMM Exchange Networks — Zhavoronkov | Production architecture = enumerate then allocate; gas in the edge weight; k=20; NP-hardness with gas |
| [2603.27172](https://arxiv.org/abs/2603.27172) | On the convergence rate of the one-hop transfer algorithm — Jiang, Wen | `O(Nκ log 1/ε)`, and the experiment showing `κ` barely matters |
| [2604.02909](https://arxiv.org/abs/2604.02909) | Concave Continuation: Linking Routing to Arbitrage — Jiang, Wen | Extends the transfer algorithm to negative allocations; computable via a Uniswap-V3-style price API |
| [2603.02844](https://arxiv.org/abs/2603.02844) | Optimal Routing across CFMMs with Gas Fees — Escudero, Lara, Sama | Gas as activation costs: MIP, relaxation, KKT, no-trade conditions. Theory only |
| [2502.08258](https://arxiv.org/abs/2502.08258) | Marginal Price Optimization — Loesch, Richardson | Newton in price space; Jacobian `O(N)`; 20–100× over Clarabel |
| [2404.00765](https://arxiv.org/abs/2404.00765) | Convex Network Flows — Diamandis, Angeris, Edelman | What was built and benchmarked; L-BFGS-B, not SPG |
| [2302.04938](https://arxiv.org/abs/2302.04938) | An Efficient Algorithm for Optimal Routing Through CFMMs — Diamandis et al. | The dual decomposition this line of work started from; `CFMMRouter.jl` |
