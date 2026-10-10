//! Convex flow routing: optimise over the whole pool network at once.
//!
//! Every other splitting algorithm here enumerates a handful of paths and then divides the order
//! between them. This one does not enumerate paths at all. It solves for a **price per token**,
//! lets each pool decide independently how much flow it wants at those prices, and reads the
//! route off the resulting flow. The optimum is therefore not confined to a pre-chosen path set.
//!
//! The formulation is the convex flow problem of Diamandis, Angeris and Edelman
//! (arXiv:2404.00765) in its zero-edge-utility case, which that paper identifies as CFMM
//! routing. Three properties make it fit here rather than merely apply here:
//!
//! - **Net flow per node is the route format.** [`Route::validate`](crate::types::Route::validate)
//!   groups swaps into branch collections keyed by `token_in`, pooling all flow arriving at a token
//!   and re-splitting it. That is conservation at token nodes, so a flow solution needs no
//!   decomposition into disjoint paths to be expressible — only a walk to name the legs.
//! - **The per-edge oracle is `ProtocolSim`.** The subproblem needs a gain function and its
//!   one-sided derivatives; `get_amount_out` is the first and `spot_price` on the state it returns
//!   is the second. See the `edge` module.
//! - **The dual variables survive a block.** They converge to no-arbitrage prices, which drift
//!   rather than jump, so the previous block's answer is this block's starting point. See
//!   `warm_start`.
//!
//! # Cost
//!
//! The expensive part is the per-edge root finding, and most edges never reach it: the no-flow test
//! in `Edge::carries_no_flow` rejects an edge from its fee-inclusive spot price alone,
//! which the graph already carries. What remains is bounded by
//! `config::MAX_EDGE_SOLVES` and by the worker's deadline, and the solver returns its best
//! point in every case rather than failing.
//!
//! # What this does not do
//!
//! Gas does not enter the objective — it is a fixed cost per edge used, which the convex flow
//! framework handles only in its follow-up (arXiv:2408.12761) and which is not implemented here.
//! Ranking against other worker pools still happens on `amount_out_net_gas`, so a route this
//! algorithm wins with is gas-aware at the router even though the search was not. The leg cap in
//! `config::MAX_LEGS` is the crude stand-in for that missing term.

mod config;
mod dual;
mod edge;

use std::time::{Duration, Instant};

use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::{debug, instrument};
use tycho_simulation::tycho_common::models::token::Token;

use self::{
    config::{EDGE_VALUE_SLACK, MAX_LEGS, MIN_LEG_FRACTION},
    dual::DualProblem,
    edge::{Edge, EdgeFlow},
};
use super::{
    most_liquid::DepthAndPrice,
    paths,
    paths::read_market,
    sim_meter,
    split_primitives::{
        build_split_route, normalize_fractions, HopDescriptor, PathAllocation, SimulatedHop,
    },
    Address, Algorithm, AlgorithmConfig, AlgorithmError, NoPathReason, SolveRequest,
};
use crate::{
    derived::{computation::ComputationRequirements, types::TokenGasPrices},
    feed::market_data::MarketDataView,
    graph::{EdgeData, GraphQueryFilter, RouteSearch, TopologyGraph, TopologyGraphManager},
    types::{quote::RouteExclusions, ComponentId, Order, RouteResult},
};

/// Routes by solving the convex flow problem over the whole reachable pool set.
pub struct ConvexFlowAlgorithm {
    /// The hop bounds and connector tokens every route search runs under.
    query: GraphQueryFilter,
    timeout: Duration,
}

impl ConvexFlowAlgorithm {
    /// Creates a `ConvexFlowAlgorithm` from an `AlgorithmConfig`.
    pub(crate) fn with_config(config: AlgorithmConfig) -> Self {
        Self {
            query: GraphQueryFilter {
                min_hops: config.min_hops(),
                max_hops: config.max_hops(),
                connector_tokens: config.connector_tokens().cloned(),
            },
            timeout: config.timeout(),
        }
    }
}

impl Default for ConvexFlowAlgorithm {
    fn default() -> Self {
        Self::with_config(AlgorithmConfig::default())
    }
}

/// How long a solve has left.
///
/// Defined here rather than shared with the other algorithms so this module depends on nothing
/// of theirs: it is four lines, and a borrowed one would mean widening another algorithm's
/// visibility to suit this one.
#[derive(Clone, Copy)]
pub(crate) struct Deadline {
    start: Instant,
    timeout: Duration,
}

impl Deadline {
    /// A budget of `timeout` from `start`.
    pub(crate) fn new(start: Instant, timeout: Duration) -> Self {
        Self { start, timeout }
    }

    /// Whether the solve has run past its budget.
    pub(crate) fn expired(&self) -> bool {
        self.start.elapsed() > self.timeout
    }

    /// How long the solve has been running, for the lines that report it.
    pub(crate) fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// A gas amount in output-token terms, or `None` when the output token has no price.
///
/// The same conversion the other algorithms apply, so the net figures this one reports are
/// comparable with theirs at cross-pool ranking. Duplicated rather than shared for the reason
/// above; it is four lines of arithmetic and the convention it encodes is fixed by the wire
/// format, not by any one algorithm.
fn gas_cost_in_token(
    total_gas: &BigUint,
    gas_price_wei: &BigUint,
    token_prices: Option<&TokenGasPrices>,
    token_out: &Address,
) -> Option<BigUint> {
    let price = token_prices?.get(token_out)?;
    if price.denominator.is_zero() {
        return None;
    }
    Some(total_gas * gas_price_wei * &price.numerator / &price.denominator)
}

/// The flow network one solve runs over: its edges, and the token behind each node index.
struct Network {
    edges: Vec<Edge>,
    tokens: Vec<Token>,
    node_in: usize,
    node_out: usize,
}

impl Network {
    /// Builds the network from every pool on every path the graph admits between the order's
    /// tokens.
    ///
    /// The hop bounds, the connector-token allowlist and the request's exclusions are all applied
    /// by the graph's own search, so this reuses that rather than re-deriving reachability. Pools
    /// appearing on several paths are unioned: the flow problem is over edges, and an edge is one
    /// pool traded one way however many paths happen to cross it.
    fn build(
        graph: &TopologyGraph<DepthAndPrice>,
        market: &MarketDataView<'_>,
        order: &Order,
        search: RouteSearch<'_>,
        exclusions: &RouteExclusions,
    ) -> Result<Self, AlgorithmError> {
        let token_paths = graph
            .paths_between(order.token_in(), order.token_out(), search)
            .map_err(|_| {
                AlgorithmError::no_path(
                    order.token_in().clone(),
                    order.token_out().clone(),
                    NoPathReason::SourceTokenNotInGraph,
                )
            })?;

        let mut node_of: FxHashMap<Address, usize> = FxHashMap::default();
        let mut tokens: Vec<Token> = Vec::new();
        let mut seen: FxHashSet<(ComponentId, Address, Address)> = FxHashSet::default();
        let mut edges: Vec<Edge> = Vec::new();

        for token_path in &token_paths {
            for path in graph.expand_path(token_path, None, exclusions) {
                for (from, edge_data, to) in path.iter() {
                    let key = (edge_data.component_id.clone(), from.clone(), to.clone());
                    if !seen.insert(key) {
                        continue;
                    }
                    let (Some(token_in), Some(token_out)) =
                        (market.get_token(from), market.get_token(to))
                    else {
                        continue;
                    };
                    let node_in = node_index(&mut node_of, &mut tokens, token_in.clone());
                    let node_out = node_index(&mut node_of, &mut tokens, token_out.clone());

                    // No edge weight means the derived pass has not priced this pair yet, and
                    // the no-flow test has nothing to read. Skipping is the safe reading: the
                    // edge rejoins the network on the next block that prices it.
                    let Some((marginal_at_zero, max_in)) =
                        edge_bounds(market, edge_data, token_in, token_out)
                    else {
                        continue;
                    };
                    edges.push(Edge {
                        component_id: edge_data.component_id.clone(),
                        token_in: token_in.clone(),
                        token_out: token_out.clone(),
                        node_in,
                        node_out,
                        marginal_at_zero,
                        max_in,
                    });
                }
            }
        }

        let (Some(&node_in), Some(&node_out)) =
            (node_of.get(order.token_in()), node_of.get(order.token_out()))
        else {
            return Err(AlgorithmError::no_path(
                order.token_in().clone(),
                order.token_out().clone(),
                NoPathReason::NoGraphPath,
            ));
        };
        if edges.is_empty() {
            return Err(AlgorithmError::no_path(
                order.token_in().clone(),
                order.token_out().clone(),
                NoPathReason::NoGraphPath,
            ));
        }
        Ok(Self { edges, tokens, node_in, node_out })
    }
}

/// Index of `token` in the price vector, assigning one if it is new.
fn node_index(
    node_of: &mut FxHashMap<Address, usize>,
    tokens: &mut Vec<Token>,
    token: Token,
) -> usize {
    let address = token.address.clone();
    if let Some(&index) = node_of.get(&address) {
        return index;
    }
    let index = tokens.len();
    tokens.push(token);
    node_of.insert(address, index);
    index
}

/// Converts a decimal-adjusted marginal price into raw output units per raw input unit.
///
/// `ProtocolSim::spot_price` is a rate between *human* units — tycho's implementations pass each
/// token's decimals into the reserve ratio — while every amount in this module, and every derived
/// token price, is in raw units. The two differ by `10^(dec_out - dec_in)`, which for WETH into
/// USDC is a factor of a trillion.
fn to_raw_marginal(human: f64, decimals_in: u32, decimals_out: u32) -> f64 {
    human * 10f64.powi(decimals_out as i32 - decimals_in as i32)
}

/// The edge's marginal price at zero trade, in raw output units per raw input unit, and the
/// largest input the pool accepts. `None` when either is unavailable and the edge cannot take
/// part.
///
/// The graph caches a fee-inclusive spot price on each edge, but only once a derived-data pass
/// has run — `EdgeData::data` is documented as "None if not yet computed", and it is empty on a
/// worker's first solves and for any pair the derived pass has not reached. Treating the cache as
/// required would idle the whole network in exactly those cases, so a miss falls back to reading
/// the price off the pool. That costs no `get_amount_out`: `spot_price` on the current state is
/// the marginal price at zero trade by definition.
fn edge_bounds(
    market: &MarketDataView<'_>,
    edge_data: &EdgeData<DepthAndPrice>,
    token_in: &Token,
    token_out: &Token,
) -> Option<(f64, BigUint)> {
    let sim = market.get_simulation_state(&edge_data.component_id)?;
    let human = match edge_data.data.as_ref() {
        Some(weight) => weight.spot_price,
        None => sim
            .spot_price(token_in, token_out)
            .ok()?,
    };
    // `spot_price` is decimal-adjusted — a rate between human units — while every amount here
    // and every derived token price is in raw units. Comparing the two directly is wrong by
    // `10^(dec_out - dec_in)`, which for WETH into USDC is a factor of a trillion, so this
    // conversion is not a refinement but the thing that makes the no-flow test mean anything.
    let marginal_at_zero = to_raw_marginal(human, token_in.decimals, token_out.decimals);
    if !(marginal_at_zero.is_finite() && marginal_at_zero > 0.0) {
        return None;
    }
    let (max_in, _) = sim
        .get_limits(token_in.address.clone(), token_out.address.clone())
        .ok()?;
    (!max_in.is_zero()).then_some((marginal_at_zero, max_in))
}

/// Reference prices, one per token, denominated in the output token.
///
/// These set the unit each token's price is measured in, so the coordinator starts at all-ones
/// and its gradient comes out dimensionless. A good reference is therefore not just a faster
/// start but the thing that makes the problem well conditioned at all.
///
/// Derived data already holds a per-token price against the gas token, recomputed every block.
/// Those are the no-arbitrage prices the duals converge to, so rebasing them onto the output
/// token puts the coordinator near its answer before the first iteration. A token the table does
/// not price starts at the output token's own price, which the no-flow test then prices out of
/// the active set on the first evaluation if it does not belong there.
fn warm_start(network: &Network, prices: Option<&TokenGasPrices>) -> Vec<f64> {
    let mut reference = vec![f64::NAN; network.tokens.len()];
    reference[network.node_out] = 1.0;

    // Derived prices first, where the table has them.
    if let Some(prices) = prices {
        if let Some(out_price) = prices
            .get(&network.tokens[network.node_out].address)
            .and_then(price_to_f64)
        {
            for (index, token) in network.tokens.iter().enumerate() {
                if index == network.node_out {
                    continue;
                }
                if let Some(price) = prices
                    .get(&token.address)
                    .and_then(price_to_f64)
                {
                    let rebased = price / out_price;
                    if rebased.is_finite() && rebased > 0.0 {
                        reference[index] = rebased;
                    }
                }
            }
        }
    }

    propagate_reference(network, &mut reference);
    for price in &mut reference {
        if !(price.is_finite() && *price > 0.0) {
            *price = 1.0;
        }
    }
    reference
}

/// Fills in tokens the price table does not cover, by walking the network's own spot prices.
///
/// A missing reference cannot simply default to one. The reference sets the token's unit, and a
/// token quoted in 18 decimals against an output token quoted in 6 is then wrong by twelve
/// orders of magnitude — which puts the edge value cap, and with it every net flow through that
/// token, as far out. An edge's marginal price relates its two tokens directly, so one priced
/// neighbour is enough: for an edge `a → b` paying `m` units of `b` per unit of `a`, one unit of
/// `a` is worth `m` units of `b`. Only the order of magnitude matters here, so a few sweeps of
/// this is enough and no averaging is needed.
fn propagate_reference(network: &Network, reference: &mut [f64]) {
    // One sweep can only carry a price one hop, and an edge may be visited before its source is
    // priced, so repeat until nothing new is learned.
    for _ in 0..network.tokens.len().min(8) {
        let mut learned = false;
        for edge in &network.edges {
            let marginal = edge.marginal_at_zero;
            if !(marginal.is_finite() && marginal > 0.0) {
                continue;
            }
            let (known_in, known_out) =
                (reference[edge.node_in].is_finite(), reference[edge.node_out].is_finite());
            if known_out && !known_in {
                let price = marginal * reference[edge.node_out];
                if price.is_finite() && price > 0.0 {
                    reference[edge.node_in] = price;
                    learned = true;
                }
            } else if known_in && !known_out {
                let price = reference[edge.node_in] / marginal;
                if price.is_finite() && price > 0.0 {
                    reference[edge.node_out] = price;
                    learned = true;
                }
            }
        }
        if !learned {
            break;
        }
    }
}

/// A derived token price as a float, or `None` when the fraction cannot be represented.
fn price_to_f64(
    price: &tycho_simulation::tycho_core::simulation::protocol_sim::Price,
) -> Option<f64> {
    let numerator = price.numerator.to_f64()?;
    let denominator = price.denominator.to_f64()?;
    (denominator != 0.0)
        .then(|| numerator / denominator)
        .filter(|p| p.is_finite() && *p > 0.0)
}

/// Caps each edge's input at the amount of its input token the order is worth.
///
/// Without this an edge's bracket is whatever `get_limits` reports, which on a deep pool is far
/// beyond anything this order could push through it. A dual iterate is free to be infeasible, so
/// the root finding will happily propose that trade, and the resulting net flows are large enough
/// to make the line search meaningless. Value is conserved along a route up to fees, so the
/// order's own value bounds every edge on it.
fn cap_edges_to_order_value(network: &mut Network, reference: &[f64], order: &Order) {
    let Some(amount) = order
        .amount()
        .to_f64()
        .filter(|a| a.is_finite())
    else {
        return;
    };
    let order_value = amount * reference[network.node_in];
    if !(order_value.is_finite() && order_value > 0.0) {
        return;
    }
    for edge in &mut network.edges {
        let token_price = reference[edge.node_in];
        if !(token_price.is_finite() && token_price > 0.0) {
            continue;
        }
        // A cap that rounds below one raw unit means this order is not worth a single unit of
        // the edge's input token at the reference prices. Leaving the edge uncapped there is the
        // worst available answer: an iterate may then propose the pool's entire `get_limits`,
        // and the net flow that reports is scaled by the very reference price that made the cap
        // sub-unit, so the dual's residual lands tens of orders of magnitude out. One raw unit
        // is the smallest honest bracket. An infinite cap is a different thing and keeps meaning
        // no constraint, which `f64 as u128` saturates into.
        let cap = order_value * EDGE_VALUE_SLACK / token_price;
        let cap = if cap >= 1.0 { cap } else { 1.0 };
        let cap = BigUint::from(cap as u128);
        if cap < edge.max_in {
            edge.max_in = cap;
        }
    }
}

/// Walks the solved flow into legs the route format can express.
///
/// The flow is already conserved at every token, which is what `Route::validate` checks, so this
/// is a decomposition rather than a reconstruction: repeatedly walk from the input token to the
/// output token along edges that still carry flow, take the bottleneck, and subtract it. A walk
/// that revisits a token is a circulation — arbitrage the route must not depend on — and is
/// abandoned rather than emitted.
fn decompose(network: &Network, flows: &[EdgeFlow], order: &Order) -> Vec<PathAllocation> {
    let mut remaining: Vec<BigUint> = flows
        .iter()
        .map(|flow| flow.amount_in.clone())
        .collect();
    let total = order
        .amount()
        .to_f64()
        .filter(|t| *t > 0.0)
        .unwrap_or(1.0);

    let mut out_edges: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (index, edge) in network.edges.iter().enumerate() {
        out_edges
            .entry(edge.node_in)
            .or_default()
            .push(index);
    }

    let order_amount = order.amount().clone();
    let mut allocated = BigUint::zero();
    let mut allocations = Vec::new();
    while allocations.len() < MAX_LEGS && allocated < order_amount {
        let Some(walk) = walk_once(network, &remaining, &out_edges) else {
            break;
        };
        // The bottleneck is what every edge on the walk can still carry.
        let bottleneck = walk
            .iter()
            .map(|&index| remaining[index].clone())
            .min()
            .unwrap_or_else(BigUint::zero);
        if bottleneck.is_zero() {
            break;
        }

        // A dual iterate need not be primal feasible, so the solved flows need not divide the
        // order exactly — the residual says how far off they are. The legs must divide it
        // exactly all the same, or `build_split_route` is handed fractions summing past one and
        // refuses them. Clamping each walk to what is left, and normalising at the end, is the
        // same conservation-by-construction the convex flow paper's own routing experiment uses
        // in place of a reconstruction step.
        let bottleneck = bottleneck.min(&order_amount - &allocated);
        if bottleneck.is_zero() {
            break;
        }
        let fraction = bottleneck.to_f64().unwrap_or(0.0) / total;
        for &index in &walk {
            remaining[index] -= &bottleneck;
        }
        allocated += &bottleneck;
        if fraction < MIN_LEG_FRACTION {
            // Too small to pay its own gas, and the split convention cannot express it. The flow
            // is already subtracted, so the walk is not retried.
            continue;
        }

        let hops: Vec<SimulatedHop> = walk
            .iter()
            .map(|&index| {
                let edge = &network.edges[index];
                SimulatedHop {
                    descriptor: HopDescriptor::new(
                        edge.component_id.clone(),
                        edge.token_in.clone(),
                        edge.token_out.clone(),
                    ),
                    amount_out: flows[index].amount_out.clone(),
                    gas: flows[index].gas.clone(),
                }
            })
            .collect();
        allocations.push(PathAllocation {
            hops,
            flow_fraction: fraction,
            amount_in: bottleneck,
            amount_out: BigUint::zero(),
            marginal_price_product: 0.0,
        });
    }
    // Rounding and the clamp above both leave the fractions a little short of one; the split
    // convention needs them to account for the whole order.
    if !allocations.is_empty() {
        let mut fractions: Vec<f64> = allocations
            .iter()
            .map(|a| a.flow_fraction)
            .collect();
        if normalize_fractions(&mut fractions).is_ok() {
            for (allocation, fraction) in allocations.iter_mut().zip(fractions) {
                allocation.flow_fraction = fraction;
            }
        }
    }
    allocations
}

/// One greedy walk from the input token to the output token over edges that still carry flow.
///
/// Returns the edge indices in order, or `None` when no such walk remains. Tokens already on the
/// walk are not revisited, so the result is a simple path.
fn walk_once(
    network: &Network,
    remaining: &[BigUint],
    out_edges: &FxHashMap<usize, Vec<usize>>,
) -> Option<Vec<usize>> {
    let mut walk = Vec::new();
    let mut visited: FxHashSet<usize> = FxHashSet::default();
    visited.insert(network.node_in);
    let mut node = network.node_in;

    while node != network.node_out {
        // Prefer the edge carrying the most, so the first walks take the trunk of the flow and
        // the leg cap spends itself on the branches that matter.
        let next = out_edges
            .get(&node)?
            .iter()
            .copied()
            .filter(|&index| !remaining[index].is_zero())
            .filter(|&index| !visited.contains(&network.edges[index].node_out))
            .max_by(|&a, &b| remaining[a].cmp(&remaining[b]))?;
        walk.push(next);
        node = network.edges[next].node_out;
        visited.insert(node);
        if walk.len() > network.tokens.len() {
            return None;
        }
    }
    (!walk.is_empty()).then_some(walk)
}

impl Algorithm for ConvexFlowAlgorithm {
    type GraphType = TopologyGraph<DepthAndPrice>;
    type GraphManager = TopologyGraphManager<DepthAndPrice>;

    fn name(&self) -> &str {
        "convex_flow"
    }

    #[instrument(level = "debug", skip_all, fields(order_id = %request.order().id()))]
    async fn find_best_route(
        &self,
        request: SolveRequest<'_, Self::GraphType>,
    ) -> Result<RouteResult, AlgorithmError> {
        let deadline = Deadline::new(Instant::now(), self.timeout);
        let parts = request.into_parts();
        let order = parts.order;
        if !order.is_sell() {
            return Err(AlgorithmError::ExactOutNotSupported);
        }

        // The hop bounds are this algorithm's, the exclusions are the request's.
        let search = RouteSearch { bounds: &self.query, exclusions: &parts.exclusions };
        let token_prices = match parts.derived.as_ref() {
            Some(derived) => derived
                .read()
                .await
                .token_prices()
                .cloned(),
            None => None,
        };

        let view = read_market(&parts.market, parts.label).await?;
        let mut network = Network::build(parts.graph, &view, order, search, &parts.exclusions)?;
        // Take an owned subset of just the pools this solve touches and release the market lock
        // before the coordinator starts: every later step is simulation-bound, and holding a read
        // lock across it would stall the feed's writer for the whole solve.
        let component_ids: FxHashSet<&ComponentId> = network
            .edges
            .iter()
            .map(|edge| &edge.component_id)
            .collect();
        let market = view.extract_subset_with_overlay(&component_ids);
        drop(view);

        // Bracketing is not optional: `edge` meters its swaps, and counts left unreported
        // accumulate on the worker thread for the life of the process.
        sim_meter::start_solve();

        // The warm start defines each token's unit rather than seeding an iterate: the solver
        // works in ratios to it, so its own starting point is all-ones by construction.
        let reference = warm_start(&network, token_prices.as_ref());
        cap_edges_to_order_value(&mut network, &reference, order);
        let problem = DualProblem::new(
            &network.edges,
            network.tokens.len(),
            network.node_in,
            network.node_out,
            order.amount(),
            reference,
        )?;
        let solution = problem.minimise(problem.start(), &market, &deadline)?;
        debug!(
            iterations = solution.iterations,
            edge_solves = solution.edge_solves,
            residual = solution.residual,
            edges = network.edges.len(),
            tokens = network.tokens.len(),
            "convex flow solved",
        );

        sim_meter::report("convex_flow", &market, || deadline.elapsed().as_millis() as u64);

        let allocations = decompose(&network, &solution.flows, order);
        if allocations.is_empty() {
            return Err(AlgorithmError::InsufficientLiquidity);
        }

        // A split is only worth returning if it beats carrying the order on one path. The flow
        // can spread across more routes than `MAX_LEGS` allows, and the legs that survive the cut
        // are then scaled up to cover the whole order — which pushes more through each pool than
        // the solver priced, and can pay less than simply taking the best route. Water-fill keeps
        // the same floor for the same reason.
        let single = allocations
            .first()
            .map(|best| {
                let mut only = best.clone();
                only.flow_fraction = 1.0;
                vec![only]
            })
            .unwrap_or_default();

        // `build_split_route` re-executes the plan against shared state and applies the
        // remainder-split convention, so the route it returns is the authoritative one — the
        // solver's own per-edge amounts were priced independently and do not account for two
        // legs meeting in the same pool.
        // Price both and keep the better. `build_split_route` re-executes each plan against
        // shared state, so these are comparable on the amounts that would really be paid.
        let split = build_split_route(&allocations, &market, order).and_then(|route| {
            route
                .validate()
                .map(|()| route)
                .map_err(Into::into)
        });
        let single = (allocations.len() > 1)
            .then(|| {
                build_split_route(&single, &market, order).and_then(|route| {
                    route
                        .validate()
                        .map(|()| route)
                        .map_err(Into::into)
                })
            })
            .transpose()
            .ok()
            .flatten();

        let route = match (split, single) {
            (Ok(split), Some(single)) => {
                if single.amount_out(order.token_out()) > split.amount_out(order.token_out()) {
                    single
                } else {
                    split
                }
            }
            (Ok(split), None) => split,
            (Err(_), Some(single)) => single,
            (Err(error), None) => return Err(error),
        };

        // A route that pays nothing is not a route. The coordinator returns the best point it
        // reached and that point need not be primal feasible, so an order too small to move any
        // pool can still walk into legs whose every output truncates to zero. Reporting no route
        // is both the honest answer and the useful one: the router ranks on output, and a
        // zero-output candidate can only displace a real one in the logs.
        if route
            .amount_out(order.token_out())
            .is_zero()
        {
            return Err(AlgorithmError::InsufficientLiquidity);
        }

        // Gas does not steer the search, but the result must still be comparable with the other
        // worker pools, which rank on output net of gas. Converted through the same helper the
        // other split algorithm uses, so the two are measured the same way.
        let gas_price = paths::fetch_gas_price(&market)?;
        let gross = route.amount_out(order.token_out());
        let gas_cost = gas_cost_in_token(
            &route.total_gas(),
            &gas_price,
            token_prices.as_ref(),
            order.token_out(),
        );
        let net = BigInt::from(gross) -
            gas_cost
                .map(BigInt::from)
                .unwrap_or_else(BigInt::zero);
        Ok(RouteResult::new(route, net, gas_price))
    }

    fn computation_requirements(&self) -> ComputationRequirements {
        // Spot prices feed the no-flow test and are what makes the solve affordable, so they must
        // be current. Token prices only seed the warm start: a stale vector costs iterations,
        // never correctness, and waiting for one would be worse than starting from a flat guess.
        ComputationRequirements::none()
            .allow_stale("token_prices")
            .expect("Conflicting Computation Requirements")
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

#[cfg(test)]
mod tests {
    use tycho_simulation::tycho_common::Bytes;

    use super::*;

    fn token(byte: u8, symbol: &str) -> Token {
        Token::new(
            &Bytes::from(vec![byte; 20]),
            symbol,
            18,
            0,
            &[Some(10_000)],
            Default::default(),
            100,
        )
    }

    fn edge(node_in: usize, node_out: usize, id: &str) -> Edge {
        Edge {
            component_id: id.to_string(),
            token_in: token(node_in as u8 + 1, "IN"),
            token_out: token(node_out as u8 + 1, "OUT"),
            node_in,
            node_out,
            marginal_at_zero: 1.0,
            max_in: BigUint::from(1_000_000u32),
        }
    }

    fn network(edges: Vec<Edge>, node_count: usize) -> Network {
        Network {
            edges,
            tokens: (0..node_count)
                .map(|i| token(i as u8 + 1, "T"))
                .collect(),
            node_in: 0,
            node_out: node_count - 1,
        }
    }

    fn flow(amount_in: u32, amount_out: u32) -> EdgeFlow {
        EdgeFlow {
            amount_in: BigUint::from(amount_in),
            amount_out: BigUint::from(amount_out),
            gas: BigUint::from(100_000u32),
        }
    }

    fn sell_order(amount: u32) -> Order {
        Order::new(
            Bytes::from(vec![1u8; 20]),
            Bytes::from(vec![3u8; 20]),
            BigUint::from(amount),
            crate::types::OrderSide::Sell,
            Bytes::from(vec![9u8; 20]),
        )
    }

    /// A derived token price as the exact fraction the price table stores.
    fn price(value: u32) -> tycho_simulation::tycho_core::simulation::protocol_sim::Price {
        tycho_simulation::tycho_core::simulation::protocol_sim::Price {
            numerator: BigUint::from(value),
            denominator: BigUint::from(1u32),
        }
    }

    // ---- End-to-end solves on hand-computable markets -------------------------------------
    //
    // Every other test here exercises one piece in isolation. These run the whole algorithm on a
    // market small enough to work out by hand, so a failure says the solver is wrong rather than
    // that some component is.

    use crate::{
        algorithm::test_utils::{
            setup_market_weighted_boxed, token as mk_token, ConstantProductSim,
        },
        graph::GraphManager,
    };

    /// A zero-fee constant-product pool, reserves given in raw units.
    fn pool(
        reserve_0: u128,
        reserve_1: u128,
    ) -> Box<dyn tycho_simulation::tycho_common::simulation::protocol_sim::ProtocolSim> {
        Box::new(ConstantProductSim {
            reserve_0: BigUint::from(reserve_0),
            reserve_1: BigUint::from(reserve_1),
            gas: 100_000,
        })
    }

    /// A fee-bearing constant-product pool, reserves in raw units.
    ///
    /// Every other pool in this module is fee-free, which is the one thing a real market never
    /// is. The subproblem's optimality condition compares a quoted marginal price against the
    /// curve `get_amount_out` walks, so a fee only one of the two accounts for solves for the
    /// wrong root — by the ratio of the fee to the order's own price impact, which for a small
    /// order is a factor of hundreds. A zero-fee double cannot see that at all.
    fn fee_pool(
        reserve_0: u128,
        reserve_1: u128,
        fee_bps: u32,
    ) -> Box<dyn tycho_simulation::tycho_common::simulation::protocol_sim::ProtocolSim> {
        Box::new(crate::algorithm::test_utils::FeeConstantProductSim {
            reserve_0: BigUint::from(reserve_0),
            reserve_1: BigUint::from(reserve_1),
            gas: 100_000,
            fee_bps,
        })
    }

    /// What a fee-bearing pool pays for `amount_in`, by the formula `fee_pool` uses.
    fn fee_cp_out(amount_in: u128, reserve_in: u128, reserve_out: u128, fee_bps: u32) -> BigUint {
        let net_in =
            BigUint::from(amount_in) * BigUint::from(10_000 - fee_bps) / BigUint::from(10_000u32);
        &net_in * BigUint::from(reserve_out) / (BigUint::from(reserve_in) + &net_in)
    }

    /// What a pool pays for `amount_in`, by the same constant-product formula the mock uses:
    /// `out = in * reserve_out / (reserve_in + in)`.
    ///
    /// In `BigUint` rather than `u128`: at wei magnitudes the numerator alone exceeds `u128`
    /// (1e18 × 2e21 is 2e39 against a ceiling of 3.4e38), and a test helper that overflows is
    /// indistinguishable from the algorithm failing.
    fn cp_out(amount_in: u128, reserve_in: u128, reserve_out: u128) -> BigUint {
        let amount_in = BigUint::from(amount_in);
        &amount_in * BigUint::from(reserve_out) / (BigUint::from(reserve_in) + &amount_in)
    }

    fn sell(token_in: &Token, token_out: &Token, amount: u128) -> Order {
        crate::algorithm::test_utils::order(
            token_in,
            token_out,
            amount,
            crate::types::OrderSide::Sell,
        )
    }

    /// One pool, one direction, no alternative. The optimum is forced: send the whole order
    /// through, because output is increasing in input and nothing else can carry any of it.
    ///
    /// This is the test that says whether the coordinator works at all. Everything else the
    /// algorithm does — splitting, multi-hop, ranking — is refinement on top of being able to
    /// drive a single edge to the order amount.
    #[tokio::test]
    async fn test_solve_single_pool_sends_the_whole_order() {
        let token_a = mk_token(0x0A, "A");
        let token_c = mk_token(0x0C, "C");
        // a/c reserves, ordered by address: A < C, so reserve_0 is A.
        let (market, manager) =
            setup_market_weighted_boxed(vec![("p", &token_a, &token_c, pool(1_000, 2_000))]);

        let amount = 100u128;
        let order = sell(&token_a, &token_c, amount);
        let algorithm = ConvexFlowAlgorithm::default();

        let result = algorithm
            .find_best_route(SolveRequest::new(manager.graph(), market, &order))
            .await
            .expect("the only route must be found");

        let expected = cp_out(amount, 1_000, 2_000);
        assert_eq!(
            result
                .route()
                .amount_out(&token_c.address),
            expected,
            "the whole order must go through the only pool"
        );
    }

    /// Two pools serving the same pair, one twice as deep. Constant-product output is strictly
    /// concave, so splitting beats sending everything through the better pool — by how much
    /// depends on the solver, but it must not be worse.
    #[tokio::test]
    async fn test_solve_two_parallel_pools_beats_the_best_single() {
        let token_a = mk_token(0x0A, "A");
        let token_c = mk_token(0x0C, "C");
        let (market, manager) = setup_market_weighted_boxed(vec![
            ("deep", &token_a, &token_c, pool(10_000, 20_000)),
            ("shallow", &token_a, &token_c, pool(2_000, 4_000)),
        ]);

        let amount = 1_000u128;
        let order = sell(&token_a, &token_c, amount);
        let algorithm = ConvexFlowAlgorithm::default();

        let result = algorithm
            .find_best_route(SolveRequest::new(manager.graph(), market, &order))
            .await
            .expect("both pools serve the pair");

        let best_single = cp_out(amount, 10_000, 20_000);
        assert!(
            result
                .route()
                .amount_out(&token_c.address) >=
                best_single,
            "a split must not pay less than the best single pool"
        );
    }

    /// The only route is two hops. Tests that flow is conserved at the intermediate token: what
    /// the first pool pays is exactly what the second is given.
    #[tokio::test]
    async fn test_solve_two_hop_route() {
        let token_a = mk_token(0x0A, "A");
        let token_b = mk_token(0x0B, "B");
        let token_c = mk_token(0x0C, "C");
        let (market, manager) = setup_market_weighted_boxed(vec![
            ("ab", &token_a, &token_b, pool(1_000, 1_000)),
            ("bc", &token_b, &token_c, pool(1_000, 1_000)),
        ]);

        let amount = 100u128;
        let order = sell(&token_a, &token_c, amount);
        let algorithm = ConvexFlowAlgorithm::default();

        let result = algorithm
            .find_best_route(SolveRequest::new(manager.graph(), market, &order))
            .await
            .expect("the two-hop route must be found");

        let mid = cp_out(amount, 1_000, 1_000);
        let mid: u128 = mid.try_into().expect("toy amount fits");
        let expected = cp_out(mid, 1_000, 1_000);
        assert_eq!(
            result
                .route()
                .amount_out(&token_c.address),
            expected,
            "the whole order must traverse both hops"
        );
    }

    // ---- Conditioning: the same solve, scaled up one dimension at a time ------------------
    //
    // The three solves above use toy magnitudes. Real pools hold 1e18-scale reserves, quote in
    // different decimals, and come in dozens per pair. These vary one of those at a time against
    // the single-pool case, whose answer stays forced however large the numbers get, so whichever
    // one fails names the thing that breaks on a real market.

    /// Single pool at realistic magnitudes: 1e18-scale reserves, 18-decimal tokens.
    #[tokio::test]
    async fn test_solve_single_pool_at_wei_magnitudes() {
        let token_a = mk_token(0x0A, "A");
        let token_c = mk_token(0x0C, "C");
        let (reserve_a, reserve_c) = (1_000e18 as u128, 2_000e18 as u128);
        let (market, manager) = setup_market_weighted_boxed(vec![(
            "p",
            &token_a,
            &token_c,
            pool(reserve_a, reserve_c),
        )]);

        let amount = 1e18 as u128;
        let order = sell(&token_a, &token_c, amount);
        let result = ConvexFlowAlgorithm::default()
            .find_best_route(SolveRequest::new(manager.graph(), market, &order))
            .await
            .expect("the only route must be found at wei magnitudes");

        let expected = cp_out(amount, reserve_a, reserve_c);
        let got = result
            .route()
            .amount_out(&token_c.address);
        // The subproblem resolves the input to a fraction of the pool's limit rather than
        // exactly, so compare within a basis point rather than to the wei.
        let tolerance = &expected / 10_000u32;
        let diff = if got > expected { &got - &expected } else { &expected - &got };
        assert!(diff <= tolerance, "expected ~{expected}, got {got}");
    }

    /// Solves `A -> C` over a market with `middles` distinct intermediate tokens, each offering
    /// its own two-hop route, alongside one direct pool. Returns the output, or the error.
    ///
    /// Token count scales with `middles` while the hop count stays at two, which is the shape
    /// the real market presents: dozens of tokens reachable inside the hop limit, not long
    /// chains. The direct pool means a correct solver can never do worse than a known amount,
    /// however many middles it is offered.
    async fn solve_with_middles(
        middles: usize,
        amount: u128,
        distinct_prices: bool,
    ) -> Result<BigUint, AlgorithmError> {
        let token_a = mk_token(0x01, "A");
        let token_c = mk_token(0x02, "C");
        let mids: Vec<Token> = (0..middles)
            .map(|i| mk_token(0x10 + i as u8, "M"))
            .collect();

        let direct = 10_000e18 as u128;
        let mut ids: Vec<String> = vec!["direct".to_string()];
        for i in 0..middles {
            ids.push(format!("a_m{i}"));
            ids.push(format!("m{i}_c"));
        }

        let mut specs: Vec<(
            &str,
            &Token,
            &Token,
            Box<dyn tycho_simulation::tycho_common::simulation::protocol_sim::ProtocolSim>,
        )> = vec![(ids[0].as_str(), &token_a, &token_c, pool(direct, direct))];
        for (i, mid) in mids.iter().enumerate() {
            // Each leg is deep enough that the two-hop route is competitive with the direct one,
            // so the optimum genuinely wants to spread across them. `distinct_prices` perturbs
            // each middle's reserves so no two routes are priced exactly alike.
            let skew = if distinct_prices { 1 + i as u128 } else { 0 };
            specs.push((
                ids[1 + 2 * i].as_str(),
                &token_a,
                mid,
                pool(direct, direct + skew * (direct / 1_000)),
            ));
            specs.push((
                ids[2 + 2 * i].as_str(),
                mid,
                &token_c,
                pool(direct, direct - skew * (direct / 2_000)),
            ));
        }
        let (market, manager) = setup_market_weighted_boxed(specs);

        let order = sell(&token_a, &token_c, amount);
        ConvexFlowAlgorithm::default()
            .find_best_route(SolveRequest::new(manager.graph(), market, &order))
            .await
            .map(|result| {
                result
                    .route()
                    .amount_out(&token_c.address)
            })
    }

    /// Walks the token count up until the solve stops working, and names where.
    ///
    /// A correct solver must clear the direct pool's output at every size: the direct pool is in
    /// every one of these markets, so refusing to route, or routing for less, means the extra
    /// choices made the answer worse rather than better.
    #[tokio::test]
    async fn test_solve_scales_with_token_count() {
        let amount = 1e18 as u128;
        let floor = cp_out(amount, 10_000e18 as u128, 10_000e18 as u128);

        let mut failures: Vec<String> = Vec::new();
        for middles in [1usize, 2, 4, 8, 16] {
            match solve_with_middles(middles, amount, false).await {
                Ok(out) if out >= floor => {}
                Ok(out) => failures.push(format!(
                    "{middles} middles ({} tokens): paid {out}, below the direct pool's {floor}",
                    middles + 2
                )),
                Err(error) => {
                    failures.push(format!("{middles} middles ({} tokens): {error}", middles + 2))
                }
            }
        }
        assert!(
            failures.is_empty(),
            "convex flow broke as the token count grew:\n  {}",
            failures.join("\n  ")
        );
    }

    /// The same sweep with every route priced differently, so no two allocations are equally
    /// good and each edge subproblem has a unique optimum.
    ///
    /// Run beside the degenerate sweep this tells apart two very different findings: a solver
    /// that cannot handle many tokens, and one that cannot handle ties. Dual decomposition is
    /// known to return an infeasible point when the subproblems have flat optima — the edge
    /// flows are each optimal but need not add up — which is the case the convex flow paper
    /// restores feasibility for in a second phase this does not yet implement.
    #[tokio::test]
    async fn test_solve_scales_with_token_count_distinct_prices() {
        let amount = 1e18 as u128;
        let floor = cp_out(amount, 10_000e18 as u128, 10_000e18 as u128);

        let mut failures: Vec<String> = Vec::new();
        for middles in [1usize, 2, 4, 8, 16] {
            match solve_with_middles(middles, amount, true).await {
                Ok(out) if out >= floor => {}
                Ok(out) => failures.push(format!(
                    "{middles} middles ({} tokens): paid {out}, below the direct pool's {floor}",
                    middles + 2
                )),
                Err(error) => {
                    failures.push(format!("{middles} middles ({} tokens): {error}", middles + 2))
                }
            }
        }
        assert!(
            failures.is_empty(),
            "convex flow broke with distinct prices:\n  {}",
            failures.join("\n  ")
        );
    }

    /// Walks the *magnitude* up on a fixed two-hop market and names where it stops working.
    ///
    /// Single-hop solves hold at wei magnitudes and a two-hop solve holds at toy ones, so the
    /// interaction is what matters: at an intermediate token the first hop's output has to be
    /// the second hop's input, and `f64` carries about sixteen significant digits. The sweep
    /// shows whether conservation survives once the amounts outrun that.
    ///
    /// It starts at `1e6`, a thousand raw units against a million-unit pool. A decade below that
    /// the order is one raw unit, the first hop pays zero for it by integer division, and the
    /// expected output of the whole route is zero — there is no route to find, which
    /// `test_two_hop_pays_nothing_for_a_single_raw_unit` pins separately.
    #[tokio::test]
    async fn test_two_hop_scales_with_magnitude() {
        let token_a = mk_token(0x0A, "A");
        let token_b = mk_token(0x0B, "B");
        let token_c = mk_token(0x0C, "C");

        let mut outcomes: Vec<String> = Vec::new();
        for exponent in [6u32, 9, 12, 15, 18, 21, 24] {
            let reserve = 10u128.pow(exponent);
            let amount = reserve / 1_000;
            let (market, manager) = setup_market_weighted_boxed(vec![
                ("ab", &token_a, &token_b, pool(reserve, reserve)),
                ("bc", &token_b, &token_c, pool(reserve, reserve)),
            ]);
            let order = sell(&token_a, &token_c, amount);
            let mid = cp_out(amount, reserve, reserve);
            let mid: u128 = mid.try_into().expect("fits");
            let expected = cp_out(mid, reserve, reserve);

            let result = ConvexFlowAlgorithm::default()
                .find_best_route(SolveRequest::new(manager.graph(), market, &order))
                .await;
            match result {
                Ok(route) => {
                    let got = route
                        .route()
                        .amount_out(&token_c.address);
                    let tolerance = &expected / 100u32;
                    let diff = if got > expected { &got - &expected } else { &expected - &got };
                    if diff > tolerance {
                        outcomes.push(format!("1e{exponent}: paid {got}, expected ~{expected}"));
                    }
                }
                Err(error) => outcomes.push(format!("1e{exponent}: {error}")),
            }
        }
        assert!(
            outcomes.is_empty(),
            "two-hop solves break as amounts grow:\n  {}",
            outcomes.join("\n  ")
        );
    }

    /// A one-raw-unit order over two hops: the first pool pays zero for it by integer division,
    /// so there is no flow worth taking and the solver must say so.
    ///
    /// The point of pinning it is what the alternative looks like. A flow that swallows the whole
    /// order and pays nothing out conserves at every token, so it is perfectly feasible, and a
    /// coordinator that ranks points on conservation prefers it to every other point it will
    /// ever see. It also puts the dual value below the optimum it is supposed to bound. The
    /// subproblem refuses it instead, in `Edge::worth_trading`.
    #[tokio::test]
    async fn test_two_hop_pays_nothing_for_a_single_raw_unit() {
        let token_a = mk_token(0x0A, "A");
        let token_b = mk_token(0x0B, "B");
        let token_c = mk_token(0x0C, "C");
        let (market, manager) = setup_market_weighted_boxed(vec![
            ("ab", &token_a, &token_b, pool(1_000, 1_000)),
            ("bc", &token_b, &token_c, pool(1_000, 1_000)),
        ]);

        let order = sell(&token_a, &token_c, 1);
        let result = ConvexFlowAlgorithm::default()
            .find_best_route(SolveRequest::new(manager.graph(), market, &order))
            .await;

        assert!(
            matches!(result, Err(AlgorithmError::InsufficientLiquidity)),
            "a route that pays nothing is not a route: {result:?}"
        );
    }

    /// The single-pool solve, with a fee on the pool.
    ///
    /// The optimum is still forced — one pool, one direction — so this says whether the
    /// subproblem's optimality condition holds when the marginal price and `get_amount_out`
    /// both carry a fee. A solver comparing a fee-free marginal against a fee-bearing curve
    /// solves for a root that is wrong by the fee, which on an order this size is most of the
    /// pool.
    #[tokio::test]
    async fn test_solve_single_fee_pool_sends_the_whole_order() {
        let token_a = mk_token(0x0A, "A");
        let token_c = mk_token(0x0C, "C");
        let (reserve_a, reserve_c) = (1_000e18 as u128, 2_000e18 as u128);
        let fee_bps = 30;
        let (market, manager) = setup_market_weighted_boxed(vec![(
            "p",
            &token_a,
            &token_c,
            fee_pool(reserve_a, reserve_c, fee_bps),
        )]);

        let amount = 1e18 as u128;
        let order = sell(&token_a, &token_c, amount);
        let result = ConvexFlowAlgorithm::default()
            .find_best_route(SolveRequest::new(manager.graph(), market, &order))
            .await
            .expect("the only route must be found through a fee-bearing pool");

        let expected = fee_cp_out(amount, reserve_a, reserve_c, fee_bps);
        let got = result
            .route()
            .amount_out(&token_c.address);
        let tolerance = &expected / 10_000u32;
        let diff = if got > expected { &got - &expected } else { &expected - &got };
        assert!(diff <= tolerance, "expected ~{expected}, got {got}");
    }

    /// Fee-bearing pools of different depths and different fees, which is what a real pair looks
    /// like. The split must still beat sending everything through the cheapest one.
    #[tokio::test]
    async fn test_solve_fee_pools_split_beats_the_best_single() {
        let token_a = mk_token(0x0A, "A");
        let token_c = mk_token(0x0C, "C");
        let deep = 10_000e18 as u128;
        let (market, manager) = setup_market_weighted_boxed(vec![
            ("deep_5bp", &token_a, &token_c, fee_pool(deep, 2 * deep, 5)),
            ("mid_30bp", &token_a, &token_c, fee_pool(deep / 2, deep, 30)),
            ("shallow_100bp", &token_a, &token_c, fee_pool(deep / 10, deep / 5, 100)),
        ]);

        let amount = 100e18 as u128;
        let order = sell(&token_a, &token_c, amount);
        let result = ConvexFlowAlgorithm::default()
            .find_best_route(SolveRequest::new(manager.graph(), market, &order))
            .await
            .expect("three pools serve the pair");

        let best_single = fee_cp_out(amount, deep, 2 * deep, 5);
        assert!(
            result
                .route()
                .amount_out(&token_c.address) >=
                best_single,
            "a split across fee-bearing pools must not pay less than the cheapest single pool"
        );
    }

    /// A two-hop route through fee-bearing pools, which is where a fee the marginal price does
    /// not account for compounds: the second hop is fed whatever the first one really paid.
    #[tokio::test]
    async fn test_solve_fee_two_hop_conserves_at_the_intermediate_token() {
        let token_a = mk_token(0x0A, "A");
        let token_b = mk_token(0x0B, "B");
        let token_c = mk_token(0x0C, "C");
        let reserve = 10_000e18 as u128;
        let fee_bps = 30;
        let (market, manager) = setup_market_weighted_boxed(vec![
            ("ab", &token_a, &token_b, fee_pool(reserve, reserve, fee_bps)),
            ("bc", &token_b, &token_c, fee_pool(reserve, reserve, fee_bps)),
        ]);

        let amount = 1e18 as u128;
        let order = sell(&token_a, &token_c, amount);
        let result = ConvexFlowAlgorithm::default()
            .find_best_route(SolveRequest::new(manager.graph(), market, &order))
            .await
            .expect("the two-hop route must be found");

        let mid: u128 = fee_cp_out(amount, reserve, reserve, fee_bps)
            .try_into()
            .expect("fits");
        let expected = fee_cp_out(mid, reserve, reserve, fee_bps);
        let got = result
            .route()
            .amount_out(&token_c.address);
        let tolerance = &expected / 1_000u32;
        let diff = if got > expected { &got - &expected } else { &expected - &got };
        assert!(diff <= tolerance, "expected ~{expected}, got {got}");
    }

    /// Sweeps how small the order is relative to pool depth, on a fixed two-hop market.
    ///
    /// This is the dimension the real market lives in: a $1,000 order against a multi-million
    /// dollar pool is a ten-thousandth of its depth or less. A solver that only works on trades
    /// comparable to the pool is no use here, so the sweep finds the ratio at which it stops.
    #[tokio::test]
    async fn test_two_hop_scales_with_relative_trade_size() {
        let token_a = mk_token(0x0A, "A");
        let token_b = mk_token(0x0B, "B");
        let token_c = mk_token(0x0C, "C");
        let reserve = 10_000e18 as u128;

        let mut outcomes: Vec<String> = Vec::new();
        for divisor in [10u128, 100, 1_000, 10_000, 100_000, 1_000_000] {
            let amount = reserve / divisor;
            let (market, manager) = setup_market_weighted_boxed(vec![
                ("ab", &token_a, &token_b, pool(reserve, reserve)),
                ("bc", &token_b, &token_c, pool(reserve, reserve)),
            ]);
            let order = sell(&token_a, &token_c, amount);
            let result = ConvexFlowAlgorithm::default()
                .find_best_route(SolveRequest::new(manager.graph(), market, &order))
                .await;
            if let Err(error) = result {
                outcomes.push(format!("order = depth/{divisor}: {error}"));
            }
        }
        assert!(
            outcomes.is_empty(),
            "two-hop solves break as the order shrinks against pool depth:\n  {}",
            outcomes.join("\n  ")
        );
    }

    /// The same sweep carried as far as the fixture can express, so the scaling limit is a known
    /// number rather than something rediscovered on the next benchmark.
    ///
    /// 128 middles is 130 tokens and 257 edges. The sweep stops there because `mk_token` derives
    /// an address from a single byte and 256 middles runs out of them, not because the solve
    /// does: whatever moves this bound has to widen the fixture first.
    #[tokio::test]
    async fn test_solve_scales_past_thirty_tokens() {
        let amount = 1e18 as u128;
        let floor = cp_out(amount, 10_000e18 as u128, 10_000e18 as u128);
        for middles in [32usize, 64, 128] {
            let out = solve_with_middles(middles, amount, false)
                .await
                .unwrap_or_else(|error| panic!("{middles} middles: {error}"));
            assert!(out >= floor, "{middles} middles: paid {out}, below the direct pool's {floor}");
        }
    }

    /// An edge whose input token is worth more than the whole order must still be bracketed.
    ///
    /// The recorded market has these: a token priced high enough against the order that its
    /// value cap rounds below a single raw unit. Skipping the cap there leaves the edge with
    /// whatever `get_limits` reports, which is how a 10-token solve came back with a net-flow
    /// residual of 2.8e52 — the bracket was a deep pool's entire limit, and the reference price
    /// that made the cap sub-unit scaled the resulting flow right back up.
    #[test]
    fn test_cap_edges_floors_at_one_raw_unit() {
        // Only an intermediate token can have a sub-unit cap: on an edge out of `token_in` the
        // reference price cancels, leaving the cap at `EDGE_VALUE_SLACK` times the order.
        let mut net = network(vec![edge(0, 1, "a"), edge(1, 2, "b")], 3);
        let limit = net.edges[1].max_in.clone();
        // One raw unit of the middle token is worth a billion orders.
        let reference = vec![1.0, 1e9, 1.0];
        cap_edges_to_order_value(&mut net, &reference, &sell_order(1));

        assert_eq!(
            net.edges[0].max_in,
            BigUint::from(EDGE_VALUE_SLACK as u32),
            "an edge out of token_in is capped at the order itself, plus the slack"
        );
        assert!(net.edges[1].max_in < limit, "the edge must be bracketed, not left at get_limits");
        assert_eq!(net.edges[1].max_in, BigUint::from(1u32), "one raw unit is the floor");
    }

    /// Many edges *and* a tiny order, which is the quadrant the real market sits in and the one
    /// the sweeps above leave out: `test_solve_scales_with_token_count` varies the token count at
    /// a fixed relative size, and `test_two_hop_scales_with_relative_trade_size` varies the size
    /// on a fixed two-hop market.
    ///
    /// The direct pool is in every one of these markets, so clearing its output is the floor
    /// however many alternatives the solver is offered and however little of the pool the order
    /// is worth. At these sizes splitting is worth almost nothing — the direct pool *is* the
    /// answer — so what this really asserts is that the extra choices never make it worse.
    #[tokio::test]
    async fn test_solve_small_order_against_many_edges() {
        let depth = 10_000e18;
        let mut failures: Vec<String> = Vec::new();
        for middles in [32usize, 64] {
            for divisor in [1e4f64, 1e5, 1e6] {
                let amount = (depth / divisor) as u128;
                let floor = cp_out(amount, depth as u128, depth as u128);
                match solve_with_middles(middles, amount, false).await {
                    Ok(out) if out >= floor => {}
                    Ok(out) => failures.push(format!(
                        "{middles} middles, order = depth/{divisor:.0e}: paid {out}, below the \
                         direct pool's {floor}"
                    )),
                    Err(error) => failures
                        .push(format!("{middles} middles, order = depth/{divisor:.0e}: {error}")),
                }
            }
        }
        assert!(
            failures.is_empty(),
            "convex flow broke on a small order against many edges:\n  {}",
            failures.join("\n  ")
        );
    }

    /// The subproblem has to resolve a flow far below the bracket it searches.
    ///
    /// `cap_edges_to_order_value` brackets every edge by the order's own value, and a leg
    /// carrying a ten-thousandth of the order is an ordinary leg — `MIN_LEG_FRACTION` admits
    /// exactly that. A subproblem that resolves its bracket to a fixed fraction of itself, as
    /// twelve bisection steps do, reports zero for anything below a thousandth of it. That is
    /// not a rounding error in the answer, it is a staircase where the dual's gradient should
    /// be: net flow then only moves in steps a thousand times larger than the leg the
    /// coordinator is trying to size, and no step length can converge on it.
    #[tokio::test]
    async fn test_edge_resolves_flow_far_below_its_bracket() {
        let token_a = mk_token(0x0A, "A");
        let token_c = mk_token(0x0C, "C");
        let reserve = 10_000e18 as u128;
        let (market, manager) =
            setup_market_weighted_boxed(vec![("p", &token_a, &token_c, pool(reserve, reserve))]);

        let amount = 1e18 as u128;
        let order = sell(&token_a, &token_c, amount);
        let view = read_market(&market, None)
            .await
            .expect("market reads");
        let search = RouteSearch {
            bounds: &ConvexFlowAlgorithm::default().query,
            exclusions: &RouteExclusions::default(),
        };
        let network =
            Network::build(manager.graph(), &view, &order, search, &RouteExclusions::default())
                .expect("network builds");
        let ids: FxHashSet<&ComponentId> = network
            .edges
            .iter()
            .map(|edge| &edge.component_id)
            .collect();
        let state = view.extract_subset_with_overlay(&ids);
        drop(view);

        // The bracket the coordinator would hand this edge, and a target a ten-thousandth of the
        // order inside it — a fortieth of a thousandth of the bracket, so bisection at twelve
        // steps cannot see it at all.
        let bracket = amount * EDGE_VALUE_SLACK as u128;
        let wanted = amount / 10_000;
        let mut edge = network.edges[0].clone();
        edge.max_in = BigUint::from(bracket);

        // For this pool `h⁺(w) = R² / (R + w)²`, so the price ratio that puts the optimum at
        // `wanted` is that evaluated there.
        let ratio = (reserve as f64 / (reserve + wanted) as f64).powi(2);
        let (flow, sims) = edge
            .solve(ratio, 1.0, &state)
            .expect("the pool prices every trade inside its own limit");

        let got = flow.amount_in.to_f64().expect("fits");
        let error = (got - wanted as f64).abs() / wanted as f64;
        assert!(
            error < 1e-3,
            "wanted ~{wanted} in, got {} ({} simulations, relative error {error:.3e})",
            flow.amount_in,
            sims
        );
    }

    /// A pool traded in both directions at once is not a trade anyone can execute.
    ///
    /// The paper gives each CFMM one signed flow variable, so a reverse trade is negative `w` on
    /// the same edge and the two directions are mutually exclusive by construction. This
    /// implementation splits each pool into two directed edges with `w >= 0`, which makes that
    /// exclusivity an assumption rather than a fact. If it does not hold, the "flow" the solver
    /// returns is not realisable and whatever conservation it satisfies is fictitious.
    #[tokio::test]
    async fn test_no_pool_is_traded_in_both_directions() {
        let token_a = mk_token(0x0A, "A");
        let token_b = mk_token(0x0B, "B");
        let token_c = mk_token(0x0C, "C");
        let r = 10_000e18 as u128;
        let (market, manager) = setup_market_weighted_boxed(vec![
            ("ab", &token_a, &token_b, pool(r, r)),
            ("bc", &token_b, &token_c, pool(r, r)),
            ("ac", &token_a, &token_c, pool(r, r)),
        ]);
        let order = sell(&token_a, &token_c, r / 1_000);
        let route = ConvexFlowAlgorithm::default()
            .find_best_route(SolveRequest::new(manager.graph(), market, &order))
            .await
            .expect("solvable");

        // Within the returned route, a component must appear in one direction only.
        let mut directions: FxHashMap<&str, FxHashSet<(Bytes, Bytes)>> = FxHashMap::default();
        for swap in route.route().swaps() {
            directions
                .entry(swap.component_id())
                .or_default()
                .insert((swap.token_in().clone(), swap.token_out().clone()));
        }
        for (component, dirs) in &directions {
            assert_eq!(dirs.len(), 1, "{component} is traded in {} directions", dirs.len());
        }
    }

    /// Checks the analytic gradient against a central finite difference of the objective.
    ///
    /// Every piece downstream — the Armijo line search, the Barzilai-Borwein step, the
    /// stationarity test — assumes `gradient` is the derivative of `value`. That was derived on
    /// paper and verified against the source, but never checked numerically, and the coordinator
    /// is observed flying apart on any instance with more than a couple of free prices. A wrong
    /// gradient looks exactly like that.
    ///
    /// Evaluated away from the warm start, because the warm start sits on every edge's no-flow
    /// kink, where the derivative genuinely does not exist.
    ///
    /// The step is chosen against the amounts, not the prices. Flows are integers, so the dual
    /// value is a step function below the price change that moves a flow by one raw unit, and a
    /// central difference taken inside that scale measures rounding instead of a derivative: on
    /// this market the relative error falls from 1.4 at a step of `1e-7` to `2e-15` at `1e-3`,
    /// the wrong way round for truncation error. A step this coarse is still well inside the
    /// region where edges carry flow, and the second-order envelope error it does admit stays
    /// four orders below the tolerance here.
    ///
    /// The market is at wei magnitudes for the same reason read from the other side. The
    /// identity being tested — that the gradient is the subproblem's own optimal flow — holds
    /// only where the subproblem can find that optimum, and its difference quotient needs the
    /// pool's output to move by more than the pool's integer rounding. A toy market cannot
    /// supply that, and the gradient there is not wrong so much as undefined.
    #[tokio::test]
    async fn test_gradient_matches_finite_difference() {
        let (a, b, c) = (mk_token(0x0A, "A"), mk_token(0x0B, "B"), mk_token(0x0C, "C"));
        let (market, manager) = setup_market_weighted_boxed(vec![
            ("ab", &a, &b, pool(1_000e18 as u128, 2_000e18 as u128)),
            ("bc", &b, &c, pool(2_000e18 as u128, 1_500e18 as u128)),
            ("ac", &a, &c, pool(1_000e18 as u128, 1_400e18 as u128)),
        ]);
        let order = sell(&a, &c, 10e18 as u128);
        let algorithm = ConvexFlowAlgorithm::default();

        let view = read_market(&market, None)
            .await
            .expect("market reads");
        let search =
            RouteSearch { bounds: &algorithm.query, exclusions: &RouteExclusions::default() };
        let network =
            Network::build(manager.graph(), &view, &order, search, &RouteExclusions::default())
                .expect("network builds");
        let ids: FxHashSet<&ComponentId> = network
            .edges
            .iter()
            .map(|edge| &edge.component_id)
            .collect();
        let state = view.extract_subset_with_overlay(&ids);
        drop(view);

        let reference = warm_start(&network, None);
        let problem = DualProblem::new(
            &network.edges,
            network.tokens.len(),
            network.node_in,
            network.node_out,
            order.amount(),
            reference,
        )
        .expect("dual builds");

        // Off the kink: the input token priced well below its reference, so edges carry flow and
        // the objective is locally smooth.
        let mut point = vec![1.0; network.tokens.len()];
        point[network.node_in] = 0.4;
        point[network.node_out] = 1.0;

        let mut budget = usize::MAX;
        let analytic = problem
            .evaluate(&point, &state, &mut budget)
            .expect("evaluates")
            .gradient;

        let h = 1e-3;
        for node in 0..network.tokens.len() {
            if node == network.node_out {
                continue; // pinned; its gradient is deliberately zeroed
            }
            let mut up = point.clone();
            let mut down = point.clone();
            up[node] += h;
            down[node] -= h;
            let f_up = problem
                .evaluate(&up, &state, &mut budget)
                .expect("evaluates")
                .value;
            let f_down = problem
                .evaluate(&down, &state, &mut budget)
                .expect("evaluates")
                .value;
            let numeric = (f_up - f_down) / (2.0 * h);
            let scale = numeric
                .abs()
                .max(analytic[node].abs())
                .max(1e-6);
            let relative = (numeric - analytic[node]).abs() / scale;
            assert!(
                relative < 1e-3,
                "node {node}: analytic {:.6e}, finite difference {:.6e} (relative error {relative:.3})",
                analytic[node],
                numeric
            );
        }
    }

    /// The decimal conversion, tested directly.
    ///
    /// Not end to end: `ConstantProductSim` reports `reserve_out / reserve_in` as a raw ratio,
    /// where tycho's real implementations divide through each token's decimals
    /// (`spot_price_from_reserves(.., base.decimals, quote.decimals)`). A solve against the
    /// double would therefore exercise the double's convention rather than the one production
    /// uses, so the conversion is pinned on its own and the decimal path is covered end to end
    /// only by the recorded-market benchmark.
    #[test]
    fn test_raw_marginal_conversion() {
        // WETH (18) into USDC (6) at 2000 USDC per WETH: one wei buys 2e-9 raw USDC, because
        // 1e18 wei buys 2000e6 = 2e9 raw USDC.
        let raw = to_raw_marginal(2_000.0, 18, 6);
        assert!((raw - 2e-9).abs() < 1e-18, "got {raw}");
        assert!((raw * 1e18 - 2e9).abs() < 1.0, "1e18 wei must buy 2000e6 raw USDC");
    }

    #[test]
    fn test_raw_marginal_conversion_is_identity_for_equal_decimals() {
        assert_eq!(to_raw_marginal(3.5, 18, 18), 3.5);
    }

    #[test]
    fn test_raw_marginal_conversion_reverses_direction() {
        // USDC (6) into WETH (18) at 1/2000: one raw USDC buys 5e8 wei.
        let raw = to_raw_marginal(1.0 / 2_000.0, 6, 18);
        assert!((raw - 5e8).abs() < 1.0, "got {raw}");
    }

    /// Many pools on one pair, at wei magnitudes. Output must still clear the best single pool.
    #[tokio::test]
    async fn test_solve_many_parallel_pools_at_wei_magnitudes() {
        let token_a = mk_token(0x0A, "A");
        let token_c = mk_token(0x0C, "C");
        let pools: Vec<(String, u128, u128)> = (1..=12)
            .map(|i| (format!("p{i}"), i as u128 * 100e18 as u128, i as u128 * 200e18 as u128))
            .collect();
        let specs: Vec<(
            &str,
            &Token,
            &Token,
            Box<dyn tycho_simulation::tycho_common::simulation::protocol_sim::ProtocolSim>,
        )> = pools
            .iter()
            .map(|(id, a, c)| (id.as_str(), &token_a, &token_c, pool(*a, *c)))
            .collect();
        let (market, manager) = setup_market_weighted_boxed(specs);

        let amount = 1e18 as u128;
        let order = sell(&token_a, &token_c, amount);
        let result = ConvexFlowAlgorithm::default()
            .find_best_route(SolveRequest::new(manager.graph(), market, &order))
            .await
            .expect("twelve pools serve the pair");

        let best_single = cp_out(amount, 1_200e18 as u128, 2_400e18 as u128);
        assert!(
            result
                .route()
                .amount_out(&token_c.address) >=
                best_single,
            "a split across twelve pools must not pay less than the deepest one alone"
        );
    }

    #[test]
    fn test_warm_start_rebases_onto_output_token() {
        let net = network(vec![edge(0, 1, "p")], 2);
        let mut prices = FxHashMap::default();
        prices.insert(net.tokens[0].address.clone(), price(4));
        prices.insert(net.tokens[1].address.clone(), price(2));
        let start = warm_start(&net, Some(&prices));
        // The input token is worth twice the output token, and the output token is the numéraire.
        assert_eq!(start[0], 2.0);
        assert_eq!(start[1], 1.0);
    }

    #[test]
    fn test_warm_start_falls_back_to_flat_without_prices() {
        let net = network(vec![edge(0, 1, "p")], 2);
        assert_eq!(warm_start(&net, None), vec![1.0, 1.0]);
    }

    #[test]
    fn test_warm_start_ignores_unusable_output_price() {
        let net = network(vec![edge(0, 1, "p")], 2);
        let mut prices = FxHashMap::default();
        prices.insert(net.tokens[0].address.clone(), price(4));
        prices.insert(net.tokens[1].address.clone(), price(0));
        assert_eq!(warm_start(&net, Some(&prices)), vec![1.0, 1.0]);
    }

    #[test]
    fn test_warm_start_skips_unpriced_token() {
        let net = network(vec![edge(0, 1, "a"), edge(1, 2, "b")], 3);
        let mut prices = FxHashMap::default();
        prices.insert(net.tokens[2].address.clone(), price(1));
        let start = warm_start(&net, Some(&prices));
        assert_eq!(start[1], 1.0, "an unpriced token keeps the default");
    }

    #[test]
    fn test_decompose_single_path() {
        let net = network(vec![edge(0, 1, "a"), edge(1, 2, "b")], 3);
        let flows = vec![flow(1_000, 900), flow(900, 800)];
        let allocations = decompose(&net, &flows, &sell_order(1_000));
        assert_eq!(allocations.len(), 1);
        assert_eq!(allocations[0].hops.len(), 2);
        assert!((allocations[0].flow_fraction - 0.9).abs() < 0.2);
    }

    #[test]
    fn test_decompose_two_parallel_paths() {
        // Two disjoint pools from token 0 to token 2, carrying 600 and 400.
        let net = network(vec![edge(0, 2, "a"), edge(0, 2, "b")], 3);
        let flows = vec![flow(600, 590), flow(400, 395)];
        let allocations = decompose(&net, &flows, &sell_order(1_000));
        assert_eq!(allocations.len(), 2, "each pool becomes its own leg");
        let total: f64 = allocations
            .iter()
            .map(|a| a.flow_fraction)
            .sum();
        assert!((total - 1.0).abs() < 1e-6, "fractions cover the order");
    }

    #[test]
    fn test_decompose_drops_dust_leg() {
        // The second pool carries far less than MIN_LEG_FRACTION of the order.
        let net = network(vec![edge(0, 2, "a"), edge(0, 2, "b")], 3);
        let flows = vec![flow(1_000_000, 990_000), flow(1, 1)];
        let allocations = decompose(&net, &flows, &sell_order(1_000_001));
        assert_eq!(allocations.len(), 1, "a leg too small to pay its gas is dropped");
    }

    #[test]
    fn test_decompose_ignores_idle_edges() {
        let net = network(vec![edge(0, 2, "a"), edge(0, 2, "b")], 3);
        let flows = vec![flow(1_000, 990), EdgeFlow::none()];
        let allocations = decompose(&net, &flows, &sell_order(1_000));
        assert_eq!(allocations.len(), 1);
    }

    #[test]
    fn test_decompose_abandons_circulation() {
        // A cycle 0→1→0 carrying flow, with no edge reaching the output token. A walk must not
        // loop forever, and must emit nothing rather than an arbitrage-dependent route.
        let net = network(vec![edge(0, 1, "a"), edge(1, 0, "b")], 3);
        let flows = vec![flow(500, 500), flow(500, 500)];
        let allocations = decompose(&net, &flows, &sell_order(500));
        assert!(allocations.is_empty(), "a circulation is not a route");
    }

    #[test]
    fn test_decompose_caps_leg_count() {
        // More parallel pools than MAX_LEGS, each carrying an equal share.
        let edges: Vec<Edge> = (0..MAX_LEGS + 4)
            .map(|i| edge(0, 2, &format!("p{i}")))
            .collect();
        let flows: Vec<EdgeFlow> = (0..MAX_LEGS + 4)
            .map(|_| flow(1_000, 990))
            .collect();
        let net = network(edges, 3);
        let allocations = decompose(&net, &flows, &sell_order(1_000 * (MAX_LEGS as u32 + 4)));
        assert_eq!(allocations.len(), MAX_LEGS, "leg count is bounded");
    }

    #[test]
    fn test_decompose_empty_flow_yields_nothing() {
        let net = network(vec![edge(0, 2, "a")], 3);
        let allocations = decompose(&net, &[EdgeFlow::none()], &sell_order(1_000));
        assert!(allocations.is_empty());
    }
}
