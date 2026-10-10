//! The single-pair gas-aware optimum, as a benchmark reference.
//!
//! Every bps figure this harness reports is measured against another algorithm — `BF_d2` by
//! default. That answers "did this change help" and cannot answer "how much is left on the
//! table", which is the question that decides whether routing work is worth doing at all. This
//! module answers the second one for the orders where it can: it computes the best net output
//! obtainable on a single pair, with gas charged per pool used, and registers itself as an
//! algorithm so the existing reports measure everything else against it.
//!
//! It is a reference, not a router. It is far too slow to serve a quote, it is registered only by
//! the benchmark binaries, and it answers a deliberately narrower question than the algorithms it
//! measures:
//!
//! - **Direct pools only.** No multi-hop. An algorithm that routes `A → B → C` can legitimately
//!   beat this number, and when one does, that is the value of multi-hop on that order rather than
//!   a bug here.
//! - **The [`MAX_POOLS`] deepest pools for the pair.** The optimum over every pool is a subset
//!   search, and the search is what costs; the literature's own audits work at this scale (Xi and
//!   Moallemi evaluate four Uniswap pools).
//! - **Exact within those bounds.** Every subset of the candidate pools is tried, each one's
//!   allocation solved by bisecting a common marginal price, and each one priced by
//!   `build_split_route` — the same re-execution every other config's route goes through, so the
//!   numbers are comparable by construction rather than by argument.
//!
//! # What it computes
//!
//! For a subset `S` of pools, the optimal split equalises marginal prices: there is one price `λ`
//! at which each pool `i` wants `wᵢ(λ)`, and the allocation that maximises output is the one
//! where `Σᵢ wᵢ(λ) = amount`. `Σᵢ wᵢ(λ)` falls as `λ` rises, so one bisection finds it. Gas makes
//! the choice of `S` discrete — each pool used costs a swap whatever it carries — which is why
//! this enumerates subsets instead of solving one convex problem. That discreteness is not an
//! artifact: with fixed costs per pool the routing problem is a minimum concave-cost flow, which
//! is NP-hard, and enumerating a small candidate set is the standard response.
//!
//! `wᵢ(λ)` comes from [`SwapConstraint::PoolTargetPrice`], which asks a pool directly for the
//! swap that moves its marginal price to `λ`. The query is deliberately not shared with
//! `convex_flow`'s copy of it: a reference that shares code with what it measures cannot catch
//! that code being wrong.

use std::time::{Duration, Instant};

use fynd_core::{
    algorithm::{
        most_liquid::DepthAndPrice,
        paths::read_market,
        split_primitives::{
            build_split_route, normalize_fractions, HopDescriptor, PathAllocation, SimulatedHop,
        },
    },
    derived::types::TokenGasPrices,
    feed::market_data::MarketState,
    graph::{EdgeData, TopologyGraph, TopologyGraphManager},
    rustc_hash::FxHashSet,
    types::RouteResult,
    Algorithm, AlgorithmConfig, AlgorithmError, ComputationRequirements, NoPathReason, Order,
    SolveRequest,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{Float, One, ToPrimitive, Zero};
use tycho_simulation::{
    evm::query_pool_swap::query_pool_swap,
    tycho_common::{
        models::{token::Token, Address},
        simulation::errors::SimulationError,
    },
    tycho_core::simulation::protocol_sim::{Price, QueryPoolSwapParams, SwapConstraint},
};

/// How many of the pair's pools the optimum is taken over.
///
/// The subset search is `2^MAX_POOLS`, so this is the cost knob. Eight because six was not
/// enough: on the wide scenario set `water_fill` beat the reference on 23 orders by splitting
/// across more pools than it considered, and a reference something routinely beats is not one.
/// The busiest pair in the recorded market is served by eleven pools, so this is still a cut —
/// read the column as a ceiling over the eight best-paying pools, not over all of them.
const MAX_POOLS: usize = 8;

/// How many of the pair's pools are probed before the cut to [`MAX_POOLS`].
///
/// The cut has to be on what a pool pays for *this* order, which costs one swap each to find
/// out, so the probing itself needs a bound. Depth orders the pools before probing, since a
/// pool too thin to matter will not win on output either.
const MAX_PROBED_POOLS: usize = 24;

/// Bisection steps on the common marginal price, per subset.
///
/// Each step costs one pool query per pool in the subset. Twenty-four halvings resolve `λ` to one
/// part in sixteen million of its starting bracket, which is far past what the split fractions
/// need: the route is priced by re-execution afterwards, so `λ` only has to be good enough to
/// propose the right split.
const BISECTION_STEPS: usize = 24;

/// Lowest marginal price the bisection will consider, as a fraction of the highest.
///
/// Nine decades below the best pool's starting price. A `λ` under that would mean the order is
/// large enough to exhaust every pool on the pair, which the cap on each pool's input handles.
const PRICE_FLOOR_FRACTION: f64 = 1e-9;

/// The single-pair gas-aware optimum.
pub struct GasAwareOptimum {
    timeout: Duration,
}

impl GasAwareOptimum {
    /// Builds the reference from an `AlgorithmConfig`, which supplies only its timeout — the hop
    /// bounds and connector tokens mean nothing to a direct-pool search.
    pub fn with_config(config: AlgorithmConfig) -> Self {
        Self { timeout: config.timeout() }
    }
}

/// One pool the optimum may use.
struct PoolCandidate {
    component_id: String,
    token_in: Token,
    token_out: Token,
    /// `h⁺(0)`, the marginal price net of the pool's fee, in raw output units per raw input unit.
    marginal_at_zero: f64,
    /// The most this pool will take, from `get_limits`, capped to the order.
    max_in: BigUint,
    /// What this pool alone pays for the whole order, which ranks the candidates.
    alone: BigUint,
}

impl Algorithm for GasAwareOptimum {
    type GraphType = TopologyGraph<DepthAndPrice>;
    type GraphManager = TopologyGraphManager<DepthAndPrice>;

    fn name(&self) -> &str {
        "gas_aware_optimum"
    }

    async fn find_best_route(
        &self,
        request: SolveRequest<'_, Self::GraphType>,
    ) -> Result<RouteResult, AlgorithmError> {
        let started = Instant::now();
        let parts = request.into_parts();
        let order = parts.order;
        let no_path = |reason| {
            AlgorithmError::no_path(order.token_in().clone(), order.token_out().clone(), reason)
        };
        if !order.is_sell() {
            return Err(no_path(NoPathReason::NoGraphPath));
        }

        let token_prices = match parts.derived.as_ref() {
            Some(derived) => derived
                .read()
                .await
                .token_prices()
                .cloned(),
            None => None,
        };
        let view = read_market(&parts.market, parts.label).await?;

        // The pair's direct pools, deepest first.
        let (Some(from), Some(to)) = (
            parts
                .graph
                .get_token_ix(order.token_in()),
            parts
                .graph
                .get_token_ix(order.token_out()),
        ) else {
            return Err(no_path(NoPathReason::SourceTokenNotInGraph));
        };
        let mut pools: Vec<&EdgeData<DepthAndPrice>> = parts
            .graph
            .pools_between(from, to)
            .iter()
            .filter(|edge| {
                !parts
                    .exclusions
                    .excludes_pool(&edge.component_id)
            })
            .collect();
        pools.sort_by(|left, right| {
            depth_of(right)
                .partial_cmp(&depth_of(left))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pools.truncate(MAX_PROBED_POOLS);
        if pools.is_empty() {
            return Err(no_path(NoPathReason::NoGraphPath));
        }

        let component_ids: FxHashSet<&String> = pools
            .iter()
            .map(|edge| &edge.component_id)
            .collect();
        let market = view.extract_subset_with_overlay(&component_ids);
        let (Some(token_in), Some(token_out)) =
            (view.get_token(order.token_in()), view.get_token(order.token_out()))
        else {
            return Err(no_path(NoPathReason::SourceTokenNotInGraph));
        };
        let (token_in, token_out) = (token_in.clone(), token_out.clone());
        drop(view);

        let gas_price = fynd_core::algorithm::paths::fetch_gas_price(&market)?;
        let mut candidates: Vec<PoolCandidate> = pools
            .iter()
            .filter_map(|edge| candidate(&market, &edge.component_id, &token_in, &token_out, order))
            .collect();
        if candidates.is_empty() {
            return Err(no_path(NoPathReason::NoGraphPath));
        }
        // Keep the pools that pay the most for this order, which is not the same as the deepest:
        // depth is measured at a fixed price move, and the order is whatever size it is.
        candidates.sort_by(|left, right| right.alone.cmp(&left.alone));
        candidates.truncate(MAX_POOLS);

        // Every subset, each priced by re-execution. The empty subset is not a route.
        let deadline = self.timeout;
        let mut best: Option<(BigInt, fynd_core::Route)> = None;
        for mask in 1u32..(1u32 << candidates.len()) {
            if started.elapsed() > deadline {
                break;
            }
            let subset: Vec<&PoolCandidate> = candidates
                .iter()
                .enumerate()
                .filter(|(index, _)| mask & (1 << index) != 0)
                .map(|(_, candidate)| candidate)
                .collect();
            let Some(allocations) = allocate(&subset, order, &market) else {
                continue;
            };
            let Ok(route) = build_split_route(&allocations, &market, order) else {
                continue;
            };
            if route.validate().is_err() {
                continue;
            }

            // Net of gas, the figure the router ranks on, with gas charged per pool the subset
            // uses. `route.total_gas()` already sums the legs that were executed.
            let gross = route.amount_out(order.token_out());
            if gross.is_zero() {
                continue;
            }
            let net = BigInt::from(gross) -
                gas_in_output_units(
                    &route.total_gas(),
                    &gas_price,
                    token_prices.as_ref(),
                    order.token_out(),
                )
                .map(BigInt::from)
                .unwrap_or_else(BigInt::zero);
            if best
                .as_ref()
                .is_none_or(|(best_net, _)| net > *best_net)
            {
                best = Some((net, route));
            }
        }

        let Some((net, route)) = best else {
            return Err(no_path(NoPathReason::NoGraphPath));
        };
        Ok(RouteResult::new(route, net, gas_price))
    }

    fn computation_requirements(&self) -> ComputationRequirements {
        ComputationRequirements::none()
            .allow_stale("token_prices")
            .expect("conflicting computation requirements")
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// A pool's depth, for the deepest-first cut. Zero for an edge the derived pass has not priced.
fn depth_of(edge: &EdgeData<DepthAndPrice>) -> f64 {
    edge.data
        .as_ref()
        .map_or(0.0, |weight| weight.depth)
}

/// Describes one pool: its fee-net marginal price, its cap, and what it pays on its own.
///
/// Gas is deliberately not measured here. Each subset's route is re-executed to be priced, and
/// that re-execution reports the gas the route really costs, which is both more accurate than a
/// probe and the figure the router ranks on. Gas enters the optimum through the net objective the
/// subsets are compared on, which is what makes this gas-*aware* rather than gas-adjusted.
///
/// Returns `None` for a pool that cannot price this pair at all.
fn candidate(
    market: &MarketState,
    component_id: &str,
    token_in: &Token,
    token_out: &Token,
    order: &Order,
) -> Option<PoolCandidate> {
    let sim = market.get_simulation_state(component_id)?;

    // The fee-net marginal price, as `ComponentDepthComputation::net_marginal_price` takes it and
    // as `PoolTargetPrice` compares against: `spot_price` marks the fee up for some protocols and
    // not others, and the lower of the two directions is net of it either way.
    let forward = sim
        .spot_price(token_in, token_out)
        .ok()?;
    let net = match sim.spot_price(token_out, token_in) {
        Ok(reverse) if reverse.is_finite() && reverse > 0.0 => forward.min(1.0 / reverse),
        _ => forward,
    };
    let marginal_at_zero = net * 10f64.powi(token_out.decimals as i32 - token_in.decimals as i32);
    if !(marginal_at_zero.is_finite() && marginal_at_zero > 0.0) {
        return None;
    }

    // No pool needs more than the whole order.
    let (limit, _) = sim
        .get_limits(token_in.address.clone(), token_out.address.clone())
        .ok()?;
    let max_in = limit.min(order.amount().clone());
    if max_in.is_zero() {
        return None;
    }

    let alone = sim
        .get_amount_out(max_in.clone(), token_in, token_out)
        .ok()?
        .amount;

    Some(PoolCandidate {
        component_id: component_id.to_string(),
        token_in: token_in.clone(),
        token_out: token_out.clone(),
        marginal_at_zero,
        max_in,
        alone,
    })
}

/// The allocation across `subset` that equalises marginal prices and spends the whole order.
///
/// Bisects the common marginal price `λ`: `Σ wᵢ(λ)` falls as `λ` rises, so the `λ` whose flows sum
/// to the order is unique. Returns `None` when the subset cannot take the order at all.
fn allocate(
    subset: &[&PoolCandidate],
    order: &Order,
    market: &MarketState,
) -> Option<Vec<PathAllocation>> {
    let amount = order
        .amount()
        .to_f64()
        .filter(|a| a.is_finite() && *a > 0.0)?;

    // At the highest starting price nothing flows; nine decades below it, everything does.
    let mut high = subset
        .iter()
        .map(|candidate| candidate.marginal_at_zero)
        .fold(f64::MIN_POSITIVE, f64::max);
    let mut low = high * PRICE_FLOOR_FRACTION;

    let mut flows: Vec<(BigUint, BigUint)> = Vec::new();
    for _ in 0..BISECTION_STEPS {
        let middle = (low + high) / 2.0;
        flows = subset
            .iter()
            .map(|candidate| flow_at_price(candidate, middle, market))
            .collect();
        let total: f64 = flows
            .iter()
            .filter_map(|(amount_in, _)| amount_in.to_f64())
            .sum();
        if total > amount {
            low = middle;
        } else {
            high = middle;
        }
    }
    // The last evaluated point may sit either side of the order; the fractions are normalised
    // below and the route is re-executed, so what matters is the split, not the total.
    let total: f64 = flows
        .iter()
        .filter_map(|(amount_in, _)| amount_in.to_f64())
        .sum();
    if !(total.is_finite() && total > 0.0) {
        return None;
    }

    let mut allocations = Vec::new();
    for (candidate, (amount_in, amount_out)) in subset.iter().zip(&flows) {
        let fraction = amount_in.to_f64()? / total;
        if !(fraction.is_finite() && fraction > 0.0) {
            continue;
        }
        allocations.push(PathAllocation {
            hops: vec![SimulatedHop {
                descriptor: HopDescriptor::new(
                    candidate.component_id.clone(),
                    candidate.token_in.clone(),
                    candidate.token_out.clone(),
                ),
                amount_out: amount_out.clone(),
                gas: BigUint::zero(),
            }],
            flow_fraction: fraction,
            amount_in: amount_in.clone(),
            amount_out: amount_out.clone(),
            marginal_price_product: 0.0,
        });
    }
    if allocations.is_empty() {
        return None;
    }
    // The split convention needs the fractions to account for the whole order exactly. Dividing
    // by the total leaves them a rounding short, and `build_split_route` refuses that, which is
    // what left every multi-pool subset unpriced and the "optimum" equal to the best single pool.
    let mut fractions: Vec<f64> = allocations
        .iter()
        .map(|allocation| allocation.flow_fraction)
        .collect();
    normalize_fractions(&mut fractions).ok()?;
    for (allocation, fraction) in allocations.iter_mut().zip(fractions) {
        allocation.flow_fraction = fraction;
    }
    Some(allocations)
}

/// What one pool wants to carry at marginal price `target`, from the pool itself.
///
/// Zero when the pool is already at or below the price, when it cannot be asked, or when the
/// answer is not worth trading. Deliberately a separate implementation from `convex_flow`'s.
fn flow_at_price(
    candidate: &PoolCandidate,
    target: f64,
    market: &MarketState,
) -> (BigUint, BigUint) {
    let idle = (BigUint::zero(), BigUint::zero());
    if target >= candidate.marginal_at_zero {
        return idle;
    }
    let Some(sim) = market.get_simulation_state(&candidate.component_id) else {
        return idle;
    };
    let Some(price) = raw_price(target) else {
        return idle;
    };
    let params = QueryPoolSwapParams::new(
        candidate.token_in.clone(),
        candidate.token_out.clone(),
        SwapConstraint::PoolTargetPrice {
            target: price,
            tolerance: 0.0,
            min_amount_in: None,
            max_amount_in: Some(candidate.max_in.clone()),
        },
    );
    let queried = match sim.query_pool_swap(&params) {
        Err(SimulationError::FatalError(message))
            if message == "query_pool_swap not implemented" =>
        {
            query_pool_swap(sim, &params)
        }
        result => result,
    };
    match queried {
        // A protocol with a closed form ignores `max_amount_in`, and a target below what the cap
        // reaches is refused outright; either way the cap is the answer.
        Ok(swap) if swap.amount_in() <= &candidate.max_in => {
            (swap.amount_in().clone(), swap.amount_out().clone())
        }
        _ => sim
            .get_amount_out(candidate.max_in.clone(), &candidate.token_in, &candidate.token_out)
            .map(|result| (candidate.max_in.clone(), result.amount))
            .unwrap_or(idle),
    }
}

/// A gas amount in raw output-token units, or `None` when the output token has no price.
fn gas_in_output_units(
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

/// A raw price as the exact fraction [`SwapConstraint`] takes.
fn raw_price(value: f64) -> Option<Price> {
    if !(value.is_finite() && value > 0.0) {
        return None;
    }
    let (mantissa, exponent, sign) = value.integer_decode();
    if sign < 0 {
        return None;
    }
    let mut numerator = BigUint::from(mantissa);
    let mut denominator = BigUint::one();
    let shift = usize::from(exponent.unsigned_abs());
    if exponent >= 0 {
        numerator <<= shift;
    } else {
        denominator <<= shift;
    }
    Some(Price { numerator, denominator })
}
