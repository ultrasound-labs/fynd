//! The per-edge arbitrage subproblem.
//!
//! The convex flow dual decomposes into one independent problem per edge: given prices on the
//! edge's two tokens, how much flow does this pool want to carry? For a two-node edge with gain
//! function `h` that is the scalar problem
//!
//! ```text
//! maximise  −η₁·w + η₂·h(w)   over  0 ≤ w ≤ w_max
//! ```
//!
//! `h` is concave, so the objective is concave in `w` and the optimum satisfies
//!
//! ```text
//! η₂·h⁺(w⋆) ≤ η₁ ≤ η₂·h⁻(w⋆)
//! ```
//!
//! which says in words: *trade until the pool's marginal price has fallen to the price ratio the
//! coordinator is offering*.
//!
//! # The pool answers this directly
//!
//! That sentence is the contract of [`SwapConstraint::PoolTargetPrice`] — "the swap required to
//! move the pool's marginal price down to a target" — so this module asks for it rather than
//! searching for it. Around twenty protocols implement `query_pool_swap`; Uniswap V2 inverts its
//! own reserve formula, Uniswap V3 and V4 use the pool's native `sqrt_price_limit` swap, and
//! anything without a closed form falls back to tycho's generic search. The cost is one call for
//! an analytic pool and up to thirty `get_amount_out` calls for the rest, which the returned
//! `price_points` reports so the caller can bill it.
//!
//! Three things come with the oracle that a search over `get_amount_out` cannot have:
//!
//! - **The protocol's own definition of marginal price**, so the fee is handled by the protocol
//!   that charges it. `ProtocolSim::spot_price` is not that quantity — it is contractually
//!   `pre_fee_price / (1 - fee)`, a buy price marked *up* by the fee, where this subproblem needs
//!   the derivative of `get_amount_out`, which is the same price marked *down* by it. Searching on
//!   `spot_price` solves for a root that is wrong by `1/(1 − fee)²`, and for an order whose price
//!   impact is a basis point against a thirty basis point pool that is not a correction, it is the
//!   whole answer.
//! - **Exactness where the curve is discrete.** A tick pool's marginal price steps, so there is
//!   often no trade that lands on the target; the protocol's own implementation knows which side of
//!   the tick to stop on, and a numerical search over amounts does not.
//! - **One call instead of a bracket.** The search this replaced spent two simulations per trial
//!   point building a difference quotient, because no usable derivative was available.
//!
//! The no-flow test stays, and still costs nothing: an edge whose marginal price at zero trade is
//! already below the price ratio wants no flow, and the cached price answers that without a call.

use num_bigint::BigUint;
use num_traits::{Float, One, ToPrimitive, Zero};
use tycho_simulation::{
    evm::query_pool_swap::query_pool_swap,
    tycho_common::{models::token::Token, simulation::errors::SimulationError},
    tycho_core::simulation::protocol_sim::{Price, QueryPoolSwapParams, SwapConstraint},
};

use super::config::TARGET_PRICE_TOLERANCE;
use crate::{
    algorithm::{sim_meter, sim_meter::MeteredProtocolSim, AlgorithmError},
    feed::market_data::MarketState,
    types::ComponentId,
};

/// The stage every swap this module makes is booked under.
const EDGE_SUBPROBLEM_STAGE: sim_meter::StageLabel = "convex_flow_edge";

/// What one `query_pool_swap` is billed as when it reports no search history.
///
/// A pool that answers analytically runs no simulation at all, but it is not free either, so it
/// is charged the same as one swap rather than nothing.
const ANALYTIC_QUERY_COST: usize = 1;

/// One directed edge of the flow network: a pool, traded one way.
///
/// Both directions of a pool are separate edges, each with flow `w ≥ 0`. Modelling it this way
/// means only the `h⁺(0)` side of the no-flow interval is ever tested — the `h⁻(0)` side is the
/// reverse edge's own test — and the subproblem never has to consider negative flow.
#[derive(Clone)]
pub(crate) struct Edge {
    /// The pool this edge trades through.
    pub component_id: ComponentId,
    /// Token going in.
    pub token_in: Token,
    /// Token coming out.
    pub token_out: Token,
    /// Index of `token_in` in the solver's price vector.
    pub node_in: usize,
    /// Index of `token_out` in the solver's price vector.
    pub node_out: usize,
    /// `h⁺(0)`: the marginal price net of the pool's fee, raw token_out per raw token_in, at zero
    /// trade.
    ///
    /// Net of the fee is the whole point, and is why this is not a plain `spot_price`: see
    /// `edge_bounds` in the parent module for how it is derived, and the module docs for what
    /// goes wrong when it is not.
    pub marginal_at_zero: f64,
    /// Largest input the pool accepts for this pair, from `get_limits`, capped to the order's own
    /// value by `cap_edges_to_order_value`.
    pub max_in: BigUint,
}

/// What an edge carries at a given price pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EdgeFlow {
    /// Input the edge takes.
    pub amount_in: BigUint,
    /// Output it pays.
    pub amount_out: BigUint,
    /// Gas the swap costs, zero when the edge carries nothing.
    ///
    /// Zero also when the flow came from the price oracle, which reports no gas. Nothing
    /// downstream reads it: `decompose` copies it into a `SimulatedHop`, and `build_split_route`
    /// re-executes every hop and takes the gas from that, which is the figure the route carries
    /// and the router ranks on.
    pub gas: BigUint,
}

impl EdgeFlow {
    /// The edge is not used.
    pub fn none() -> Self {
        Self { amount_in: BigUint::zero(), amount_out: BigUint::zero(), gas: BigUint::zero() }
    }

    /// Whether this edge carries anything.
    pub fn is_active(&self) -> bool {
        !self.amount_in.is_zero()
    }
}

impl Edge {
    /// Whether the optimum leaves this edge unused, decided without a call.
    ///
    /// Flow is worth starting only while the output is worth more than the input it consumes,
    /// i.e. while `η₂·h⁺(0) > η₁`. Written as a price ratio that is the paper's no-flow
    /// condition for the forward direction: the edge sits idle whenever `η₁/η₂ ≥ h⁺(0)`.
    ///
    /// Exact, now that `marginal_at_zero` is the fee-net marginal price rather than a quoted
    /// one. On a quoted price this test admitted edges that could not pay, and the subproblem had
    /// to spend simulations discovering it.
    ///
    /// A non-positive output price would make the comparison meaningless, so the edge is treated
    /// as idle; the coordinator keeps prices non-negative and the output token's price pinned at
    /// one, so this only guards against a degenerate warm start.
    pub fn carries_no_flow(&self, price_in: f64, price_out: f64) -> bool {
        if price_out <= 0.0 || !self.marginal_at_zero.is_finite() {
            return true;
        }
        price_in / price_out >= self.marginal_at_zero
    }

    /// Solves this edge's subproblem at the given prices.
    ///
    /// Returns the flow the edge wants to carry, and what the call cost in simulations. An edge
    /// rejected by [`Self::carries_no_flow`] costs nothing and never reaches here.
    ///
    /// # Errors
    ///
    /// Returns [`AlgorithmError::SimulationFailed`] when the pool cannot price a swap to the
    /// target, and [`AlgorithmError::DataNotFound`] when the component has left the market since
    /// the edge was built.
    pub fn solve(
        &self,
        price_in: f64,
        price_out: f64,
        market: &MarketState,
    ) -> Result<(EdgeFlow, usize), AlgorithmError> {
        if self.carries_no_flow(price_in, price_out) || self.max_in.is_zero() {
            return Ok((EdgeFlow::none(), 0));
        }
        let target_marginal = price_in / price_out;

        // An input the coordinator has priced at zero, or not priced at all. The subproblem is
        // unbounded there — free input, concave output — so what answers it is the
        // `EDGE_VALUE_SLACK` cap, a tuning constant rather than anything about the market. It is
        // also not a price the oracle can be asked about.
        let Some(target) = raw_price(target_marginal) else {
            return self.at_limit(price_in, price_out, market);
        };

        let sim = market
            .get_simulation_state(&self.component_id)
            .ok_or_else(|| AlgorithmError::DataNotFound {
                kind: "simulation state",
                id: Some(self.component_id.clone()),
            })?;
        let params = QueryPoolSwapParams::new(
            self.token_in.clone(),
            self.token_out.clone(),
            SwapConstraint::PoolTargetPrice {
                target,
                tolerance: TARGET_PRICE_TOLERANCE,
                min_amount_in: None,
                max_amount_in: Some(self.max_in.clone()),
            },
        );

        // The generic search is tycho's own fallback for a pool with no closed form, and it is
        // reached the same way `ComponentDepthComputation` reaches it: by the error the default
        // trait method returns.
        let queried = match sim.query_pool_swap(&params) {
            Err(SimulationError::FatalError(message))
                if message == "query_pool_swap not implemented" =>
            {
                query_pool_swap(sim, &params)
            }
            result => result,
        };

        // Two things send this to the cap, and they are the same answer.
        //
        // The query can refuse the target as unreachable, which its contract says means the
        // target is above the pool's spot price or below what the pool's limit can reach. The
        // first cannot happen here — `carries_no_flow` already compared the target against the
        // fee-net marginal, which is at or below the quoted spot the query tests — so an
        // unreachable target means the optimum is past the bracket.
        //
        // Or the query can answer past the cap, because a protocol with a closed form ignores
        // `max_amount_in` and swaps to the price it was asked for: Uniswap V3 and V4 take the
        // target and nothing else. Honouring the cap is this module's job either way, since it
        // is what keeps an infeasible intermediate iterate finite.
        //
        // `at_limit` re-simulates at the cap, so a pool that is genuinely broken rather than
        // saturated still returns its error from there.
        let swap = match queried {
            Ok(swap) if swap.amount_in() <= &self.max_in => swap,
            _ => {
                let (flow, spent) = self.at_limit(price_in, price_out, market)?;
                return Ok((flow, spent + ANALYTIC_QUERY_COST));
            }
        };

        // What the answer cost. An analytic pool reports no search history; the generic search
        // reports one price point per probe it made, which is one `get_amount_out` each.
        let sims = swap
            .price_points()
            .as_ref()
            .map_or(ANALYTIC_QUERY_COST, |points| points.len().max(ANALYTIC_QUERY_COST));

        let flow = EdgeFlow {
            amount_in: swap.amount_in().clone(),
            amount_out: swap.amount_out().clone(),
            gas: BigUint::zero(),
        };
        Ok((self.worth_trading(flow, price_in, price_out), sims))
    }

    /// The flow at the pool's own limit, for the degenerate case of an unpriced input.
    fn at_limit(
        &self,
        price_in: f64,
        price_out: f64,
        market: &MarketState,
    ) -> Result<(EdgeFlow, usize), AlgorithmError> {
        let flow = self.evaluate(&self.max_in, market)?;
        Ok((self.worth_trading(flow, price_in, price_out), 1))
    }

    /// Drops a flow worth less at these prices than not trading at all.
    ///
    /// The subproblem maximises `η₂·h(w) − η₁·w`, which is zero at zero flow, so a flow is only
    /// its solution while that stays positive. The oracle stops on a price and does not test
    /// that, and two cases make it fail: a pool whose marginal price is already at the target
    /// pays nothing for the smallest trade it will accept, and the unpriced-input branch above
    /// answers with a cap rather than an optimum.
    ///
    /// It is not a small correction. A flow that consumes the order and pays nothing conserves
    /// at every node, so it is perfectly feasible and the coordinator will prefer it to
    /// everything else; it also drives the dual value below the optimum it is supposed to bound.
    fn worth_trading(&self, flow: EdgeFlow, price_in: f64, price_out: f64) -> EdgeFlow {
        let (Some(amount_in), Some(amount_out)) =
            (flow.amount_in.to_f64(), flow.amount_out.to_f64())
        else {
            return EdgeFlow::none();
        };
        if price_out * amount_out > price_in * amount_in {
            flow
        } else {
            EdgeFlow::none()
        }
    }

    /// Simulates `amount_in` through this edge.
    fn evaluate(
        &self,
        amount_in: &BigUint,
        market: &MarketState,
    ) -> Result<EdgeFlow, AlgorithmError> {
        let sim = market
            .get_simulation_state(&self.component_id)
            .ok_or_else(|| AlgorithmError::DataNotFound {
                kind: "simulation state",
                id: Some(self.component_id.clone()),
            })?;

        let result = sim
            .get_amount_out_metered(
                &self.component_id,
                EDGE_SUBPROBLEM_STAGE,
                amount_in.clone(),
                &self.token_in,
                &self.token_out,
            )
            .map_err(|e| AlgorithmError::SimulationFailed {
                component_id: self.component_id.clone(),
                error: e.to_string(),
            })?;

        Ok(EdgeFlow { amount_in: amount_in.clone(), amount_out: result.amount, gas: result.gas })
    }
}

/// A raw price as the exact fraction [`SwapConstraint`] takes.
///
/// Mirrors `to_raw_price` in `derived::computations::component_depth` without its decimal
/// adjustment: every price in this module is already raw output units per raw input unit, which
/// is the convention the constraint wants. `None` for a price the constraint cannot carry, which
/// leaves the caller to treat the input as unpriced.
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

#[cfg(test)]
mod tests {
    use num_traits::ToPrimitive;

    use super::*;

    fn token(byte: u8, symbol: &str) -> Token {
        Token::new(
            &tycho_simulation::tycho_common::Bytes::from(vec![byte; 20]),
            symbol,
            18,
            0,
            &[Some(10_000)],
            Default::default(),
            100,
        )
    }

    fn edge(marginal_at_zero: f64, max_in: u64) -> Edge {
        Edge {
            component_id: "pool-1".to_string(),
            token_in: token(0x01, "A"),
            token_out: token(0x02, "B"),
            node_in: 0,
            node_out: 1,
            marginal_at_zero,
            max_in: BigUint::from(max_in),
        }
    }

    #[test]
    fn test_no_flow_when_input_priced_above_marginal() {
        let edge = edge(2.0, 1_000);
        // Input costs 3 output units; the pool only pays 2 at the margin.
        assert!(edge.carries_no_flow(3.0, 1.0));
    }

    #[test]
    fn test_flow_when_input_priced_below_marginal() {
        let edge = edge(2.0, 1_000);
        assert!(!edge.carries_no_flow(1.0, 1.0));
    }

    #[test]
    fn test_no_flow_at_exact_marginal() {
        // At equality the first unit breaks even, so the edge stays idle. This is the boundary
        // the paper's bid-ask reading describes.
        let edge = edge(2.0, 1_000);
        assert!(edge.carries_no_flow(2.0, 1.0));
    }

    #[test]
    fn test_no_flow_on_non_positive_output_price() {
        let edge = edge(2.0, 1_000);
        assert!(edge.carries_no_flow(1.0, 0.0));
        assert!(edge.carries_no_flow(1.0, -1.0));
    }

    #[test]
    fn test_no_flow_on_non_finite_marginal() {
        assert!(edge(f64::NAN, 1_000).carries_no_flow(1.0, 1.0));
        assert!(edge(f64::INFINITY, 1_000).carries_no_flow(1.0, 1.0));
    }

    #[test]
    fn test_edge_flow_none_is_inactive() {
        let flow = EdgeFlow::none();
        assert!(!flow.is_active());
        assert!(flow.amount_out.is_zero());
        assert!(flow.gas.is_zero());
    }

    #[test]
    fn test_edge_flow_with_input_is_active() {
        let flow = EdgeFlow {
            amount_in: BigUint::from(1u32),
            amount_out: BigUint::zero(),
            gas: BigUint::zero(),
        };
        assert!(flow.is_active());
    }

    #[test]
    fn test_raw_price_round_trips() {
        for value in [1.0f64, 2.5, 1e-9, 2e-9, 1e18, 0.3333333333333333] {
            let price = raw_price(value).expect("a positive finite price converts");
            let numerator = price.numerator.to_f64().expect("fits");
            let denominator = price
                .denominator
                .to_f64()
                .expect("fits");
            let recovered = numerator / denominator;
            assert!((recovered - value).abs() <= value * 1e-15, "{value} came back as {recovered}");
        }
    }

    #[test]
    fn test_raw_price_rejects_unusable_values() {
        assert!(raw_price(0.0).is_none());
        assert!(raw_price(-1.0).is_none());
        assert!(raw_price(f64::NAN).is_none());
        assert!(raw_price(f64::INFINITY).is_none());
    }

    #[test]
    fn test_worth_trading_drops_a_flow_that_pays_nothing() {
        let edge = edge(2.0, 1_000);
        let flow = EdgeFlow {
            amount_in: BigUint::from(10u32),
            amount_out: BigUint::zero(),
            gas: BigUint::zero(),
        };
        assert!(!edge
            .worth_trading(flow, 1.0, 1.0)
            .is_active());
    }

    #[test]
    fn test_worth_trading_keeps_a_profitable_flow() {
        let edge = edge(2.0, 1_000);
        let flow = EdgeFlow {
            amount_in: BigUint::from(10u32),
            amount_out: BigUint::from(21u32),
            gas: BigUint::zero(),
        };
        assert!(edge
            .worth_trading(flow, 1.0, 1.0)
            .is_active());
    }
}
