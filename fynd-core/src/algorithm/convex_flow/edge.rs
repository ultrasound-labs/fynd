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
//! where `h⁺` and `h⁻` are the right and left derivatives. The derivatives are not assumed to
//! agree: a tick AMM is piecewise and need not be differentiable where the optimum lands.
//!
//! Two things make this affordable against a worker's simulation budget. The **no-flow test**
//! rejects an edge using only its marginal price at zero trade — the fee-inclusive spot price
//! the graph already carries — so an edge the optimum does not use costs nothing. Edges that
//! survive it are bracketed by `get_limits` before any bisection step, so no simulation is spent
//! discovering the search interval.

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use tycho_simulation::tycho_common::models::token::Token;

use super::config::{BISECTION_REL_TOLERANCE, MAX_BISECTION_STEPS};
use crate::{
    algorithm::{sim_meter, sim_meter::MeteredProtocolSim, AlgorithmError},
    feed::market_data::MarketState,
    types::ComponentId,
};

/// The stage every swap this module makes is booked under.
const EDGE_SUBPROBLEM_STAGE: sim_meter::StageLabel = "convex_flow_edge";

/// One directed edge of the flow network: a pool, traded one way.
///
/// Both directions of a pool are separate edges, each with flow `w ≥ 0`. Modelling it this way
/// means only the `h⁺(0)` side of the no-flow interval is ever tested — the `h⁻(0)` side is the
/// reverse edge's own test — and the bisection never has to consider negative flow.
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
    /// `h⁺(0)`: the fee-inclusive marginal price, token_out per token_in, at zero trade.
    ///
    /// Taken from the graph's edge weight, which the derived-data pass refreshes every block, so
    /// reading it costs no simulation.
    pub marginal_at_zero: f64,
    /// Largest input the pool accepts for this pair, from `get_limits`.
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
    /// Whether the optimum leaves this edge unused, decided without simulating.
    ///
    /// Flow is worth starting only while the output is worth more than the input it consumes,
    /// i.e. while `η₂·h⁺(0) > η₁`. Written as a price ratio that is the paper's no-flow
    /// condition for the forward direction: the edge sits idle whenever `η₁/η₂ ≥ h⁺(0)`.
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
    /// Returns the flow the edge wants to carry, and how many simulations it took. A caller
    /// tracking a budget adds the count; an edge rejected by [`Self::carries_no_flow`] costs
    /// nothing and never reaches here.
    ///
    /// # Errors
    ///
    /// Returns [`AlgorithmError::SimulationFailed`] when the pool cannot price a trade it
    /// admitted through `get_limits`, and [`AlgorithmError::DataNotFound`] when the component
    /// has left the market since the edge was built.
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

        // The bracket is known before any simulation: zero is always feasible and the pool's own
        // limit bounds the other end.
        let mut low = BigUint::zero();
        let mut high = self.max_in.clone();
        let mut best = EdgeFlow::none();
        let mut sims = 0usize;

        let tolerance = self.bisection_tolerance();
        for _ in 0..MAX_BISECTION_STEPS {
            if &high - &low <= tolerance {
                break;
            }
            let mid = (&low + &high) >> 1u32;
            if mid.is_zero() {
                break;
            }
            let (flow, marginal) = self.evaluate(&mid, market)?;
            sims += 1;

            // `h⁺` is non-increasing in `w` because `h` is concave: while the marginal price at
            // `mid` still beats the price ratio, every unit up to `mid` was worth taking, so the
            // optimum is at or above it.
            if marginal > target_marginal {
                best = flow;
                low = mid;
            } else {
                high = mid;
            }
        }

        // The whole bracket may clear the target, in which case the pool's limit is the optimum
        // and the loop never evaluated it.
        if !best.is_active() && !low.is_zero() {
            let (flow, _) = self.evaluate(&low, market)?;
            sims += 1;
            best = flow;
        }
        Ok((best, sims))
    }

    /// Smallest input difference bisection bothers to resolve.
    ///
    /// Scaled off the edge's own limit so a pool quoting in 6-decimal units and one quoting in
    /// 18 get the same relative precision, with a floor of one raw unit since amounts are
    /// integers.
    fn bisection_tolerance(&self) -> BigUint {
        let scaled = self
            .max_in
            .to_f64()
            .map(|limit| limit * BISECTION_REL_TOLERANCE)
            .filter(|t| t.is_finite() && *t >= 1.0)
            .map(|t| BigUint::from(t as u128));
        scaled.unwrap_or_else(BigUint::one)
    }

    /// Simulates `amount_in` through this edge, returning the flow and the pool's marginal price
    /// after the trade — the `h⁺(w)` the optimality condition compares against.
    fn evaluate(
        &self,
        amount_in: &BigUint,
        market: &MarketState,
    ) -> Result<(EdgeFlow, f64), AlgorithmError> {
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

        // The post-trade state's spot price is the marginal price at this point on the curve.
        // This is the derivative the convex literature assumes a closed-form invariant for, and
        // the reason a black-box VM pool can be routed through here at all.
        let marginal = result
            .new_state
            .spot_price(&self.token_in, &self.token_out)
            .map_err(|e| AlgorithmError::SimulationFailed {
                component_id: self.component_id.clone(),
                error: e.to_string(),
            })?;

        let flow =
            EdgeFlow { amount_in: amount_in.clone(), amount_out: result.amount, gas: result.gas };
        Ok((flow, marginal))
    }
}

#[cfg(test)]
mod tests {
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
    fn test_bisection_tolerance_scales_with_limit() {
        // 1e18 raw units at 1e-9 relative precision leaves a 1e9-unit bracket.
        let large = edge(2.0, u64::MAX).bisection_tolerance();
        assert!(large > BigUint::one());
    }

    #[test]
    fn test_bisection_tolerance_floors_at_one_unit() {
        // A tiny limit would scale below a single raw unit, which is not a resolvable amount.
        assert_eq!(edge(2.0, 10).bisection_tolerance(), BigUint::one());
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
}
