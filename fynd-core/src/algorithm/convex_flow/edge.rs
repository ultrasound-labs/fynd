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
//! # Where the derivative comes from
//!
//! Not from `ProtocolSim::spot_price`. That is tempting — it is one cheap call and the graph
//! already caches it — and it is the wrong quantity. Its contract is "the amount of `quote`
//! required to buy exactly 1 unit of `base`, accounting for the protocol fee (i.e.
//! `price = pre_fee_price / (1.0 - fee)`)": a *buy* price marked **up** by the fee, where the
//! derivative of `get_amount_out` on the sell side is the pre-fee price marked **down** by it.
//! Thirteen protocol families in tycho-simulation implement it through the same
//! `add_fee_markup` helper, so across the market
//!
//! ```text
//! spot_price = h⁺(0) / (1 − fee)²
//! ```
//!
//! For a 30 bp pool that is 60 bps of error in a quantity whose useful signal — the price impact
//! of an ordinary order against pool depth — is one basis point. Solving `spot_price(w) = η₁/η₂`
//! drives every edge to about `fee × reserves` of flow whatever the order is worth, which is the
//! over-trading the recorded market showed. Correcting it with `(1 − fee)²` is not available
//! either: `ProtocolSim::fee` is `todo!()` for VM pools and panics the worker.
//!
//! So the subproblem reads the derivative off the gain function it is actually optimising: a
//! difference quotient of `get_amount_out` over a step a few thousandths of the trial point.
//! That costs a second simulation per trial and is worth it twice over — it is the real curve,
//! and it is local. The secant from zero is not a substitute: `h(w)/w > η₁/η₂` says the
//! objective is still *positive* at `w`, not that it is still *rising*, and for a
//! constant-product curve its root sits at exactly twice the optimum.
//!
//! A difference quotient over `[w, w+δ]` is `h⁺(w + δ/2)` to second order, so using it as
//! `h⁺(w)` displaces the answer by `δ/2` and nothing worse. That is the whole reason to prefer
//! it to a quoted price: the error it admits is a step size this module chooses, where the
//! error a quoted price admits is a fee the market chooses, amplified by the pool's depth.
//!
//! Three things make this affordable against a worker's simulation budget. The **no-flow test**
//! rejects an edge using only its marginal price at zero trade — the spot price the graph
//! already carries — so an edge the optimum does not use mostly costs nothing; the fee markup
//! above makes that test admit a few edges it should not, which is the safe direction, since it
//! spends simulations rather than losing routes. Edges that survive it are bracketed by
//! `get_limits` before any step, so no simulation is spent discovering the search interval. And
//! the root inside that bracket is found by **regula falsi on `1/√`**, which for a
//! constant-product curve is linear in `w` and so lands on the answer in one or two simulations
//! where bisection needs twelve.
//!
//! Resolution matters as much as cost. A bracket is the whole order's worth of the input token,
//! and an edge carrying one part in ten thousand of a split is an ordinary leg, so a subproblem
//! that resolves flow to a fixed fraction of its bracket reports zero for legs the route needs.
//! Interpolation resolves a smooth curve to machine precision within the step cap; the cap is
//! there for curves it cannot predict.

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use tycho_simulation::tycho_common::models::token::Token;

use super::config::{EDGE_REL_TOLERANCE, MAX_EDGE_ITERATIONS};
use crate::{
    algorithm::{sim_meter, sim_meter::MeteredProtocolSim, AlgorithmError},
    feed::market_data::MarketState,
    types::ComponentId,
};

/// The stage every swap this module makes is booked under.
const EDGE_SUBPROBLEM_STAGE: sim_meter::StageLabel = "convex_flow_edge";

/// Denominator the interpolated bracket fraction is expressed in, so the trial point is computed
/// in integer arithmetic and a bracket too wide for `f64` cannot overflow it.
const FRACTION_SCALE: u64 = 1_000_000_000;

/// The difference quotient's step, as a fraction of the point it is taken at.
const DERIVATIVE_STEP_DIVISOR: u64 = 4096;

/// How many raw output units the difference quotient must span to be worth believing.
///
/// Outputs are integers, so a quotient whose numerator is a handful of units carries a rounding
/// error of a sizeable fraction of itself, and a slope wrong by tens of percent idles an edge
/// that should carry flow. Below this the subproblem falls back on the average rate from zero.
/// Any real order is orders of magnitude clear of it: a leg paying 64 raw units of an 18-decimal
/// token is 6.4e-17 of one token.
const MIN_RESOLVED_OUTPUT_UNITS: f64 = 64.0;

/// One directed edge of the flow network: a pool, traded one way.
///
/// Both directions of a pool are separate edges, each with flow `w ≥ 0`. Modelling it this way
/// means only the `h⁺(0)` side of the no-flow interval is ever tested — the `h⁻(0)` side is the
/// reverse edge's own test — and the search never has to consider negative flow.
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
    /// The pool's quoted spot price, token_out per token_in, at zero trade.
    ///
    /// Taken from the graph's edge weight, which the derived-data pass refreshes every block, so
    /// reading it costs no simulation. It is an *upper bound* on `h⁺(0)` and not `h⁺(0)` itself —
    /// `spot_price = h⁺(0) / (1 − fee)²`, see the module docs — so it filters edges and guides
    /// interpolation, and never decides how much an edge carries.
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

/// Which end of the bracket the last accepted step moved.
///
/// Regula falsi keeps one endpoint for many steps when the curve is not the shape the transform
/// assumes, and then converges linearly from one side. The Illinois safeguard halves the stale
/// endpoint's residual once it has been kept twice, which pulls the next interpolation across.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stale {
    Neither,
    Low,
    High,
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
    /// The search keeps a bracket whose low end is the largest input whose increment paid for
    /// itself and whose high end is the smallest that did not, both evaluated, starting from
    /// `(0, 0)` and the pool's own limit. Each step tests one trial point by the secant from the
    /// low end: `η₂·(h(trial) − h(low)) > η₁·(trial − low)`. Accepting on that test means every
    /// returned flow has a strictly higher subproblem objective than the point before it, and
    /// the first point is zero flow, so what comes back is always worth trading.
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
        // An input the coordinator has priced at zero, or not priced at all. The subproblem is
        // unbounded there — free input, concave output — so what answers it is the
        // `EDGE_VALUE_SLACK` cap, a tuning constant rather than anything about the market.
        //
        // That is worth stating plainly, because the cap is doing more work than the dual is.
        // Every toy solve in this module's tests reaches its answer through this branch: the
        // coordinator drives the input price to zero, the edge offers its whole cap, and
        // `decompose` clamps that back to the order — which on a single-pool market is exactly
        // "send everything" and is why those tests pass. Returning no flow here instead, which
        // is the defensible reading of an unbounded subproblem, fails five of them.
        //
        // It is also why the real market is not helped by making this branch cheap. Walking to
        // the cap through the search costs twenty-four simulations and reaching it directly
        // costs one, and on the recorded market that saving raises the solve rate by half while
        // tripling the orders it loses on, because the budget it frees is spent on further
        // excursions to a zero price. The sim budget was acting as an accidental regulariser.
        if target_marginal <= 0.0 || target_marginal.is_nan() {
            return self.at_limit(price_in, price_out, market);
        }

        // The bracket is known before any simulation: zero is always feasible and the pool's
        // own limit bounds the other end.
        let mut low = BigUint::zero();
        let mut high = self.max_in.clone();
        let (high_flow, high_marginal, mut sims) = self.marginal_at(&high, market)?;

        if high_marginal > target_marginal {
            // The whole bracket is still worth feeding, so the limit is the optimum. A deep pool
            // against a small order lands here, and the bracket is the order's own value, so
            // this is the cap binding rather than the pool saturating.
            return Ok((self.worth_trading(high_flow, price_in, price_out), sims));
        }

        // `marginal_at_zero` bounds `h⁺(0)` from above, so it is a usable guide for the first
        // interpolation even though it is not the derivative.
        let mut low_residual = residual(self.marginal_at_zero, target_marginal);
        let mut high_residual = residual(high_marginal, target_marginal);
        let mut best: Option<EdgeFlow> = None;
        let mut stale = Stale::Neither;

        let tolerance = self.bracket_tolerance();
        for _ in 0..MAX_EDGE_ITERATIONS {
            if &high - &low <= tolerance {
                break;
            }
            let trial = interpolate(&low, &high, low_residual, high_residual);
            let (flow, marginal, spent) = self.marginal_at(&trial, market)?;
            sims += spent;

            // `h⁺` is non-increasing in `w` because `h` is concave: while the marginal at the
            // trial still beats the price ratio, every unit up to it was worth taking, so the
            // optimum is at or above it.
            if marginal > target_marginal {
                low = trial;
                low_residual = residual(marginal, target_marginal);
                best = Some(flow);
                if stale == Stale::Low {
                    high_residual = high_residual.map(|r| r * 0.5);
                }
                stale = Stale::Low;
            } else {
                high = trial;
                high_residual = residual(marginal, target_marginal);
                if stale == Stale::High {
                    low_residual = low_residual.map(|r| r * 0.5);
                }
                stale = Stale::High;
            }
        }
        match best {
            Some(flow) => Ok((self.worth_trading(flow, price_in, price_out), sims)),
            None => Ok((EdgeFlow::none(), sims)),
        }
    }

    /// The flow at `amount_in` and the curve's slope there, with the simulations it cost.
    ///
    /// The slope is a difference quotient of `get_amount_out` over `DERIVATIVE_STEP_DIVISOR` of
    /// the point, taken forward where the pool's limit allows it and backward at the limit
    /// itself. Two simulations, and the second one is what makes the quantity the derivative of
    /// the function being optimised rather than a price quoted alongside it.
    fn marginal_at(
        &self,
        amount_in: &BigUint,
        market: &MarketState,
    ) -> Result<(EdgeFlow, f64, usize), AlgorithmError> {
        let (flow, out) = self.evaluate(amount_in, market)?;
        let step = self.derivative_step(amount_in);
        let forward = amount_in + &step;

        let (probe, backward) =
            if forward <= self.max_in { (forward, false) } else { (amount_in - &step, true) };
        let (_, probe_out_value) = self.evaluate(&probe, market)?;
        let span = step
            .to_f64()
            .filter(|s| s.is_finite() && *s > 0.0)
            .ok_or_else(|| AlgorithmError::Other("derivative step does not fit a float".into()))?;
        let difference = match backward {
            false => probe_out_value - out,
            true => out - probe_out_value,
        };

        // A pool whose output barely moves across the step cannot be differentiated here: its
        // integer rounding is a large fraction of the difference. That is a property of the
        // amounts rather than of the pool, and no real order is near it, so rather than trust a
        // slope that is tens of percent wrong — which idles an edge that should carry flow —
        // fall back on the average rate from zero. By concavity that is an upper bound on the
        // slope, so a market too coarse to measure gets over-trading instead of abandonment, and
        // `Self::worth_trading` still holds the result to paying for itself.
        let marginal = if difference.abs() < MIN_RESOLVED_OUTPUT_UNITS {
            let amount = amount_in
                .to_f64()
                .filter(|a| a.is_finite() && *a > 0.0);
            match amount {
                Some(amount) => out / amount,
                None => 0.0,
            }
        } else {
            difference / span
        };
        Ok((flow, marginal, 2))
    }

    /// How far apart the two points of the difference quotient sit.
    ///
    /// A few thousandths of the trial point, floored at one raw unit since amounts are integers.
    /// Small enough that displacing the answer by half of it does not matter, large enough that
    /// the output difference is not lost to the pool's own integer rounding.
    fn derivative_step(&self, amount_in: &BigUint) -> BigUint {
        (amount_in / BigUint::from(DERIVATIVE_STEP_DIVISOR)).max(BigUint::one())
    }

    /// The flow at the pool's own limit, for the degenerate case of an unpriced input.
    fn at_limit(
        &self,
        price_in: f64,
        price_out: f64,
        market: &MarketState,
    ) -> Result<(EdgeFlow, usize), AlgorithmError> {
        let (flow, _) = self.evaluate(&self.max_in, market)?;
        Ok((self.worth_trading(flow, price_in, price_out), 1))
    }

    /// Drops a flow worth less at these prices than not trading at all.
    ///
    /// The subproblem maximises `η₂·h(w) − η₁·w`, which is zero at zero flow, so a flow is only
    /// its solution while that stays positive. The search above accepts only steps that raise
    /// it, so this is the backstop for the two paths that do not go through a step: the
    /// bracket-wide secant, and the free-input case in [`Self::at_limit`].
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

    /// Smallest input difference the root finding bothers to resolve.
    ///
    /// Scaled off the edge's own limit so a pool quoting in 6-decimal units and one quoting in
    /// 18 get the same relative precision, with a floor of one raw unit since amounts are
    /// integers.
    fn bracket_tolerance(&self) -> BigUint {
        let scaled = self
            .max_in
            .to_f64()
            .map(|limit| limit * EDGE_REL_TOLERANCE)
            .filter(|t| t.is_finite() && *t >= 1.0)
            .map(|t| BigUint::from(t as u128));
        scaled.unwrap_or_else(BigUint::one)
    }

    /// Simulates `amount_in` through this edge, returning the flow and its output as a float.
    ///
    /// No spot price is read here. The subproblem's derivative comes from secants of this
    /// function, for the reason in the module docs.
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

        let amount_out = result
            .amount
            .to_f64()
            .filter(|out| out.is_finite())
            .ok_or_else(|| AlgorithmError::Other("pool output does not fit a float".to_string()))?;
        let flow =
            EdgeFlow { amount_in: amount_in.clone(), amount_out: result.amount, gas: result.gas };
        Ok((flow, amount_out))
    }
}

/// How far a marginal price sits from the target, in the coordinate the interpolation is linear
/// in. Negative while the edge is still worth feeding, positive once it is past its optimum.
///
/// That coordinate is `1/√h⁺`. For a constant-product curve `h⁺(w) = R_in·R_out/(R_in+w)²`, so
/// `1/√h⁺(w) = (R_in+w)/√(R_in·R_out)` is exactly linear in `w` and one interpolation lands on
/// the root; a tick curve is this shape piecewise, and a weighted or VM curve is monotone in the
/// same direction, which is all regula falsi needs. `None` for a price the pool cannot quote,
/// which leaves the next step to the halving safeguard.
fn residual(marginal: f64, target_marginal: f64) -> Option<f64> {
    if !(marginal.is_finite() && marginal > 0.0) ||
        !(target_marginal.is_finite() && target_marginal > 0.0)
    {
        return None;
    }
    Some(marginal.sqrt().recip() - target_marginal.sqrt().recip())
}

/// The next trial point: where the secant through the bracket's two residuals crosses zero,
/// halving the bracket instead whenever that is not a usable interior point.
///
/// Computed as an integer fraction of the bracket so a bracket wider than `f64` can represent is
/// still split exactly, and clamped strictly inside so a step always makes progress.
fn interpolate(
    low: &BigUint,
    high: &BigUint,
    low_residual: Option<f64>,
    high_residual: Option<f64>,
) -> BigUint {
    let span = high - low;
    let fraction = match (low_residual, high_residual) {
        (Some(below), Some(above)) => (-below) / (above - below),
        _ => 0.5,
    };
    let scaled = if fraction.is_finite() && fraction > 0.0 && fraction < 1.0 {
        (fraction * FRACTION_SCALE as f64) as u64
    } else {
        FRACTION_SCALE / 2
    };
    let offset = (&span * BigUint::from(scaled.clamp(1, FRACTION_SCALE - 1))) /
        BigUint::from(FRACTION_SCALE);
    // A span of one raw unit never reaches here: the tolerance test stops first, and it floors
    // at one. So both clamps leave a point strictly inside the bracket.
    low + offset
        .max(BigUint::one())
        .min(&span - BigUint::one())
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
    fn test_bracket_tolerance_scales_with_limit() {
        // 1e18 raw units at 1e-9 relative precision leaves a 1e9-unit bracket.
        let large = edge(2.0, u64::MAX).bracket_tolerance();
        assert!(large > BigUint::one());
    }

    #[test]
    fn test_bracket_tolerance_floors_at_one_unit() {
        // A tiny limit would scale below a single raw unit, which is not a resolvable amount.
        assert_eq!(edge(2.0, 10).bracket_tolerance(), BigUint::one());
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
    fn test_residual_sign_follows_the_optimum() {
        // Above the target the edge is still worth feeding, which is the negative side.
        assert!(residual(4.0, 1.0).expect("priceable") < 0.0);
        assert!(residual(1.0, 4.0).expect("priceable") > 0.0);
        assert_eq!(residual(2.0, 2.0), Some(0.0));
    }

    #[test]
    fn test_residual_rejects_unquotable_prices() {
        assert_eq!(residual(0.0, 1.0), None);
        assert_eq!(residual(f64::NAN, 1.0), None);
        assert_eq!(residual(1.0, -1.0), None);
    }

    #[test]
    fn test_interpolate_lands_on_the_root_of_a_linear_residual() {
        // Residuals -1 at 0 and +3 at 4000 put the root a quarter of the way along.
        let trial = interpolate(&BigUint::zero(), &BigUint::from(4_000u32), Some(-1.0), Some(3.0));
        assert_eq!(trial, BigUint::from(1_000u32));
    }

    #[test]
    fn test_interpolate_halves_without_usable_residuals() {
        let trial = interpolate(&BigUint::zero(), &BigUint::from(1_000u32), None, Some(1.0));
        assert_eq!(trial, BigUint::from(500u32));
        // Residuals that do not straddle zero would extrapolate outside the bracket.
        let trial = interpolate(&BigUint::zero(), &BigUint::from(1_000u32), Some(1.0), Some(3.0));
        assert_eq!(trial, BigUint::from(500u32));
    }

    #[test]
    fn test_interpolate_stays_inside_the_bracket() {
        // A root all but on an endpoint must still move the trial off it, or the search stalls.
        let span = 1_000_000u32;
        let trial = interpolate(&BigUint::zero(), &BigUint::from(span), Some(-1e-18), Some(1.0));
        assert!(trial > BigUint::zero() && trial < BigUint::from(span));
        let trial = interpolate(&BigUint::zero(), &BigUint::from(span), Some(-1.0), Some(1e-18));
        assert!(trial > BigUint::zero() && trial < BigUint::from(span));
    }

    #[test]
    fn test_interpolate_splits_a_bracket_wider_than_f64() {
        // Integer arithmetic throughout: a span past f64's range still splits exactly.
        let high = BigUint::from(1u32) << 1_200u32;
        let trial = interpolate(&BigUint::zero(), &high, Some(-1.0), Some(1.0));
        assert_eq!(trial, &high / 2u32);
    }
}
