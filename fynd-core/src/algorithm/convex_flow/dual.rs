//! The dual problem and the coordinator that minimises it.
//!
//! # The problem
//!
//! For an exact-input sell order the network utility is `U(y) = y_out − I_S(y)` over
//! `S = { y : y_in ≥ −amount, y_j ≥ 0 for j ≠ in }`, and every edge utility is zero — the
//! configuration the convex flow paper identifies as the CFMM routing case. Its conjugate is
//!
//! ```text
//! Ū(ν) = ν_in · amount      finite only where ν ≥ 0 and ν_out ≥ 1
//! ```
//!
//! which leaves the reduced dual with **one variable per token**, not two per pool:
//!
//! ```text
//! minimise  ν_in · amount + Σᵢ fᵢ(Aᵢᵀν)      subject to ν ≥ 0, ν_out = 1
//! ```
//!
//! `ν_out` is pinned rather than merely bounded: the objective is decreasing in it, so the bound
//! is always active, and fixing it also removes the formulation's scale invariance. Every other
//! price is then denominated in `token_out`.
//!
//! # The gradient
//!
//! `∇fᵢ` is the edge's own optimal flow, so the gradient needs no differentiation — it falls out
//! of the subproblems the objective already solved:
//!
//! ```text
//! ∂g/∂ν_j = amount·[j = in] + Σ_{e: out(e)=j} h_e(w⋆) − Σ_{e: in(e)=j} w⋆_e
//! ```
//!
//! which is net flow at node `j`. Driving it to zero *is* flow conservation, and that is why the
//! coordinator needs only a gradient and never a Jacobian.
//!
//! # The coordinator
//!
//! Spectral projected gradient: Barzilai–Borwein step lengths with a non-monotone Armijo line
//! search, projected onto `ν ≥ 0`. BB gets most of the practical benefit of a quasi-Newton
//! method on this problem while needing no curvature history and no new dependency, which
//! matters more here than the last few percent of convergence rate — the budget is spent in the
//! edge subproblems, not in the coordinator's arithmetic.

use num_bigint::BigUint;
use num_traits::ToPrimitive;
use tracing::debug;

use super::{
    super::convex_flow::Deadline,
    config::{
        ARMIJO_COEFFICIENT, BACKTRACK_FACTOR, INITIAL_STEP, LINE_SEARCH_MEMORY, MAX_BACKTRACKS,
        MAX_EDGE_SOLVES, MAX_ITERATIONS, MAX_STEP, MIN_STEP, STATIONARITY_TOLERANCE,
    },
    edge::{Edge, EdgeFlow},
};
use crate::{algorithm::AlgorithmError, feed::market_data::MarketState};

/// One evaluation of the dual: its value, its gradient, and the flows behind both.
pub(crate) struct DualPoint {
    /// Dual objective value.
    pub value: f64,
    /// Net flow at each node, which is the dual gradient.
    pub gradient: Vec<f64>,
    /// What each edge wanted to carry at these prices, parallel to the edge list.
    pub flows: Vec<EdgeFlow>,
    /// Sum of squared net flows, as a scale-free feasibility measure.
    pub residual: f64,
    /// Whether every edge was solved, or the simulation budget cut the pass short.
    ///
    /// An incomplete pass leaves the unreached edges idle, so its flows are honest — they are a
    /// flow the network could carry — but its *value* is missing those edges' contributions and
    /// comes out lower than it should. Values from passes over different edge sets are not
    /// comparable, which is what this flag protects the line search from.
    pub complete: bool,
}

/// The point the coordinator will return, ranked by primal feasibility first.
///
/// Ranking on the dual value is the obvious reading of "best so far" and is the wrong one here.
/// The dual value bounds the order's output, so over the whole region the iterates visit it
/// varies by the slippage the order pays — basis points — while the net-flow residual over that
/// same region varies by eight orders of magnitude. The route is built from the flows, and flows
/// that do not conserve cannot be walked into legs, so the comparison has to be able to see that
/// difference: feasibility first, and then the dual value, which is the bound on the order's
/// output and so the right way to choose between two flows that both conserve.
#[derive(Clone, Copy)]
struct Candidate {
    residual: f64,
    value: f64,
    carries_flow: bool,
}

impl Candidate {
    fn of(point: &DualPoint) -> Self {
        Self {
            residual: point.residual,
            value: point.value,
            carries_flow: point
                .flows
                .iter()
                .any(EdgeFlow::is_active),
        }
    }

    /// Whether the point conserves well enough for its dual value to be the thing that matters.
    ///
    /// The threshold is [`STATIONARITY_TOLERANCE`] squared because the residual is a sum of
    /// squared net flows and the tolerance is a norm, so this is the same condition the
    /// coordinator stops on.
    fn conserves(&self) -> bool {
        self.residual <= STATIONARITY_TOLERANCE * STATIONARITY_TOLERANCE
    }

    /// Whether this point is a better one to return than `other`.
    ///
    /// A point where every edge is idle comes first for conservation — the whole order unspent
    /// is in perfect balance — and it is the one point that provably cannot become a route,
    /// since there is no flow to walk from `token_in` to `token_out`. On the recorded market the
    /// iterates spend most of their time alternating between that point and a saturated one, so
    /// ranking on the residual alone returns the idle point and the solve reports no route where
    /// a flow that over-trades would have clamped into a real, honestly-priced one. Carrying
    /// flow therefore outranks conserving, and the residual decides among the points that do.
    fn improves_on(&self, other: &Self) -> bool {
        match (self.carries_flow, other.carries_flow) {
            (true, false) => return true,
            (false, true) => return false,
            _ => {}
        }
        match (self.conserves(), other.conserves()) {
            (true, true) => self.value < other.value,
            (true, false) => true,
            (false, true) => false,
            (false, false) => self.residual < other.residual,
        }
    }
}

/// The reduced dual for one order over one edge set.
pub(crate) struct DualProblem<'a> {
    edges: &'a [Edge],
    num_nodes: usize,
    node_in: usize,
    node_out: usize,
    amount_in: f64,
    /// `ν⁰`, the warm-start prices that define each token's unit. Never zero.
    reference: Vec<f64>,
    /// The order's value in output-token units, `amount · ν⁰_in`. Divides the objective and the
    /// gradient so both are dimensionless.
    value_scale: f64,
}

/// What the coordinator settled on.
pub(crate) struct Solution {
    /// Edge flows at the best point found.
    pub flows: Vec<EdgeFlow>,
    /// Coordinator iterations run.
    pub iterations: usize,
    /// Edge subproblems solved, the solve's real cost.
    pub edge_solves: usize,
    /// Net-flow residual at the returned point.
    pub residual: f64,
}

impl<'a> DualProblem<'a> {
    /// A dual over `edges`, selling `amount_in` of the token at `node_in` for the one at
    /// `node_out`.
    pub fn new(
        edges: &'a [Edge],
        num_nodes: usize,
        node_in: usize,
        node_out: usize,
        amount_in: &BigUint,
        reference: Vec<f64>,
    ) -> Result<Self, AlgorithmError> {
        let amount = amount_in
            .to_f64()
            .filter(|a| a.is_finite() && *a > 0.0)
            .ok_or_else(|| {
                AlgorithmError::Other("order amount does not fit a float".to_string())
            })?;
        // A zero or unusable reference price would divide the whole scheme by zero, so each one
        // falls back to the output token's unit rather than failing the solve.
        let mut reference = reference;
        reference.resize(num_nodes, 1.0);
        for price in &mut reference {
            if !(price.is_finite() && *price > 0.0) {
                *price = 1.0;
            }
        }
        let value_scale = (amount * reference[node_in]).max(f64::MIN_POSITIVE);
        Ok(Self { edges, num_nodes, node_in, node_out, amount_in: amount, reference, value_scale })
    }

    /// The starting point: every node at its reference price.
    pub fn start(&self) -> Vec<f64> {
        vec![1.0; self.num_nodes]
    }

    /// Evaluates the dual and its gradient at `prices`, solving every edge that is not rejected
    /// by the no-flow test.
    ///
    /// `budget` is decremented by the simulations spent; evaluation stops early once it is
    /// exhausted, leaving the remaining edges idle. That degrades the answer rather than failing
    /// the solve, which is what the worker contract asks for.
    pub fn evaluate(
        &self,
        scaled_prices: &[f64],
        market: &MarketState,
        budget: &mut usize,
    ) -> Result<DualPoint, AlgorithmError> {
        // The edge subproblems price in real units; only the coordinator sees `u`.
        let prices: Vec<f64> = scaled_prices
            .iter()
            .zip(&self.reference)
            .map(|(u, reference)| u * reference)
            .collect();

        let mut net_flow = vec![0.0; self.num_nodes];
        let mut flows = Vec::with_capacity(self.edges.len());
        let mut complete = true;
        let mut value = prices[self.node_in] * self.amount_in;
        net_flow[self.node_in] = self.amount_in;

        for edge in self.edges {
            let flow = if *budget == 0 {
                complete = false;
                EdgeFlow::none()
            } else {
                let (flow, sims) =
                    edge.solve(prices[edge.node_in], prices[edge.node_out], market)?;
                *budget = budget.saturating_sub(sims);
                flow
            };

            if flow.is_active() {
                let w = flow.amount_in.to_f64().unwrap_or(0.0);
                let out = flow.amount_out.to_f64().unwrap_or(0.0);
                value += prices[edge.node_out] * out - prices[edge.node_in] * w;
                net_flow[edge.node_in] -= w;
                net_flow[edge.node_out] += out;
            }
            flows.push(flow);
        }

        // Net flow per node is the gradient in `ν`; the chain rule to `u` multiplies by `ν⁰`,
        // which also values every node in the output token. Dividing by the order's value makes
        // the components comparable with each other and with a step length of one.
        let mut gradient = vec![0.0; self.num_nodes];
        for (node, raw) in net_flow.iter().enumerate() {
            gradient[node] = raw * self.reference[node] / self.value_scale;
        }
        gradient[self.node_out] = 0.0;
        let residual = gradient
            .iter()
            .map(|g| g * g)
            .sum::<f64>();
        Ok(DualPoint { value: value / self.value_scale, gradient, flows, residual, complete })
    }

    /// Minimises the dual from `start`, returning the most feasible point it reached.
    ///
    /// Stops on convergence, on the iteration cap, on the simulation budget, or on the deadline —
    /// and in every case returns the best point seen rather than an error, because a worker that
    /// runs out of time still owes the router a route. "Best" is [`Candidate`]'s ordering, and
    /// every point the solver evaluates is a candidate, line-search trials included: the
    /// backtracking is what crosses the region where edges first take flow, so that is where the
    /// feasible points are found.
    pub fn minimise(
        &self,
        start: Vec<f64>,
        market: &MarketState,
        deadline: &Deadline,
    ) -> Result<Solution, AlgorithmError> {
        let mut budget = MAX_EDGE_SOLVES;
        let mut prices = self.project(start);
        let mut point = self.evaluate(&prices, market, &mut budget)?;

        let mut best = Candidate::of(&point);
        let mut best_flows = point.flows.clone();

        let mut step = INITIAL_STEP;
        let mut history = vec![point.value];
        let mut iterations = 0usize;

        for iteration in 0..MAX_ITERATIONS {
            iterations = iteration + 1;
            if deadline.expired() || budget == 0 {
                debug!(iteration, budget, "convex flow stopping early");
                break;
            }

            // Projected steepest-descent direction at the current BB step length.
            let trial = self.project(
                prices
                    .iter()
                    .zip(&point.gradient)
                    .map(|(p, g)| p - step * g)
                    .collect(),
            );
            let direction: Vec<f64> = trial
                .iter()
                .zip(&prices)
                .map(|(t, p)| t - p)
                .collect();
            let slope: f64 = direction
                .iter()
                .zip(&point.gradient)
                .map(|(d, g)| d * g)
                .sum();
            if !slope.is_finite() {
                debug!(iteration, "convex flow gradient is not finite");
                break;
            }
            // Stationarity is tested at a fixed reference step, not at the current one: the
            // Barzilai-Borwein length shrinks as iterates settle, which shrinks `direction` and
            // with it `slope`, so a test on `slope` reports convergence whenever the step gets
            // small whatever the gradient still says.
            if self.stationarity(&prices, &point.gradient) <= STATIONARITY_TOLERANCE {
                debug!(iteration, "convex flow converged");
                break;
            }

            // Non-monotone Armijo: accept against the worst recent value so the iterates can
            // cross a ridge instead of stalling on it.
            let reference = history
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut alpha = 1.0;
            let mut accepted = None;
            for _ in 0..MAX_BACKTRACKS {
                let candidate: Vec<f64> = prices
                    .iter()
                    .zip(&direction)
                    .map(|(p, d)| p + alpha * d)
                    .collect();
                let evaluated = self.evaluate(&candidate, market, &mut budget)?;
                if Candidate::of(&evaluated).improves_on(&best) {
                    best = Candidate::of(&evaluated);
                    best_flows = evaluated.flows.clone();
                }
                if !evaluated.complete {
                    debug!(iteration, "convex flow ran out of simulations mid line search");
                    break;
                }
                if evaluated.value <= reference + ARMIJO_COEFFICIENT * alpha * slope {
                    accepted = Some((candidate, evaluated));
                    break;
                }
                if deadline.expired() || budget == 0 {
                    break;
                }
                alpha *= BACKTRACK_FACTOR;
            }
            let Some((next_prices, next_point)) = accepted else {
                debug!(iteration, "convex flow line search exhausted");
                break;
            };

            // Barzilai-Borwein step for the next iteration, from the secant pair this one made.
            let s: Vec<f64> = next_prices
                .iter()
                .zip(&prices)
                .map(|(n, p)| n - p)
                .collect();
            let y: Vec<f64> = next_point
                .gradient
                .iter()
                .zip(&point.gradient)
                .map(|(n, p)| n - p)
                .collect();
            let ss: f64 = s.iter().map(|v| v * v).sum();
            let sy: f64 = s
                .iter()
                .zip(&y)
                .map(|(a, b)| a * b)
                .sum();
            step = if sy > 0.0 { (ss / sy).clamp(MIN_STEP, MAX_STEP) } else { MAX_STEP };

            prices = next_prices;
            point = next_point;

            history.push(point.value);
            if history.len() > LINE_SEARCH_MEMORY {
                history.remove(0);
            }
        }

        Ok(Solution {
            flows: best_flows,
            iterations,
            edge_solves: MAX_EDGE_SOLVES - budget,
            residual: best.residual,
        })
    }

    /// How far the iterate is from stationary, by a projected gradient step of fixed length one.
    fn stationarity(&self, prices: &[f64], gradient: &[f64]) -> f64 {
        let projected = self.project(
            prices
                .iter()
                .zip(gradient)
                .map(|(p, g)| p - g)
                .collect(),
        );
        projected
            .iter()
            .zip(prices)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// Projects onto the feasible set: every price non-negative, the output token's pinned at
    /// one. A non-finite coordinate is reset rather than propagated, so one bad subproblem
    /// cannot poison the whole vector.
    fn project(&self, mut prices: Vec<f64>) -> Vec<f64> {
        prices.resize(self.num_nodes, 0.0);
        for (index, price) in prices.iter_mut().enumerate() {
            if !price.is_finite() || *price < 0.0 {
                *price = 0.0;
            }
            if index == self.node_out {
                *price = 1.0;
            }
        }
        prices
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn problem(edges: &[Edge]) -> DualProblem<'_> {
        DualProblem::new(edges, 3, 0, 2, &BigUint::from(1_000u32), vec![1.0, 1.0, 1.0])
            .expect("amount fits")
    }

    #[test]
    fn test_project_pins_output_price_and_clamps_negatives() {
        let edges: Vec<Edge> = Vec::new();
        let dual = problem(&edges);
        let projected = dual.project(vec![-5.0, 2.0, 0.25]);
        assert_eq!(projected[0], 0.0, "negative price clamped to zero");
        assert_eq!(projected[1], 2.0, "free price untouched");
        assert_eq!(projected[2], 1.0, "output price pinned to one");
    }

    #[test]
    fn test_project_resets_non_finite_prices() {
        let edges: Vec<Edge> = Vec::new();
        let dual = problem(&edges);
        let projected = dual.project(vec![f64::NAN, f64::INFINITY, 1.0]);
        assert_eq!(projected[0], 0.0);
        assert_eq!(projected[1], 0.0);
    }

    #[test]
    fn test_project_resizes_short_vector() {
        let edges: Vec<Edge> = Vec::new();
        let dual = problem(&edges);
        let projected = dual.project(vec![1.0]);
        assert_eq!(projected.len(), 3, "a short warm start is padded, not rejected");
    }

    #[test]
    fn test_amount_rejects_non_representable_order() {
        let edges: Vec<Edge> = Vec::new();
        // 2^4096 overflows f64 and must be refused rather than silently become infinity.
        let huge = BigUint::from(1u32) << 4096u32;
        assert!(DualProblem::new(&edges, 2, 0, 1, &huge, vec![1.0, 1.0]).is_err());
    }

    #[test]
    fn test_empty_edge_set_grades_to_input_amount() {
        let edges: Vec<Edge> = Vec::new();
        let dual = problem(&edges);
        let market = MarketState::default();
        let mut budget = 10;
        let point = dual
            .evaluate(&[1.0, 1.0, 1.0], &market, &mut budget)
            .expect("no edges to evaluate");
        // With no edges the dual is Ū(ν) = ν_in · amount, and the gradient at token_in is the
        // whole unspent order — both divided by the order's own value, so both are exactly one.
        assert_eq!(point.value, 1.0);
        assert_eq!(point.gradient[0], 1.0, "a wholly unspent order is a unit of net flow");
        assert_eq!(point.gradient[2], 0.0, "pinned coordinate never steers");
        assert_eq!(budget, 10, "no edges means no simulations");
    }

    #[test]
    fn test_minimise_on_empty_edge_set_returns_a_point() {
        let edges: Vec<Edge> = Vec::new();
        let dual = problem(&edges);
        let market = MarketState::default();
        let deadline = Deadline::new(Instant::now(), Duration::from_millis(50));
        let solution = dual
            .minimise(vec![1.0, 1.0, 1.0], &market, &deadline)
            .expect("solvable");
        assert!(solution.flows.is_empty());
        assert_eq!(solution.edge_solves, 0, "no edges means no simulations");
    }

    #[test]
    fn test_minimise_respects_expired_deadline() {
        let edges: Vec<Edge> = Vec::new();
        let dual = problem(&edges);
        let market = MarketState::default();
        // A deadline already in the past: the loop must break on its first check.
        let deadline =
            Deadline::new(Instant::now() - Duration::from_secs(10), Duration::from_millis(1));
        let solution = dual
            .minimise(vec![1.0, 1.0, 1.0], &market, &deadline)
            .expect("returns best-so-far rather than erroring");
        assert_eq!(solution.iterations, 1, "stopped on the first deadline check");
    }
}
