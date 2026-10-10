//! Tuning constants for the convex flow solver.

/// Maximum coordinator iterations before the solver returns its best feasible point.
///
/// The reference implementation converges in roughly 20 iterations on a 2,500-market instance
/// with zero edge utilities, which is the case this solver runs in. The cap is a backstop for a
/// badly scaled instance, not an expected stopping point.
pub(crate) const MAX_ITERATIONS: usize = 60;

/// Relative duality-gap target. Below this the coordinator stops.
pub(crate) const GAP_TOLERANCE: f64 = 1e-7;

/// Maximum bisection steps for one edge subproblem.
///
/// Each step costs one `get_amount_out`, so this is the dominant simulation cost. The bracket
/// halves every step, so 12 steps resolve an edge's input to one part in 4096 of
/// its limit. The coordinator re-solves every edge each iteration from updated prices, so
/// resolving one edge beyond this buys less than spending the same simulations on another
/// iteration does.
pub(crate) const MAX_BISECTION_STEPS: usize = 12;

/// Bisection stops once the bracket is this fraction of the edge's upper bound.
pub(crate) const BISECTION_REL_TOLERANCE: f64 = 1e-9;

/// Cap on edge subproblem solves across one whole solve.
///
/// The no-flow test (`edge::Edge::carries_no_flow`) removes most edges before any simulation, so
/// this bounds the tail. It has to cover several whole evaluations — one pass over a few hundred
/// edges is already thousands of simulations — so the deadline, not this, is the usual stop.
pub(crate) const MAX_EDGE_SOLVES: usize = 20_000;

/// Barzilai-Borwein step length clamped to this range, so one badly scaled iteration cannot
/// throw the price vector far from the feasible region.
pub(crate) const MIN_STEP: f64 = 1e-12;
/// See [`MIN_STEP`].
pub(crate) const MAX_STEP: f64 = 1e6;

/// Initial step length when there is no previous iterate to derive one from.
///
/// Small on purpose. The warm start puts every edge exactly on its no-flow boundary — the
/// reference prices are the marginal prices — so the step that matters is the small perturbation
/// that tips edges into carrying flow, and for an order a ten-thousandth of a pool's depth that
/// perturbation is of the same order. Starting at one and halving down to it costs a dual
/// evaluation per halving, which is thousands of simulations; starting near it costs none.
/// Barzilai-Borwein takes over from the second iteration, so this only has to be right once.
pub(crate) const INITIAL_STEP: f64 = 1.0;

/// Non-monotone line search: the trial step is accepted against the worst objective in this
/// many recent iterations rather than the last one, which lets the method cross narrow ridges.
pub(crate) const LINE_SEARCH_MEMORY: usize = 5;

/// Maximum backtracks per line search before the step is given up on.
///
/// Each backtrack is a whole dual evaluation, so this is expensive and bounded. Combined with
/// [`INITIAL_STEP`] it reaches perturbations down to about `1e-3 · 2^-24`, which covers orders
/// far smaller against pool depth than anything the dataset holds.
pub(crate) const MAX_BACKTRACKS: usize = 40;

/// Backtracking contraction factor.
pub(crate) const BACKTRACK_FACTOR: f64 = 0.5;

/// Sufficient-decrease coefficient for the Armijo condition.
pub(crate) const ARMIJO_COEFFICIENT: f64 = 1e-4;

/// Flow below this fraction of the order amount is dropped from the final route.
///
/// A leg this small cannot pay its own gas, and the split convention in `Route::validate` has no
/// way to express a zero-amount branch.
pub(crate) const MIN_LEG_FRACTION: f64 = 1e-4;

/// Maximum legs in a reconstructed route. Each leg is gas and calldata, and the encoder's split
/// shapes do not benefit from a long tail of tiny branches.
pub(crate) const MAX_LEGS: usize = 8;

/// How much more than the order's own value a single edge may be asked to carry.
///
/// `get_limits` reports what a pool could theoretically absorb, which for a deep pool is orders
/// of magnitude beyond anything this order could send through it. Bisecting over that bracket
/// lets an intermediate iterate propose a trade far larger than the order, which is both
/// physically meaningless and numerically destructive — the dual's intermediate points need not
/// be feasible, but they do need to be finite. Value is conserved along a route up to fees, so
/// no edge can carry more than the order is worth; the slack covers reference prices that are
/// merely approximate.
pub(crate) const EDGE_VALUE_SLACK: f64 = 4.0;
