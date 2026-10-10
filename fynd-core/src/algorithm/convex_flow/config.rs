//! Tuning constants for the convex flow solver.

/// Maximum coordinator iterations before the solver returns its best feasible point.
///
/// Each iteration costs one edge-subproblem pass plus one per line-search backtrack, and the
/// deadline stops the solve long before this on a large network. The cap is a backstop against a
/// badly scaled instance looping forever, not an expected stopping point: on a 129-edge network
/// the coordinator needs upwards of a hundred iterations to drive the net-flow residual to zero,
/// and since the returned point is the most feasible one seen
/// ([`dual::DualProblem::minimise`](super::dual::DualProblem::minimise)), extra iterations can
/// only improve the answer.
pub(crate) const MAX_ITERATIONS: usize = 200;

/// Stopping tolerance on the projected-gradient norm, which is net flow as a fraction of the
/// order.
///
/// This is a feasibility test, not a duality gap: the dual gradient *is* net flow, so a norm
/// below this means every token conserves to within this fraction of the order — a tenth of a
/// basis point, well inside what `MIN_LEG_FRACTION` can express.
///
/// It has to stay above what the edge subproblems can resolve, and those errors do not cancel
/// across edges, so the reachable floor is roughly `sqrt(edges)` times one edge's own error. A
/// tolerance below the floor is never reached, and the coordinator then spends its whole budget
/// circling a point it has already found: the shipped `1e-7` against a bisected subproblem's
/// `1e-3` is what left runs ending on the iteration cap with hundreds of iterations spent after
/// the answer was in hand.
pub(crate) const STATIONARITY_TOLERANCE: f64 = 1e-4;

/// Maximum root-finding steps for one edge subproblem.
///
/// Each step costs one `get_amount_out`, so this is the dominant simulation cost. The subproblem
/// is solved by regula falsi rather than bisection
/// ([`edge::Edge::solve`](super::edge::Edge::solve)), which converges superlinearly and so spends
/// most of its steps below this cap; the cap matters only for a curve the interpolation cannot
/// predict.
pub(crate) const MAX_EDGE_ITERATIONS: usize = 12;

/// Root finding stops once the bracket is this fraction of the edge's upper bound.
pub(crate) const EDGE_REL_TOLERANCE: f64 = 1e-9;

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
/// Small on purpose. The warm start puts every edge at or near its no-flow boundary — the
/// reference prices are the marginal prices — so the step that matters is the small perturbation
/// that tips edges into carrying flow, and for an order a ten-thousandth of a pool's depth that
/// perturbation is of the same order. Overshooting is far more expensive than undershooting: a
/// step too long is paid for one dual evaluation at a time by the line search, while a step too
/// short is corrected for free by the Barzilai-Borwein length from the second iteration on.
/// Starting at one costs ten to fourteen backtracks on a mid-sized network, which is most of the
/// simulation budget spent before the first accepted step.
pub(crate) const INITIAL_STEP: f64 = 1e-6;

/// Non-monotone line search: the trial step is accepted against the worst objective in this
/// many recent iterations rather than the last one, which lets the method cross narrow ridges.
pub(crate) const LINE_SEARCH_MEMORY: usize = 5;

/// Maximum backtracks per line search before the step is given up on.
///
/// Each backtrack is a whole dual evaluation, so this is expensive and bounded. Combined with
/// [`INITIAL_STEP`] it reaches perturbations down to about `1e-6 · 2^-40`, which covers orders
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
/// of magnitude beyond anything this order could send through it. Bracketing over that range
/// lets an intermediate iterate propose a trade far larger than the order, which is both
/// physically meaningless and numerically destructive — the dual's intermediate points need not
/// be feasible, but they do need to be finite. Value is conserved along a route up to fees, so
/// no edge can carry more than the order is worth; the slack covers reference prices that are
/// merely approximate.
///
/// The slack is not free: it sets the net-flow gradient at the warm start, where every edge
/// draining an intermediate token proposes its whole cap against one unit of unspent order. It
/// is kept anyway. Measured against this solver's own tuning, narrowing it to the order exactly
/// loses the widest instances — 65 and 129 edges against an order a hundred-thousandth of pool
/// depth end 5 to 10% unallocated — because the room an infeasible intermediate iterate needs is
/// the room this constant grants. It cost resolution when the subproblem bisected a fixed number
/// of times, since a wider bracket then meant a coarser answer; it costs none now that
/// [`MAX_EDGE_ITERATIONS`] interpolates.
pub(crate) const EDGE_VALUE_SLACK: f64 = 4.0;
