//! Builds a trade dataset that covers the scenarios the recorded one does not.
//!
//! `aggregator_trades_10k.json` is real aggregator flow, and that is both its value and its
//! blind spot: it is thousand-dollar orders on the busiest pairs. Measured against a true
//! gas-aware optimum, every shipped algorithm's allocation on that dataset is within a hundredth
//! of a basis point of optimal, and the optimum itself uses a single pool on 140 of the 148
//! orders it can answer. That is not a statement about the algorithms. It is a statement about
//! the orders: at that size, against that depth, with gas charged per pool, there is nothing to
//! split. A benchmark that only asks easy questions cannot tell a good router from a lucky one.
//!
//! This generator asks the other questions, by sweeping the two axes the recorded dataset holds
//! fixed:
//!
//! - **Size, relative to the liquidity that exists.** Each order is a fraction of its pair's own
//!   depth, from a ten-thousandth of it up to twice it. An absolute size means nothing — a hundred
//!   thousand dollars is noise on WETH/USDC and the whole market on a thin pair — so the ladder is
//!   relative, which is what makes a thin pair and a deep one comparable.
//! - **Liquidity, across tiers.** Pairs are ranked by depth and sampled from the top, middle and
//!   bottom of the distribution, so the set contains the pairs where routing is hard as well as the
//!   ones where it is easy.
//!
//! Both directions of every pair are emitted, because depth is not symmetric and neither are the
//! failures.
//!
//! # What "depth" means here
//!
//! The depth of a pool, as the derived pass computes it: the input after which the pool's
//! marginal price, net of its fee, has fallen by a fixed fraction. Summed over the pools serving
//! a pair, in the input token's own units. So "an order at 1.0 of the pair's depth" is an order
//! large enough to move every pool on that pair by that fraction — unambiguously a large order,
//! whatever the pair is worth in dollars.
//!
//! # Output
//!
//! The same JSON the benchmark already reads, so nothing downstream changes:
//!
//! ```bash
//! cargo bench --bench scenarios -- --out scenarios.json
//! ./scripts/bench.sh --name wide --orders 0 --jobs 1 --trades scenarios.json --configs OPT,WF_d3
//! ```
//!
//! `amount_usd` is filled in so the reports and `analysis/bench-analyze.py` can slice by size. It
//! is derived from the market's own token prices, which are denominated in the gas token, times
//! one assumed gas-token price (`--eth-usd`). Only the absolute scale comes from that number; the
//! ratios between orders are the market's.

use std::{collections::HashMap, path::PathBuf};

use fynd_core::{
    derived::types::ComponentDepths, rustc_hash::FxHashMap, AlgorithmRegistry, PoolConfig, Solver,
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use tycho_simulation::tycho_common::models::Address;

use crate::{
    crate_path, default_fixture_path, load_market, symbol_table, token_label, wei_per_token,
    Market, READY_TIMEOUT,
};

/// Where each order sits on its pair's depth.
///
/// Spans six orders of magnitude on purpose. The bottom of the ladder is the regime the recorded
/// dataset already covers, kept so a run on this set is comparable with one on that; the top is
/// an order larger than the liquidity serving its pair, which is where a router either splits,
/// routes around, or fails.
const DEPTH_FRACTIONS: &[f64] = &[1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1.0, 2.0];

/// One band of pairs to sample from: where it starts in the depth ranking, and how many pools a
/// pair must have to belong to it.
struct Tier {
    name: &'static str,
    /// How far down the depth ranking the sample starts. Zero is the deepest pair in the market.
    position: f64,
    /// Fewest pools a pair may have. The axis matters as much as depth does: a pair served by one
    /// pool has no split to get right, so it cannot tell two routing algorithms apart, and a pair
    /// served by a dozen is where the whole question lives.
    min_pools: usize,
}

/// The bands the dataset is built from.
///
/// Not evenly spaced, because the depth distribution is not: it falls by six orders of magnitude
/// over the first tenth of the ranking and the median pair in this market is a single dust pool.
/// The run prints the deciles it sampled from, both overall and restricted to multi-pool pairs,
/// which is how these were chosen and how they should be rechosen when the fixture is
/// re-recorded.
///
/// `deep-single` is deliberate rather than an oversight: the deepest pairs in this market are
/// single-pool wrappers and stable curves whose price barely moves, and a router has to be right
/// about those too — including declining to split when there is nothing to split across.
const TIERS: &[Tier] = &[
    Tier { name: "deep-multi", position: 0.0, min_pools: 4 },
    Tier { name: "deep-single", position: 0.0, min_pools: 1 },
    Tier { name: "mid-multi", position: 0.15, min_pools: 2 },
    Tier { name: "thin", position: 0.45, min_pools: 1 },
];

/// Smallest order the generator will emit, in wei of value.
///
/// Below roughly a thousandth of an ETH an order is dust: the gas to touch one pool outweighs any
/// routing decision, every algorithm returns the same single leg, and the row measures nothing.
const MIN_ORDER_WEI: f64 = 1e15;

/// What the generator was asked to build.
#[derive(clap::Parser, Debug)]
#[command(about = "Builds a trade dataset spanning order sizes and liquidity tiers")]
pub struct ScenarioFlags {
    /// The market recording the scenarios are derived from.
    #[arg(long)]
    pub fixture: Option<PathBuf>,

    /// Where to write the dataset. Defaults to `scenarios.json` in the repository root, beside
    /// the recorded dataset, because `cargo bench` runs this binary with the crate as its working
    /// directory and the benchmark script runs from the root.
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Pairs sampled from each liquidity tier.
    #[arg(long, default_value_t = 20)]
    pub pairs_per_tier: usize,

    /// Gas-token price used to label orders in `amount_usd`.
    ///
    /// Only sets the scale of a field the reports slice by; no routing decision reads it.
    #[arg(long, default_value_t = 3000.0)]
    pub eth_usd: f64,

    /// Accepted and ignored, because `cargo bench` appends it to the binary's arguments.
    #[arg(long, hide = true)]
    pub bench: bool,
}

/// One directional pair the generator may emit orders for.
struct PairDepth {
    token_in: Address,
    token_out: Address,
    /// Summed pool depth, in the input token's smallest unit.
    depth: f64,
    /// The same depth valued in wei, which is what ranks pairs against each other.
    depth_wei: f64,
    pools: usize,
}

/// Builds the dataset and writes it.
///
/// # Errors
///
/// Returns the reason the market could not be read or the file could not be written.
pub async fn run() -> Result<(), String> {
    let flags = <ScenarioFlags as clap::Parser>::parse();
    let fixture = flags
        .fixture
        .clone()
        .unwrap_or_else(default_fixture_path);
    let out = flags
        .out
        .clone()
        .unwrap_or_else(|| crate_path("../scenarios.json"));
    println!("scenario dataset");
    println!("  fixture          {}", fixture.display());

    let market = load_market(&fixture).await;
    let solver = solver_for(&market).await?;
    let symbols = symbol_table();

    let derived_data = solver.derived_data();
    let (pairs, wei) = {
        let derived = derived_data.read().await;
        let wei = wei_per_token(&derived);
        let depths = derived
            .component_depths()
            .ok_or_else(|| "the market produced no component depths".to_string())?;
        (collect_pairs(depths, &wei), wei)
    };
    if pairs.is_empty() {
        return Err("the market produced no priced pairs with depth".to_string());
    }
    println!("  priced pairs     {}", pairs.len());
    print_distribution(&pairs);

    let mut trades = Vec::new();
    let mut summary: Vec<(String, usize, usize)> = Vec::new();

    for tier in TIERS {
        let sampled = sample_tier(&pairs, tier, flags.pairs_per_tier);
        let before = trades.len();
        for pair in &sampled {
            for fraction in DEPTH_FRACTIONS {
                let amount = pair.depth * fraction;
                let value_wei = amount *
                    wei.get(&pair.token_in)
                        .copied()
                        .unwrap_or(0.0);
                if !(amount >= 1.0 && value_wei >= MIN_ORDER_WEI) {
                    continue;
                }
                trades.push(trade_json(
                    &pair.token_in,
                    &pair.token_out,
                    &to_raw(amount),
                    value_wei / 1e18 * flags.eth_usd,
                ));
            }
        }
        summary.push((
            format!(
                "{} (rank {:.0}%+, {}+ pools)",
                tier.name,
                tier.position * 100.0,
                tier.min_pools
            ),
            sampled.len(),
            trades.len() - before,
        ));
        for pair in sampled.iter().take(2) {
            println!(
                "  {:<12} {} -> {} ({} pools, {:.4} ETH deep)",
                tier.name,
                token_label(&pair.token_in, &symbols),
                token_label(&pair.token_out, &symbols),
                pair.pools,
                pair.depth_wei / 1e18,
            );
        }
    }

    if trades.is_empty() {
        return Err("every candidate order fell below the dust floor".to_string());
    }
    let json = serde_json::to_string(&trades).map_err(|error| error.to_string())?;
    std::fs::write(&out, json)
        .map_err(|error| format!("could not write {}: {error}", out.display()))?;
    let written = out.canonicalize().unwrap_or(out);

    println!();
    println!("  {:<34} {:>6} {:>8}", "tier", "pairs", "orders");
    for (label, pairs, orders) in &summary {
        println!("  {label:<34} {pairs:>6} {orders:>8}");
    }
    println!("  {:<34} {:>6} {:>8}", "total", "", trades.len());
    println!();
    println!("  wrote {}", written.display());
    println!("  run it with:");
    println!(
        "    ./scripts/bench.sh --name wide --orders 0 --jobs 1 --trades {} \\",
        written.display()
    );
    println!("      --timeout-ms 60000 --configs OPT,WF_d3,BF_d3,ML_d3,PFW_d3");
    Ok(())
}

/// A solver over the recording, for its derived data.
///
/// One worker on a cheap algorithm: nothing here solves an order, the solver is built because
/// replaying a recording and running the derived computations is what it does.
async fn solver_for(market: &Market) -> Result<Solver, String> {
    let pool: PoolConfig = toml::from_str("algorithm = \"most_liquid\"\nnum_workers = 1\n")
        .map_err(|error| format!("the generator's own pool config does not parse: {error}"))?;
    let solver = Solver::from_recording_with(
        market.chain,
        market.updates.clone(),
        HashMap::from([("scenarios".to_string(), pool)]),
        market.market_gas_price.clone(),
        &AlgorithmRegistry::new(),
    )
    .await
    .map_err(|error| error.to_string())?;
    solver
        .wait_until_ready(READY_TIMEOUT)
        .await
        .map_err(|error| format!("solver never became ready: {error}"))?;
    Ok(solver)
}

/// Prints where the depth sits across the ranking, which is what the tier positions are chosen
/// against.
fn print_distribution(pairs: &[PairDepth]) {
    let multi: Vec<&PairDepth> = pairs
        .iter()
        .filter(|pair| pair.pools >= 2)
        .collect();
    println!(
        "  depth by rank    (ETH, deepest first; {} of {} pairs have more than one pool)",
        multi.len(),
        pairs.len()
    );
    println!("    {:>5} {:>16} {:>16}", "rank", "all pairs", "multi-pool only");
    for decile in 0..=10 {
        let at = |list: &[&PairDepth]| -> String {
            if list.is_empty() {
                return "-".to_string();
            }
            let index = ((decile as f64 / 10.0) * (list.len() - 1) as f64).round() as usize;
            format!("{:.4} ({}p)", list[index].depth_wei / 1e18, list[index].pools)
        };
        let all: Vec<&PairDepth> = pairs.iter().collect();
        println!("    {:>4}% {:>16} {:>16}", decile * 10, at(&all), at(&multi));
    }
}

/// Sums the per-pool depths into one figure per directional pair.
fn collect_pairs(depths: &ComponentDepths, wei: &HashMap<Address, f64>) -> Vec<PairDepth> {
    let mut totals: FxHashMap<(Address, Address), (f64, usize)> = FxHashMap::default();
    for ((_, token_in, token_out), depth) in depths {
        let Some(depth) = depth
            .to_f64()
            .filter(|d| d.is_finite() && *d > 0.0)
        else {
            continue;
        };
        let entry = totals
            .entry((token_in.clone(), token_out.clone()))
            .or_insert((0.0, 0));
        entry.0 += depth;
        entry.1 += 1;
    }

    let mut pairs: Vec<PairDepth> = totals
        .into_iter()
        .filter_map(|((token_in, token_out), (depth, pools))| {
            // A pair with no price for its input token cannot be ranked against the others, and
            // its orders could not be labelled either.
            let unit = wei
                .get(&token_in)
                .copied()
                .filter(|u| u.is_finite() && *u > 0.0)?;
            Some(PairDepth { token_in, token_out, depth, depth_wei: depth * unit, pools })
        })
        .collect();
    pairs.sort_by(|left, right| {
        right
            .depth_wei
            .partial_cmp(&left.depth_wei)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs
}

/// Takes `count` pairs from one tier: those with enough pools, starting the given fraction of the
/// way down their own depth ranking.
///
/// The ranking is deepest first, so a position of zero samples the deepest pairs that qualify.
/// Clamped so a short list still yields a sample rather than nothing.
fn sample_tier<'a>(pairs: &'a [PairDepth], tier: &Tier, count: usize) -> Vec<&'a PairDepth> {
    let eligible: Vec<&PairDepth> = pairs
        .iter()
        .filter(|pair| pair.pools >= tier.min_pools)
        .collect();
    if eligible.is_empty() || count == 0 {
        return Vec::new();
    }
    let span = eligible.len().saturating_sub(count);
    let start = (tier.position.clamp(0.0, 1.0) * span as f64).round() as usize;
    eligible
        .into_iter()
        .skip(start.min(span))
        .take(count)
        .collect()
}

/// An amount in the input token's smallest unit, as the dataset writes it.
fn to_raw(amount: f64) -> BigUint {
    // `f64` carries sixteen digits and an amount can be thirty, so the string goes through a
    // decimal exponent rather than `as u128`, which saturates.
    let rounded = format!("{amount:.0}");
    rounded
        .parse()
        .unwrap_or_else(|_| BigUint::from(0u8))
}

/// One dataset record, in the shape `trades::load_trade_orders` reads.
fn trade_json(
    token_in: &Address,
    token_out: &Address,
    amount: &BigUint,
    amount_usd: f64,
) -> serde_json::Value {
    serde_json::json!({
        "orders": [{
            "id": "",
            "token_in": token_in.to_string(),
            "token_out": token_out.to_string(),
            "amount": amount.to_string(),
            "side": "sell",
            "sender": "0x0000000000000000000000000000000000000001",
            "receiver": serde_json::Value::Null,
        }],
        "amount_usd": amount_usd,
        "options": { "timeout_ms": 5000, "min_responses": serde_json::Value::Null, "max_gas": serde_json::Value::Null },
    })
}
