"""Diagnostic: compute actual YES win rates for top-5% markets by price range.

This script queries the raw data directly (no engine) to determine:
1. Whether the sports-heavy top-5% sample has different calibration vs general data
2. What win rates each strategy actually encounters at their trigger price levels

Output tells us whether anomalous backtest results are due to:
  (a) Sports calibration differing from general Kalshi calibration, or
  (b) Engine bugs (look-ahead bias, accounting errors)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb

# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "kalshi"
TRADES_DIR = DATA_DIR / "trades"
MARKETS_DIR = DATA_DIR / "markets"
MARKET_SAMPLE = 0.05  # top 5% by volume (same as run_validation.py)

# Price ranges matching each strategy's trigger condition
RANGES = {
    "fade_cheap (NO signal: YES 5–15¢)": (0.05, 0.15),
    "buy_low   (YES signal: YES <20¢)": (0.01, 0.20),
    "yes_at_30s(YES signal: YES 30–37¢)": (0.30, 0.37),
    "coin_flip (YES signal: YES 48–52¢)": (0.48, 0.52),
}

BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def color_pct(actual: float, implied: float) -> str:
    diff = actual - implied
    c = GREEN if diff > 0.01 else (RED if diff < -0.01 else YELLOW)
    return f"{c}{actual:.1%}{RESET} (implied {implied:.1%}, diff {diff:+.1%})"


def main() -> None:
    con = duckdb.connect()

    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}Calibration Diagnostic  |  Kalshi top {MARKET_SAMPLE:.0%} by volume{RESET}")
    print(f"{BOLD}{'=' * 72}{RESET}\n")

    # ── Step 1: Select top-5% markets by volume ──────────────────────────────
    print("Loading market volumes...")
    vol_rows = con.execute(
        f"""
        SELECT ticker, COUNT(*) AS cnt
        FROM '{TRADES_DIR}/*.parquet'
        GROUP BY ticker
        ORDER BY cnt DESC
        """
    ).fetchall()
    total_markets = len(vol_rows)
    k = max(1, int(total_markets * MARKET_SAMPLE))
    top_markets = [row[0] for row in vol_rows[:k]]
    print(f"  Total markets: {total_markets:,} → top {MARKET_SAMPLE:.0%}: {k:,} markets\n")

    ids_str = ", ".join(f"'{m}'" for m in top_markets)

    # ── Step 2: Load market results ───────────────────────────────────────────
    print("Loading market results...")
    result_rows = con.execute(
        f"""
        SELECT ticker, result
        FROM '{MARKETS_DIR}/*.parquet'
        WHERE ticker IN ({ids_str})
          AND status = 'finalized'
          AND result IN ('yes', 'no')
        """
    ).fetchall()
    market_result: dict[str, str] = dict(result_rows)
    resolved_count = len(market_result)
    yes_count = sum(1 for r in market_result.values() if r == "yes")
    print(f"  Resolved markets: {resolved_count:,}  ({yes_count:,} YES = {yes_count / resolved_count:.1%})\n")

    # ── Step 3: For each strategy's price range, find markets where the FIRST
    #    trade in that market fell in range, and compute YES win rate ──────────
    print("Querying first-trade prices per market...")
    first_trade_rows = con.execute(
        f"""
        SELECT ticker, yes_price / 100.0 AS yes_price
        FROM (
            SELECT ticker, yes_price,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY created_time, trade_id) AS rn
            FROM '{TRADES_DIR}/*.parquet'
            WHERE ticker IN ({ids_str})
        ) t
        WHERE rn = 1
        """
    ).fetchall()
    first_price: dict[str, float] = dict(first_trade_rows)
    print(f"  Markets with first trade: {len(first_price):,}\n")

    # ── Step 4: Also compute win rate for markets where ANY trade fell in range
    #    (closer to what the strategy actually encounters) ───────────────────
    print("Querying markets that had at least one trade in each price range...")
    any_trade_rows = con.execute(
        f"""
        SELECT DISTINCT ticker, yes_price / 100.0 AS yes_price
        FROM '{TRADES_DIR}/*.parquet'
        WHERE ticker IN ({ids_str})
        """
    ).fetchall()

    # Build set per market of all observed prices
    from collections import defaultdict

    market_prices: dict[str, list[float]] = defaultdict(list)
    for ticker, price in any_trade_rows:
        market_prices[ticker].append(price)

    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}Win Rate Analysis by Price Range{RESET}")
    print(f"{BOLD}{'=' * 72}{RESET}\n")

    for label, (lo, hi) in RANGES.items():
        mid_price = (lo + hi) / 2

        # Method A: First trade in range
        in_range_first = [t for t, p in first_price.items() if lo <= p <= hi and t in market_result]
        yes_wins_first = sum(1 for t in in_range_first if market_result[t] == "yes")
        wr_first = yes_wins_first / len(in_range_first) if in_range_first else float("nan")

        # Method B: Any trade in range (≥1 trade)
        in_range_any = [
            t for t, prices in market_prices.items() if any(lo <= p <= hi for p in prices) and t in market_result
        ]
        yes_wins_any = sum(1 for t in in_range_any if market_result[t] == "yes")
        wr_any = yes_wins_any / len(in_range_any) if in_range_any else float("nan")

        print(f"  {BOLD}{label}{RESET}")
        print(f"    Price range:   {lo:.0%} – {hi:.0%} YES  (implied YES win ≈ {mid_price:.0%})")
        print(f"    First-trade:   {len(in_range_first):>5,} markets → YES win rate = {color_pct(wr_first, mid_price)}")
        print(f"    Any-trade:     {len(in_range_any):>5,} markets → YES win rate = {color_pct(wr_any, mid_price)}")
        print()

    # ── Step 5: Overall calibration curve ─────────────────────────────────────
    print(f"{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}Full Calibration Curve (top-5% markets, first-trade price buckets){RESET}")
    print(f"{BOLD}{'=' * 72}{RESET}\n")
    print(f"  {'Price bucket':>14}  {'Markets':>8}  {'YES wins':>9}  {'Win rate':>9}  {'Implied':>8}  {'Edge':>7}")
    print(f"  {'-' * 14}  {'-' * 8}  {'-' * 9}  {'-' * 9}  {'-' * 8}  {'-' * 7}")

    buckets = [(i / 100, (i + 5) / 100) for i in range(5, 100, 5)]
    for lo, hi in buckets:
        in_range = [t for t, p in first_price.items() if lo <= p < hi and t in market_result]
        if not in_range:
            continue
        wins = sum(1 for t in in_range if market_result[t] == "yes")
        wr = wins / len(in_range)
        mid = (lo + hi) / 2
        edge = wr - mid
        edge_str = f"{edge:+.1%}"
        edge_color = GREEN if edge > 0.01 else (RED if edge < -0.01 else "")
        print(
            f"  {lo:.0%}–{hi:.0%}          {len(in_range):>8,}  {wins:>9,}  "
            f"{wr:>9.1%}  {mid:>8.0%}  {edge_color}{edge_str:>7}{RESET}"
        )

    print()


if __name__ == "__main__":
    main()
