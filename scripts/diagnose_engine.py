"""Targeted engine diagnostic: trace coin_flip fills on a small sample.

Runs coin_flip on the 10 largest-volume Kalshi markets with verbose logging
to expose fill prices, win/loss outcomes, and PnL per market.

Also checks:
  - taker_side distribution in the data
  - yes_price + no_price sum (should be ~100 cents)
  - Actual fill prices vs trigger prices for coin_flip
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb

DATA_DIR = Path(__file__).parent.parent / "data" / "kalshi"
TRADES_DIR = DATA_DIR / "trades"
MARKETS_DIR = DATA_DIR / "markets"
MARKET_SAMPLE = 0.05

BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def main() -> None:
    con = duckdb.connect()

    # ── Get top-5% markets ────────────────────────────────────────────────────
    print(f"\n{BOLD}Loading top-5% markets...{RESET}")
    vol_rows = con.execute(
        f"""
        SELECT ticker, COUNT(*) AS cnt
        FROM '{TRADES_DIR}/*.parquet'
        GROUP BY ticker
        ORDER BY cnt DESC
        """
    ).fetchall()
    k = max(1, int(len(vol_rows) * MARKET_SAMPLE))
    top_markets = [row[0] for row in vol_rows[:k]]
    print(f"  {len(top_markets):,} markets selected (top {MARKET_SAMPLE:.0%})\n")

    ids_str = ", ".join(f"'{m}'" for m in top_markets)

    # ── Check taker_side distribution ────────────────────────────────────────
    print(f"{BOLD}Taker-side distribution in top-5% markets:{RESET}")
    taker_rows = con.execute(
        f"""
        SELECT taker_side, COUNT(*) AS cnt,
               AVG(yes_price / 100.0) AS avg_yes_price
        FROM '{TRADES_DIR}/*.parquet'
        WHERE ticker IN ({ids_str})
        GROUP BY taker_side
        """
    ).fetchall()
    total_trades = sum(r[1] for r in taker_rows)
    for taker_side, cnt, avg_yes in taker_rows:
        print(f"  taker={taker_side:4s}: {cnt:>10,} ({cnt / total_trades:.1%}) avg_yes={avg_yes:.3f}")

    # ── Check yes+no price sum ────────────────────────────────────────────────
    print(f"\n{BOLD}Price sum check (yes_price + no_price should ≈ 100):{RESET}")
    sum_rows = con.execute(
        f"""
        SELECT AVG(yes_price + no_price)   AS avg_sum,
               MIN(yes_price + no_price)   AS min_sum,
               MAX(yes_price + no_price)   AS max_sum,
               STDDEV(yes_price + no_price) AS std_sum
        FROM '{TRADES_DIR}/*.parquet'
        WHERE ticker IN ({ids_str})
        """
    ).fetchone()
    print(f"  avg={sum_rows[0]:.4f}  min={sum_rows[1]}  max={sum_rows[2]}  std={sum_rows[3]:.4f}")

    # ── Simulate coin_flip fills ──────────────────────────────────────────────
    # For each market, find:
    # (a) The trigger trade: first trade where taker=YES AND yes_price in [48,52]
    #     (YES-taker trade that triggers on_trade) -- BUT actually, on_trade
    #     fires for ALL trades, not just YES-taker. Let me be more precise.
    #
    # The strategy triggers on_trade when yes_price in [48,52], regardless of
    # taker_side. Then it places a Buy YES order at that price. The order fills
    # on the NEXT trade where taker=NO AND yes_price <= order.price.
    #
    # So: trigger = first trade with yes_price in [48,52]
    #     fill    = first subsequent trade with taker=NO AND yes_price <= trigger_price

    print(f"\n{BOLD}Coin-flip fill simulation (top-5% markets, YES 48–52¢):{RESET}")
    print("  Analyzing which markets trigger, fill, and what the fill price is...")

    # Get all trades for top markets, sorted by time
    print("  Fetching all relevant trades (this may take a moment)...")
    trade_rows = con.execute(
        f"""
        SELECT ticker, yes_price / 100.0, no_price / 100.0, taker_side, created_time
        FROM '{TRADES_DIR}/*.parquet'
        WHERE ticker IN ({ids_str})
        ORDER BY created_time, trade_id
        """
    ).fetchall()

    # Load results
    result_rows = con.execute(
        f"""
        SELECT ticker, result
        FROM '{MARKETS_DIR}/*.parquet'
        WHERE ticker IN ({ids_str})
          AND status = 'finalized'
          AND result IN ('yes', 'no')
        """
    ).fetchall()
    market_result = dict(result_rows)

    print(f"  Trades loaded: {len(trade_rows):,}")

    # Simulate per market
    from collections import defaultdict

    market_trades: dict[str, list] = defaultdict(list)
    for row in trade_rows:
        market_trades[row[0]].append(row[1:])  # (yes_price, no_price, taker_side, ts)

    triggered = 0
    filled = 0
    not_filled = 0  # triggered but never filled
    fill_prices: list[float] = []
    trigger_prices: list[float] = []
    pnls: list[float] = []
    fill_delay_trades: list[int] = []  # how many trades until fill after trigger

    LO, HI = 0.48, 0.52

    for mid, trades in market_trades.items():
        ordered = False
        order_price = None
        trigger_idx = None

        for i, (yes_price, _no_price, taker_side, _ts) in enumerate(trades):
            if not ordered:
                # Check if we trigger (any trade in 48-52¢ range)
                if LO <= yes_price <= HI:
                    ordered = True
                    order_price = yes_price
                    trigger_idx = i
                    triggered += 1
            else:
                # Check if we fill: taker=NO and yes_price <= order_price
                if taker_side == "no" and yes_price <= order_price:
                    fill_price = yes_price
                    filled += 1
                    fill_prices.append(fill_price)
                    trigger_prices.append(order_price)
                    fill_delay_trades.append(i - trigger_idx)

                    # Compute P&L (simplified: $100 notional, 1% commission)
                    qty = int(100 / order_price)
                    if qty > 0:
                        cost = fill_price * qty
                        commission = cost * 0.01
                        if mid in market_result:
                            resolution = market_result[mid]
                            payout = qty * 1.0 if resolution == "yes" else 0.0
                        else:
                            payout = fill_price * qty  # mark-to-market (no resolution)
                        pnl = payout - cost - commission
                        pnls.append(pnl)
                    break  # one order per market

        else:
            if ordered:
                not_filled += 1  # triggered but never filled

    print(f"\n  Trigger price range: {LO:.0%}–{HI:.0%}")
    print(f"  Markets triggered:   {triggered:,} ({triggered / len(market_trades):.1%} of {len(market_trades):,})")
    print(f"  Orders filled:       {filled:,} ({filled / triggered:.1%} of triggered)")
    print(f"  Orders never filled: {not_filled:,}")

    if fill_prices:
        avg_trigger = sum(trigger_prices) / len(trigger_prices)
        avg_fill = sum(fill_prices) / len(fill_prices)
        avg_delay = sum(fill_delay_trades) / len(fill_delay_trades)
        print(f"\n  Avg trigger price:   {avg_trigger:.4f} ({avg_trigger:.2%})")
        print(f"  Avg fill price:      {avg_fill:.4f} ({avg_fill:.2%})")
        print(f"  Price improvement:   {avg_trigger - avg_fill:+.4f} ({avg_trigger - avg_fill:+.2%})")
        print(f"  Avg fills-to-fill:   {avg_delay:.1f} trades after trigger")

    if pnls:
        wins = [p for p in pnls if p > 0]
        win_rate = len(wins) / len(pnls)
        avg_pnl = sum(pnls) / len(pnls)
        total_pnl = sum(pnls)
        print("\n  Simulated P&L (resolved markets only):")
        print(f"  Markets with P&L:  {len(pnls):,}")
        print(f"  Win rate:          {win_rate:.1%}")
        print(f"  Avg P&L:          ${avg_pnl:+.3f}")
        print(f"  Total P&L:        ${total_pnl:+,.2f}")
        print(f"  Return on $10k:   {total_pnl / 10000:+.2%}")

    # ── Compare with ALL-TRADE trigger (no taker filter on fill) ─────────────
    print(f"\n{BOLD}Bonus: If we could fill on ANY next trade (not just NO-taker):{RESET}")
    filled_any = 0
    pnls_any: list[float] = []

    for mid, trades in market_trades.items():
        ordered = False
        order_price = None

        for _i, (yes_price, _no_price, _taker_side, _ts) in enumerate(trades):
            if not ordered:
                if LO <= yes_price <= HI:
                    ordered = True
                    order_price = yes_price
            else:
                # Fill on VERY NEXT trade (regardless of taker side or price)
                fill_price = yes_price
                filled_any += 1
                qty = int(100 / order_price)
                if qty > 0:
                    cost = fill_price * qty
                    commission = cost * 0.01
                    if mid in market_result:
                        payout = qty * 1.0 if market_result[mid] == "yes" else 0.0
                    else:
                        payout = fill_price * qty
                    pnl = payout - cost - commission
                    pnls_any.append(pnl)
                break

    if pnls_any:
        wins_a = [p for p in pnls_any if p > 0]
        avg_pnl_any = sum(pnls_any) / len(pnls_any)
        total_ret_any = sum(pnls_any) / 10000
        print(
            f"  Filled: {filled_any:,}, win rate: {len(wins_a) / len(pnls_any):.1%},"
            f" avg P&L: ${avg_pnl_any:+.3f}, total return: {total_ret_any:+.2%}"
        )

    print()


if __name__ == "__main__":
    main()
