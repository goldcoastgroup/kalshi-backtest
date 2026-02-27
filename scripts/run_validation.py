"""Non-interactive strategy validation runner.

Runs all strategies against Kalshi (top 5% by volume) and prints a
ranked results table for comparing predicted vs actual engine output.

Usage:
    uv run run_validation.py

Transaction cost model: orders placed in on_trade fill immediately at the
trigger trade's price (liquidity_cap=False — one complete fill per market).
Commission: 1% of notional. Position sizing is fixed-notional ($100 per
market) so that per-trade EV is uniform across price levels.

NOTE: liquidity_cap=False means fill sizes are not bounded by the trade's
reported volume. This is calibration-testing mode: the absolute return
figures are not realistic, but the per-trade EV direction and win rates
are valid for testing whether the calibration edge is real.

Hypotheses — TOP 5% BY VOLUME (sports-heavy: NBA/NFL/NHL/soccer):

The top-5% sample is dominated by sports markets, whose calibration differs
dramatically from the full Kalshi dataset:
  - YES <20¢: actual win rate ~26% (implied 10-20%) — massive underdog edge
  - YES 30-37¢: actual win rate ~39% (implied 33-37%) — 6pp edge
  - YES 48-52¢: actual win rate ~38% (first-touch selection; expect <50%)
  - YES 85-95¢: actual win rate ~40% (implied 92%) — favorites UNDERPERFORM

Sports-market expected ranking (best → worst):
    1. buy_low    → POSITIVE  — underdogs win 26% at <20¢ vs 10-20% implied
    2. yes_at_30s → POSITIVE  — 6pp edge at 30-37¢ still holds in sports
    3. coin_flip  → NEGATIVE  — 1% commission + first-touch selection bias
    4. fade_cheap → NEGATIVE  — buying NO at 85-95¢ when YES wins ~40%

General-calibration ranking (run with no market_sample filter):
    1. yes_at_30s → POSITIVE  — 2pp edge survives 1% commission
    2. fade_cheap → SMALL POS — 1pp NO edge at 5-15¢ YES
    3. coin_flip  → SLIGHT NEG — 1% commission drag only
    4. buy_low    → NEGATIVE  — wrong direction on general-population data
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root without editable install
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.feeds.kalshi import KalshiFeed
from src.backtesting.rust_engine import Engine
from src.backtesting.strategies.buy_low import BuyLowStrategy
from src.backtesting.strategies.coin_flip import CoinFlipStrategy
from src.backtesting.strategies.fade_cheap import FadeCheapStrategy
from src.backtesting.strategies.yes_at_30s import YesAt30sStrategy

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

MARKET_SAMPLE = 0.05  # top 5% by volume — enough data, fast enough to run
INITIAL_CASH = 10_000.0
NOTIONAL = 100.0  # $ per trade — fixed notional for uniform risk across price levels


def _color_return(r: float) -> str:
    c = GREEN if r > 0 else (YELLOW if r > -0.05 else RED)
    return f"{c}{r:+.2%}{RESET}"


def run_strategy(strategy, feed_factory, label: str) -> dict:
    print(f"\n{BOLD}▶ Running: {label}{RESET}  ({strategy.description})")
    feed = feed_factory()
    engine = Engine(feed=feed, strategy=strategy, market_sample=MARKET_SAMPLE, progress=True, liquidity_cap=False)
    result = engine.run()
    m = result.metrics
    return {
        "name": label,
        "total_return": m.get("total_return", 0),
        "win_rate": m.get("win_rate", 0),
        "profit_factor": m.get("profit_factor", 0),
        "num_fills": int(m.get("num_fills", 0)),
        "markets_traded": result.num_markets_traded,
        "markets_resolved": result.num_markets_resolved,
        "final_equity": result.final_equity,
        "sharpe": m.get("sharpe_ratio", 0),
        "avg_pnl": m.get("avg_trade_pnl", 0),
    }


def main():
    strategies = [
        ("buy_low     [PREDICTED: + sports]", BuyLowStrategy(initial_cash=INITIAL_CASH, notional=NOTIONAL)),
        ("yes_at_30s  [PREDICTED: +]", YesAt30sStrategy(initial_cash=INITIAL_CASH, notional=NOTIONAL)),
        ("coin_flip   [PREDICTED: ~0]", CoinFlipStrategy(initial_cash=INITIAL_CASH, notional=NOTIONAL)),
        ("fade_cheap  [PREDICTED: --sports]", FadeCheapStrategy(initial_cash=INITIAL_CASH, notional=NOTIONAL)),
    ]

    feed_factory = KalshiFeed

    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(
        f"{BOLD}Strategy Validation Run  |  Kalshi  |  Top {MARKET_SAMPLE:.0%} by volume"
        f"  |  ${NOTIONAL:.0f} notional/trade{RESET}"
    )
    print(f"{BOLD}{'=' * 72}{RESET}")
    print(
        "\nHypotheses for top-5% by volume (sports-heavy: NBA/NFL/NHL/soccer):\n"
        "  buy_low    : YES <20¢ wins ~26% in sports (implied 10-20%) → STRONGLY POSITIVE\n"
        "  yes_at_30s : YES 30-37¢ wins ~39% in sports (implied 33-37%) → POSITIVE\n"
        "  coin_flip  : YES 48-52¢ at first touch → selection bias + 1% commission → NEGATIVE\n"
        "  fade_cheap : Buy NO at 85-95¢ when YES wins ~40% in sports → STRONGLY NEGATIVE\n"
        "\n  NOTE: General Kalshi calibration predicts yes_at_30s>fade_cheap>coin_flip>buy_low."
        "\n  Sports markets invert buy_low (underdogs) and fade_cheap (favorites underperform).\n"
    )

    results = []
    for label, strategy in strategies:
        try:
            r = run_strategy(strategy, feed_factory, label)
            results.append(r)
        except Exception as e:
            print(f"{RED}ERROR running {label}: {e}{RESET}")

    print(f"\n\n{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}RESULTS SUMMARY{RESET}")
    print(f"{BOLD}{'=' * 72}{RESET}")

    header = (
        f"  {'Strategy':<28}  {'Return':>8}  {'WinRate':>7}  {'PF':>6}"
        f"  {'Mkts':>5}  {'Res':>5}  {'AvgP&L':>8}  {'Sharpe':>7}"
    )
    print(f"\n{DIM}{header}{RESET}")
    print(f"  {'-' * 28}  {'-' * 8}  {'-' * 7}  {'-' * 6}  {'-' * 5}  {'-' * 5}  {'-' * 8}  {'-' * 7}")

    for r in results:
        wr_str = f"{r['win_rate']:.1%}"
        pf_str = f"{r['profit_factor']:.3f}"
        avg_str = f"${r['avg_pnl']:+.3f}"
        sharpe_str = f"{r['sharpe']:+.3f}"
        print(
            f"  {r['name']:<28}  {_color_return(r['total_return']):>8}  "
            f"{wr_str:>7}  {pf_str:>6}  {r['markets_traded']:>5,}  {r['markets_resolved']:>5,}"
            f"  {avg_str:>8}  {sharpe_str:>7}"
        )

    print()

    if len(results) == 4:
        sorted_results = sorted(results, key=lambda x: x["total_return"], reverse=True)
        # Sports-market expected ranking (top-5% is NBA/NFL/NHL/soccer dominated)
        expected_order = ["buy_low", "yes_at_30s", "coin_flip", "fade_cheap"]
        actual_order = [r["name"].split()[0] for r in sorted_results]

        print(f"{BOLD}Ranking check (sports-market hypothesis):{RESET}")
        print(f"  Expected (best→worst): {' > '.join(expected_order)}")
        print(f"  Actual   (best→worst): {' > '.join(actual_order)}")

        yes_return = next((r["total_return"] for r in results if "yes_at_30s" in r["name"]), None)
        low_return = next((r["total_return"] for r in results if "buy_low" in r["name"]), None)
        flip_return = next((r["total_return"] for r in results if "coin_flip" in r["name"]), None)
        fade_return = next((r["total_return"] for r in results if "fade_cheap" in r["name"]), None)

        # Sanity check 1: buy_low must beat fade_cheap in sports markets
        # (underdogs win 26% at <20¢; sports favorites underperform at 85-95¢ YES)
        if low_return is not None and fade_return is not None:
            lo, fa = f"{low_return:+.2%}", f"{fade_return:+.2%}"
            if low_return > fade_return:
                print(f"\n  {GREEN}✓ SPORTS SANITY CHECK PASSED:{RESET} buy_low ({lo}) > fade_cheap ({fa})")
            else:
                print(f"\n  {RED}✗ SPORTS SANITY CHECK FAILED:{RESET} buy_low ({lo}) ≤ fade_cheap ({fa})")
                print("    Unexpected if running top-5% sports-heavy sample.")

        # Sanity check 2: yes_at_30s must beat coin_flip
        # (30-37¢ calibration edge should outperform zero-edge + commission drag)
        if yes_return is not None and flip_return is not None:
            ye, fl = f"{yes_return:+.2%}", f"{flip_return:+.2%}"
            if yes_return > flip_return:
                print(f"\n  {GREEN}✓ EDGE SANITY CHECK PASSED:{RESET} yes_at_30s ({ye}) > coin_flip ({fl})")
            else:
                print(f"\n  {RED}✗ EDGE SANITY CHECK FAILED:{RESET} yes_at_30s ({ye}) ≤ coin_flip ({fl})")
                print("    The 6pp 30-37¢ edge should survive 1% commission in sports markets.")

        # Cost model check: coin_flip must be negative (zero-edge + commission)
        if flip_return is not None:
            fl = f"{flip_return:+.2%}"
            if flip_return < 0:
                print(f"\n  {GREEN}✓ COST MODEL CHECK PASSED:{RESET} coin_flip ({fl}) — commission drag confirmed.")
            else:
                print(
                    f"\n  {RED}✗ COST MODEL CHECK FAILED:{RESET} coin_flip is positive ({fl})"
                    " — possible look-ahead bias."
                )

        # Primary hypothesis: buy_low is profitable in sports markets
        if low_return is not None:
            lo = f"{low_return:+.2%}"
            if low_return > 0:
                print(f"\n  {GREEN}✓ PRIMARY SPORTS HYPOTHESIS CONFIRMED:{RESET} buy_low is profitable ({lo}).")
                print("    Sports underdogs win ~26% at <20¢ vs 10-20% implied — massive edge.")
            else:
                print(f"\n  {YELLOW}⚠ PRIMARY SPORTS HYPOTHESIS NOT MET:{RESET} buy_low ({lo}) — negative.")
                print("    Check whether sample contains enough NBA/NFL/NHL/soccer markets.")

        # Secondary hypothesis: yes_at_30s is profitable
        if yes_return is not None:
            ye = f"{yes_return:+.2%}"
            if yes_return > 0:
                print(f"\n  {GREEN}✓ SECONDARY HYPOTHESIS CONFIRMED:{RESET} yes_at_30s is profitable ({ye}).")
            else:
                print(f"\n  {YELLOW}⚠ SECONDARY HYPOTHESIS NOT MET:{RESET} yes_at_30s ({ye}) — negative.")
                print("    The 6pp edge at 30-37¢ may not survive 1% commission in this sample.")

    print()


if __name__ == "__main__":
    main()
