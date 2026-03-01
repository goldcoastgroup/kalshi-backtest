# KXRT Market Profile Analysis Suite — Design

**Date:** 2026-03-01
**Goal:** Characterize KXRT (Kalshi × Rotten Tomatoes) prediction markets in the last 5 days before close, producing metrics that inform a market-making strategy (primarily: when to quote, how wide, and what directional flow looks like).

---

## Data Pipeline

All 9 analyses share the same base query pattern:

1. Load `data/kalshi/trades/*.parquet`, filter `ticker LIKE 'KXRT%'`
2. Join `data/kalshi/markets/*.parquet` on `ticker` to get `close_time`
3. Compute `hours_to_close = date_diff('hour', created_time, close_time)`
4. Filter to `0 <= hours_to_close <= 120` (last 5 days = 120 hours)

A shared helper in `src/analysis/kalshi/util/kxrt_trades.py` will expose this as a reusable DuckDB SQL fragment (CTE string) so each analysis can extend it without duplication. Prices stay in integer cents in the raw query; converted to float `[0, 1]` only when computing price-level metrics.

All analyses accept optional `trades_dir` / `markets_dir` constructor args for testability (matching existing analysis conventions).

---

## Analyses

Files live in `src/analysis/kalshi/`. All follow the `Analysis` base class pattern: implement `run()` → return `AnalysisOutput(figure, data, chart)`.

The shared x-axis for time-series analyses is **6-hour bins** from 120h-to-close down to 0h-to-close (20 bins), ordered so time flows left-to-right (far from close on the left, near close on the right).

### 1. `kxrt_volume_by_hours_to_close_rt`
- **Chart:** BAR
- **X:** Hours-to-close bucket (6h bins, 0–120h)
- **Y:** Total contracts traded
- **Purpose:** Reveals whether volume spikes near expiry or is spread evenly — primary input for "when to quote"

### 2. `kxrt_volume_by_hour_of_day_rt`
- **Chart:** BAR
- **X:** UTC hour of day (0–23)
- **Y:** Total contracts traded
- **Purpose:** Intraday liquidity windows — which hours have meaningful flow

### 3. `kxrt_trade_arrival_rate_rt`
- **Chart:** LINE
- **X:** Hours-to-close (6h bins)
- **Y:** Trades per hour (total trades in bin / bin width in hours)
- **Purpose:** Normalized trade frequency — complements #1 by counting events not volume

### 4. `kxrt_avg_trade_size_rt`
- **Chart:** LINE
- **X:** Hours-to-close (6h bins)
- **Y:** Average contracts per trade
- **Purpose:** Whether order sizes grow or shrink near expiry — informs minimum quote size

### 5. `kxrt_taker_imbalance_rt`
- **Chart:** STACKED_BAR
- **X:** Hours-to-close (6h bins)
- **Y:** YES-taker % / NO-taker % (stacked to 100%)
- **Purpose:** Directional pressure over time — is one side dominant near close?

### 6. `kxrt_price_volatility_rt`
- **Chart:** LINE
- **X:** Hours-to-close (6h bins)
- **Y:** Std dev of VWAP within each bin (cents)
- **Purpose:** How much price moves within a window — proxy for minimum viable quote width

### 7. `kxrt_trade_price_impact_rt`
- **Chart:** LINE
- **X:** Hours-to-close (6h bins)
- **Y:** Mean absolute price change trade-to-trade (cents), averaged across all KXRT markets
- **Purpose:** Per-trade adverse selection cost estimate

### 8. `kxrt_effective_spread_rt`
- **Chart:** LINE
- **X:** Hours-to-close (6h bins)
- **Y:** Median consecutive trade price gap (cents)
- **Purpose:** Realized spread proxy — lower bound on the spread you need to break even

### 9. `kxrt_volume_by_strike_rt`
- **Chart:** BAR (grouped or stacked by taker side)
- **X:** Strike level (45 / 60 / 75 / 90 / other)
- **Y:** Total contracts
- **Purpose:** Which strike markets carry the most liquidity — focus quoting effort there

---

## File Structure

```
prediction-market-analysis/
└── src/
    └── analysis/
        └── kalshi/
            ├── util/
            │   └── kxrt_trades.py          # shared CTE + helpers
            ├── kxrt_volume_by_hours_to_close_rt.py
            ├── kxrt_volume_by_hour_of_day_rt.py
            ├── kxrt_trade_arrival_rate_rt.py
            ├── kxrt_avg_trade_size_rt.py
            ├── kxrt_taker_imbalance_rt.py
            ├── kxrt_price_volatility_rt.py
            ├── kxrt_trade_price_impact_rt.py
            ├── kxrt_effective_spread_rt.py
            └── kxrt_volume_by_strike_rt.py
```

---

## Testing

Each analysis gets a test in `tests/` using the existing session-scoped fixture pattern. Fixtures inject small synthetic KXRT trades + markets parquet files via `trades_dir` / `markets_dir` constructor args. Tests verify that `run()` returns a non-null figure and non-empty data frame — no golden-file comparisons needed.
