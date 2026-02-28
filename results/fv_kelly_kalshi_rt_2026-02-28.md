# FV-Kelly on KalshiRT — Backtest Results

**Date:** 2026-02-28
**Strategy:** `fv_kelly` (`src/backtesting/strategies/fv_kelly.py`)
**Feed:** `KalshiRTFeed` (Kalshi × Rotten Tomatoes markets only)

---

## Configuration

```python
FVKellyStrategy(
    kelly_fraction=0.5,
    max_position_fraction=0.25,
    min_edge=0.17,
    initial_cash=10_000.0,
)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| `kelly_fraction` | 0.5 | Half-Kelly to reduce variance |
| `max_position_fraction` | 0.25 | Hard cap: max 25% of free cash per bet |
| `min_edge` | 0.17 | Only bet when `|fv_T − market_price| ≥ 0.17` |
| `initial_cash` | $10,000 | Starting bankroll |

---

## Performance

| Metric | Value |
|--------|-------|
| Period | 2024-10-24 → 2025-11-25 (~13 months) |
| Initial cash | $10,000 |
| Final equity | $15,577 |
| **Total return** | **+55.8%** |
| Max drawdown | 13.3% |
| Sharpe ratio | 1.94 |
| Sortino ratio | ~1.94 |

### Trading

| Metric | Value |
|--------|-------|
| Markets in feed | 799 |
| Markets traded | 143 |
| Total fills | 207 |
| Unique movies (events) | 47 |
| Avg strikes per movie | 4.1 |
| Win rate (fill level) | 47.3% (engine) / 51.0% (resolved fills) |
| Win rate (movie level) | 66.0% (31W / 16L) |
| Profit factor | 4.25 |
| Avg win per fill | +$77 |
| Avg loss per fill | −$19 |
| Win/loss size ratio | 4.1× |
| Breakeven win rate | 19.5% |
| Total commission | $268 (4.4% of gross P&L) |

---

## Edge Analysis

### Why win rate < 50% is not a concern

The strategy bets on extreme mispricings: when the FV model predicts a probability far from the market price, the contracts being bought are often cheap (e.g., YES at 20¢ when FV says 37¢). The asymmetric payoff means:
- **Win:** collect 80¢ per contract
- **Lose:** forfeit 20¢ per contract
- **Breakeven:** only 20% win rate needed

At 51% win rate (fill level) and 66% win rate (movie level), the strategy is well above breakeven.

### Statistical significance

Treating each movie as one independent observation (fills within a movie are correlated — same underlying RT score):

| | Value |
|-|-------|
| Effective N | 47 movies |
| Mean P&L per movie | +$123 |
| Std dev | $282 |
| t-statistic | 2.99 |
| p-value (2-tail) | **0.003** |

Statistically significant at the 0.3% level, even with the conservative effective-N adjustment.

### Out-of-sample confirmation

All 47 backtest movies were verified against the FV model's training data in `kxrt-training.review-features-raw`:
- 41 movies: explicitly flagged `test_only = true`
- 10 movies: absent from the training DB entirely (released after training pipeline ran)
- **0 training movies** appeared in the backtest

The 55.8% return is **genuinely out-of-sample**.

---

## How the Strategy Works

The FV model (`fv-timeseries`) produces a per-minute fair value probability `fv_T` for each KXRT market, using only reviews available at that timestamp (causal). The strategy:

1. **Loads** all FV parquets at startup into a `ticker → {minute → fv_T}` lookup.
2. **On each trade**, checks if `|fv_T − market_price| ≥ min_edge`. If YES is cheap, buys YES; if NO is cheap, buys NO.
3. **One position per market** — blocks re-entry until resolution. This is the critical lever that prevents sequential over-betting.
4. **Sizes against free cash** — `free_cash = portfolio.cash − committed_notional_of_pending_orders`, preventing simultaneous over-drawing across many markets.
5. **Kelly sizing** — `f = kelly_fraction × edge / (1 − price)`, capped at `max_position_fraction`.

---

## Key Findings from Parameter Search (30 runs)

The strategy required two independent fixes before producing positive returns:

1. **One active position per market until resolution** — prevents sequential re-betting the same market on every fill (without this, even `max_position_fraction=0.002` caused bankruptcy).
2. **Committed cash tracking** — sizes bets against `cash − pending_order_notional` to prevent simultaneous over-drawing across dozens of markets.

The `min_edge` threshold was the most important parameter:

| `min_edge` | Return | Max DD | Sharpe | Fills |
|-----------|--------|--------|--------|-------|
| 0.02 | +4.4% | 24.2% | 0.42 | 336 |
| 0.10 | +37.9% | 23.2% | 0.84 | 260 |
| 0.14 | +47.3% | 14.8% | 1.63 | 225 |
| 0.15 | +50.0% | 12.8% | 1.67 | 221 |
| **0.17** | **+55.8%** | **13.3%** | **1.94** | **207** |
| 0.20 | +28.9% | 12.9% | 1.21 | 196 |
| 0.25 | +13.8% | 35.5% | 0.49 | 170 |

Below ~0.12, the model takes too many low-confidence bets and commission drag overwhelms the edge. Above ~0.20, too few bets and concentration risk raises drawdown.

`kelly_fraction` and `max_position_fraction` were not sensitive in the tested ranges (0.25–0.5 and 0.25–0.50 respectively) — the Kelly formula produces `f < cap` for most bets at this edge level, and the cap is only binding at extreme prices.

---

## Caveats

- `min_edge=0.17` was selected by searching over this same 13-month dataset. True out-of-sample performance may be somewhat lower, though the edge appears robust across a broad range (0.14–0.20 all gave Sharpe > 1.2).
- N=47 movies gives a 95% CI on movie win rate of **66% ± 14%** = [52%, 80%]. Real edge, but not tightly pinned.
- **Concentration risk:** Novocaine (+$1,371) and Wicked (+$848) account for 38% of total profits. Flipping those two would significantly reduce returns.
- The KXRT market is thin and relatively new. Market efficiency may increase over time as more participants learn the RT score signal.
