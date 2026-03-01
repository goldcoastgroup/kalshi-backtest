# FV-Kelly — min_kelly_fraction vs flat min_edge

**Date:** 2026-03-01
**Strategy:** `fv_kelly` (`src/backtesting/strategies/fv_kelly.py`)
**Feed:** `KalshiRTFeed` (Kalshi × Rotten Tomatoes markets only)
**Initial cash:** $10,000

---

## Motivation

The original flat `min_edge` gate is price-blind: a 0.17 edge on an 83¢ contract means
something very different than on a 50¢ contract. The Kelly fraction `f* = edge / (1 - p)`
already normalizes for price and is the natural unit for gauging trade quality. This
experiment replaces the flat gate with a `min_kelly_fraction` threshold, then adds back
a small raw `min_edge` floor to suppress model noise at extreme prices.

---

## Baseline (flat min_edge, current defaults)

| Param | Value |
|---|---|
| `min_edge` | 0.172 |
| `min_edge_no` | 0.160 |

| Metric | Value |
|---|---|
| Return | **109.86%** |
| Max drawdown | **14.48%** |
| Sharpe | **1.207** |
| Win rate | 51.5% |
| Profit factor | 4.055 |
| Fills | 274 |
| Commission | $407 |
| Final equity | $20,986.35 |

---

## Phase 1 — min_kelly_fraction only (no raw edge floor)

First attempt: replace `min_edge` entirely with `min_kelly_fraction = f* >= threshold`.

| min_kf | Return | Max DD | Sharpe | Win rate | PF | Fills | Equity |
|---|---|---|---|---|---|---|---|
| 0.25 | 81.12% | 54.58% | 1.312 | 81.0% | 3.734 | 667 | $18,112 |
| 0.35 | 76.36% | 61.41% | 1.343 | 82.0% | 3.086 | 661 | $17,636 |
| 0.40 | 80.04% | 52.43% | 1.349 | 82.7% | 3.254 | 658 | $18,004 |
| 0.50 | 80.17% | 86.27% | 1.321 | 84.8% | 2.963 | 658 | $18,018 |
| 0.60 | 69.73% | 67.07% | 1.448 | 87.5% | 3.031 | 646 | $16,973 |
| 0.75 | 79.58% | 61.91% | 1.427 | 90.2% | 6.387 | 644 | $17,958 |
| 1.00 | 1.19%  | 41.52% | 0.417 | 97.8% | 6614  | 46  | $10,119 |

**Finding:** min_kf alone fails. Fills stay ~640–667 regardless of threshold because extreme-price
markets (e.g. 90¢ YES needing only 5¢ raw edge to hit kf=0.50) flood in. Those bets have high
win rates but explosive drawdown when they lose. The flat `min_edge` was inadvertently acting as
a noise floor at extreme prices.

---

## Phase 2 — dual gate: min_kelly_fraction + min_edge floor

Add back a modest raw edge floor to suppress extreme-price noise while using min_kf as the
primary quality bar.

| min_kf | min_edge | Return | Max DD | Sharpe | Win rate | PF | Fills | Equity |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 0.05 | 89.56%  | 31.26% | 0.924 | 63.3% | 2.931 | 354 | $18,956 |
| 0.25 | 0.05 | 105.87% | 30.20% | 1.005 | 66.5% | 3.581 | 349 | $20,587 |
| 0.30 | 0.05 | 111.19% | 33.45% | 1.020 | 65.4% | 3.991 | 341 | $21,119 |
| 0.20 | 0.08 | 108.83% | 27.47% | 1.029 | 61.0% | 3.178 | 328 | $20,883 |
| 0.25 | 0.08 | 117.33% | 27.30% | 1.077 | 64.3% | 3.628 | 322 | $21,733 |
| **0.30** | **0.08** | **119.50%** | **27.20%** | **1.092** | **62.9%** | **3.946** | **313** | **$21,950** |
| 0.20 | 0.10 | 94.31%  | 24.69% | 0.972 | 57.0% | 2.872 | 307 | $19,431 |
| 0.25 | 0.10 | 107.04% | 24.73% | 1.053 | 60.7% | 3.504 | 300 | $20,704 |
| 0.30 | 0.10 | 109.36% | 24.69% | 1.070 | 60.1% | 3.720 | 291 | $20,936 |

---

## Conclusions

The dual gate (`min_kf=0.30, min_edge=0.08`) is the best new configuration:
- **+9.6pp more return** than baseline (119.5% vs 109.9%)
- **~2x higher drawdown** (27.2% vs 14.5%) — the main cost
- Slightly lower Sharpe (1.092 vs 1.207)
- More fills (313 vs 274) — captures mid-price bets the flat gate over-rejected

The flat `min_edge=0.172` baseline still wins on Sharpe and drawdown. The dual gate is
strictly better on raw return but worse on risk-adjusted metrics.

**Decision:** Reverted to flat `min_edge` defaults. The dual gate is worth revisiting once
the drawdown source is understood (likely correlated bets on same-movie markets at adjacent
strikes, or position sizing at mid-range prices).
