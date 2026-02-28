# FV-Kelly — NO Direction & Mid-Price Tuning

**Date:** 2026-02-28
**Strategy:** `fv_kelly` (extended with `min_edge_no` and `min_edge_mid`)
**Feed:** `KalshiRTFeed`
**Based on:** `results/fv_kelly_kalshi_rt_2026-02-28.md`

---

## Motivation

The baseline strategy uses a single `min_edge=0.17` threshold for all bets. This test
explored whether relaxing that threshold in two specific situations would add profitable trades:

1. **NO direction** — NO bets may have a different edge profile than YES bets.
2. **Mid-price band (31–41c)** — contracts priced near 35c might be more accurately modeled
   than extreme-price contracts.

---

## Configuration Tested

New parameters added to `FVKellyStrategy`:

| Parameter | Description |
|-----------|-------------|
| `min_edge_no` | Minimum edge for NO bets (relaxed vs base `min_edge`) |
| `min_edge_mid` | Minimum edge when bought contract is priced in `[mid_price_lo, mid_price_hi]` |
| `mid_price_lo` | Lower bound of mid-price band (default 0.31) |
| `mid_price_hi` | Upper bound of mid-price band (default 0.41) |

The effective threshold for any bet is `min()` of all applicable thresholds.

---

## Results Summary

| Config | Return | Max DD | Sharpe | Fills |
|--------|--------|--------|--------|-------|
| Baseline (`min_edge=0.17` everywhere) | +55.8% | 13.3% | 1.936 | 207 |
| `no=0.16` only | +59.7% | 14.3% | 1.989 | 209 |
| `no=0.15` only | +56.6% | 13.8% | 1.866 | 209 |
| `no=0.14` only | +54.4% | 13.9% | 1.833 | 213 |
| `no=0.13` only | +27.9% | 14.5% | 1.035 | 217 |
| `no=0.16, mid=0.16, 31–41c` | **+62.4%** | **13.9%** | **2.041** | **209** |
| `no=0.16, mid=0.14, 20–80c` | +52.0% | 14.6% | 1.655 | 212 |

The mid-price band range (28–45c tested) was completely insensitive — all variations
gave identical fills (209) and identical returns. The mid-price condition adds no new bets
at any threshold tested; its small return improvement (+2.7 pp over `no=0.16` alone)
comes from slightly earlier entry timing on existing bets that happen to cross through
the 31–41c range.

---

## Conclusion: Results Not Adopted

**The sample is too small (N=47 movies, 209 fills) to distinguish signal from noise.**

The improvement from baseline to best config (+6.6 pp return, +0.105 Sharpe) is within
the expected sampling variance given N=47. Key concerns:

- `no=0.13` cliff (27.9% return, major degradation) just 3 steps below the apparent
  optimum at `no=0.16` — the edge response is sharp and unstable.
- The mid-price gain (59.7% → 62.4%) comes from timing effects on 2 fills, not new bets.
  This is noise, not a real signal about mid-price accuracy.
- Parameter search over this dataset inflates apparent performance regardless of direction.

**Decision: revert to baseline `min_edge=0.17` for all bets until a larger dataset
is available to validate direction- and price-specific thresholds.**
