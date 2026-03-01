# Model vs Market Calibration: Price Level × Time to Close

**Date:** 2026-03-01
**Data:** 658 resolved KXRT markets, 21,867 hourly snapshots
**metric:** model_err = actual_YES_rate − fv_T;  market_err = actual_YES_rate − vwap_yes
  negative = overestimate;  MAE = mean absolute error (lower is better)

---

## Summary

The market beats the model at every time horizon and every price level, with one narrow
exception (5–7 days out, the two tie on overall MAE). The model's error is 2–3× larger
than the market's throughout. Crucially, the model does not gain an advantage closer to
close — if anything, the market becomes *more* accurate while the model stays noisy.

---

## Calibration by Time to Market Close

| time to close | n      | actual | fv_T  | price | model err | market err | winner |
|---|---|---|---|---|---|---|---|
| 0–6h          |    457 | 0.411  | 0.504 | 0.422 | **−0.093** | −0.011     | market |
| 6–12h         |    772 | 0.417  | 0.482 | 0.415 | **−0.065** | +0.002     | market |
| 12–24h        |  1,966 | 0.478  | 0.545 | 0.488 | **−0.067** | −0.010     | market |
| 1–2 days      |  3,468 | 0.450  | 0.513 | 0.460 | **−0.063** | −0.011     | market |
| 2–3 days      |  4,546 | 0.458  | 0.521 | 0.457 | **−0.063** | +0.002     | market |
| 3–5 days      |  8,306 | 0.433  | 0.521 | 0.460 | **−0.089** | −0.027     | market |
| 5–7 days      |  2,352 | 0.418  | 0.465 | 0.465 | −0.048     | −0.048     | **TIE** |

**Key finding:** The market is better at every horizon. The model's worst performance is
near close (0–6h, model_err = −0.093 vs market_err = −0.011). At 5–7 days out they
tie, but the model never leads.

---

## Spotlight: 5 Days Out (72–120h)

Overall weighted MAE — model: **0.0887**  market: **0.0273**  → market 3.2× better

| price     | n     | actual | fv_T  | price | model err  | market err | winner |
|---|---|---|---|---|---|---|---|
| 0–20¢     | 3,170 | 0.042  | 0.105 | 0.056 | −0.064     | −0.014     | market |
| 20–40¢    |   886 | 0.253  | 0.451 | 0.297 | **−0.198** | −0.044     | market |
| 40–60¢    |   817 | 0.481  | 0.616 | 0.501 | −0.135     | −0.020     | market |
| 60–80¢    |   972 | 0.573  | 0.777 | 0.699 | **−0.204** | −0.126     | market |
| 80–100¢   | 2,461 | 0.929  | 0.950 | 0.932 | −0.021     | −0.003     | market |

The model is furthest off in the 20–40¢ and 60–80¢ price ranges (−0.198 and −0.204).
Market error is consistently small (within 4–13pp) at all price levels.

---

## Spotlight: 1 Day Out (12–24h)

Overall weighted MAE — model: **0.0672**  market: **0.0280**  → market 2.4× better

| price     | n   | actual | fv_T  | price | model err  | market err | winner |
|---|---|---|---|---|---|---|---|
| 0–20¢     | 867 | 0.006  | 0.074 | 0.035 | −0.068     | −0.029     | market |
| 20–40¢    |  95 | 0.316  | 0.543 | 0.284 | **−0.227** | +0.032     | market |
| 40–60¢    |  72 | 0.583  | 0.687 | 0.479 | −0.103     | **+0.104** | **MODEL** |
| 60–80¢    | 101 | 0.594  | 0.883 | 0.717 | **−0.289** | −0.123     | market |
| 80–100¢   | 831 | 0.966  | 0.984 | 0.958 | −0.018     | +0.009     | market |

The model beats the market in the 40–60¢ range at 1 day out (n=72, small sample).
Everywhere else the market is better, and worst-case the model is off by −0.289 in
the 60–80¢ range.

---

## Full Cross-Tab: Model Error vs Market Error (price × time)

### 0–20¢
| time    |  n    | actual | fv_T  | price | model err | market err | winner |
|---|---|---|---|---|---|---|---|
| 5–7d    |   844 | 0.077  | 0.078 | 0.060 | −0.001    | +0.017     | **MODEL** |
| 3–5d    | 3,170 | 0.042  | 0.105 | 0.056 | −0.064    | −0.014     | market |
| 2–3d    | 1,892 | 0.018  | 0.065 | 0.045 | −0.047    | −0.026     | market |
| 1–2d    | 1,517 | 0.011  | 0.056 | 0.039 | −0.045    | −0.029     | market |
| 12–24h  |   867 | 0.006  | 0.074 | 0.035 | −0.068    | −0.029     | market |
| 6–12h   |   431 | 0.012  | 0.114 | 0.034 | −0.102    | −0.022     | market |
| 0–6h    |   259 | 0.008  | 0.155 | 0.030 | −0.147    | −0.022     | market |

### 20–40¢
| time    |  n  | actual | fv_T  | price | model err  | market err | winner |
|---|---|---|---|---|---|---|---|
| 5–7d    | 298 | 0.211  | 0.330 | 0.290 | −0.119     | −0.078     | market |
| 3–5d    | 886 | 0.253  | 0.451 | 0.297 | −0.198     | −0.044     | market |
| 2–3d    | 416 | 0.264  | 0.448 | 0.296 | −0.183     | −0.032     | market |
| 1–2d    | 212 | 0.302  | 0.439 | 0.284 | −0.137     | +0.018     | market |
| 12–24h  |  95 | 0.316  | 0.543 | 0.284 | **−0.227** | +0.032     | market |
| 6–12h   |  26 | 0.308  | 0.730 | 0.299 | −0.422     | +0.009     | market |

### 40–60¢
| time    |  n  | actual | fv_T  | price | model err | market err | winner |
|---|---|---|---|---|---|---|---|
| 5–7d    | 214 | 0.318  | 0.496 | 0.495 | −0.179    | −0.177     | market |
| 3–5d    | 817 | 0.481  | 0.616 | 0.501 | −0.135    | −0.020     | market |
| 2–3d    | 347 | 0.608  | 0.690 | 0.496 | −0.082    | +0.112     | **MODEL** |
| 1–2d    | 234 | 0.538  | 0.668 | 0.498 | −0.130    | +0.040     | market |
| 12–24h  |  72 | 0.583  | 0.687 | 0.479 | −0.103    | +0.104     | **MODEL** |

### 60–80¢
| time    |  n  | actual | fv_T  | price | model err  | market err | winner |
|---|---|---|---|---|---|---|---|
| 5–7d    | 307 | 0.577  | 0.683 | 0.700 | −0.106     | **−0.124** | **MODEL** |
| 3–5d    | 972 | 0.573  | 0.777 | 0.699 | −0.204     | −0.126     | market |
| 2–3d    | 365 | 0.745  | 0.867 | 0.703 | −0.122     | +0.042     | market |
| 1–2d    | 292 | 0.627  | 0.850 | 0.705 | −0.223     | −0.078     | market |
| 12–24h  | 101 | 0.594  | 0.883 | 0.717 | **−0.289** | −0.123     | market |

### 80–100¢
| time    |  n    | actual | fv_T  | price | model err | market err | winner |
|---|---|---|---|---|---|---|---|
| 5–7d    |   689 | 0.884  | 0.892 | 0.923 | −0.008    | **−0.039** | **MODEL** |
| 3–5d    | 2,461 | 0.929  | 0.950 | 0.932 | −0.021    | −0.003     | market |
| 2–3d    | 1,526 | 0.954  | 0.985 | 0.943 | −0.030    | +0.011     | market |
| 1–2d    | 1,213 | 0.965  | 0.986 | 0.952 | −0.021    | +0.013     | market |
| 12–24h  |   831 | 0.966  | 0.984 | 0.958 | −0.018    | +0.009     | market |
| 6–12h   |   302 | 0.980  | 0.968 | 0.956 | +0.012    | +0.024     | **MODEL** |

---

## Conclusions

1. **The market is better calibrated than the model at every time horizon overall.**
   MAE ratios: 3.2× worse at 5 days out, 2.4× worse at 1 day out.

2. **The model does not improve relative to the market as time passes.** Near close
   (0–6h), the model's error actually peaks (−0.093 vs market's −0.011). The market
   incorporates new information faster and more accurately.

3. **At 5–7 days out, the model and market tie on overall MAE (both −0.048).**
   This is the only regime where the model is competitive — very early, before the
   market has aggregated much signal. But even here the model isn't ahead.

4. **The model wins in isolated (price, time) cells**, mostly at extremes with small
   sample sizes (n < 100). The notable exceptions are:
   - 5–7d, 0–20¢: model slightly better (−0.001 vs +0.017)
   - 5–7d, 60–80¢: model slightly better (−0.106 vs −0.124)
   - 5–7d, 80–100¢: model slightly better (−0.008 vs −0.039)
   - 2–3d and 12–24h, 40–60¢: model better (small n)

5. **The model's worst regime is 60–80¢ near close** (−0.289 at 12–24h). At that
   price level and time, the market is at 71.7% and the model at 88.3%, but actual
   is only 59.4%. The model is dramatically over-confident.

6. **Implication for strategy:** The model's edge is not from being better-calibrated
   than the market — the market is consistently more accurate. The model's value, if
   any, comes from identifying specific directional mispricings (especially at low
   prices and early timeframes) where its review-trajectory signal differs from the
   market consensus, not from having a more accurate absolute probability estimate.
