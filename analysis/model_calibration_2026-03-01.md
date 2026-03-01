# FV Model Calibration Analysis

**Date:** 2026-03-01
**Data:** 658 resolved KXRT markets, 21,867 hourly (ticker, timestamp) snapshots
**Source:** `fv-timeseries/data/*.parquet` joined with `data/kalshi/markets/*.parquet` and `data/kalshi/trades/*.parquet`

---

## Summary

The fv model systematically overestimates YES probability across all price levels, by 10–20pp in the mid-range. The market is far better calibrated. When the strategy sees a 17pp apparent edge (fv − market_price), the real edge (actual outcome − market_price) is only ~7pp on average and can be negative in certain price buckets. The flat `min_edge=0.17` threshold works better than `min_kelly_fraction` not because it is theoretically sound, but because it accidentally acts as a price ceiling that concentrates bets in the regime where the model's overconfidence does the least harm.

---

## Calibration by fv_T Bucket

*"When the model says fv_T = X%, does the market actually resolve YES X% of the time?"*

| fv_T range | n (snapshots) | actual YES | model fv_T | market VWAP | model error | market error |
|---|---|---|---|---|---|---|
| 0–10%    | 6,791 | 0.6%  | 1.7%  | 3.2%  | −1.2pp | −2.7pp |
| 10–20%   | 1,217 | 6.7%  | 14.5% | 13.1% | −7.7pp | −6.4pp |
| 20–30%   |   948 | 8.8%  | 24.5% | 20.1% | **−15.8pp** | −11.3pp |
| 30–40%   |   718 | 21.6% | 35.3% | 25.2% | **−13.7pp** | −3.6pp |
| 40–50%   |   760 | 33.2% | 44.8% | 34.2% | **−11.6pp** | −1.0pp |
| 50–60%   |   821 | 38.7% | 55.0% | 43.5% | **−16.3pp** | −4.8pp |
| 60–70%   |   905 | 47.1% | 65.2% | 50.0% | **−18.1pp** | −2.9pp |
| 70–80%   | 1,036 | 55.0% | 75.1% | 58.5% | **−20.1pp** | −3.4pp |
| 80–90%   | 1,481 | 67.6% | 85.3% | 70.3% | **−17.7pp** | −2.7pp |
| 90–100%  | 7,186 | 93.8% | 98.0% | 91.7% | −4.2pp | +2.1pp |

**Key observations:**
- Model error is negative in every bucket — the model uniformly overestimates YES probability.
- Worst overestimation is in the 70–80% fv_T range: model says 75%, market says 58%, actual is 55%.
- Market error is much smaller than model error throughout. The market is consistently closer to reality.
- Only at the extremes (0–10%, 90–100%) does the model's calibration approach acceptable levels.

---

## Calibration by Market Price Bucket

*"Is the market itself well-calibrated? What does the actual resolution rate look like at each price level?"*

| price bucket | n | actual YES | avg VWAP | avg fv_T | market edge | model edge |
|---|---|---|---|---|---|---|
| 0–10¢   | 7,454 | 1.0%  | 2.7%  | 4.9%  | −1.7pp | −3.9pp |
| 10–20¢  | 1,526 | 12.5% | 14.5% | 26.2% | −2.1pp | −13.7pp |
| 20–30¢  | 1,076 | 19.4% | 24.9% | 40.0% | −5.5pp | −20.6pp |
| 30–40¢  |   865 | 33.5% | 34.9% | 48.9% | −1.3pp | −15.4pp |
| 40–50¢  |   879 | 45.8% | 44.8% | 59.1% | +1.0pp | −13.3pp |
| 50–60¢  |   809 | 54.4% | 55.2% | 66.6% | −0.8pp | −12.3pp |
| 60–70¢  | 1,020 | 54.4% | 65.1% | 75.6% | **−10.7pp** | **−21.2pp** |
| 70–80¢  | 1,031 | 68.5% | 75.2% | 83.5% | −6.8pp | −15.0pp |
| 80–90¢  | 1,643 | 84.3% | 85.5% | 90.1% | −1.2pp | −5.8pp |
| 90–100¢ | 5,564 | 97.4% | 96.7% | 98.1% | +0.7pp | −0.7pp |

**Key observations:**
- The market is much better calibrated than the model across all buckets.
- The 60–70¢ bucket is the market's worst region: priced at 65%, actual only 54% (−10.7pp). Also where model error is worst (−21.2pp).
- At 90–100¢, the market has slight positive edge (+0.7pp) and model error is near zero (−0.7pp).

---

## Apparent Edge vs Real Edge

*"When fv_T − market_price = X (the strategy's entry signal), what is the actual realized edge?"*

Across all snapshots where apparent edge ≥ 0.17 (the strategy's default filter):
- **Average apparent edge:** 30.8pp
- **Average real edge (actual − market):** 7.1pp
- **Model overclaims by:** ~22pp

| apparent edge bucket | n | actual YES | avg fv_T | avg price | apparent edge | real edge | model overclaim |
|---|---|---|---|---|---|---|---|
| < −20%   |   340 | 35.0% | 31.8% | 60.9% | −29.1pp | −25.9pp | −3.2pp |
| −20:−10% |   673 | 37.3% | 38.6% | 52.5% | −13.9pp | −15.2pp | +1.3pp |
| −10:0%   | 7,132 | 14.0% | 15.9% | 18.1% | −2.2pp  | −4.2pp  | +1.9pp |
| 0:5%     | 6,355 | 73.0% | 76.0% | 74.2% | +1.9pp  | −1.2pp  | +3.1pp |
| 5:10%    | 2,228 | 56.7% | 65.9% | 58.5% | +7.4pp  | −1.8pp  | +9.2pp |
| 10:17%   | 1,860 | 49.0% | 63.4% | 50.2% | +13.2pp | **−1.2pp** | +14.4pp |
| 17:25%   | 1,362 | 48.2% | 66.1% | 45.4% | +20.7pp | **+2.9pp** | +17.8pp |
| 25:40%   | 1,295 | 46.9% | 68.2% | 37.2% | +31.1pp | **+9.7pp** | +21.4pp |
| >40%     |   622 | 36.3% | 77.6% | 25.1% | +52.5pp | **+11.3pp** | +41.2pp |

**Key observation:** Even the best apparent-edge bucket (>40%) only delivers 11.3pp real edge, against a claimed 52.5pp. The model overclaims by a factor of ~4–5x throughout.

---

## Real Edge by Price Level (where apparent edge is 10–30%)

*The price level where the strategy places most bets, broken down by market price.*

| price | n | actual YES | avg fv_T | avg price | apparent edge | real edge | model overclaim |
|---|---|---|---|---|---|---|---|
| 0–10¢   | 547 | 5.9%  | 22.0% | 4.9%  | +17.1pp | +0.9pp  | +16.1pp |
| 10–20¢  | 427 | 16.6% | 33.3% | 14.6% | +18.7pp | +2.0pp  | +16.7pp |
| 20–30¢  | 321 | 25.9% | 44.6% | 25.2% | +19.5pp | +0.7pp  | +18.8pp |
| 30–40¢  | 286 | 36.7% | 54.7% | 35.1% | +19.7pp | +1.7pp  | +18.0pp |
| 40–50¢  | 307 | 49.8% | 64.7% | 45.0% | +19.7pp | +4.9pp  | +14.9pp |
| 50–60¢  | 349 | 57.9% | 75.3% | 55.2% | +20.1pp | +2.7pp  | +17.5pp |
| 60–70¢  | 511 | 62.6% | 85.9% | 65.3% | +20.7pp | **−2.7pp** | +23.3pp |
| 70–80¢  | 544 | 76.1% | 92.7% | 75.1% | +17.6pp | +1.0pp  | +16.6pp |
| 80–90¢  | 538 | 90.3% | 97.5% | 84.4% | +13.1pp | +5.9pp  | +7.2pp |

Real edge is nearly flat (0.7–5.9pp) and slightly negative at 60–70¢, across a range where apparent edge is consistently ~20pp. The model's overclaim is similar in magnitude to the apparent edge throughout.

---

## Why flat min_edge Outperforms min_kelly_fraction

This is not a finding about filter logic — it is a finding about model calibration.

1. **min_edge=0.17 is a structural price ceiling.** No market above ~83¢ can generate 17pp raw edge without fv > 1.0 (impossible). This concentrates bets at low prices (0–30¢) where, despite tiny per-snapshot real edge, the asymmetric payoff structure means rare wins are large.

2. **min_kelly_fraction removes that ceiling.** At 97¢, `kf = (fv − 0.97) / 0.03`. Even a 3¢ model overestimate passes kf=1.0, trigging a 25% position size on a bet with near-zero real edge. The model's 4pp overclaim in the 90–100% fv_T bucket translates directly into 14 large losses (−25% free cash each) out of 361 bets, driving the high drawdown.

3. **The fix is model recalibration, not filter adjustment.** If the model's fv_T values were properly calibrated (actual = predicted), min_kelly_fraction would be the theoretically correct filter. Until calibration is addressed, any filter that admits high-price bets will be exploiting a regime where the model's edge signal is noise.

---

## Next Steps

- Recalibrate fv_T using isotonic regression or Platt scaling against historical outcomes.
- Evaluate whether calibrated fv_T + min_kelly_fraction outperforms flat min_edge.
- Investigate why the 60–70¢ market price bucket has the worst market calibration (−10.7pp) — possible structural bias in how Kalshi prices these markets.
