# Backtest Analysis — in-sample-195bf64

## Top-Line Metrics

| Metric | Value |
|--------|-------|
| PnL | $+6,361.54 |
| Return % | +63.62% |
| Win Rate | 59.0% |
| Expected Win Rate | 34.3% |
| **Win Rate Over Expected** | **+24.6%** |
| Mean Edge at Fill | 0.1657 |
| Median Edge at Fill | 0.2093 |
| Qty-Weighted Edge | 0.1740 |
| % Positive Edge Fills | 91.1% |
| Total Fees | $333.82 |
| Turnover | $24,931.80 |
| Total Fills | 1,061 |
| Total Positions | 39 |
| Profit Factor | 67.12 |

## Per-Event Breakdown

| Event | Instruments | PnL | Win Rate |
|-------|-------------|-----|----------|
| KXRT-BRI | 7 | $+718.87 | 71% |
| KXRT-HOP | 8 | $+1,042.76 | 62% |
| KXRT-PRO | 5 | $+1,351.83 | 40% |
| KXRT-REA | 8 | $+1,524.90 | 75% |
| KXRT-REM | 4 | $+177.06 | 75% |
| KXRT-WIL | 7 | $+1,879.94 | 29% |

## Top Instruments

| Instrument | PnL | Avg Edge | Fills | PnL/$ |
|------------|-----|----------|-------|-------|
| KXRT-WIL-60 | $+1,833.16 | 0.2497 | 30 | 0.3398 |
| KXRT-REA-75 | $+1,383.18 | 0.2119 | 130 | 0.3587 |
| KXRT-PRO-95 | $+1,321.16 | 0.2023 | 229 | 0.2759 |
| KXRT-HOP-94 | $+848.77 | 0.2357 | 25 | 0.4694 |
| KXRT-BRI-60 | $+616.46 | 0.1954 | 131 | 0.2986 |

## Bottom Instruments

| Instrument | PnL | Avg Edge | Fills | PnL/$ |
|------------|-----|----------|-------|-------|
| KXRT-PRO-90 | $-6.45 | 0.2398 | 3 | -1.5505 |
| KXRT-WIL-80 | $-7.43 | 0.1553 | 4 | -0.2873 |
| KXRT-WIL-70 | $-14.51 | 0.0354 | 11 | -0.0652 |
| KXRT-BRI-50 | $-24.76 | 0.1017 | 15 | -0.1510 |
| KXRT-WIL-65 | $-34.20 | 0.1386 | 36 | -0.1280 |

## Win Rate vs Expected

| Probability Bucket | Count | Actual WR | Expected WR | Delta |
|--------------------|-------|-----------|-------------|-------|
| 0.03 | 13 | 38% | 1% | +37.1% |
| 0.07 | 2 | 50% | 6% | +44.4% |
| 0.12 | 1 | 0% | 11% | -11.0% |
| 0.17 | 4 | 25% | 17% | +7.8% |
| 0.23 | 1 | 100% | 23% | +76.8% |
| 0.28 | 1 | 0% | 29% | -29.0% |
| 0.33 | 2 | 50% | 32% | +17.9% |
| 0.57 | 1 | 0% | 60% | -60.0% |
| 0.62 | 1 | 100% | 64% | +35.9% |
| 0.68 | 2 | 100% | 66% | +33.8% |
| 0.72 | 3 | 100% | 74% | +25.6% |
| 0.78 | 5 | 100% | 77% | +23.0% |
| 0.82 | 2 | 100% | 82% | +18.2% |
| 0.88 | 1 | 100% | 86% | +14.0% |

## Capital Efficiency

- Starting balance: $10,000.00
- Final balance: $16,361.54
- Total capital deployed: $24,927.87
- PnL per dollar deployed: 0.0042
- PnL / Turnover: 25.52%

## Settlement vs Trade Exits

- Settled at expiry: PnL $+5,790.56
- Closed by trade: PnL $+904.80
