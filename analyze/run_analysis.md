---
name: run_analysis
description: Run full backtest analysis with in-sample vs out-of-sample tearsheet comparison
---

# Run Analysis

Generate comprehensive tearsheet reports comparing in-sample performance (6 training events) vs out-of-sample performance (KXRT-FRO holdout event).

## Steps

### 1. Clean previous reports

Delete all files in `analyze/` except this skill file (`run_analysis.md`):

```bash
find analyze/ -type f ! -name 'run_analysis.md' -delete
```

### 2. Run in-sample analysis

Run this Python snippet:

```bash
cd /home/bgram/dev/kalshi-backtest && .venv/bin/python -c "
import train
import prepare
prepare.run_backtest_analysis(strategy_factory=train.HybridFV, output_dir='analyze')
"
```

This generates `analyze/in-sample-{hash}.html`, `analyze/in-sample-{hash}.md`, and `analyze/in-sample-{hash}.json`.

### 3. Run out-of-sample analysis

Run this Python snippet:

```bash
cd /home/bgram/dev/kalshi-backtest && .venv/bin/python -c "
import train
import prepare
prepare.run_backtest_analysis(strategy_factory=train.HybridFV, event_tickers=['KXRT-FRO'], output_dir='analyze')
"
```

This generates `analyze/oos-{hash}.html`, `analyze/oos-{hash}.md`, and `analyze/oos-{hash}.json`.

### 4. Read and compare

Read both markdown files from the `analyze/` directory. Present a comparison report to the user covering:

- **Top-line metrics side by side**: PnL, Return %, Win Rate Over Expected, Mean Edge, Turnover
- **Overfitting check**: Compare win_rate_over_expected between in-sample and OOS. A large positive in-sample delta with a negative OOS delta is a red flag.
- **Edge quality**: Compare mean edge, % positive edge fills, maker/taker breakdown
- **Per-event context**: How does the single OOS event compare to individual in-sample events?
- **Capital efficiency**: PnL per dollar deployed, turnover ratios

### 5. Commit reports

```bash
git add analyze/
git commit -m "analysis: tearsheet reports for $(git rev-parse --short HEAD)"
```
