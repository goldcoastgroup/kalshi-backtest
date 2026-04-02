# Kalshi Backtest Research Program

## 1. Setup (once per run)

1. Agree on a **run tag** with the user (e.g. `apr02`).
2. Create and switch to branch `autoresearch/<tag>`.
3. Read `prepare.py` and `train.py` to understand the current state.
4. Verify the local data cache exists:
   ```bash
   ls .cache/backtest_data.pkl
   ```
   If missing, run `uv run train.py` once to build it (fetches from MongoDB + Kalshi API, then pickles locally). The `.cache/` directory is gitignored and persists across branch changes, so data loading is only slow on the very first run (~10s). Subsequent runs deserialize from pickle in ~1s.
5. Initialize `results.tsv`:
   ```
   commit	pnl	return_pct	total_fees	total_fills	win_rate	status	description
   ```

## 2. What You Can Modify

- **Only `train.py`** — everything else is frozen.
- You may change: strategy logic, strategy-specific config constants (spread, position limits, sizing, etc.), strategy class.
- You may NOT: modify `prepare.py`, modify `engine/`, add new dependencies, change the fee model, change `EVENT_TICKERS`, or change `STARTING_BALANCE` (these are frozen in `prepare.py`).

### Hard constraints — never violate these

- **Prices must stay in [0.01, 0.99]** — the engine enforces this; submitting outside this range will reject.
- **Quantities must be positive whole numbers** — the engine enforces this.
- **Do not modify the engine or prepare.py** — these define the exchange simulation and are frozen.
- **Fee model is fixed**: taker fee = `ceil(0.07 * qty * price * (1-price) * 100) / 100`, maker fee = 0.
- **No hindsight bias.** Every strategy decision must be based on information available *before* the backtest runs — never on the results of a previous run. This is the most important constraint. Concretely:
  - **Never hardcode instrument-specific behavior.** You cannot say "skip KXRT-BRI-55 because it lost money." You *can* say "reduce size when `hours_left < 12` because late-stage markets are harder to exit" — that is a universal rule the strategy discovers from attributes it observes at decision time.
  - **Filter by attributes, not by identity.** If certain instruments lose money, ask *why* — is it low liquidity? FV near 0.5? High gamma? Then encode that attribute-based rule in the strategy so it generalizes to unseen instruments.
  - **The strategy must be the same function for every instrument.** The strategy class is instantiated once per instrument with identical logic. Differences in behavior should come from the data the strategy observes at runtime (FV, book state, greeks, `hours_left`, `cur_score`, `total_reviews`), not from the instrument's ticker string.
  - **Test your reasoning:** before committing a change, ask "would this rule make sense on a new movie I've never backtested?" If the answer is no, it's overfitting.

## 3. How Experiments Work

Each experiment edits `train.py`, runs it, and evaluates the result.

- `train.py` defines a Strategy subclass and config constants.
- The strategy receives callbacks: `on_data(FairValueData)`, `on_book_update(instrument_id, timestamp_ns)`, `on_fill(Fill)`.
- The strategy can call: `submit_order()`, `cancel_order()`, `modify_order()`, `best_bid()`, `best_ask()`, `get_position()`, `get_balance()`, `get_free_balance()`.
- At the bottom, `train.py` calls `prepare.run_backtest(strategy_factory=MyStrategy)`.
- The standardized output block is printed at the end:
  ```
  ---
  pnl:              +139.33
  return_pct:       +1.39
  total_fees:       12.45
  total_fills:      287
  win_rate:         66.7
  n_instruments:    24
  run_seconds:      3.2
  total_seconds:    8.5
  ```

## 4. The Experiment Loop

```
LOOP FOREVER:
1. Check git status (current branch, clean working tree)
2. Edit train.py with your next idea
3. git add train.py && git commit -m "experiment: <short description>"
4. Run: timeout 300 uv run python train.py > run.log 2>&1
5. Extract: grep "^pnl:" run.log
6. If empty (crash or timeout):
   - tail -n 50 run.log
   - If timeout (exit code 124): strategy is too slow, simplify and re-run
   - If trivial fix (typo, import): fix, commit, re-run
   - If fundamental: log as crash, git reset, move on
7. Record in results.tsv:
   commit<TAB>pnl<TAB>return_pct<TAB>total_fees<TAB>total_fills<TAB>win_rate<TAB>status<TAB>description
8. If pnl IMPROVED (higher than previous best):
   - Status: keep
   - This commit becomes the new baseline
9. If pnl SAME or WORSE:
   - Status: discard
   - git reset --hard <previous best commit>
10. GOTO 1
```

### Critical rules

- **NEVER STOP.** Loop indefinitely until the user interrupts. There is always another idea to try. Do not pause to ask questions, summarize, or wait for approval. Just keep running experiments.
- **Every run has a 5-minute hard cap.** Use `timeout 300` on every run. If a strategy exceeds 5 minutes, it is too slow — treat it as a crash, discard, and design a faster approach.
- **Do not deliberate between experiments.** Commit, run, record, move on. Thinking time is wasted time — you learn more from running an experiment than from theorizing about it.

## 5. Available Data

The strategy receives two types of data via callbacks:

### FairValueData (on_data callback)
- `fv`: fair value probability [0, 1] — model estimate of YES outcome probability
- `theta`: time-decay Greek (how fast FV changes per hour with no new info)
- `gamma_pos`: FV sensitivity to a positive (fresh) review
- `gamma_neg`: FV sensitivity to a negative (rotten) review
- `new_review`: True at first minute of an epoch triggered by a new review
- `hours_left`: hours until market close
- `cur_score`: current Tomatometer score [0, 1]
- `total_reviews`: cumulative review count

### OrderBook (on_book_update callback)
- Access via `self.best_bid()` → `(price, size)` and `self.best_ask()` → `(price, size)`
- Real L2 orderbook snapshots and deltas from Kalshi

### Order Types
- `TimeInForce.GTC` — Good till canceled (rests on book as maker order)
- `TimeInForce.FOK` — Fill or kill (aggressive taker, fills immediately or rejects)
- `TimeInForce.IOC` — Immediate or cancel (fills what it can, cancels remainder)
- `TimeInForce.POST_ONLY` — Maker only (rejects if it would cross the spread)

### Order Modification & Queue Position
Kalshi queue semantics are modeled in the engine:
- **Decrease quantity** (`modify_order` with smaller qty): preserves queue position. Use this when you want to reduce exposure without losing your place in line.
- **Increase quantity** (`modify_order` with larger qty): resets queue position to back of line.
- **Practical implication**: if you have a resting order to buy 10 and now want to buy 20, it is better to submit a *new* order for 10 more (preserving the original order's queue position) rather than modifying the existing order to 20 (which would lose queue priority). Conversely, if you want to reduce from 10 to 5, modify in place to keep your queue spot.

## 6. Research Directions

### Strategy Archetypes to Explore

1. **Market Making (FV-based)**
   - Quote bid/ask around fair value with a spread
   - Use theta/gamma for dynamic spread adjustment
   - Kelly criterion for position sizing
   - Inventory management: widen spread when position grows

2. **Directional (FV vs Market)**
   - When FV significantly differs from mid-market, take aggressive positions
   - Scale size by confidence (FV distance from 0.5, review count, hours left)
   - Short skew: only buy when FV > 0.5, only sell when FV < 0.5

3. **Event-Driven (Review Reactions)**
   - Use `new_review` flag and `gamma_pos`/`gamma_neg` to predict post-review moves
   - Trade aggressively when a new review shifts FV but market hasn't adjusted

4. **Hybrid Approaches**
   - Combine market making (passive) with directional signals (aggressive)
   - Use book imbalance as short-term signal, FV as medium-term signal

### Parameter Tuning
- Spread width, position limits, Kelly fraction
- Cooldown periods, minimum size thresholds
- Event ticker selection (which movies to trade)
- Balance allocation

### Key Considerations
- Binary options: prices bounded [0.01, 0.99], settle at $0 or $1
- Fully collateralized: buying costs `price`, shorting costs `(1-price)`
- Fees eat into thin edges — need sufficient spread to cover
- Settlement PnL: open positions settle at market close
- Variance scaling: edge near 0.5 has highest variance (`p*(1-p)`)

## 7. Tips

- **Start small**: One change per experiment. Don't combine strategy change + parameter change.
- **Baseline first**: Make sure the baseline runs cleanly before iterating.
- **Watch fees**: A strategy with many small trades can lose money on fees alone.
- **Position limits matter**: Uncapped positions amplify both gains and losses.
- **Check per-instrument PnL**: Some movies are easier to trade than others.
- **Log everything**: The results.tsv description should be clear enough to understand what changed.
- **FV is powerful**: The fair value model is calibrated — strategies that use it tend to outperform pure order flow strategies.
