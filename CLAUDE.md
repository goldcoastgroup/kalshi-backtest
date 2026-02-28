# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build the Rust extension (required after any Rust changes)
make build-rust                          # cd crates/backtesting_engine && maturin develop --release

# Run a backtest
make backtest                            # interactive strategy picker
make backtest -- --strategy buy_low      # specific strategy

# Run a fronttest (live market feed, paper broker)
make fronttest

# Testing
make test                                # uv run pytest tests/ -v
uv run pytest tests/ -v -k <name>        # single test

# Linting / formatting
make lint                                # ruff check + ruff format --check
make format                              # ruff check --fix + ruff format

# Analysis subproject (delegates to prediction-market-analysis/Makefile)
make analyze
make run <analysis_name>
```

## Architecture

### Python / Rust split

The hot simulation loop (broker, portfolio, lifecycle events) runs as a compiled Rust PyO3 extension (`crates/backtesting_engine/`). Strategy callbacks always execute in Python. The Python wrapper lives at `src/backtesting/rust_engine.py`; it exposes the same `Engine` API as the pure-Python fallback.

**After any change to Rust source, you must run `make build-rust` before testing.**

### Strategy plugin system

Subclass `Strategy` (`src/backtesting/strategy.py`) and drop the file in `src/backtesting/strategies/`. `Strategy.load()` auto-discovers all non-abstract subclasses at runtime — no manual registration required.

Minimum implementation:

```python
class MyStrategy(Strategy):
    def __init__(self):
        super().__init__(name="my_strategy", initial_cash=10_000.0)

    def on_trade(self, trade: TradeEvent) -> None:
        self.buy_yes(trade.market_id, price=0.45, quantity=10)
```

Lifecycle hooks (all optional): `initialize()`, `finalize()`, `on_fill()`, `on_market_open()`, `on_market_close()`, `on_market_resolve()`.

### Feed abstraction

All feeds implement `BaseFeed` (`src/backtesting/feeds/base.py`). Strategies receive normalized `TradeEvent` objects regardless of platform — Kalshi cents are divided by 100 during feed normalization so all prices are `float` in `[0.0, 1.0]`.

Available feeds:
- `KalshiFeed` / `KalshiLiveFeed` — `src/backtesting/feeds/kalshi.py`
- `PolymarketFeed` / `PolymarketLiveFeed` — `src/backtesting/feeds/polymarket.py`

### Fill semantics (CLOB — prevents look-ahead bias)

A resting limit order fills only when **both** conditions are met:

| Order | Price condition | Taker condition |
|---|---|---|
| BUY YES | `yes_price ≤ limit` | `taker_side == NO` (seller hit the bid) |
| SELL YES | `yes_price ≥ limit` | `taker_side == YES` (buyer lifted the ask) |
| BUY NO | `no_price ≤ limit` | `taker_side == YES` (NO seller is YES buyer) |
| SELL NO | `no_price ≥ limit` | `taker_side == NO` (NO buyer lifted ask) |

Requiring the taker-side condition prevents filling on a price that merely *passed through* the limit — matching how Kalshi and Polymarket CLOBs actually work.

### Commission models

Auto-detected from platform; can be overridden in `Engine(commission_rate=..., flat_commission=...)`.

- **Kalshi** (default): `commission = rate × P × (1−P) × qty` — `flat_commission=False`, `rate=0.07`
- **Polymarket** (default): `commission = rate × P × qty` — `flat_commission=True`, `rate=0.001`

### Data layout

`data/` is a symlink to `prediction-market-analysis/data/`. Historical data lives in chunked Parquet files under `data/{kalshi,polymarket}/`. DuckDB glob queries span all chunks: `FROM 'data/kalshi/trades/*.parquet'`.

#### `data/kalshi/markets/*.parquet` schema

| Column | Type | Notes |
|---|---|---|
| `ticker` | VARCHAR | Market identifier |
| `event_ticker` | VARCHAR | Parent event identifier |
| `market_type` | VARCHAR | e.g. `"binary"` |
| `title` | VARCHAR | Human-readable market title |
| `yes_sub_title` | VARCHAR | YES outcome description |
| `no_sub_title` | VARCHAR | NO outcome description |
| `status` | VARCHAR | e.g. `"active"`, `"closed"` |
| `yes_bid` | BIGINT | Integer cents (0–100) |
| `yes_ask` | BIGINT | Integer cents (0–100) |
| `no_bid` | BIGINT | Integer cents (0–100) |
| `no_ask` | BIGINT | Integer cents (0–100) |
| `last_price` | BIGINT | Integer cents (0–100) |
| `volume` | BIGINT | Total contracts traded |
| `volume_24h` | BIGINT | 24h contract volume |
| `open_interest` | BIGINT | Open interest |
| `result` | VARCHAR | `"yes"` / `"no"` / `""` (unresolved) |
| `created_time` | TIMESTAMP WITH TIME ZONE | |
| `open_time` | TIMESTAMP WITH TIME ZONE | |
| `close_time` | TIMESTAMP WITH TIME ZONE | |
| `_fetched_at` | TIMESTAMP_NS | Collection metadata timestamp |

#### `data/kalshi/trades/*.parquet` schema

| Column | Type | Notes |
|---|---|---|
| `trade_id` | VARCHAR | UUID |
| `ticker` | VARCHAR | Market identifier |
| `count` | BIGINT | Number of contracts |
| `yes_price` | BIGINT | Integer cents (0–100); divide by 100 for float |
| `no_price` | BIGINT | Integer cents (0–100); always `100 - yes_price` |
| `taker_side` | VARCHAR | `"yes"` or `"no"` |
| `created_time` | TIMESTAMP WITH TIME ZONE | Trade timestamp |
| `_fetched_at` | TIMESTAMP_NS | Collection metadata timestamp |

Files are chunked by sequential ID range (e.g. `trades_0_10000.parquet`). All prices in raw parquet are **integer cents**; the feed normalizer divides by 100.

### MongoDB databases

This project uses two MongoDB databases related to Kalshi × Rotten Tomatoes (KXRT) prediction markets, where Kalshi markets are created on whether a movie will achieve a given Tomatometer score threshold.

#### `kxrt` — live/operational database

**`events`** — One document per active or finalized Kalshi RT event (movie).

| Field | Type | Notes |
|---|---|---|
| `_id` | String | Kalshi event ticker, e.g. `KXRTSOULM8TE` |
| `title` | String | Movie title |
| `status` | String | e.g. `"active"`, `"finalized"` |
| `market_open` | Date | When Kalshi markets opened |
| `market_close` | Date | When Kalshi markets closed |
| `premiere` | Date | Movie theatrical premiere date |
| `rt_link` | String | Full Rotten Tomatoes URL |
| `media_path` | String | RT media path, e.g. `"/m/soulm8te"` |
| `ems_id` | String | EMS UUID linking to movies/reviews |
| `last_modified` | Date | Last pipeline update time |
| `markets` | Array | Sub-markets, each with `ticker` (String), `strike` (Number — RT score threshold, e.g. 45/60/75/90), `status` (String) |

**`movies`** — Movies that have/had active events (subset of `kxrt-training.movies-all`).

| Field | Type | Notes |
|---|---|---|
| `_id` | String | EMS UUID (matches `ems_id` in reviews) |
| `title` | String | Movie title |
| `premiere` | Date | Theatrical premiere date |
| `rt_link` | String | Rotten Tomatoes URL |
| `media_path` | String | RT media path |
| `creation_time` | Date | When this record was created |

**`reviews`** — Individual RT critic reviews, collected in real-time during the market window.

| Field | Type | Notes |
|---|---|---|
| `_id` | String | RT review ID |
| `timestamp` | Date | When review was published/collected |
| `score` | Number | `1` = Fresh, `0` = Rotten |
| `metadata.ems_id` | String | Links to `movies._id` and `events.ems_id` |
| `metadata.movie_title` | String | |
| `metadata.publication_name` | String | Outlet name |
| `metadata.critic_name` | String | Critic name |
| `metadata.review_url` | String | URL to the review |

**`bad-events`** — Blocklist of event tickers with bad/invalid data. Only field is `_id` (String). Used exclusively by the webscraper to skip known-bad events during ingestion — bad events are never written to `kxrt.events`, so application code does not need to cross-reference this collection.

**`tv-cache`** — Currently empty; likely a cache for TV-related market data.

---

#### `kxrt-training` — historical/ML training database

**`movies-all`** — Superset of `kxrt.movies`; all movies regardless of whether they had a Kalshi event.

| Field | Type | Notes |
|---|---|---|
| `_id` | String | EMS UUID |
| `title` | String | |
| `premiere` | Date | |
| `rt_link` | String | |
| `media_path` | String | |
| `creation_time` | Date | |
| `persisted_reviews` | Boolean | Whether reviews were collected into `reviews-all` |
| `persisted_reviews_tz` | Boolean | Whether TZ-corrected reviews were collected into `reviews-all-tz` |

**`reviews-all`** — Flattened version of `kxrt.reviews` (metadata promoted to top level) for all movies.

| Field | Type | Notes |
|---|---|---|
| `_id` | String | RT review ID |
| `creation_date` | Date | Review publication date |
| `score` | Number | `1` = Fresh, `0` = Rotten |
| `ems_id` | String | Links to movie |
| `movie_title` | String | |
| `publication_name` | String | |
| `critic_name` | String | |
| `review_url` | String \| null | |

**`reviews-all-tz`** — Same schema as `reviews-all` but with timezone-corrected `creation_date`. `critic_name` can be null. Used for more accurate temporal feature engineering.

**`review-features-raw`** — Pre-computed ML training features. One document per (movie, time-step `t`).

| Field | Type | Notes |
|---|---|---|
| `_id` | String | `"{movie_id}-{t}"` |
| `movie_id` | String | EMS UUID |
| `movie_title` | String \| Number | Movie title (occasional type anomaly) |
| `t` | Number | Time step index (integer, e.g. days from market open) |
| `premiere_t` | Number | `t` value at theatrical premiere |
| `first_review_t` | Number | `t` of first review (NaN if none yet) |
| `total_reviews` | Number | Cumulative reviews as of time `t` |
| `total_positive` | Number | Cumulative fresh reviews as of time `t` |
| `total_score` | Number | `total_positive / total_reviews` |
| `count` | Number | Reviews in this time window/step |
| `count_positive` | Number | Fresh reviews in this window |
| `pre_window_reviews` | Number | Reviews published before the market open window |
| `pre_window_positive` | Number | Fresh reviews before window |
| `pre_window_score` | Number | Score before window |
| `test_only` | Boolean \| Number | If true, this movie is held out for testing only |

---

### Subproject relationship

`prediction-market-analysis/` is an independent Python project with its own `pyproject.toml`, `Makefile`, and `CLAUDE.md`. It handles data collection and market analysis. The backtesting project imports its data via the `data/` symlink. Unknown `make` targets in the root Makefile are forwarded to the subproject's Makefile.
