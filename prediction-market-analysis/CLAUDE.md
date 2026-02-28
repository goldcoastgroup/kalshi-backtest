# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup (installs tools + downloads ~36GB of data)
make setup

# Run analyses interactively
make analyze

# Run a specific analysis by name
make run <analysis_name>

# Data collection (interactive menu)
make index

# Testing
make test                        # Run all tests
uv run pytest tests/ -v -k <name>  # Run a single test

# Linting
make lint                        # Check for issues
make format                      # Auto-fix issues
```

## Architecture

This is a plugin-based framework for collecting and analyzing prediction market data from Kalshi and Polymarket.

### Plugin Discovery

Both `Analysis` (`src/common/analysis.py`) and `Indexer` (`src/common/indexer.py`) are auto-discovered at runtime. Any class inheriting from `Analysis` placed in `src/analysis/**/*.py` or from `Indexer` in `src/indexers/**/*.py` is automatically registered — no manual imports needed.

### Adding a New Analysis

Subclass `Analysis`, set `name` and `description`, and implement `run()` returning `AnalysisOutput`. Place the file in `src/analysis/{kalshi,polymarket,comparison}/`. See `docs/ANALYSIS.md` for full templates and common DuckDB query patterns.

```python
class MyAnalysis(Analysis):
    name = "my_analysis"
    description = "What it does"

    def run(self) -> AnalysisOutput:
        base_dir = Path(__file__).parent.parent.parent.parent
        con = duckdb.connect()
        df = con.execute(f"SELECT ... FROM '{base_dir}/data/kalshi/trades/*.parquet'").df()
        fig, ax = plt.subplots()
        # ... build figure ...
        chart = bar_chart(...)  # optional web chart config
        return AnalysisOutput(figure=fig, data=df, chart=chart)
```

`Analysis.save()` handles all export formats automatically (PNG at 300 DPI, PDF, CSV, JSON). Use `self.progress("message")` as a context manager for spinners on slow operations.

### Storage Layer

Data lives in `data/{kalshi,polymarket}/` as chunked Parquet files (10,000 records/chunk). DuckDB glob queries span all chunks: `FROM 'data/kalshi/trades/*.parquet'`. `ParquetStorage` (`src/common/storage.py`) handles deduplication and cursor-based resumable indexing.

### Kalshi Data Model

- **Markets**: `ticker`, `event_ticker`, `status` (`open`/`closed`/`finalized`), `result` (`yes`/`no`/empty), prices in cents (1–99)
- **Trades**: `trade_id`, `ticker`, `yes_price`, `no_price` (always `100 - yes_price`), `taker_side` (`yes`/`no`), `count`

### Polymarket Data Model

- **Trades (CTF/NegRisk)**: On-chain `OrderFilled` events from Polygon; amounts use 6 decimals (USDC)
- **Legacy FPMM Trades**: Pre-2022 `FPMMBuy`/`FPMMSell` events; `amount`/`outcome_tokens` stored as strings to avoid overflow
- Block → timestamp mapping in `data/polymarket/blocks/*.parquet`

### Kalshi Category Utilities

`src/analysis/kalshi/util/categories.py` maps `event_ticker` prefixes to high-level groups (Sports, Politics, Crypto, etc.):

```python
from src.analysis.kalshi.util.categories import get_group, GROUP_COLORS
get_group("NFLGAME")  # "Sports"
```

### Chart Interface

`src/common/interfaces/chart.py` provides typed helpers for generating web-renderable chart configs alongside matplotlib figures. Available: `line_chart()`, `bar_chart()`, `area_chart()`, `pie_chart()`, `scatter_chart()`, `heatmap()`, `treemap()`. These are serialized to JSON alongside the figure.

### Testing Patterns

Tests use session-scoped pytest fixtures (`tests/conftest.py`) that create temporary Parquet files. Analysis classes accept custom `trades_dir`/`markets_dir` constructor arguments to point at fixture data. Use `@pytest.mark.slow` for slow tests.
