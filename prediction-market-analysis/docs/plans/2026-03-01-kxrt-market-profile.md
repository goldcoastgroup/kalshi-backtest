# KXRT Market Profile Analysis Suite — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build 9 independent `Analysis` subclasses that profile KXRT prediction markets in the last 5 days before close, informing market-making strategy around liquidity timing, spread sizing, and flow direction.

**Architecture:** A shared DuckDB CTE helper (`src/analysis/kalshi/util/kxrt_trades.py`) provides the base query (filter trades to `KXRT%` tickers, join markets for `close_time`, compute `hours_to_close`, keep 0–119h window). Each of the 9 analyses extends this CTE with its own aggregation. Tests use synthetic KXRT fixture data injected via `trades_dir` / `markets_dir` constructor args; the existing `test_analysis_run.py` auto-discovery handles all 9 analyses automatically once routing is updated.

**Tech Stack:** Python, DuckDB, pandas, matplotlib, PyArrow, pytest

---

### Task 1: Create shared KXRT trades CTE utility

**Files:**
- Create: `src/analysis/kalshi/util/kxrt_trades.py`

**Step 1: Create the file**

```python
# src/analysis/kalshi/util/kxrt_trades.py
"""Shared DuckDB CTE for KXRT market trades in the last 5 days before close."""

from __future__ import annotations


def kxrt_base_cte(trades_dir: str, markets_dir: str) -> str:
    """Return a WITH-clause fragment for KXRT trades within 0–119 hours of market close.

    The CTE is named ``kxrt_trades`` and exposes all trade columns plus:
      - hours_to_close (INTEGER): whole hours remaining until market close

    Usage::

        sql = f'''
            WITH {kxrt_base_cte(self.trades_dir, self.markets_dir)}
            SELECT (hours_to_close / 6) * 6 AS bucket, SUM(count) AS vol
            FROM kxrt_trades GROUP BY bucket
        '''
    """
    return f"""
        kxrt_trades AS (
            SELECT
                t.*,
                date_diff('hour', t.created_time, m.close_time) AS hours_to_close
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE t.ticker LIKE 'KXRT%'
              AND date_diff('hour', t.created_time, m.close_time) BETWEEN 0 AND 119
        )
    """
```

**Step 2: Verify the module imports cleanly**

```bash
cd prediction-market-analysis && uv run python -c "from src.analysis.kalshi.util.kxrt_trades import kxrt_base_cte; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add prediction-market-analysis/src/analysis/kalshi/util/kxrt_trades.py
git commit -m "feat: add kxrt_base_cte shared DuckDB utility"
```

---

### Task 2: Add KXRT test fixtures and update auto-discovery routing

**Files:**
- Modify: `prediction-market-analysis/tests/conftest.py`
- Modify: `prediction-market-analysis/tests/test_analysis_run.py`

**Step 1: Add fixture helpers and session fixtures to conftest.py**

Add the following two helper functions **after** `_make_kalshi_markets` (around line 60):

```python
def _make_kxrt_trades() -> pd.DataFrame:
    """Build KXRT trades spread across 0–119 hours before a fixed close_time (240 rows)."""
    close_time = pd.Timestamp("2024-06-10 12:00:00", tz="UTC")
    tickers = ["KXRTMOVIETEST-45", "KXRTMOVIETEST-60"]
    rows = []
    trade_id = 0
    for ticker in tickers:
        for hours_offset in range(1, 121):  # 1h–120h before close → hours_to_close 0–119
            trade_id += 1
            rows.append(
                {
                    "trade_id": str(trade_id),
                    "ticker": ticker,
                    "count": 10,
                    "yes_price": 50,
                    "no_price": 50,
                    "taker_side": "yes" if trade_id % 2 == 0 else "no",
                    "created_time": close_time - pd.Timedelta(hours=hours_offset),
                    "_fetched_at": close_time,
                }
            )
    return pd.DataFrame(rows)


def _make_kxrt_markets() -> pd.DataFrame:
    """Build KXRT markets with close_time (2 rows)."""
    close_time = pd.Timestamp("2024-06-10 12:00:00", tz="UTC")
    return pd.DataFrame(
        [
            {
                "ticker": "KXRTMOVIETEST-45",
                "event_ticker": "KXRTMOVIETEST",
                "status": "finalized",
                "result": "yes",
                "volume": 1000,
                "close_time": close_time,
            },
            {
                "ticker": "KXRTMOVIETEST-60",
                "event_ticker": "KXRTMOVIETEST",
                "status": "finalized",
                "result": "no",
                "volume": 800,
                "close_time": close_time,
            },
        ]
    )
```

Then add two session-scoped fixtures **after** `kalshi_markets_dir`:

```python
@pytest.fixture(scope="session")
def kxrt_trades_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("kxrt_trades")
    _make_kxrt_trades().to_parquet(d / "trades.parquet")
    return d


@pytest.fixture(scope="session")
def kxrt_markets_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("kxrt_markets")
    _make_kxrt_markets().to_parquet(d / "markets.parquet")
    return d
```

Then update the `all_fixture_dirs` fixture signature and body to include the two new fixtures:

```python
@pytest.fixture(scope="session")
def all_fixture_dirs(
    kalshi_trades_dir: Path,
    kalshi_markets_dir: Path,
    kxrt_trades_dir: Path,
    kxrt_markets_dir: Path,
    polymarket_trades_dir: Path,
    polymarket_legacy_trades_dir: Path,
    polymarket_markets_dir: Path,
    polymarket_blocks_dir: Path,
    collateral_lookup_path: Path,
) -> dict[str, Path]:
    """Bundle all fixture directories for easy access."""
    return {
        "kalshi_trades_dir": kalshi_trades_dir,
        "kalshi_markets_dir": kalshi_markets_dir,
        "kxrt_trades_dir": kxrt_trades_dir,
        "kxrt_markets_dir": kxrt_markets_dir,
        "polymarket_trades_dir": polymarket_trades_dir,
        "polymarket_legacy_trades_dir": polymarket_legacy_trades_dir,
        "polymarket_markets_dir": polymarket_markets_dir,
        "polymarket_blocks_dir": polymarket_blocks_dir,
        "collateral_lookup_path": collateral_lookup_path,
    }
```

**Step 2: Update `_build_kwargs` in test_analysis_run.py to route KXRT analyses**

Add `is_kxrt` detection and put its checks **before** the existing `is_kalshi` checks:

```python
def _build_kwargs(cls: type[Analysis], fixture_dirs: dict[str, Path]) -> dict[str, Path]:
    """Map constructor params to fixture paths based on platform module."""
    sig = inspect.signature(cls.__init__)
    params = [p for p in sig.parameters if p != "self"]

    module = cls.__module__
    is_kxrt = "kxrt" in cls.__name__.lower()
    is_kalshi = ".kalshi." in module
    is_polymarket = ".polymarket." in module

    kwargs: dict[str, Path] = {}
    for param in params:
        # Direct match — comparison module params use explicit platform prefixes
        if param in fixture_dirs:
            kwargs[param] = fixture_dirs[param]
        elif is_kxrt and param == "trades_dir":
            kwargs[param] = fixture_dirs["kxrt_trades_dir"]
        elif is_kxrt and param == "markets_dir":
            kwargs[param] = fixture_dirs["kxrt_markets_dir"]
        elif is_kalshi and param == "trades_dir":
            kwargs[param] = fixture_dirs["kalshi_trades_dir"]
        elif is_kalshi and param == "markets_dir":
            kwargs[param] = fixture_dirs["kalshi_markets_dir"]
        elif is_polymarket and param == "trades_dir":
            kwargs[param] = fixture_dirs["polymarket_trades_dir"]
        elif is_polymarket and param == "legacy_trades_dir":
            kwargs[param] = fixture_dirs["polymarket_legacy_trades_dir"]
        elif is_polymarket and param == "markets_dir":
            kwargs[param] = fixture_dirs["polymarket_markets_dir"]
        elif is_polymarket and param == "blocks_dir":
            kwargs[param] = fixture_dirs["polymarket_blocks_dir"]

    return kwargs
```

**Step 3: Run tests to confirm existing suite still passes**

```bash
cd prediction-market-analysis && uv run pytest tests/ -v
```

Expected: all existing tests pass, no new failures.

**Step 4: Commit**

```bash
git add prediction-market-analysis/tests/conftest.py prediction-market-analysis/tests/test_analysis_run.py
git commit -m "test: add KXRT fixture helpers and auto-discovery routing"
```

---

### Task 3: kxrt_volume_by_hours_to_close_rt

**Files:**
- Create: `prediction-market-analysis/src/analysis/kalshi/kxrt_volume_by_hours_to_close_rt.py`

**Step 1: Create the analysis file**

```python
"""KXRT market volume by hours remaining to close."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.kalshi.util.kxrt_trades import kxrt_base_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class KxrtVolumeByHoursToCloseRtAnalysis(Analysis):
    """Trading volume across the final 5 days of KXRT markets, by 6-hour bucket."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kxrt_volume_by_hours_to_close_rt",
            description="KXRT market volume by hours remaining to close",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()
        with self.progress("Querying KXRT trades"):
            df = con.execute(
                f"""
                WITH {kxrt_base_cte(self.trades_dir, self.markets_dir)}
                SELECT
                    (hours_to_close / 6) * 6 AS hours_bucket,
                    SUM(count) AS total_contracts
                FROM kxrt_trades
                GROUP BY hours_bucket
                ORDER BY hours_bucket DESC
                """
            ).df()
        return AnalysisOutput(figure=self._figure(df), data=df, chart=self._chart(df))

    def _figure(self, df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels = [f"{int(b)}-{int(b) + 6}h" for b in df["hours_bucket"]]
        ax.bar(labels, df["total_contracts"], color="#4C72B0")
        ax.set_xlabel("Hours to Close")
        ax.set_ylabel("Total Contracts")
        ax.set_title("KXRT Volume by Hours to Close (Last 5 Days)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def _chart(self, df: pd.DataFrame) -> ChartConfig:
        return ChartConfig(
            type=ChartType.BAR,
            data=[
                {
                    "hours_to_close": f"{int(r['hours_bucket'])}-{int(r['hours_bucket']) + 6}h",
                    "total_contracts": int(r["total_contracts"]),
                }
                for _, r in df.iterrows()
            ],
            xKey="hours_to_close",
            yKeys=["total_contracts"],
            title="KXRT Volume by Hours to Close",
            xLabel="Hours to Close",
            yLabel="Total Contracts",
            yUnit=UnitType.NUMBER,
        )
```

**Step 2: Run tests — auto-discovery picks up the new analysis**

```bash
cd prediction-market-analysis && uv run pytest tests/test_analysis_run.py -v -k "KxrtVolumeByHoursToClose"
```

Expected: `PASSED`

**Step 3: Commit**

```bash
git add prediction-market-analysis/src/analysis/kalshi/kxrt_volume_by_hours_to_close_rt.py
git commit -m "feat: add kxrt_volume_by_hours_to_close_rt analysis"
```

---

### Task 4: kxrt_volume_by_hour_of_day_rt

**Files:**
- Create: `prediction-market-analysis/src/analysis/kalshi/kxrt_volume_by_hour_of_day_rt.py`

**Step 1: Create the analysis file**

```python
"""KXRT market volume by UTC hour of day."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.kalshi.util.kxrt_trades import kxrt_base_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class KxrtVolumeByHourOfDayRtAnalysis(Analysis):
    """KXRT market volume aggregated by UTC hour of day."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kxrt_volume_by_hour_of_day_rt",
            description="KXRT market volume by UTC hour of day",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()
        with self.progress("Querying KXRT trades"):
            df = con.execute(
                f"""
                WITH {kxrt_base_cte(self.trades_dir, self.markets_dir)}
                SELECT
                    HOUR(created_time AT TIME ZONE 'UTC') AS hour_of_day,
                    SUM(count) AS total_contracts
                FROM kxrt_trades
                GROUP BY hour_of_day
                ORDER BY hour_of_day
                """
            ).df()
        return AnalysisOutput(figure=self._figure(df), data=df, chart=self._chart(df))

    def _figure(self, df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(df["hour_of_day"], df["total_contracts"], color="#4C72B0")
        ax.set_xticks(range(0, 24))
        ax.set_xlabel("UTC Hour of Day")
        ax.set_ylabel("Total Contracts")
        ax.set_title("KXRT Volume by Hour of Day (UTC)")
        plt.tight_layout()
        return fig

    def _chart(self, df: pd.DataFrame) -> ChartConfig:
        return ChartConfig(
            type=ChartType.BAR,
            data=[
                {
                    "hour_of_day": f"{int(r['hour_of_day']):02d}:00",
                    "total_contracts": int(r["total_contracts"]),
                }
                for _, r in df.iterrows()
            ],
            xKey="hour_of_day",
            yKeys=["total_contracts"],
            title="KXRT Volume by Hour of Day",
            xLabel="UTC Hour",
            yLabel="Total Contracts",
            yUnit=UnitType.NUMBER,
        )
```

**Step 2: Run tests**

```bash
cd prediction-market-analysis && uv run pytest tests/test_analysis_run.py -v -k "KxrtVolumeByHourOfDay"
```

Expected: `PASSED`

**Step 3: Commit**

```bash
git add prediction-market-analysis/src/analysis/kalshi/kxrt_volume_by_hour_of_day_rt.py
git commit -m "feat: add kxrt_volume_by_hour_of_day_rt analysis"
```

---

### Task 5: kxrt_trade_arrival_rate_rt

**Files:**
- Create: `prediction-market-analysis/src/analysis/kalshi/kxrt_trade_arrival_rate_rt.py`

**Step 1: Create the analysis file**

```python
"""KXRT trade arrival rate (trades per hour) by hours to close."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.kalshi.util.kxrt_trades import kxrt_base_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class KxrtTradeArrivalRateRtAnalysis(Analysis):
    """Trade arrival rate (trades per hour) across the final 5 days of KXRT markets."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kxrt_trade_arrival_rate_rt",
            description="KXRT trade arrival rate by hours to close",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()
        with self.progress("Querying KXRT trades"):
            df = con.execute(
                f"""
                WITH {kxrt_base_cte(self.trades_dir, self.markets_dir)}
                SELECT
                    (hours_to_close / 6) * 6 AS hours_bucket,
                    COUNT(*) AS trade_count,
                    COUNT(*) / 6.0 AS trades_per_hour
                FROM kxrt_trades
                GROUP BY hours_bucket
                ORDER BY hours_bucket DESC
                """
            ).df()
        return AnalysisOutput(figure=self._figure(df), data=df, chart=self._chart(df))

    def _figure(self, df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels = [f"{int(b)}-{int(b) + 6}h" for b in df["hours_bucket"]]
        ax.plot(labels, df["trades_per_hour"], marker="o", color="#4C72B0")
        ax.set_xlabel("Hours to Close")
        ax.set_ylabel("Trades per Hour")
        ax.set_title("KXRT Trade Arrival Rate by Hours to Close")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def _chart(self, df: pd.DataFrame) -> ChartConfig:
        return ChartConfig(
            type=ChartType.LINE,
            data=[
                {
                    "hours_to_close": f"{int(r['hours_bucket'])}-{int(r['hours_bucket']) + 6}h",
                    "trades_per_hour": round(float(r["trades_per_hour"]), 2),
                }
                for _, r in df.iterrows()
            ],
            xKey="hours_to_close",
            yKeys=["trades_per_hour"],
            title="KXRT Trade Arrival Rate",
            xLabel="Hours to Close",
            yLabel="Trades per Hour",
            yUnit=UnitType.NUMBER,
        )
```

**Step 2: Run tests**

```bash
cd prediction-market-analysis && uv run pytest tests/test_analysis_run.py -v -k "KxrtTradeArrivalRate"
```

Expected: `PASSED`

**Step 3: Commit**

```bash
git add prediction-market-analysis/src/analysis/kalshi/kxrt_trade_arrival_rate_rt.py
git commit -m "feat: add kxrt_trade_arrival_rate_rt analysis"
```

---

### Task 6: kxrt_avg_trade_size_rt

**Files:**
- Create: `prediction-market-analysis/src/analysis/kalshi/kxrt_avg_trade_size_rt.py`

**Step 1: Create the analysis file**

```python
"""KXRT average contracts per trade by hours to close."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.kalshi.util.kxrt_trades import kxrt_base_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class KxrtAvgTradeSizeRtAnalysis(Analysis):
    """Average contracts per trade across the final 5 days of KXRT markets."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kxrt_avg_trade_size_rt",
            description="KXRT average trade size by hours to close",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()
        with self.progress("Querying KXRT trades"):
            df = con.execute(
                f"""
                WITH {kxrt_base_cte(self.trades_dir, self.markets_dir)}
                SELECT
                    (hours_to_close / 6) * 6 AS hours_bucket,
                    AVG(count) AS avg_trade_size
                FROM kxrt_trades
                GROUP BY hours_bucket
                ORDER BY hours_bucket DESC
                """
            ).df()
        return AnalysisOutput(figure=self._figure(df), data=df, chart=self._chart(df))

    def _figure(self, df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels = [f"{int(b)}-{int(b) + 6}h" for b in df["hours_bucket"]]
        ax.plot(labels, df["avg_trade_size"], marker="o", color="#4C72B0")
        ax.set_xlabel("Hours to Close")
        ax.set_ylabel("Avg Contracts per Trade")
        ax.set_title("KXRT Average Trade Size by Hours to Close")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def _chart(self, df: pd.DataFrame) -> ChartConfig:
        return ChartConfig(
            type=ChartType.LINE,
            data=[
                {
                    "hours_to_close": f"{int(r['hours_bucket'])}-{int(r['hours_bucket']) + 6}h",
                    "avg_trade_size": round(float(r["avg_trade_size"]), 2),
                }
                for _, r in df.iterrows()
            ],
            xKey="hours_to_close",
            yKeys=["avg_trade_size"],
            title="KXRT Average Trade Size",
            xLabel="Hours to Close",
            yLabel="Avg Contracts per Trade",
            yUnit=UnitType.NUMBER,
        )
```

**Step 2: Run tests**

```bash
cd prediction-market-analysis && uv run pytest tests/test_analysis_run.py -v -k "KxrtAvgTradeSize"
```

Expected: `PASSED`

**Step 3: Commit**

```bash
git add prediction-market-analysis/src/analysis/kalshi/kxrt_avg_trade_size_rt.py
git commit -m "feat: add kxrt_avg_trade_size_rt analysis"
```

---

### Task 7: kxrt_taker_imbalance_rt

**Files:**
- Create: `prediction-market-analysis/src/analysis/kalshi/kxrt_taker_imbalance_rt.py`

**Step 1: Create the analysis file**

```python
"""KXRT YES vs NO taker imbalance by hours to close."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.kalshi.util.kxrt_trades import kxrt_base_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class KxrtTakerImbalanceRtAnalysis(Analysis):
    """YES vs NO taker direction across the final 5 days of KXRT markets."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kxrt_taker_imbalance_rt",
            description="KXRT YES vs NO taker imbalance by hours to close",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()
        with self.progress("Querying KXRT trades"):
            df = con.execute(
                f"""
                WITH {kxrt_base_cte(self.trades_dir, self.markets_dir)}
                SELECT
                    (hours_to_close / 6) * 6 AS hours_bucket,
                    100.0 * SUM(CASE WHEN taker_side = 'yes' THEN count ELSE 0 END) / SUM(count) AS yes_pct,
                    100.0 * SUM(CASE WHEN taker_side = 'no'  THEN count ELSE 0 END) / SUM(count) AS no_pct
                FROM kxrt_trades
                GROUP BY hours_bucket
                ORDER BY hours_bucket DESC
                """
            ).df()
        return AnalysisOutput(figure=self._figure(df), data=df, chart=self._chart(df))

    def _figure(self, df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels = [f"{int(b)}-{int(b) + 6}h" for b in df["hours_bucket"]]
        ax.bar(labels, df["yes_pct"], label="YES taker", color="#2196F3")
        ax.bar(labels, df["no_pct"], bottom=df["yes_pct"], label="NO taker", color="#F44336")
        ax.set_xlabel("Hours to Close")
        ax.set_ylabel("Share of Volume (%)")
        ax.set_title("KXRT Taker Imbalance by Hours to Close")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def _chart(self, df: pd.DataFrame) -> ChartConfig:
        return ChartConfig(
            type=ChartType.STACKED_BAR_100,
            data=[
                {
                    "hours_to_close": f"{int(r['hours_bucket'])}-{int(r['hours_bucket']) + 6}h",
                    "yes_pct": round(float(r["yes_pct"]), 1),
                    "no_pct": round(float(r["no_pct"]), 1),
                }
                for _, r in df.iterrows()
            ],
            xKey="hours_to_close",
            yKeys=["yes_pct", "no_pct"],
            title="KXRT Taker Imbalance",
            xLabel="Hours to Close",
            yLabel="Share (%)",
            yUnit=UnitType.PERCENT,
            colors={"yes_pct": "#2196F3", "no_pct": "#F44336"},
        )
```

**Step 2: Run tests**

```bash
cd prediction-market-analysis && uv run pytest tests/test_analysis_run.py -v -k "KxrtTakerImbalance"
```

Expected: `PASSED`

**Step 3: Commit**

```bash
git add prediction-market-analysis/src/analysis/kalshi/kxrt_taker_imbalance_rt.py
git commit -m "feat: add kxrt_taker_imbalance_rt analysis"
```

---

### Task 8: kxrt_price_volatility_rt

**Files:**
- Create: `prediction-market-analysis/src/analysis/kalshi/kxrt_price_volatility_rt.py`

**Step 1: Create the analysis file**

```python
"""KXRT price volatility (std dev of traded prices) by hours to close."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.kalshi.util.kxrt_trades import kxrt_base_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class KxrtPriceVolatilityRtAnalysis(Analysis):
    """Std dev of traded yes_price (cents) within 6-hour windows across KXRT markets."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kxrt_price_volatility_rt",
            description="KXRT price volatility by hours to close",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()
        with self.progress("Querying KXRT trades"):
            df = con.execute(
                f"""
                WITH {kxrt_base_cte(self.trades_dir, self.markets_dir)}
                SELECT
                    (hours_to_close / 6) * 6 AS hours_bucket,
                    COALESCE(STDDEV(yes_price), 0) AS price_vol_cents
                FROM kxrt_trades
                GROUP BY hours_bucket
                ORDER BY hours_bucket DESC
                """
            ).df()
        return AnalysisOutput(figure=self._figure(df), data=df, chart=self._chart(df))

    def _figure(self, df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels = [f"{int(b)}-{int(b) + 6}h" for b in df["hours_bucket"]]
        ax.plot(labels, df["price_vol_cents"], marker="o", color="#4C72B0")
        ax.set_xlabel("Hours to Close")
        ax.set_ylabel("Price Std Dev (cents)")
        ax.set_title("KXRT Price Volatility by Hours to Close")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def _chart(self, df: pd.DataFrame) -> ChartConfig:
        return ChartConfig(
            type=ChartType.LINE,
            data=[
                {
                    "hours_to_close": f"{int(r['hours_bucket'])}-{int(r['hours_bucket']) + 6}h",
                    "price_vol_cents": round(float(r["price_vol_cents"]), 2),
                }
                for _, r in df.iterrows()
            ],
            xKey="hours_to_close",
            yKeys=["price_vol_cents"],
            title="KXRT Price Volatility",
            xLabel="Hours to Close",
            yLabel="Price Std Dev (cents)",
            yUnit=UnitType.CENTS,
        )
```

**Step 2: Run tests**

```bash
cd prediction-market-analysis && uv run pytest tests/test_analysis_run.py -v -k "KxrtPriceVolatility"
```

Expected: `PASSED`

**Step 3: Commit**

```bash
git add prediction-market-analysis/src/analysis/kalshi/kxrt_price_volatility_rt.py
git commit -m "feat: add kxrt_price_volatility_rt analysis"
```

---

### Task 9: kxrt_trade_price_impact_rt

**Files:**
- Create: `prediction-market-analysis/src/analysis/kalshi/kxrt_trade_price_impact_rt.py`

**Step 1: Create the analysis file**

```python
"""KXRT mean absolute trade-to-trade price impact by hours to close."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.kalshi.util.kxrt_trades import kxrt_base_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class KxrtTradePriceImpactRtAnalysis(Analysis):
    """Mean absolute consecutive trade price change per 6-hour bucket across KXRT markets."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kxrt_trade_price_impact_rt",
            description="KXRT trade-to-trade price impact by hours to close",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()
        with self.progress("Querying KXRT trades"):
            df = con.execute(
                f"""
                WITH {kxrt_base_cte(self.trades_dir, self.markets_dir)},
                trade_sequence AS (
                    SELECT
                        (hours_to_close / 6) * 6 AS hours_bucket,
                        yes_price,
                        LAG(yes_price) OVER (PARTITION BY ticker ORDER BY created_time) AS prev_price
                    FROM kxrt_trades
                )
                SELECT
                    hours_bucket,
                    AVG(ABS(yes_price - prev_price)) AS avg_price_impact_cents
                FROM trade_sequence
                WHERE prev_price IS NOT NULL
                GROUP BY hours_bucket
                ORDER BY hours_bucket DESC
                """
            ).df()
        return AnalysisOutput(figure=self._figure(df), data=df, chart=self._chart(df))

    def _figure(self, df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels = [f"{int(b)}-{int(b) + 6}h" for b in df["hours_bucket"]]
        ax.plot(labels, df["avg_price_impact_cents"], marker="o", color="#4C72B0")
        ax.set_xlabel("Hours to Close")
        ax.set_ylabel("Avg |ΔPrice| (cents)")
        ax.set_title("KXRT Trade-to-Trade Price Impact by Hours to Close")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def _chart(self, df: pd.DataFrame) -> ChartConfig:
        return ChartConfig(
            type=ChartType.LINE,
            data=[
                {
                    "hours_to_close": f"{int(r['hours_bucket'])}-{int(r['hours_bucket']) + 6}h",
                    "avg_price_impact_cents": round(float(r["avg_price_impact_cents"]), 2),
                }
                for _, r in df.iterrows()
            ],
            xKey="hours_to_close",
            yKeys=["avg_price_impact_cents"],
            title="KXRT Trade Price Impact",
            xLabel="Hours to Close",
            yLabel="Avg |ΔPrice| (cents)",
            yUnit=UnitType.CENTS,
        )
```

**Step 2: Run tests**

```bash
cd prediction-market-analysis && uv run pytest tests/test_analysis_run.py -v -k "KxrtTradePriceImpact"
```

Expected: `PASSED`

**Step 3: Commit**

```bash
git add prediction-market-analysis/src/analysis/kalshi/kxrt_trade_price_impact_rt.py
git commit -m "feat: add kxrt_trade_price_impact_rt analysis"
```

---

### Task 10: kxrt_effective_spread_rt

**Files:**
- Create: `prediction-market-analysis/src/analysis/kalshi/kxrt_effective_spread_rt.py`

**Step 1: Create the analysis file**

```python
"""KXRT effective spread proxy (median consecutive trade price gap) by hours to close."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.kalshi.util.kxrt_trades import kxrt_base_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class KxrtEffectiveSpreadRtAnalysis(Analysis):
    """Median consecutive trade price gap as a realized spread proxy across KXRT markets."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kxrt_effective_spread_rt",
            description="KXRT effective spread proxy by hours to close",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()
        with self.progress("Querying KXRT trades"):
            df = con.execute(
                f"""
                WITH {kxrt_base_cte(self.trades_dir, self.markets_dir)},
                trade_sequence AS (
                    SELECT
                        (hours_to_close / 6) * 6 AS hours_bucket,
                        yes_price,
                        LAG(yes_price) OVER (PARTITION BY ticker ORDER BY created_time) AS prev_price
                    FROM kxrt_trades
                )
                SELECT
                    hours_bucket,
                    MEDIAN(ABS(yes_price - prev_price)) AS median_spread_cents
                FROM trade_sequence
                WHERE prev_price IS NOT NULL
                GROUP BY hours_bucket
                ORDER BY hours_bucket DESC
                """
            ).df()
        return AnalysisOutput(figure=self._figure(df), data=df, chart=self._chart(df))

    def _figure(self, df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels = [f"{int(b)}-{int(b) + 6}h" for b in df["hours_bucket"]]
        ax.plot(labels, df["median_spread_cents"], marker="o", color="#4C72B0")
        ax.set_xlabel("Hours to Close")
        ax.set_ylabel("Median |ΔPrice| (cents)")
        ax.set_title("KXRT Effective Spread Proxy by Hours to Close")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def _chart(self, df: pd.DataFrame) -> ChartConfig:
        return ChartConfig(
            type=ChartType.LINE,
            data=[
                {
                    "hours_to_close": f"{int(r['hours_bucket'])}-{int(r['hours_bucket']) + 6}h",
                    "median_spread_cents": float(r["median_spread_cents"]),
                }
                for _, r in df.iterrows()
            ],
            xKey="hours_to_close",
            yKeys=["median_spread_cents"],
            title="KXRT Effective Spread Proxy",
            xLabel="Hours to Close",
            yLabel="Median |ΔPrice| (cents)",
            yUnit=UnitType.CENTS,
        )
```

**Step 2: Run tests**

```bash
cd prediction-market-analysis && uv run pytest tests/test_analysis_run.py -v -k "KxrtEffectiveSpread"
```

Expected: `PASSED`

**Step 3: Commit**

```bash
git add prediction-market-analysis/src/analysis/kalshi/kxrt_effective_spread_rt.py
git commit -m "feat: add kxrt_effective_spread_rt analysis"
```

---

### Task 11: kxrt_volume_by_strike_rt

**Files:**
- Create: `prediction-market-analysis/src/analysis/kalshi/kxrt_volume_by_strike_rt.py`

**Step 1: Create the analysis file**

The ticker format is `KXRTEVENTTICKER-STRIKE` (e.g. `KXRTRENTALFAMILY-45`). Extract the strike with `regexp_extract(ticker, '-([0-9]+)$', 1)`.

```python
"""KXRT total volume by strike level."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.kalshi.util.kxrt_trades import kxrt_base_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class KxrtVolumeByStrikeRtAnalysis(Analysis):
    """Total KXRT trading volume broken down by strike level, split by taker side."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kxrt_volume_by_strike_rt",
            description="KXRT volume by strike level",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()
        with self.progress("Querying KXRT trades"):
            df = con.execute(
                f"""
                WITH {kxrt_base_cte(self.trades_dir, self.markets_dir)}
                SELECT
                    TRY_CAST(regexp_extract(ticker, '-([0-9]+)$', 1) AS INTEGER) AS strike,
                    SUM(CASE WHEN taker_side = 'yes' THEN count ELSE 0 END) AS yes_contracts,
                    SUM(CASE WHEN taker_side = 'no'  THEN count ELSE 0 END) AS no_contracts,
                    SUM(count) AS total_contracts
                FROM kxrt_trades
                GROUP BY strike
                ORDER BY strike
                """
            ).df()
        return AnalysisOutput(figure=self._figure(df), data=df, chart=self._chart(df))

    def _figure(self, df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [str(int(s)) if s is not None and not pd.isna(s) else "other" for s in df["strike"]]
        x = range(len(labels))
        width = 0.35
        ax.bar([i - width / 2 for i in x], df["yes_contracts"], width, label="YES taker", color="#2196F3")
        ax.bar([i + width / 2 for i in x], df["no_contracts"], width, label="NO taker", color="#F44336")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Total Contracts")
        ax.set_title("KXRT Volume by Strike Level")
        ax.legend()
        plt.tight_layout()
        return fig

    def _chart(self, df: pd.DataFrame) -> ChartConfig:
        return ChartConfig(
            type=ChartType.BAR,
            data=[
                {
                    "strike": str(int(r["strike"])) if r["strike"] is not None and not pd.isna(r["strike"]) else "other",
                    "yes_contracts": int(r["yes_contracts"]),
                    "no_contracts": int(r["no_contracts"]),
                }
                for _, r in df.iterrows()
            ],
            xKey="strike",
            yKeys=["yes_contracts", "no_contracts"],
            title="KXRT Volume by Strike Level",
            xLabel="Strike",
            yLabel="Total Contracts",
            yUnit=UnitType.NUMBER,
            colors={"yes_contracts": "#2196F3", "no_contracts": "#F44336"},
        )
```

**Step 2: Run tests**

```bash
cd prediction-market-analysis && uv run pytest tests/test_analysis_run.py -v -k "KxrtVolumeByStrike"
```

Expected: `PASSED`

**Step 3: Run the full test suite**

```bash
cd prediction-market-analysis && uv run pytest tests/ -v
```

Expected: all tests pass including the 9 new KXRT analyses.

**Step 4: Commit**

```bash
git add prediction-market-analysis/src/analysis/kalshi/kxrt_volume_by_strike_rt.py
git commit -m "feat: add kxrt_volume_by_strike_rt analysis"
```

---

### Final verification

Run all analyses against the real data to confirm they produce output:

```bash
cd prediction-market-analysis && uv run python -m main analyze kxrt_volume_by_hours_to_close_rt
```

If that passes, the full suite works. Check `output/` for the generated PNG/PDF/CSV.
