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
