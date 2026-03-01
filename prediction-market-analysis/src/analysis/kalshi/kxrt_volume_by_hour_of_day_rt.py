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
