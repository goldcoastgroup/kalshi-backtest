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
