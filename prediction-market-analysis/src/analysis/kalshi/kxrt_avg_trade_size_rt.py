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
