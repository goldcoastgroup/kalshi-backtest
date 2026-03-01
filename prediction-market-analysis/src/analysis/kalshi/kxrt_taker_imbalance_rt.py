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
