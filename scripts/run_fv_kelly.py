"""Quick runner for fv_kelly backtest — prints one-line result summary."""
import sys
import ast

from src.backtesting.feeds.kalshi_rt import KalshiRTFeed
from src.backtesting.rust_engine import Engine
from src.backtesting.strategies.fv_kelly import FVKellyStrategy

kwargs = {}
for arg in sys.argv[1:]:
    k, v = arg.split("=", 1)
    try:
        kwargs[k] = ast.literal_eval(v)
    except Exception:
        kwargs[k] = v

from src.backtesting.metrics import model_calibration_report

strategy = FVKellyStrategy(**kwargs)
feed = KalshiRTFeed()
engine = Engine(feed=feed, strategy=strategy, market_sample=None)
result = engine.run()

m = result.metrics
print(
    f"ret={m.get('total_return',0):.2%}  "
    f"dd={m.get('max_drawdown',0):.2%}  "
    f"sharpe={m.get('sharpe_ratio',0):.3f}  "
    f"wr={m.get('win_rate',0):.1%}  "
    f"implied={m.get('avg_implied_prob',0):.1%}  "
    f"edge={m.get('win_rate_vs_implied',0):+.1%}  "
    f"pf={m.get('profit_factor',0):.3f}  "
    f"fills={int(m.get('num_fills',0))}  "
    f"comm=${m.get('total_commission',0):.0f}  "
    f"equity=${result.final_equity:.2f}"
)

cal = model_calibration_report(result.fills, result.market_pnls, strategy.fv_at_fill)
if cal:
    winner = "MODEL" if cal["model_better"] else "market"
    print(f"calibration ({cal['n']} fills):  model_mae={cal['model_mae']:.3f}  market_mae={cal['market_mae']:.3f}  → {winner}")
    for bucket, d in cal["by_price"].items():
        w = "MODEL" if d["model_better"] else "market"
        print(f"  {bucket:>9}:  model={d['model_err']:+.3f} (mae={d['model_mae']:.3f})  "
              f"market={d['market_err']:+.3f} (mae={d['market_mae']:.3f})  → {w}  n={d['n']}")
