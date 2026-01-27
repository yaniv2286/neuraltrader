from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from models import xgboost_model
from backtesting.engine import run_walk_forward_backtest


def _read_tickers(csv_path: str) -> List[str]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(p)
    if df.empty:
        return []

    for col in ["ticker", "symbol", "Symbol", "Ticker"]:
        if col in df.columns:
            return [str(x).strip().upper() for x in df[col].dropna().tolist() if str(x).strip()]

    first_col = df.columns[0]
    return [str(x).strip().upper() for x in df[first_col].dropna().tolist() if str(x).strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--tickers_csv", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default="1d")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--horizon", type=int, default=5)

    parser.add_argument("--initial_train_size", type=int, default=252)
    parser.add_argument("--retrain_interval", type=int, default=20)
    parser.add_argument("--validation_size", type=int, default=60)

    parser.add_argument("--buy_threshold", type=float, default=0.01)
    parser.add_argument("--sell_threshold", type=float, default=0.01)

    parser.add_argument("--commission_bps", type=float, default=1.0)
    parser.add_argument("--slippage_bps", type=float, default=2.0)

    parser.add_argument("--target_type", type=str, default="price", choices=["price", "log_return"])
    parser.add_argument("--use_tuning", action="store_true")
    parser.add_argument("--use_feature_selection", action="store_true")
    parser.add_argument("--n_features_to_select", type=int, default=20)
    parser.add_argument("--use_rolling_scaling", action="store_true")
    parser.add_argument("--out_csv", type=str, default=None)

    args = parser.parse_args(argv)

    tickers: List[str] = []
    if args.tickers_csv:
        tickers.extend(_read_tickers(args.tickers_csv))
    if args.ticker:
        tickers.append(str(args.ticker).strip().upper())

    tickers = list(dict.fromkeys([t for t in tickers if t]))
    if not tickers:
        raise SystemExit("Provide --ticker or --tickers_csv")

    rows = []
    for t in tickers:
        try:
            res = run_walk_forward_backtest(
                ticker=t,
                timeframe=args.timeframe,
                start=args.start,
                end=args.end,
                model_module=xgboost_model,
                horizon=args.horizon,
                initial_train_size=args.initial_train_size,
                retrain_interval=args.retrain_interval,
                validation_size=args.validation_size,
                buy_threshold=args.buy_threshold,
                sell_threshold=args.sell_threshold,
                commission_bps=args.commission_bps,
                slippage_bps=args.slippage_bps,
                target_type=args.target_type,
                use_tuning=args.use_tuning,
                use_feature_selection=args.use_feature_selection,
                n_features_to_select=args.n_features_to_select,
                use_rolling_scaling=args.use_rolling_scaling,
            )

            m = res.metrics
            rows.append(
                {
                    "ticker": t,
                    "timeframe": args.timeframe,
                    "horizon": args.horizon,
                    "cagr": m.cagr,
                    "sharpe": m.sharpe,
                    "max_drawdown": m.max_drawdown,
                    "volatility": m.volatility,
                    "win_rate": m.win_rate,
                    "avg_return": m.avg_return,
                    "trades": m.trades,
                    "turnover": m.turnover,
                }
            )
            print(f"{t}: CAGR={m.cagr:.3f} Sharpe={m.sharpe:.2f} MaxDD={m.max_drawdown:.2f} Trades={m.trades}")
        except Exception as e:
            print(f"{t}: ERROR: {e}")

    out = pd.DataFrame(rows)
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
