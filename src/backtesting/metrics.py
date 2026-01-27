from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    cagr: float
    sharpe: float
    max_drawdown: float
    volatility: float
    win_rate: float
    avg_return: float
    trades: int
    turnover: float


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def compute_metrics(
    *,
    equity: pd.Series,
    period_returns: pd.Series,
    periods_per_year: float,
    turnover: Optional[pd.Series] = None,
) -> BacktestMetrics:
    equity = equity.dropna()
    period_returns = period_returns.dropna()

    if equity.empty or period_returns.empty:
        return BacktestMetrics(
            cagr=float("nan"),
            sharpe=float("nan"),
            max_drawdown=float("nan"),
            volatility=float("nan"),
            win_rate=float("nan"),
            avg_return=float("nan"),
            trades=int(period_returns.shape[0]),
            turnover=float("nan"),
        )

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    if isinstance(equity.index, pd.DatetimeIndex) and len(equity.index) >= 2:
        days = (equity.index[-1] - equity.index[0]).days
        years = max(days / 365.25, 1e-9)
        cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    else:
        years = max(len(period_returns) / periods_per_year, 1e-9)
        cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)

    avg = float(period_returns.mean())
    vol = float(period_returns.std(ddof=1))
    volatility = float(vol * np.sqrt(periods_per_year))

    if vol > 0 and np.isfinite(vol):
        sharpe = float((avg / vol) * np.sqrt(periods_per_year))
    else:
        sharpe = float("nan")

    max_dd = _max_drawdown(equity)

    win_rate = float((period_returns > 0).mean())

    if turnover is not None and not turnover.empty:
        turnover_value = float(turnover.mean())
    else:
        turnover_value = float("nan")

    return BacktestMetrics(
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_dd,
        volatility=volatility,
        win_rate=win_rate,
        avg_return=avg,
        trades=int(period_returns.shape[0]),
        turnover=turnover_value,
    )
