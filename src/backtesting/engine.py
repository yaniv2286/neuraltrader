from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from data.preprocess import build_model_input
from data.model_input import generate_model_input
from features.indicators import apply_indicators
from features.advanced_features import detect_patterns, add_price_action_features
from features.momentum import add_momentum_features
from features.regime import detect_regime
from features.regime_detector import detect_market_regime, get_regime_adjusted_thresholds
from features.bear_detector import detect_bear_market, get_bear_market_strategy
from features.macro_indicators import fetch_vix_data, fetch_interest_rates, calculate_macro_signals
from features.dynamic_thresholds import DynamicThresholds
from utils.tuning import tune_xgboost
from utils.feature_selection import select_features_rfe
from utils.scaling import rolling_scale_features
from signal_combiner import combine_signal
from .metrics import BacktestMetrics, compute_metrics


@dataclass
class BacktestResult:
    ticker: str
    timeframe: str
    horizon: int
    metrics: BacktestMetrics
    equity: pd.Series
    positions: pd.Series
    signals: pd.Series
    expected_return: pd.Series


def _periods_per_year(timeframe: str) -> float:
    tf = timeframe.lower()
    if tf == "1d":
        return 252.0
    if tf in {"1wk", "1w", "1week"}:
        return 52.0
    if tf in {"1mo", "1m", "1month"}:
        return 12.0
    return 252.0


def _build_signal_feature_frame(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf = timeframe
    required = [f"{tf}_open", f"{tf}_high", f"{tf}_low", f"{tf}_close", f"{tf}_volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame(index=df.index)

    base = df[required].copy()
    base.columns = ["open", "high", "low", "close", "volume"]
    base = base.dropna()

    if base.empty:
        return pd.DataFrame(index=df.index)

    enriched = apply_indicators(base)
    enriched = detect_patterns(enriched)
    enriched = add_price_action_features(enriched)
    enriched["regime"] = detect_regime(enriched[["close"]]).values
    enriched = enriched.dropna()

    return enriched


def _walk_forward_predictions(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    model_module: Any,
    initial_train_size: int,
    retrain_interval: int,
    validation_size: int,
    best_params: Optional[Dict[str, Any]] = None,
    selected_features: Optional[List[str]] = None,
    use_rolling_scaling: bool = False,
    target_type: str = "price",
) -> Tuple[pd.Series, pd.Series]:
    preds: List[float] = []
    rmses: List[float] = []
    idx: List[pd.Timestamp] = []

    model = None
    current_rmse = float("nan")

    for i in range(initial_train_size, len(X)):
        if model is None or ((i - initial_train_size) % retrain_interval == 0):
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]

            # Apply rolling scaling if enabled and target is log_return
            if use_rolling_scaling and target_type == "log_return":
                X_train, _ = rolling_scale_features(X_train, window=252)

            # Apply feature selection if provided
            if selected_features is not None:
                X_train = X_train[selected_features]

            model = model_module.build_model(best_params) if best_params is not None else model_module.build_model()
            if not hasattr(model, "fit"):
                raise ValueError("Model instance must implement fit/predict for backtesting")
            model.fit(X_train, y_train)

            val_n = int(min(validation_size, max(1, len(X_train) // 5)))
            X_val = X_train.iloc[-val_n:]
            y_val = y_train.iloc[-val_n:]
            if selected_features is not None:
                X_val = X_val[selected_features]
            if hasattr(model, "ensemble_predict"):
                y_val_pred = model.ensemble_predict(X_val)
            else:
                y_val_pred = model.predict(X_val)
            y_val_pred = np.array(y_val_pred).astype(float).reshape(-1)
            y_val_arr = np.array(y_val).astype(float).reshape(-1)
            current_rmse = float(np.sqrt(np.mean((y_val_pred - y_val_arr) ** 2)))

        x_row = X.iloc[i : i + 1]
        # Apply rolling scaling to the current row if needed
        if use_rolling_scaling and target_type == "log_return":
            # Use a trailing window ending at i to scale the row
            window_df = X.iloc[max(0, i-252):i]
            scaled_row, _ = rolling_scale_features(window_df, window=252)
            x_row = scaled_row.iloc[[-1]]
        if selected_features is not None:
            x_row = x_row[selected_features]
        if hasattr(model, "ensemble_predict"):
            pred = model.ensemble_predict(x_row)
        else:
            pred = model.predict(x_row)
        pred = float(np.array(pred).reshape(-1)[0])

        preds.append(pred)
        rmses.append(current_rmse)
        idx.append(X.index[i])

    return pd.Series(preds, index=pd.Index(idx)), pd.Series(rmses, index=pd.Index(idx))


def run_walk_forward_backtest(
    *,
    ticker: str,
    timeframe: str = "1d",
    start: str = "2018-01-01",
    end: str = "2025-01-01",
    model_module: Any,
    horizon: int = 5,
    initial_train_size: int = 252,
    retrain_interval: int = 20,
    validation_size: int = 60,
    buy_threshold: float = 0.01,
    sell_threshold: float = 0.01,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    target_type: str = "price",  # "price" or "log_return"
    use_tuning: bool = False,
    use_feature_selection: bool = False,
    n_features_to_select: int = 20,
    use_rolling_scaling: bool = False,
) -> BacktestResult:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    df = build_model_input(ticker, [timeframe], start, end)
    df = df.dropna()

    target_col = f"{timeframe}_close"
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    signal_features_df = _build_signal_feature_frame(df, timeframe)

    working = df.copy()
    
    # Add momentum features for trend filtering
    ohlcv_cols = [f"{timeframe}_open", f"{timeframe}_high", f"{timeframe}_low", f"{timeframe}_close", f"{timeframe}_volume"]
    if all(c in working.columns for c in ohlcv_cols):
        base_df = working[ohlcv_cols].copy()
        base_df.columns = ["open", "high", "low", "close", "volume"]
        base_df = base_df.dropna()
        if not base_df.empty:
            momentum_df = add_momentum_features(base_df)
            # Merge momentum features back to working df
            for col in ['trend_signal', 'sma_200', 'sma_50', 'rsi', 'macd_bullish']:
                if col in momentum_df.columns:
                    working[col] = momentum_df[col].reindex(working.index)
    
    # Initialize dynamic thresholds (temporarily disabled for testing)
    # dynamic_thresholds = DynamicThresholds()
    
    # Detect market regime
    market_regime, regime_strength = detect_market_regime(base_df)
    working['market_regime'] = market_regime.reindex(working.index)
    working['regime_strength'] = regime_strength.reindex(working.index)
    
    # Detect bear market strength
    bear_strength = detect_bear_market(base_df)
    working['bear_strength'] = bear_strength
    
    # Fetch macro indicators (temporarily disabled)
    try:
        # vix_data = fetch_vix_data(start, end)
        # rates_data = fetch_interest_rates(start, end)
        # macro_signals = calculate_macro_signals(base_df, vix_data, rates_data)
        # # Add macro signals to working dataframe
        # for key, value in macro_signals.items():
        #     if isinstance(value, pd.Series):
        #         working[f'macro_{key}'] = value.reindex(working.index)
        #     else:
        #         working[f'macro_{key}'] = value
        vix_data = None
        rates_data = None
        macro_signals = {}
    except Exception as e:
        print(f"Warning: Could not fetch macro data: {e}")
        vix_data = None
        rates_data = None
        macro_signals = {}
    
    X_train, y_train, X_test, y_test = generate_model_input(working, target_col, n_steps_ahead=horizon, target_type=target_type)

    if len(X_train) <= initial_train_size + 5:
        raise ValueError("Not enough data for backtest given initial_train_size")

    # Optional rolling scaling (useful for returns)
    if use_rolling_scaling and target_type == "log_return":
        X_train, _ = rolling_scale_features(X_train, window=252)

    # Optional feature selection (only for XGBoost for now)
    selected_features = None
    if use_feature_selection and model_module.__name__ == "xgboost_model":
        selected_features, _ = select_features_rfe(X_train, y_train, n_features_to_select=n_features_to_select)
        X_train = X_train[selected_features]
        # We will filter columns in the walk-forward loop as well

    # Optional tuning for XGBoost
    best_params = None
    if use_tuning and model_module.__name__ == "xgboost_model":
        val_n = int(min(validation_size, max(1, len(X_train) // 5)))
        X_tr = X_train.iloc[:initial_train_size]
        y_tr = y_train.iloc[:initial_train_size]
        X_val = X_train.iloc[initial_train_size:initial_train_size + val_n]
        y_val = y_train.iloc[initial_train_size:initial_train_size + val_n]
        best_params, _ = tune_xgboost(X_tr, y_tr, X_val, y_val)

    preds, rmses = _walk_forward_predictions(
        X=X_train,
        y=y_train,
        model_module=model_module,
        initial_train_size=initial_train_size,
        retrain_interval=retrain_interval,
        validation_size=validation_size,
        best_params=best_params,
        selected_features=selected_features,
        use_rolling_scaling=use_rolling_scaling,
        target_type=target_type,
    )

    close = working[target_col].reindex(preds.index).astype(float)
    next_close = working[target_col].shift(-1).reindex(preds.index).astype(float)
    next_ret = (next_close / close - 1.0).fillna(0.0)

    positions: List[int] = []
    signals: List[str] = []
    exp_rets: List[float] = []

    for ts in preds.index:
        current_val = working.loc[ts, target_col]
        if isinstance(current_val, pd.Series):
            current_val = current_val.iloc[-1]
        current_price = float(current_val)
        pred_price = float(preds.at[ts])

        feats: Mapping[str, Any]
        if not signal_features_df.empty and ts in signal_features_df.index:
            feats = signal_features_df.loc[ts].to_dict()
        else:
            feats = {}
        
        # Add momentum features
        if 'trend_signal' in working.columns and ts in working.index:
            feats['trend_signal'] = working.loc[ts, 'trend_signal']
        if 'sma_200' in working.columns and ts in working.index:
            feats['sma_200'] = working.loc[ts, 'sma_200']
        if 'above_sma_200' in working.columns and ts in working.index:
            feats['above_sma_200'] = working.loc[ts, 'above_sma_200']
        if 'rsi' in working.columns and ts in working.index:
            feats['rsi'] = working.loc[ts, 'rsi']

        # Add regime info
        current_regime = working.loc[ts, 'market_regime'] if 'market_regime' in working.columns and ts in working.index else None
        current_regime_strength = working.loc[ts, 'regime_strength'] if 'regime_strength' in working.columns and ts in working.index else None
        current_bear_strength = working.loc[ts, 'bear_strength'] if 'bear_strength' in working.columns and ts in working.index else None
        current_macro_risk = working.loc[ts, 'macro_macro_risk_score'] if 'macro_macro_risk_score' in working.columns and ts in working.index else working.loc[ts, 'macro_risk_score'] if 'macro_risk_score' in working.columns and ts in working.index else None
        current_macro_regime = working.loc[ts, 'macro_macro_regime'] if 'macro_macro_regime' in working.columns and ts in working.index else working.loc[ts, 'macro_regime'] if 'macro_regime' in working.columns and ts in working.index else 'NEUTRAL'
        
        # Use static thresholds for now (dynamic thresholds temporarily disabled)
        dynamic_buy = buy_threshold
        dynamic_sell = sell_threshold

        out = combine_signal(
            current_price=current_price,
            predicted_prices=[pred_price],
            model_error=float(rmses.at[ts]) if ts in rmses.index else None,
            regime=None,
            features=feats,
            buy_threshold=dynamic_buy,
            sell_threshold=dynamic_sell,
            use_trend_filter=True,
            min_trend_strength=0.25,
            market_regime=current_regime,
            regime_strength=current_regime_strength,
            bear_strength=current_bear_strength,
            macro_risk=current_macro_risk,
        )

        sig = str(out["signal"])
        if sig == "BUY":
            pos = 1
        elif sig == "SELL":
            pos = -1
        else:
            pos = 0

        positions.append(pos)
        signals.append(sig)
        exp_rets.append(float(out["expected_return"]))

    positions_s = pd.Series(positions, index=preds.index, dtype=float)
    signals_s = pd.Series(signals, index=preds.index, dtype=object)
    expected_return_s = pd.Series(exp_rets, index=preds.index, dtype=float)

    prev_pos = positions_s.shift(1).fillna(0.0)
    turnover = (positions_s - prev_pos).abs()

    cost_per_unit = (commission_bps + slippage_bps) / 10000.0
    costs = turnover * cost_per_unit

    strat_ret = (prev_pos * next_ret) - costs
    equity = (1.0 + strat_ret).cumprod()

    metrics = compute_metrics(
        equity=equity,
        period_returns=strat_ret,
        periods_per_year=_periods_per_year(timeframe),
        turnover=turnover,
    )

    return BacktestResult(
        ticker=ticker,
        timeframe=timeframe,
        horizon=horizon,
        metrics=metrics,
        equity=equity,
        positions=positions_s,
        signals=signals_s,
        expected_return=expected_return_s,
    )
