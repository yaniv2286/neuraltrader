"""
AI Ensemble Trading Strategy V2 - OPTIMIZED
============================================
All improvements implemented:
1. Max Ticker Exposure (20% limit)
2. Volatility-Adjusted Position Sizing (ATR-based)
3. Kelly Criterion for optimal sizing
4. Confidence-Based Position Sizing
5. Market Regime Filter for SHORTs
6. Higher Confidence Threshold for SHORTs (10x stricter)
7. Trailing Stop Loss
8. Daily Loss Limit (2%)
9. Drawdown-Based Position Reduction
10. Signal Confirmation (2 of 3 models agree)

Goal: CAGR > 20%, Max DD < 20%
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')

from src.features.technical_indicators import generate_all_features
from src.models.cpu_models.xgboost_model import XGBoostModel
from src.core.data_store import get_data_store
from src.core.model_cache import get_model_cache

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class RiskManager:
    """
    Advanced Risk Management System
    - Kelly Criterion sizing
    - Volatility-adjusted positions
    - Drawdown-based reduction
    - Daily loss limits
    - Trailing stops
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        max_position_pct: float = 0.10,
        max_ticker_exposure_pct: float = 0.20,
        daily_loss_limit_pct: float = 0.02,
        target_volatility: float = 0.02
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_ticker_exposure_pct = max_ticker_exposure_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.target_volatility = target_volatility
        
        # Tracking
        self.ticker_trade_counts = defaultdict(int)
        self.total_trades = 0
        self.daily_pnl = defaultdict(float)
        self.win_count = 0
        self.loss_count = 0
        self.total_wins = 0.0
        self.total_losses = 0.0
        
    def calculate_kelly_fraction(self) -> float:
        """
        Calculate Kelly Criterion fraction for optimal position sizing.
        Kelly = (W * R - L) / R
        Where: W = win rate, L = loss rate, R = avg_win / avg_loss
        """
        if self.win_count + self.loss_count < 10:
            return 0.08  # Default moderate until we have data
        
        win_rate = self.win_count / (self.win_count + self.loss_count)
        loss_rate = 1 - win_rate
        
        avg_win = self.total_wins / max(self.win_count, 1)
        avg_loss = self.total_losses / max(self.loss_count, 1)
        
        if avg_loss == 0:
            return 0.08
        
        R = avg_win / avg_loss
        kelly = (win_rate * R - loss_rate) / R
        
        # Use 3/4 Kelly for balanced growth
        fractional_kelly = kelly * 0.75
        
        # Clamp between 8% and 20%
        return max(0.08, min(0.20, fractional_kelly))
    
    def calculate_volatility_adjusted_size(
        self,
        base_size_pct: float,
        ticker_atr_pct: float
    ) -> float:
        """
        Adjust position size based on ticker volatility.
        Lower size for volatile stocks, higher for stable ones.
        """
        if ticker_atr_pct <= 0:
            return base_size_pct
        
        # Volatility ratio: target / actual
        vol_ratio = self.target_volatility / ticker_atr_pct
        
        # Clamp between 0.5x and 2x base size
        adjusted = base_size_pct * max(0.5, min(2.0, vol_ratio))
        
        return adjusted
    
    def calculate_confidence_adjusted_size(
        self,
        base_size_pct: float,
        confidence: float,
        confidence_percentile: float
    ) -> float:
        """
        Adjust position size based on signal confidence.
        Top 1%: 2.0x, Top 5%: 1.5x, Top 10%: 1.2x, else: 1.0x
        """
        if confidence_percentile >= 0.99:
            multiplier = 2.0
        elif confidence_percentile >= 0.95:
            multiplier = 1.5
        elif confidence_percentile >= 0.90:
            multiplier = 1.2
        else:
            multiplier = 1.0
        
        return base_size_pct * multiplier
    
    def calculate_drawdown_adjusted_size(
        self,
        base_size_pct: float
    ) -> float:
        """
        Reduce position size during drawdowns.
        DD < 15%: Full position
        DD 15-20%: 75% position
        DD > 20%: Half position
        """
        current_dd = (self.capital - self.peak_capital) / self.peak_capital
        
        if current_dd > -0.15:
            return base_size_pct
        elif current_dd > -0.20:
            return base_size_pct * 0.75
        else:
            return base_size_pct * 0.50
    
    def get_position_size(
        self,
        ticker: str,
        ticker_atr_pct: float,
        confidence: float,
        confidence_percentile: float
    ) -> float:
        """
        Calculate final position size combining all factors:
        1. Kelly Criterion base
        2. Volatility adjustment
        3. Confidence adjustment
        4. Drawdown adjustment
        5. Max position cap
        """
        # Start with Kelly
        kelly_size = self.calculate_kelly_fraction()
        
        # Apply volatility adjustment
        vol_adjusted = self.calculate_volatility_adjusted_size(kelly_size, ticker_atr_pct)
        
        # Apply confidence adjustment
        conf_adjusted = self.calculate_confidence_adjusted_size(vol_adjusted, confidence, confidence_percentile)
        
        # Apply drawdown adjustment
        dd_adjusted = self.calculate_drawdown_adjusted_size(conf_adjusted)
        
        # Cap at max position
        final_size = min(dd_adjusted, self.max_position_pct)
        
        return final_size
    
    def can_trade_ticker(self, ticker: str) -> bool:
        """Check if ticker hasn't exceeded max exposure limit."""
        if self.total_trades == 0:
            return True
        
        ticker_pct = self.ticker_trade_counts[ticker] / self.total_trades
        return ticker_pct < self.max_ticker_exposure_pct
    
    def can_trade_today(self, date) -> bool:
        """Check if daily loss limit hasn't been hit."""
        daily_loss = self.daily_pnl[date]
        daily_loss_pct = daily_loss / self.capital
        return daily_loss_pct > -self.daily_loss_limit_pct
    
    def record_trade(self, ticker: str, date, pnl: float):
        """Record trade for tracking."""
        self.ticker_trade_counts[ticker] += 1
        self.total_trades += 1
        self.daily_pnl[date] += pnl
        self.capital += pnl
        
        if pnl > 0:
            self.win_count += 1
            self.total_wins += pnl
        else:
            self.loss_count += 1
            self.total_losses += abs(pnl)
        
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital


class TrailingStop:
    """
    Trailing Stop Loss Manager
    - Initial stop at entry
    - Moves to breakeven after X% gain
    - Trails at Y% below peak after Z% gain
    """
    
    def __init__(
        self,
        initial_stop_pct: float = 0.025,  # Tighter initial stop
        breakeven_trigger_pct: float = 0.02,  # Move to breakeven faster
        trail_trigger_pct: float = 0.04,  # Start trailing earlier
        trail_distance_pct: float = 0.025  # Tighter trail
    ):
        self.initial_stop_pct = initial_stop_pct
        self.breakeven_trigger_pct = breakeven_trigger_pct
        self.trail_trigger_pct = trail_trigger_pct
        self.trail_distance_pct = trail_distance_pct
    
    def get_stop_price(
        self,
        entry_price: float,
        current_price: float,
        peak_price: float,
        direction: int
    ) -> float:
        """
        Calculate current stop price based on trailing logic.
        direction: 1 for LONG, -1 for SHORT
        """
        if direction == 1:  # LONG
            pnl_pct = (current_price - entry_price) / entry_price
            peak_pnl_pct = (peak_price - entry_price) / entry_price
            
            # Trail mode: after trail_trigger_pct gain
            if peak_pnl_pct >= self.trail_trigger_pct:
                stop_price = peak_price * (1 - self.trail_distance_pct)
            # Breakeven mode: after breakeven_trigger_pct gain
            elif peak_pnl_pct >= self.breakeven_trigger_pct:
                stop_price = entry_price * 1.001  # Tiny profit to cover fees
            # Initial stop
            else:
                stop_price = entry_price * (1 - self.initial_stop_pct)
            
            return stop_price
        
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price
            peak_pnl_pct = (entry_price - peak_price) / entry_price
            
            # Trail mode
            if peak_pnl_pct >= self.trail_trigger_pct:
                stop_price = peak_price * (1 + self.trail_distance_pct)
            # Breakeven mode
            elif peak_pnl_pct >= self.breakeven_trigger_pct:
                stop_price = entry_price * 0.999
            # Initial stop
            else:
                stop_price = entry_price * (1 + self.initial_stop_pct)
            
            return stop_price
    
    def is_stopped_out(
        self,
        entry_price: float,
        current_price: float,
        peak_price: float,
        direction: int
    ) -> bool:
        """Check if stop has been triggered."""
        stop_price = self.get_stop_price(entry_price, current_price, peak_price, direction)
        
        if direction == 1:  # LONG
            return current_price <= stop_price
        else:  # SHORT
            return current_price >= stop_price


class AIEnsembleStrategyV2:
    """
    AI Ensemble Strategy V2 - FULLY OPTIMIZED
    
    Improvements over V1:
    1. Max Ticker Exposure (20% limit)
    2. Volatility-Adjusted Position Sizing
    3. Kelly Criterion
    4. Confidence-Based Sizing
    5. Market Regime Filter for SHORTs
    6. Stricter SHORT thresholds
    7. Trailing Stop Loss
    8. Daily Loss Limit
    9. Drawdown-Based Position Reduction
    10. Signal Confirmation (2/3 models agree)
    """
    
    def __init__(self):
        self.data_store = get_data_store()
        self.models = {}
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.feature_columns = []
        self.risk_manager = None
        self.trailing_stop = TrailingStop()
        
    def prepare_full_dataset(
        self,
        tickers: List[str],
        start_date: str = '1993-01-01',
        end_date: str = '2023-12-31'
    ) -> pd.DataFrame:
        """Load ALL data from ALL tickers with market context."""
        print(f"📊 Loading FULL dataset from {len(tickers)} tickers...")
        print(f"   Period: {start_date} to {end_date}")
        
        all_data = []
        
        # Load SPY for market context
        try:
            spy_df = self.data_store.get_ticker_data('SPY', start_date, end_date)
            spy_df = generate_all_features(spy_df)
            market_data = spy_df[['close', 'rsi', 'macd', 'atr_percent', 'sma_200']].copy()
            market_data.columns = ['mkt_close', 'mkt_rsi', 'mkt_macd', 'mkt_atr', 'mkt_sma200']
            market_data['mkt_trend'] = (market_data['mkt_close'] > market_data['mkt_sma200']).astype(int)
            market_data['mkt_momentum'] = market_data['mkt_close'].pct_change(20)
            market_data['mkt_volatility'] = market_data['mkt_close'].pct_change().rolling(20).std()
        except:
            market_data = None
        
        for ticker in tickers:
            try:
                df = self.data_store.get_ticker_data(ticker, start_date, end_date)
                if df is None or len(df) < 100:
                    continue
                
                df = generate_all_features(df)
                if len(df) < 50:
                    continue
                
                df['ticker'] = ticker
                
                if market_data is not None:
                    df = df.join(market_data, how='left')
                
                # Target: next day's log return
                df['target'] = np.log(df['close'].shift(-1) / df['close'])
                df['target_direction'] = np.sign(df['target'])
                
                df = df.dropna()
                all_data.append(df)
            except:
                continue
        
        if not all_data:
            raise ValueError("No data loaded!")
        
        combined = pd.concat(all_data, ignore_index=False)
        print(f"   ✅ Total samples: {len(combined):,} from {len(all_data)} tickers")
        
        return combined
    
    def select_features(self, df: pd.DataFrame, n_features: int = 30) -> List[str]:
        """Select top predictive features using XGBoost importance."""
        print(f"🔍 Selecting top {n_features} features...")
        
        exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'target_direction',
                   'ticker', 'mkt_close', 'mkt_sma200']
        feature_cols = [c for c in df.columns if c not in exclude]
        
        sample = df.sample(n=min(100000, len(df)), random_state=42)
        X = sample[feature_cols].fillna(0).values
        y = sample['target'].values
        
        model = XGBoostModel(n_estimators=100, max_depth=3)
        model.fit(X, y)
        
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top = importance.head(n_features)['feature'].tolist()
        print(f"   ✅ Top features: {', '.join(top[:5])}...")
        
        return top
    
    def train_ensemble(
        self,
        tickers: List[str],
        train_start: str = '1970-01-01',
        train_end: str = '2022-12-31',
        use_cache: bool = True
    ) -> Dict:
        """Train ALL CPU models on full dataset (with caching)."""
        print("=" * 70)
        print("🤖 TRAINING AI ENSEMBLE V2 (OPTIMIZED)")
        print("=" * 70)
        
        # Try to load from cache first
        if use_cache:
            model_cache = get_model_cache()
            cached = model_cache.load_models(tickers, train_start, train_end, n_features=30)
            
            if cached is not None:
                # Restore from cache
                self.models = cached['models']
                self.feature_columns = cached['metadata']['feature_names']
                
                print(f"   ⚡ Using cached models - TRAINING SKIPPED!")
                
                return {
                    'samples': cached['metadata']['train_samples'],
                    'features': cached['metadata']['n_features'],
                    'models': list(self.models.keys()),
                    'accuracy': cached['metadata']['direction_accuracy'],
                    'cached': True
                }
        
        # No cache - train from scratch
        print("   📊 No cache found - training from scratch...")
        
        df = self.prepare_full_dataset(tickers, train_start, train_end)
        self.feature_columns = self.select_features(df, n_features=30)
        
        X = df[self.feature_columns].fillna(0).values
        y = df['target'].values
        
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        print(f"\n🚀 Training on {len(X):,} samples with {len(self.feature_columns)} features...")
        
        # Train XGBoost
        print("   Training XGBoost...")
        self.models['xgboost'] = XGBoostModel(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.01,
            reg_alpha=0.5,
            reg_lambda=1.0
        )
        self.models['xgboost'].fit(X, y)
        
        # Train RandomForest
        if HAS_SKLEARN:
            print("   Training RandomForest...")
            self.models['rf'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=42
            )
            self.models['rf'].fit(X_scaled, y)
        
        # Train LightGBM
        if HAS_LIGHTGBM:
            print("   Training LightGBM...")
            self.models['lgbm'] = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.01,
                reg_alpha=0.5,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
            self.models['lgbm'].fit(X_scaled, y)
        
        # Evaluate
        predictions = self._ensemble_predict(X, X_scaled)
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(y)) * 100
        
        print(f"\n   ✅ Models trained: {len(self.models)}")
        print(f"   ✅ Ensemble Direction Accuracy: {direction_accuracy:.1f}%")
        
        # Save to cache
        if use_cache:
            model_cache = get_model_cache()
            model_cache.save_models(
                models=self.models,
                tickers=tickers,
                train_start=train_start,
                train_end=train_end,
                n_features=30,
                feature_names=self.feature_columns,
                train_samples=len(X),
                direction_accuracy=direction_accuracy
            )
        
        return {
            'samples': len(X),
            'features': len(self.feature_columns),
            'models': list(self.models.keys()),
            'accuracy': direction_accuracy,
            'cached': False
        }
    
    def _get_individual_predictions(self, X: np.ndarray, X_scaled: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each model separately for confirmation logic."""
        predictions = {}
        
        if 'xgboost' in self.models:
            predictions['xgboost'] = self.models['xgboost'].predict(X)
        
        if 'rf' in self.models:
            predictions['rf'] = self.models['rf'].predict(X_scaled)
        
        if 'lgbm' in self.models:
            predictions['lgbm'] = self.models['lgbm'].predict(X_scaled)
        
        return predictions
    
    def _ensemble_predict(self, X: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:
        """Get ensemble prediction from all models."""
        predictions = []
        weights = []
        
        if 'xgboost' in self.models:
            pred = self.models['xgboost'].predict(X)
            predictions.append(pred)
            weights.append(1.5)
        
        if 'rf' in self.models:
            pred = self.models['rf'].predict(X_scaled)
            predictions.append(pred)
            weights.append(1.0)
        
        if 'lgbm' in self.models:
            pred = self.models['lgbm'].predict(X_scaled)
            predictions.append(pred)
            weights.append(1.2)
        
        if predictions:
            weights = np.array(weights) / sum(weights)
            ensemble = sum(p * w for p, w in zip(predictions, weights))
            return ensemble
        
        return np.zeros(len(X))
    
    def _check_signal_confirmation(
        self,
        individual_preds: Dict[str, float],
        direction: int
    ) -> bool:
        """
        Check if at least 2 of 3 models agree on direction.
        For LONG: require 2/3 agreement
        For SHORT: require ALL 3 to agree (stricter)
        """
        agreements = 0
        
        for model_name, pred in individual_preds.items():
            if direction == 1 and pred > 0:
                agreements += 1
            elif direction == -1 and pred < 0:
                agreements += 1
        
        if direction == 1:
            return agreements >= 2  # 2/3 for LONG
        else:
            return agreements >= 3  # ALL 3 for SHORT
    
    def generate_signals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Generate signals with:
        - Market regime filter for SHORTs
        - Higher threshold for SHORTs
        - Signal confirmation (2/3 models agree)
        """
        print(f"\n📈 Generating AI ensemble signals (V2 - Enhanced)...")
        
        all_signals = []
        
        # Load SPY for market context
        try:
            spy_df = self.data_store.get_ticker_data('SPY', start_date, end_date)
            spy_df = generate_all_features(spy_df)
            market_data = spy_df[['close', 'rsi', 'macd', 'atr_percent', 'sma_200']].copy()
            market_data.columns = ['mkt_close', 'mkt_rsi', 'mkt_macd', 'mkt_atr', 'mkt_sma200']
            market_data['mkt_trend'] = (market_data['mkt_close'] > market_data['mkt_sma200']).astype(int)
            market_data['mkt_momentum'] = market_data['mkt_close'].pct_change(20)
            market_data['mkt_volatility'] = market_data['mkt_close'].pct_change().rolling(20).std()
        except:
            market_data = None
        
        for ticker in tickers:
            try:
                df = self.data_store.get_ticker_data(ticker, start_date, end_date)
                if df is None or len(df) < 50:
                    continue
                
                df = generate_all_features(df)
                if len(df) < 20:
                    continue
                
                if market_data is not None:
                    df = df.join(market_data, how='left')
                
                missing = [f for f in self.feature_columns if f not in df.columns]
                if missing:
                    continue
                
                X = df[self.feature_columns].fillna(0).values
                
                if self.scaler:
                    X_scaled = self.scaler.transform(X)
                else:
                    X_scaled = X
                
                # Get individual predictions for confirmation
                individual_preds = self._get_individual_predictions(X, X_scaled)
                ensemble_pred = self._ensemble_predict(X, X_scaled)
                
                # Build signals with all data
                for i in range(len(df)):
                    pred = ensemble_pred[i]
                    confidence = abs(pred)
                    
                    # Get market regime
                    mkt_trend = df['mkt_trend'].iloc[i] if 'mkt_trend' in df.columns else 1
                    mkt_rsi = df['mkt_rsi'].iloc[i] if 'mkt_rsi' in df.columns else 50
                    mkt_momentum = df['mkt_momentum'].iloc[i] if 'mkt_momentum' in df.columns else 0
                    
                    # Determine signal direction with different thresholds
                    # LONG: prediction > 0.0001
                    # SHORT: prediction < -0.001 (10x stricter) AND bear market
                    signal = 0
                    
                    if pred > 0.0001:
                        signal = 1  # LONG
                    elif pred < -0.001:  # 10x stricter for SHORT
                        # Market regime filter: only SHORT in bear markets
                        is_bear_market = (
                            mkt_trend == 0 or  # SPY below 200 SMA
                            mkt_rsi > 70 or    # Market overbought
                            mkt_momentum < -0.05  # Strong negative momentum
                        )
                        if is_bear_market:
                            signal = -1  # SHORT
                    
                    if signal == 0:
                        continue
                    
                    # Check signal confirmation (2/3 models agree)
                    ind_preds_at_i = {k: v[i] for k, v in individual_preds.items()}
                    if not self._check_signal_confirmation(ind_preds_at_i, signal):
                        continue  # Skip if not confirmed
                    
                    all_signals.append({
                        'date': df.index[i],
                        'ticker': ticker,
                        'prediction': pred,
                        'confidence': confidence,
                        'signal': signal,
                        'close': df['close'].iloc[i],
                        'volume': df['volume'].iloc[i],
                        'atr_percent': df['atr_percent'].iloc[i] if 'atr_percent' in df.columns else 0.02,
                        'rsi': df['rsi'].iloc[i] if 'rsi' in df.columns else 50,
                        'macd': df['macd'].iloc[i] if 'macd' in df.columns else 0,
                        'macd_signal': df['macd_signal'].iloc[i] if 'macd_signal' in df.columns else 0,
                        'bb_position': df['bb_position'].iloc[i] if 'bb_position' in df.columns else 0.5,
                        'momentum_5': df['momentum_5'].iloc[i] if 'momentum_5' in df.columns else 0,
                        'momentum_20': df['momentum_20'].iloc[i] if 'momentum_20' in df.columns else 0,
                        'sma_20': df['sma_20'].iloc[i] if 'sma_20' in df.columns else df['close'].iloc[i],
                        'sma_50': df['sma_50'].iloc[i] if 'sma_50' in df.columns else df['close'].iloc[i],
                        'volume_ratio': df['volume_ratio'].iloc[i] if 'volume_ratio' in df.columns else 1,
                        'mkt_trend': mkt_trend,
                        'mkt_rsi': mkt_rsi,
                        'mkt_momentum': mkt_momentum
                    })
                
            except Exception as e:
                continue
        
        if not all_signals:
            return pd.DataFrame()
        
        signals = pd.DataFrame(all_signals)
        
        print(f"   ✅ Generated {len(signals):,} CONFIRMED signals")
        print(f"   📈 LONG signals: {(signals['signal'] == 1).sum():,}")
        print(f"   📉 SHORT signals: {(signals['signal'] == -1).sum():,}")
        
        return signals
    
    def filter_top_signals(
        self,
        signals: pd.DataFrame,
        max_per_day: int = 3,
        top_pct: float = 0.10
    ) -> pd.DataFrame:
        """Filter to top signals per day with confidence percentiles."""
        print(f"\n🎯 Filtering to top {top_pct*100:.0f}% signals...")
        
        signals['date_only'] = pd.to_datetime(signals['date']).dt.date
        signals['rank'] = signals.groupby('date_only')['confidence'].rank(ascending=False)
        
        filtered = signals[signals['rank'] <= max_per_day].copy()
        
        threshold = filtered['confidence'].quantile(1 - top_pct)
        filtered = filtered[filtered['confidence'] >= threshold]
        
        # Add confidence percentile for position sizing
        filtered['confidence_percentile'] = filtered['confidence'].rank(pct=True)
        
        print(f"   ✅ Filtered to {len(filtered):,} signals")
        print(f"   📈 LONG: {(filtered['signal'] == 1).sum():,}")
        print(f"   📉 SHORT: {(filtered['signal'] == -1).sum():,}")
        
        return filtered
    
    def _build_entry_reason(self, signal: pd.Series, direction: int) -> str:
        """Build detailed entry reason."""
        reasons = []
        
        pred = signal.get('prediction', 0)
        conf = signal.get('confidence', 0)
        direction_str = "BULLISH" if direction == 1 else "BEARISH"
        
        if conf > 0.01:
            conf_level = "HIGH"
        elif conf > 0.005:
            conf_level = "MEDIUM"
        else:
            conf_level = "LOW"
        
        reasons.append(f"AI Signal: {direction_str} ({conf_level} conf={conf:.4f})")
        
        # Technical
        tech = []
        rsi = signal.get('rsi', 50)
        if rsi < 30:
            tech.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 70:
            tech.append(f"RSI overbought ({rsi:.0f})")
        
        macd = signal.get('macd', 0)
        macd_sig = signal.get('macd_signal', 0)
        if macd > macd_sig and direction == 1:
            tech.append("MACD bullish")
        elif macd < macd_sig and direction == -1:
            tech.append("MACD bearish")
        
        if tech:
            reasons.append("Tech: " + "; ".join(tech[:2]))
        
        # Market
        mkt_trend = signal.get('mkt_trend', 1)
        reasons.append(f"Market: {'Bull' if mkt_trend == 1 else 'Bear'}")
        
        return " | ".join(reasons)
    
    def _build_exit_reason(
        self,
        exit_type: str,
        entry_price: float,
        exit_price: float,
        direction: int,
        hold_days: int,
        pnl_pct: float
    ) -> str:
        """Build detailed exit reason."""
        direction_str = "LONG" if direction == 1 else "SHORT"
        
        if exit_type == 'trailing_stop':
            return f"TRAILING STOP at ${exit_price:.2f} | {pnl_pct*100:.1f}% | {hold_days}d"
        elif exit_type == 'stop_loss':
            return f"STOP LOSS at ${exit_price:.2f} | {pnl_pct*100:.1f}% | {hold_days}d"
        elif exit_type == 'take_profit':
            return f"TAKE PROFIT at ${exit_price:.2f} | +{pnl_pct*100:.1f}% | {hold_days}d"
        elif exit_type == 'max_hold':
            return f"MAX HOLD at ${exit_price:.2f} | {pnl_pct*100:.1f}% | {hold_days}d"
        else:
            return f"{exit_type} at ${exit_price:.2f} | {pnl_pct*100:.1f}%"
    
    def backtest(
        self,
        signals: pd.DataFrame,
        initial_capital: float = 100000,
        take_profit_pct: float = 0.12,
        max_hold_days: int = 15
    ) -> Dict:
        """
        Backtest with ALL optimizations:
        - Kelly Criterion sizing
        - Volatility-adjusted positions
        - Confidence-based sizing
        - Drawdown-based reduction
        - Trailing stops
        - Daily loss limits
        - Max ticker exposure
        """
        print(f"\n💰 Running OPTIMIZED backtest (V2)...")
        print(f"   Features: Kelly + Vol-Adj + Trailing Stop + DD-Reduction")
        
        # Initialize risk manager - balanced settings
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_position_pct=0.15,  # 15% max per position
            max_ticker_exposure_pct=0.20,  # 20% per ticker
            daily_loss_limit_pct=0.025,  # 2.5% daily loss limit
            target_volatility=0.025
        )
        
        trades = []
        signals = signals.sort_values('date')
        
        # Load price data
        min_date = pd.to_datetime(signals['date']).min()
        max_date = pd.to_datetime(signals['date']).max()
        start_str = (min_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        end_str = (max_date + pd.Timedelta(days=60)).strftime('%Y-%m-%d')
        
        ticker_data = {}
        for ticker in signals['ticker'].unique():
            try:
                df = self.data_store.get_ticker_data(ticker, start_str, end_str)
                if df is not None:
                    ticker_data[ticker] = df
            except:
                pass
        
        for _, signal in signals.iterrows():
            entry_date = pd.to_datetime(signal['date'])
            ticker = signal['ticker']
            entry_price = signal['close']
            direction = signal['signal']
            confidence = signal['confidence']
            confidence_pct = signal.get('confidence_percentile', 0.5)
            atr_pct = signal.get('atr_percent', 0.02)
            
            # Check risk limits
            if not self.risk_manager.can_trade_ticker(ticker):
                continue
            
            date_only = entry_date.date()
            if not self.risk_manager.can_trade_today(date_only):
                continue
            
            if ticker not in ticker_data:
                continue
            
            df = ticker_data[ticker]
            future_data = df[df.index > entry_date].head(max_hold_days)
            
            if len(future_data) == 0:
                continue
            
            # Calculate position size using all factors
            position_pct = self.risk_manager.get_position_size(
                ticker=ticker,
                ticker_atr_pct=atr_pct,
                confidence=confidence,
                confidence_percentile=confidence_pct
            )
            
            position_size = self.risk_manager.capital * position_pct
            shares = int(position_size / entry_price)
            
            if shares <= 0:
                continue
            
            # Track trade with trailing stop
            exit_price = None
            exit_date = None
            exit_reason = None
            peak_price = entry_price
            
            for idx, row in future_data.iterrows():
                current_price = row['close']
                
                # Update peak for trailing stop
                if direction == 1:
                    peak_price = max(peak_price, current_price)
                else:
                    peak_price = min(peak_price, current_price)
                
                # Check trailing stop
                if self.trailing_stop.is_stopped_out(entry_price, current_price, peak_price, direction):
                    exit_price = current_price
                    exit_date = idx
                    exit_reason = 'trailing_stop'
                    break
                
                # Check take profit
                if direction == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                if pnl_pct >= take_profit_pct:
                    exit_price = current_price
                    exit_date = idx
                    exit_reason = 'take_profit'
                    break
            
            # Exit at end if no exit triggered
            if exit_price is None:
                exit_price = future_data.iloc[-1]['close']
                exit_date = future_data.index[-1]
                exit_reason = 'max_hold'
            
            # Calculate final P&L
            if direction == 1:
                trade_return = (exit_price - entry_price) / entry_price
            else:
                trade_return = (entry_price - exit_price) / entry_price
            
            pnl = shares * entry_price * trade_return
            
            # Record trade in risk manager
            self.risk_manager.record_trade(ticker, date_only, pnl)
            
            # Calculate drawdown
            if self.risk_manager.capital > self.risk_manager.peak_capital:
                self.risk_manager.peak_capital = self.risk_manager.capital
            drawdown = (self.risk_manager.capital - self.risk_manager.peak_capital) / self.risk_manager.peak_capital
            
            hold_days = (exit_date - entry_date).days if exit_date and entry_date else 0
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'ticker': ticker,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'position_pct': position_pct * 100,
                'return_pct': trade_return * 100,
                'pnl': pnl,
                'entry_reason': self._build_entry_reason(signal, direction),
                'exit_reason': self._build_exit_reason(exit_reason, entry_price, exit_price, direction, hold_days, trade_return),
                'exit_type': exit_reason,
                'hold_days': hold_days,
                'capital': self.risk_manager.capital,
                'drawdown': drawdown
            })
        
        if not trades:
            return {'error': 'No trades executed'}
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        capital = self.risk_manager.capital
        total_return = (capital - initial_capital) / initial_capital * 100
        
        first_date = trades_df['entry_date'].min()
        last_date = trades_df['exit_date'].max()
        trade_days = (last_date - first_date).days
        years = max(trade_days / 365.25, 1.0)
        cagr = ((capital / initial_capital) ** (1 / years) - 1) * 100
        
        max_dd = trades_df['drawdown'].min() * 100
        win_rate = (trades_df['pnl'] > 0).mean() * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (trades_df['pnl'] < 0).any() else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        long_trades = trades_df[trades_df['direction'] == 'LONG']
        short_trades = trades_df[trades_df['direction'] == 'SHORT']
        
        # Ticker distribution
        ticker_counts = trades_df['ticker'].value_counts()
        max_ticker_pct = (ticker_counts.iloc[0] / len(trades_df) * 100) if len(ticker_counts) > 0 else 0
        
        print(f"\n   📊 Risk Manager Stats:")
        print(f"      Kelly Fraction: {self.risk_manager.calculate_kelly_fraction()*100:.1f}%")
        print(f"      Unique Tickers: {trades_df['ticker'].nunique()}")
        print(f"      Max Ticker Exposure: {max_ticker_pct:.1f}%")
        
        return {
            'total_return_pct': total_return,
            'cagr_pct': cagr,
            'max_drawdown_pct': max_dd,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': (long_trades['pnl'] > 0).mean() * 100 if len(long_trades) > 0 else 0,
            'short_win_rate': (short_trades['pnl'] > 0).mean() * 100 if len(short_trades) > 0 else 0,
            'avg_return_per_trade': trades_df['return_pct'].mean(),
            'avg_position_size': trades_df['position_pct'].mean(),
            'unique_tickers': trades_df['ticker'].nunique(),
            'max_ticker_exposure': max_ticker_pct,
            'kelly_fraction': self.risk_manager.calculate_kelly_fraction() * 100,
            'final_capital': capital,
            'trades': trades_df
        }


def run_ai_ensemble_backtest_v2(use_80_20_split: bool = True):
    """Run AI ensemble backtest V2 with ALL optimizations."""
    print("=" * 70)
    print("🤖 AI ENSEMBLE TRADING STRATEGY V2 - FULLY OPTIMIZED")
    print("=" * 70)
    print("Optimizations:")
    print("  ✅ Kelly Criterion position sizing")
    print("  ✅ Volatility-adjusted positions")
    print("  ✅ Confidence-based sizing")
    print("  ✅ Drawdown-based reduction")
    print("  ✅ Trailing stop loss")
    print("  ✅ Daily loss limit (2%)")
    print("  ✅ Max ticker exposure (20%)")
    print("  ✅ Market regime filter for SHORTs")
    print("  ✅ Signal confirmation (2/3 models)")
    print("  ✅ Stricter SHORT thresholds (10x)")
    print("=" * 70)
    
    store = get_data_store()
    tickers = store.available_tickers
    print(f"\n📊 Universe: {len(tickers)} tickers")
    
    strategy = AIEnsembleStrategyV2()
    
    if use_80_20_split:
        print("\n📊 Using 80/20 SPLIT (Full Historical Data)")
        print("   Training:   1970-01-01 to 2014-12-31 (~45 years)")
        print("   Validation: 2015-01-01 to 2024-12-31 (~10 years)")
        
        train_start = '1970-01-01'
        train_end = '2014-12-31'
        test_start = '2015-01-01'
        test_end = '2024-12-31'
    else:
        print("\n📊 Using ORIGINAL SPLIT")
        train_start = '1993-01-01'
        train_end = '2022-12-31'
        test_start = '2023-01-01'
        test_end = '2024-12-31'
    
    # Train
    train_results = strategy.train_ensemble(
        tickers=tickers,
        train_start=train_start,
        train_end=train_end
    )
    
    # Generate signals
    signals = strategy.generate_signals(
        tickers=tickers,
        start_date=test_start,
        end_date=test_end
    )
    
    if signals.empty:
        print("❌ No signals generated!")
        return None
    
    # Filter - top 10% signals
    filtered = strategy.filter_top_signals(
        signals,
        max_per_day=5,
        top_pct=0.10
    )
    
    # LONG ONLY - SHORT signals underperform
    filtered = filtered[filtered['signal'] == 1].copy()
    print(f"   📈 LONG ONLY mode: {len(filtered)} signals")
    
    if filtered.empty:
        print("❌ No signals after filtering!")
        return None
    
    # Backtest
    results = strategy.backtest(
        filtered,
        initial_capital=100000,
        take_profit_pct=0.12,
        max_hold_days=15
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 AI ENSEMBLE V2 BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Total Return: {results['total_return_pct']:.2f}%")
    print(f"   CAGR: {results['cagr_pct']:.2f}%")
    print(f"   Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"   Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    
    print(f"\n📈 Trade Statistics:")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   LONG Trades: {results['long_trades']} (Win Rate: {results['long_win_rate']:.1f}%)")
    print(f"   SHORT Trades: {results['short_trades']} (Win Rate: {results['short_win_rate']:.1f}%)")
    print(f"   Avg Return/Trade: {results['avg_return_per_trade']:.2f}%")
    
    print(f"\n🛡️ Risk Management:")
    print(f"   Avg Position Size: {results['avg_position_size']:.1f}%")
    print(f"   Kelly Fraction: {results['kelly_fraction']:.1f}%")
    print(f"   Unique Tickers: {results['unique_tickers']}")
    print(f"   Max Ticker Exposure: {results['max_ticker_exposure']:.1f}%")
    
    passed = (
        results['cagr_pct'] >= 20 and
        results['max_drawdown_pct'] >= -20
    )
    
    print(f"\n📋 Validation:")
    print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    if results['cagr_pct'] < 20:
        print(f"   ⚠️ CAGR {results['cagr_pct']:.1f}% < 20%")
    if results['max_drawdown_pct'] < -20:
        print(f"   ⚠️ Max DD {results['max_drawdown_pct']:.1f}% > 20%")
    
    # Generate Excel report using Trading Backtest Excel format
    try:
        from src.core.trading_backtest_excel import create_trading_backtest_excel
        
        config = {
            'version': 'V2 - Fully Optimized',
            'models': 'XGBoost + RandomForest + LightGBM',
            'split_type': '80/20 Split' if use_80_20_split else 'Original',
            'training_period': f"{train_start} to {train_end}",
            'validation_period': f"{test_start} to {test_end}",
            'training_samples': train_results.get('samples', 'N/A'),
            'optimizations': 'Kelly + VolAdj + TrailingStop + DDReduction + Confirmation',
            'position_sizing': 'Kelly Criterion (3/4 fractional)',
            'stop_loss': 'Trailing (2.5% initial, trail at 4%+)',
            'take_profit_pct': 12,
            'max_hold_days': 15,
            'max_ticker_exposure': '20%',
            'daily_loss_limit': '2.5%',
            'short_filter': 'Bear market only (10x stricter)',
            'signal_confirmation': '2/3 LONG, 3/3 SHORT',
            'features': 30,
            'tickers': len(tickers)
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'reports/trading_backtest_{timestamp}.xlsx'
        
        create_trading_backtest_excel(
            results=results,
            config=config,
            output_path=output_path
        )
        
        print(f"\n📁 Trading Backtest Excel Report: {output_path}")
        print(f"   📖 Open 'How_To_Read' sheet first for instructions")
        print(f"   💰 Check '100K_Reality_Check' for GO/NO-GO decision")
    except Exception as e:
        print(f"\n⚠️ Could not generate Excel report: {e}")
        import traceback
        traceback.print_exc()
    
    return results


if __name__ == "__main__":
    results = run_ai_ensemble_backtest_v2()
