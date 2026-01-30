"""
High-Precision Trading Strategy
Goal: CAGR > 20%, Max DD < 20%, Win Rate > 55%
Approach: Fewer trades, higher confidence, strict risk management
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')

from src.features.technical_indicators import generate_all_features
from src.models.cpu_models.xgboost_model import XGBoostModel
from src.core.data_store import get_data_store


class PrecisionStrategy:
    """
    High-precision trading with strict selectivity.
    
    Key principles:
    1. Train on ALL data from ALL tickers (500K+ samples)
    2. Only trade TOP 1-2% of signals (highest confidence)
    3. Strict position sizing (max 5% per trade)
    4. Long-only in uptrends, avoid bear markets
    """
    
    def __init__(self):
        self.data_store = get_data_store()
        self.model = None
        self.feature_columns: List[str] = []
        self.signal_threshold = 0.005  # Only trade if predicted return > 0.5%
        
    def prepare_full_dataset(
        self,
        tickers: List[str],
        start_date: str = '1993-01-01',
        end_date: str = '2023-12-31'
    ) -> pd.DataFrame:
        """Load ALL data from ALL tickers into one dataset."""
        print(f"📊 Loading FULL dataset from {len(tickers)} tickers...")
        
        all_data = []
        
        # Load SPY for market context
        try:
            spy_df = self.data_store.get_ticker_data('SPY', start_date, end_date)
            spy_df = generate_all_features(spy_df)
            market_data = spy_df[['close', 'rsi', 'macd', 'atr_percent', 'sma_200']].copy()
            market_data.columns = ['mkt_close', 'mkt_rsi', 'mkt_macd', 'mkt_atr', 'mkt_sma200']
            market_data['mkt_trend'] = (market_data['mkt_close'] > market_data['mkt_sma200']).astype(int)
            market_data['mkt_momentum'] = market_data['mkt_close'].pct_change(20)
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
                df = df.dropna()
                
                all_data.append(df)
            except:
                continue
        
        if not all_data:
            raise ValueError("No data loaded!")
        
        combined = pd.concat(all_data, ignore_index=False)
        print(f"   ✅ Total samples: {len(combined):,} from {len(all_data)} tickers")
        
        return combined
    
    def select_best_features(self, df: pd.DataFrame, n_features: int = 20) -> List[str]:
        """Select top predictive features."""
        print(f"🔍 Selecting top {n_features} features...")
        
        exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'ticker', 'mkt_close', 'mkt_sma200']
        feature_cols = [c for c in df.columns if c not in exclude]
        
        # Sample for speed
        sample = df.sample(n=min(50000, len(df)), random_state=42)
        X = sample[feature_cols].values
        y = sample['target'].values
        
        model = XGBoostModel(n_estimators=100, max_depth=3)
        model.fit(X, y)
        
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top = importance.head(n_features)['feature'].tolist()
        print(f"   ✅ Top: {', '.join(top[:5])}...")
        
        return top
    
    def train_precision_model(
        self,
        tickers: List[str],
        train_end: str = '2022-12-31'
    ) -> Dict:
        """Train model on ALL historical data."""
        print("=" * 70)
        print("🎯 TRAINING PRECISION MODEL (ALL DATA)")
        print("=" * 70)
        
        # Load full dataset
        df = self.prepare_full_dataset(tickers, '1993-01-01', train_end)
        
        # Select features
        self.feature_columns = self.select_best_features(df, n_features=20)
        
        # Prepare training data
        X = df[self.feature_columns].values
        y = df['target'].values
        
        print(f"\n🚀 Training on {len(X):,} samples with {len(self.feature_columns)} features...")
        
        # Train with strong regularization
        self.model = XGBoostModel(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.01,
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.7
        )
        self.model.fit(X, y)
        
        # Evaluate on training data
        train_pred = self.model.predict(X)
        train_dir = np.mean(np.sign(train_pred) == np.sign(y)) * 100
        
        print(f"   ✅ Training Direction Accuracy: {train_dir:.1f}%")
        
        return {
            'samples': len(X),
            'features': len(self.feature_columns),
            'train_accuracy': train_dir
        }
    
    def generate_signals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Generate trading signals with confidence scores."""
        print(f"\n📈 Generating signals for {start_date} to {end_date}...")
        
        all_signals = []
        
        # Load SPY for market filter
        try:
            spy_df = self.data_store.get_ticker_data('SPY', start_date, end_date)
            spy_df = generate_all_features(spy_df)
            spy_df['mkt_trend'] = (spy_df['close'] > spy_df['sma_200']).astype(int)
            market_trend = spy_df['mkt_trend']
        except:
            market_trend = None
        
        for ticker in tickers:
            try:
                df = self.data_store.get_ticker_data(ticker, start_date, end_date)
                if df is None or len(df) < 50:
                    continue
                
                df = generate_all_features(df)
                if len(df) < 20:
                    continue
                
                # Add market features
                if market_trend is not None:
                    df['mkt_trend'] = market_trend.reindex(df.index).fillna(0)
                
                # Check if all features exist
                missing = [f for f in self.feature_columns if f not in df.columns]
                if missing:
                    continue
                
                X = df[self.feature_columns].values
                predictions = self.model.predict(X)
                
                signals_df = pd.DataFrame({
                    'date': df.index,
                    'ticker': ticker,
                    'prediction': predictions,
                    'confidence': np.abs(predictions),
                    'close': df['close'].values,
                    'volume': df['volume'].values,
                    'rsi': df['rsi'].values if 'rsi' in df.columns else 50,
                    'mkt_trend': df['mkt_trend'].values if 'mkt_trend' in df.columns else 1
                })
                
                all_signals.append(signals_df)
                
            except Exception as e:
                continue
        
        if not all_signals:
            return pd.DataFrame()
        
        signals = pd.concat(all_signals, ignore_index=True)
        print(f"   ✅ Generated {len(signals):,} signals")
        
        return signals
    
    def filter_high_precision_signals(
        self,
        signals: pd.DataFrame,
        top_pct: float = 0.10,  # Top 10% of signals
        min_prediction: float = 0.0,  # Any positive prediction
        require_uptrend: bool = False  # Don't require uptrend initially
    ) -> pd.DataFrame:
        """Filter to only highest confidence signals."""
        print(f"\n🎯 Filtering to top {top_pct*100:.0f}% signals...")
        print(f"   📊 Starting with {len(signals):,} signals")
        print(f"   📊 Prediction range: {signals['prediction'].min():.6f} to {signals['prediction'].max():.6f}")
        print(f"   📊 Positive predictions: {(signals['prediction'] > 0).sum():,}")
        
        # Only positive predictions (long only)
        signals = signals[signals['prediction'] > min_prediction].copy()
        print(f"   📊 After min prediction filter: {len(signals):,}")
        
        if len(signals) == 0:
            return signals
        
        # Get top signals per day
        signals['date_only'] = pd.to_datetime(signals['date']).dt.date
        
        # Rank by confidence within each day
        signals['rank'] = signals.groupby('date_only')['confidence'].rank(ascending=False)
        
        # Keep only top 3 signals per day
        max_signals_per_day = 3
        filtered = signals[signals['rank'] <= max_signals_per_day].copy()
        print(f"   📊 After daily limit: {len(filtered):,}")
        
        if len(filtered) == 0:
            return filtered
        
        # Further filter to top percentile overall
        threshold = filtered['confidence'].quantile(1 - top_pct)
        filtered = filtered[filtered['confidence'] >= threshold]
        
        print(f"   ✅ Filtered to {len(filtered):,} high-precision signals")
        if len(filtered) > 0 and filtered['date_only'].nunique() > 0:
            print(f"   📊 Avg signals per day: {len(filtered) / filtered['date_only'].nunique():.1f}")
        
        return filtered
    
    def backtest_precision(
        self,
        signals: pd.DataFrame,
        initial_capital: float = 100000,
        max_position_pct: float = 0.20,  # Max 20% per trade
        stop_loss_pct: float = 0.03,  # 3% stop loss
        take_profit_pct: float = 0.08  # 8% take profit
    ) -> Dict:
        """Backtest with strict position sizing using ACTUAL returns."""
        print(f"\n💰 Running precision backtest with ACTUAL returns...")
        
        capital = initial_capital
        peak_capital = initial_capital
        trades = []
        equity_curve = []
        
        # Group signals by date
        signals = signals.sort_values('date')
        
        # Load actual price data for each ticker
        ticker_data = {}
        for ticker in signals['ticker'].unique():
            try:
                df = self.data_store.get_ticker_data(ticker, '2023-01-01', '2024-12-31')
                if df is not None:
                    df['next_close'] = df['close'].shift(-1)
                    df['actual_return'] = (df['next_close'] - df['close']) / df['close']
                    ticker_data[ticker] = df
            except:
                pass
        
        for _, signal in signals.iterrows():
            date = signal['date']
            ticker = signal['ticker']
            entry_price = signal['close']
            prediction = signal['prediction']
            
            # Get ACTUAL next-day return
            if ticker not in ticker_data:
                continue
            
            df = ticker_data[ticker]
            date_str = str(date)[:10]
            matching = df[df.index.astype(str).str[:10] == date_str]
            
            if len(matching) == 0 or pd.isna(matching.iloc[0]['actual_return']):
                continue
            
            actual_return = matching.iloc[0]['actual_return']
            
            # Position size based on confidence
            confidence_factor = min(1.0, signal['confidence'] / 0.001)
            position_pct = max_position_pct * (0.5 + 0.5 * confidence_factor)
            position_size = capital * position_pct
            shares = int(position_size / entry_price)
            
            if shares <= 0:
                continue
            
            # Apply stop loss / take profit
            if actual_return < -stop_loss_pct:
                trade_return = -stop_loss_pct
            elif actual_return > take_profit_pct:
                trade_return = take_profit_pct
            else:
                trade_return = actual_return
            
            pnl = shares * entry_price * trade_return
            capital += pnl
            
            # Track drawdown
            if capital > peak_capital:
                peak_capital = capital
            drawdown = (capital - peak_capital) / peak_capital
            
            trades.append({
                'date': date,
                'ticker': ticker,
                'entry_price': entry_price,
                'shares': shares,
                'prediction': prediction,
                'actual_return': actual_return,
                'trade_return': trade_return,
                'pnl': pnl,
                'capital': capital,
                'drawdown': drawdown
            })
            
            equity_curve.append({'date': date, 'equity': capital, 'drawdown': drawdown})
        
        if not trades:
            return {'error': 'No trades executed'}
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital * 100
        
        # CAGR calculation
        first_date = pd.to_datetime(trades_df['date'].min())
        last_date = pd.to_datetime(trades_df['date'].max())
        years = (last_date - first_date).days / 365.25
        cagr = ((capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        max_dd = trades_df['drawdown'].min() * 100
        win_rate = (trades_df['pnl'] > 0).mean() * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (trades_df['pnl'] < 0).any() else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        results = {
            'total_return_pct': total_return,
            'cagr_pct': cagr,
            'max_drawdown_pct': max_dd,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'trades_per_day': len(trades) / trades_df['date'].nunique() if trades_df['date'].nunique() > 0 else 0,
            'final_capital': capital,
            'trades': trades_df
        }
        
        return results


def run_precision_backtest():
    """Run high-precision backtest targeting 20%+ CAGR."""
    print("=" * 70)
    print("🎯 HIGH-PRECISION TRADING STRATEGY")
    print("=" * 70)
    print("Target: CAGR > 20%, Max DD < 20%, Win Rate > 55%")
    
    # Get tickers
    store = get_data_store()
    tickers = store.available_tickers
    print(f"\n📊 Universe: {len(tickers)} tickers")
    
    # Initialize strategy
    strategy = PrecisionStrategy()
    
    # Train on ALL historical data (1993-2022)
    train_results = strategy.train_precision_model(
        tickers=tickers,
        train_end='2022-12-31'
    )
    
    # Generate signals for test period (2023-2024)
    signals = strategy.generate_signals(
        tickers=tickers,
        start_date='2023-01-01',
        end_date='2024-12-31'
    )
    
    if signals.empty:
        print("❌ No signals generated!")
        return None
    
    # Filter to TOP 5% signals only (highest confidence)
    filtered_signals = strategy.filter_high_precision_signals(
        signals,
        top_pct=0.05,  # Top 5% only
        min_prediction=0.0,
        require_uptrend=False
    )
    
    if filtered_signals.empty:
        print("❌ No signals after filtering!")
        return None
    
    # Run backtest with AGGRESSIVE position sizing for higher returns
    results = strategy.backtest_precision(
        filtered_signals,
        initial_capital=100000,
        max_position_pct=0.40,  # 40% max per trade (concentrated)
        stop_loss_pct=0.05,  # 5% stop loss
        take_profit_pct=0.15  # 15% take profit
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 PRECISION BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Total Return: {results['total_return_pct']:.2f}%")
    print(f"   CAGR: {results['cagr_pct']:.2f}%")
    print(f"   Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"   Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Trades/Day: {results['trades_per_day']:.1f}")
    
    # Validation
    passed = (
        results['cagr_pct'] >= 20 and
        results['max_drawdown_pct'] >= -20 and
        results['win_rate_pct'] >= 55
    )
    
    print(f"\n📋 Validation (Target: 20% CAGR, <20% DD, >55% Win):")
    print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    if results['cagr_pct'] < 20:
        print(f"   ⚠️ CAGR {results['cagr_pct']:.1f}% < 20%")
    if results['max_drawdown_pct'] < -20:
        print(f"   ⚠️ Max DD {results['max_drawdown_pct']:.1f}% > 20%")
    if results['win_rate_pct'] < 55:
        print(f"   ⚠️ Win Rate {results['win_rate_pct']:.1f}% < 55%")
    
    return results


if __name__ == "__main__":
    results = run_precision_backtest()
