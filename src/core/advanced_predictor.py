"""
Advanced ML Predictor - Cross-Asset Learning with Walk-Forward Optimization.
Uses ALL tickers and full historical data for better predictions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')

from src.features.technical_indicators import generate_all_features
from src.models.cpu_models.xgboost_model import XGBoostModel
from src.core.data_store import get_data_store


class AdvancedPredictor:
    """
    Cross-Asset Learning with Walk-Forward Optimization.
    
    Key improvements:
    1. Train on ALL tickers together (750K+ samples vs 2.5K)
    2. Walk-forward optimization (no look-ahead bias)
    3. Feature selection (top 20 predictors)
    4. Market regime features (SPY, VIX context)
    """
    
    def __init__(self):
        self.data_store = get_data_store()
        self.model = None
        self.feature_columns: List[str] = []
        self.top_features: List[str] = []
        self.scaler = None
        
    def prepare_cross_asset_dataset(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        add_market_features: bool = True
    ) -> pd.DataFrame:
        """
        Prepare unified dataset from ALL tickers.
        Each row is one ticker-day observation.
        """
        print(f"📊 Preparing cross-asset dataset from {len(tickers)} tickers...")
        
        all_data = []
        market_data = None
        
        # Load SPY as market benchmark
        if add_market_features:
            try:
                spy_df = self.data_store.get_ticker_data('SPY', start_date, end_date)
                spy_df = generate_all_features(spy_df)
                market_data = spy_df[['close', 'rsi', 'macd', 'bb_pct_b', 'atr_percent']].copy()
                market_data.columns = ['mkt_close', 'mkt_rsi', 'mkt_macd', 'mkt_bb', 'mkt_atr']
                market_data['mkt_trend'] = (market_data['mkt_close'] > market_data['mkt_close'].rolling(50).mean()).astype(int)
                market_data['mkt_return_20d'] = market_data['mkt_close'].pct_change(20)
            except Exception as e:
                print(f"   ⚠️ Could not load SPY for market features: {e}")
                market_data = None
        
        for ticker in tickers:
            try:
                df = self.data_store.get_ticker_data(ticker, start_date, end_date)
                if df is None or len(df) < 100:  # Reduced minimum for test periods
                    continue
                
                # Generate features
                df = generate_all_features(df)
                if len(df) < 50:  # Reduced minimum for test periods
                    continue
                
                # Add ticker identifier
                df['ticker'] = ticker
                
                # Add market features if available
                if market_data is not None:
                    df = df.join(market_data, how='left')
                
                # Create target: next day's log return
                df['target'] = np.log(df['close'].shift(-1) / df['close'])
                df['target_direction'] = np.sign(df['target'])
                
                df = df.dropna()
                all_data.append(df)
                
            except Exception as e:
                continue
        
        if not all_data:
            raise ValueError("No data loaded!")
        
        combined = pd.concat(all_data, ignore_index=False)
        print(f"   ✅ Combined dataset: {len(combined):,} samples from {len(all_data)} tickers")
        
        return combined
    
    def select_top_features(
        self,
        df: pd.DataFrame,
        n_features: int = 20
    ) -> List[str]:
        """
        Select top N features using XGBoost feature importance.
        """
        print(f"🔍 Selecting top {n_features} features...")
        
        # Get feature columns
        exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'target_direction', 
                   'ticker', 'future_return', 'mkt_close']
        feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('mkt_')]
        
        # Add market features back
        mkt_features = [c for c in df.columns if c.startswith('mkt_') and c != 'mkt_close']
        feature_cols.extend(mkt_features)
        
        # Sample for speed
        sample_size = min(50000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        X = sample_df[feature_cols].values
        y = sample_df['target'].values
        
        # Train quick model for feature importance
        model = XGBoostModel(n_estimators=100, max_depth=3)
        model.fit(X, y)
        
        # Get feature importance
        importance = model.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(n_features)['feature'].tolist()
        
        print(f"   ✅ Top features: {', '.join(top_features[:5])}...")
        
        return top_features
    
    def walk_forward_train(
        self,
        tickers: List[str],
        train_years: int = 20,  # Use 20 years of training data
        test_years: int = 2,
        start_year: int = 1993,  # Start from 1993 (SPY inception)
        end_year: int = 2024,
        n_features: int = 25,  # More features with more data
        use_expanding_window: bool = True  # Use all available history
    ) -> Dict[str, any]:
        """
        Walk-forward optimization: train on past, test on future.
        
        With use_expanding_window=True:
        - Train on ALL data from start_year to current, test on next period
        - This uses maximum available history (50+ years for some tickers)
        
        Example with expanding window:
        - Train on 1990-2014, test on 2015-2016
        - Train on 1990-2016, test on 2017-2018
        - ... continue until end_year
        """
        print("=" * 70)
        print("🚀 WALK-FORWARD CROSS-ASSET TRAINING (FULL HISTORY)")
        print("=" * 70)
        print(f"   Training mode: {'Expanding window (ALL history)' if use_expanding_window else f'Rolling {train_years} years'}")
        print(f"   Test window: {test_years} year(s)")
        print(f"   Period: {start_year} to {end_year}")
        
        all_results = []
        all_predictions = {}
        
        # Walk forward through time
        # Start testing from 2010 to have enough training data
        first_test_year = 2010
        current_year = first_test_year
        
        while current_year + test_years <= end_year + 1:
            # EXPANDING WINDOW: Use ALL data from start_year to current
            if use_expanding_window:
                train_start = f"{start_year}-01-01"
            else:
                train_start = f"{current_year - train_years}-01-01"
            
            train_end = f"{current_year - 1}-12-31"
            test_start = f"{current_year}-01-01"
            test_end = f"{current_year + test_years - 1}-12-31"
            
            years_of_training = current_year - int(train_start[:4])
            print(f"\n📅 Window: Train {train_start[:4]}-{train_end[:4]} ({years_of_training} years), Test {test_start[:4]}-{test_end[:4]}")
            
            try:
                # Prepare training data
                train_df = self.prepare_cross_asset_dataset(tickers, train_start, train_end)
                
                # Select features (on training data only!)
                if not self.top_features:
                    self.top_features = self.select_top_features(train_df, n_features)
                
                # Prepare test data
                test_df = self.prepare_cross_asset_dataset(tickers, test_start, test_end)
                
                # Train model
                X_train = train_df[self.top_features].values
                y_train = train_df['target'].values
                
                X_test = test_df[self.top_features].values
                y_test = test_df['target'].values
                y_test_dir = test_df['target_direction'].values
                
                print(f"   Training on {len(X_train):,} samples...")
                
                model = XGBoostModel(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.02,
                    reg_alpha=0.5,
                    reg_lambda=1.0
                )
                model.fit(X_train, y_train)
                
                # Predict
                predictions = model.predict(X_test)
                pred_direction = np.sign(predictions)
                
                # Evaluate
                direction_accuracy = np.mean(pred_direction == y_test_dir) * 100
                
                # Store predictions by ticker
                test_df['prediction'] = predictions
                for ticker in test_df['ticker'].unique():
                    ticker_preds = test_df[test_df['ticker'] == ticker][['prediction']].copy()
                    ticker_preds['confidence'] = np.abs(ticker_preds['prediction'])
                    if ticker not in all_predictions:
                        all_predictions[ticker] = ticker_preds
                    else:
                        all_predictions[ticker] = pd.concat([all_predictions[ticker], ticker_preds])
                
                result = {
                    'train_period': f"{train_start[:4]}-{train_end[:4]}",
                    'test_period': f"{test_start[:4]}-{test_end[:4]}",
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'direction_accuracy': direction_accuracy
                }
                all_results.append(result)
                
                print(f"   ✅ Direction Accuracy: {direction_accuracy:.1f}%")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            current_year += test_years
        
        # Summary
        avg_accuracy = 0
        if all_results:
            avg_accuracy = np.mean([r['direction_accuracy'] for r in all_results])
            print(f"\n{'=' * 70}")
            print(f"📊 WALK-FORWARD RESULTS SUMMARY")
            print(f"{'=' * 70}")
            print(f"   Windows tested: {len(all_results)}")
            print(f"   Average Direction Accuracy: {avg_accuracy:.1f}%")
            print(f"   Best: {max(r['direction_accuracy'] for r in all_results):.1f}%")
            print(f"   Worst: {min(r['direction_accuracy'] for r in all_results):.1f}%")
        
        self.feature_columns = self.top_features
        
        return {
            'results': all_results,
            'predictions': all_predictions,
            'avg_accuracy': avg_accuracy,
            'features_used': self.top_features
        }
    
    def get_predictions_for_backtest(
        self,
        walk_forward_results: Dict
    ) -> Dict[str, pd.DataFrame]:
        """
        Format predictions for BacktestEngine.
        """
        return walk_forward_results.get('predictions', {})


def run_advanced_backtest():
    """Run backtest with advanced cross-asset learning."""
    print("=" * 70)
    print("🧠 ADVANCED CROSS-ASSET ML BACKTEST")
    print("=" * 70)
    
    from src.core import BacktestEngine
    
    # Get tickers
    store = get_data_store()
    engine = BacktestEngine()
    blacklist = set(engine.config.get('blacklist', []))
    
    # Use all available tickers
    all_tickers = [t for t in store.available_tickers if t not in blacklist]
    print(f"\n📊 Universe: {len(all_tickers)} tickers")
    
    # Initialize predictor
    predictor = AdvancedPredictor()
    
    # Walk-forward training with FULL HISTORY (from 1993)
    results = predictor.walk_forward_train(
        tickers=all_tickers,
        train_years=20,
        test_years=2,
        start_year=1993,  # Use full 30+ years of data
        end_year=2024,
        n_features=25,
        use_expanding_window=True  # Use ALL available history
    )
    
    # Get predictions for backtest
    predictions = predictor.get_predictions_for_backtest(results)
    
    if not predictions:
        print("❌ No predictions generated!")
        return None
    
    # Run backtest on 2020-2024 (most recent test period)
    print(f"\n{'=' * 70}")
    print("📈 RUNNING BACKTEST WITH WALK-FORWARD PREDICTIONS")
    print("=" * 70)
    
    backtest_tickers = list(predictions.keys())
    
    result = engine.run(
        tickers=backtest_tickers,
        start_date='2020-01-01',
        end_date='2024-12-31',
        model_predictions=predictions
    )
    
    # Print results
    print(f"\n🎯 Performance Metrics:")
    print(f"   Total Return: {result.total_return_pct:.2f}%")
    print(f"   CAGR: {result.cagr_pct:.2f}%")
    print(f"   Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"   Win Rate: {result.win_rate_pct:.1f}%")
    print(f"   Profit Factor: {result.profit_factor:.2f}")
    print(f"   Total Trades: {result.total_trades}")
    
    print(f"\n📋 Validation:")
    print(f"   Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'reports/advanced_backtest_{timestamp}.xlsx'
    engine.write_excel(result, output_path)
    print(f"\n📁 Excel Output: {output_path}")
    
    return result, results


if __name__ == "__main__":
    result, wf_results = run_advanced_backtest()
