"""
Run a real backtest using trained models and the enhanced trading engine.
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys
sys.path.insert(0, '.')

from src.data.tiingo_loader import TiingoDataLoader
from src.features.indicators import apply_indicators
from src.models.cpu_models.xgboost_model import XGBoostModel
from enhanced_trading_engine import EnhancedTradingEngine

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and prepare features for model."""
    df = apply_indicators(df)
    
    # Create target: next day's return
    df['target'] = df['close'].pct_change().shift(-1)
    
    # Drop rows with NaN
    df = df.dropna()
    
    return df

def train_and_predict(ticker: str, loader: TiingoDataLoader) -> dict:
    """Train model on ticker and get predictions."""
    # Load data
    df = loader.load_ticker_data(ticker)
    if df is None or len(df) < 500:
        return None
    
    # Prepare features
    df = prepare_features(df)
    if len(df) < 300:
        return None
    
    # Split data: 70% train, 15% val, 15% test
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Feature columns (exclude target and OHLCV)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    if len(feature_cols) < 5:
        return None
    
    # Prepare X, y
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    # Train model
    model = XGBoostModel()
    model.fit(X_train, y_train)
    
    # Get predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_dir = np.mean(np.sign(train_pred) == np.sign(y_train)) * 100
    val_dir = np.mean(np.sign(val_pred) == np.sign(y_val)) * 100
    test_dir = np.mean(np.sign(test_pred) == np.sign(y_test)) * 100
    
    # Trading simulation on test set
    trades = []
    capital = 10000
    position = 0
    
    for i in range(len(test_pred) - 1):
        pred = test_pred[i]
        actual = y_test[i]
        price = test_df['close'].iloc[i]
        next_price = test_df['close'].iloc[i + 1]
        
        # Simple strategy: go long if prediction > 0
        if pred > 0 and position == 0:
            position = capital / price
            entry_price = price
        elif pred <= 0 and position > 0:
            # Close position
            pnl = position * (price - entry_price)
            capital += pnl
            trades.append({
                'entry': entry_price,
                'exit': price,
                'pnl': pnl,
                'return_pct': (price - entry_price) / entry_price * 100
            })
            position = 0
    
    # Close any remaining position
    if position > 0:
        final_price = test_df['close'].iloc[-1]
        pnl = position * (final_price - entry_price)
        capital += pnl
        trades.append({
            'entry': entry_price,
            'exit': final_price,
            'pnl': pnl,
            'return_pct': (final_price - entry_price) / entry_price * 100
        })
    
    # Calculate trading metrics
    if trades:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) * 100
        total_return = (capital - 10000) / 10000 * 100
    else:
        win_rate = 0
        total_return = 0
    
    return {
        'ticker': ticker,
        'samples': len(df),
        'train_dir': train_dir,
        'val_dir': val_dir,
        'test_dir': test_dir,
        'trades': len(trades),
        'win_rate': win_rate,
        'total_return': total_return,
        'final_capital': capital
    }

def run_backtest():
    """Run backtest on all available tickers."""
    print("🚀 REAL MODEL BACKTEST")
    print("=" * 60)
    
    loader = TiingoDataLoader()
    tickers = loader.get_available_tickers()
    print(f"Found {len(tickers)} tickers")
    
    # Initialize trading engine for safety checks
    engine = EnhancedTradingEngine()
    
    results = []
    successful = 0
    failed = 0
    
    # Test on all tickers
    sample_tickers = tickers  # All tickers
    
    print(f"\nTesting {len(sample_tickers)} tickers...")
    print("-" * 60)
    
    for i, ticker in enumerate(sample_tickers):
        # Skip blacklisted tickers
        if ticker in engine.blacklist:
            print(f"  {ticker}: SKIPPED (blacklisted)")
            continue
        
        try:
            result = train_and_predict(ticker, loader)
            if result:
                results.append(result)
                successful += 1
                status = "✅" if result['test_dir'] > 52 else "⚠️"
                print(f"  {status} {ticker}: Dir={result['test_dir']:.1f}%, Win={result['win_rate']:.1f}%, Return={result['total_return']:.1f}%")
            else:
                failed += 1
                print(f"  ❌ {ticker}: Insufficient data")
        except Exception as e:
            failed += 1
            print(f"  ❌ {ticker}: Error - {str(e)[:50]}")
    
    print("\n" + "=" * 60)
    print("📊 BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    
    if results:
        df = pd.DataFrame(results)
        
        print(f"\nTickers tested: {successful}")
        print(f"Tickers failed: {failed}")
        print(f"\nDirection Accuracy:")
        print(f"  Train: {df['train_dir'].mean():.1f}%")
        print(f"  Val:   {df['val_dir'].mean():.1f}%")
        print(f"  Test:  {df['test_dir'].mean():.1f}%")
        print(f"\nTrading Performance:")
        print(f"  Avg Win Rate: {df['win_rate'].mean():.1f}%")
        print(f"  Avg Return:   {df['total_return'].mean():.1f}%")
        print(f"  Total Trades: {df['trades'].sum()}")
        
        # Top performers
        print(f"\n🏆 TOP 5 PERFORMERS (by test direction accuracy):")
        top5 = df.nlargest(5, 'test_dir')
        for _, row in top5.iterrows():
            print(f"  {row['ticker']}: Dir={row['test_dir']:.1f}%, Win={row['win_rate']:.1f}%, Return={row['total_return']:.1f}%")
        
        # Profitable tickers
        profitable = df[df['total_return'] > 0]
        print(f"\n💰 PROFITABLE TICKERS: {len(profitable)}/{len(df)} ({len(profitable)/len(df)*100:.0f}%)")
        
        # Save results
        output_file = 'reports/real_backtest_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n📁 Results saved to {output_file}")
        
        # Update compact results with real data
        compact_df = pd.read_csv('reports/compact_results.csv')
        for _, row in df.iterrows():
            mask = compact_df['ticker'] == row['ticker']
            if mask.any():
                compact_df.loc[mask, 'test_dir'] = row['test_dir']
                compact_df.loc[mask, 'is_good'] = row['test_dir'] > 52
        compact_df.to_csv('reports/compact_results.csv', index=False)
        print("📁 Updated compact_results.csv with real test data")
        
        return df
    else:
        print("No results to display")
        return None

if __name__ == "__main__":
    results = run_backtest()
