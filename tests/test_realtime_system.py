"""
Test Real-Time Trading System
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ml.adaptive_thresholds import RealTimeThresholdOptimizer

def test_realtime_system():
    """Test the real-time adaptive threshold system"""
    print("Real-Time Trading System Test")
    print("=" * 50)
    
    # Load cached data
    cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache', 'tiingo')
    aapl_file = os.path.join(cache_dir, 'AAPL_1d_20y.csv')
    
    if not os.path.exists(aapl_file):
        print(f"✗ Data file not found: {aapl_file}")
        return
    
    print(f"✓ Loading AAPL data...")
    df = pd.read_csv(aapl_file, index_col=0, parse_dates=True)
    print(f"  Data shape: {df.shape}")
    
    # Use last 60 days for real-time simulation
    df_sim = df.tail(60)
    prices = df_sim['close'].values
    volumes = df_sim['volume'].values
    
    # Initialize adaptive optimizer
    print(f"\nInitializing adaptive optimizer...")
    optimizer = RealTimeThresholdOptimizer()
    
    # Train on historical data (use data before simulation period)
    df_train = df.iloc[-300:-60]  # 240 days training
    returns_train = df_train['close'].pct_change().dropna()
    
    print(f"Training on {len(df_train)} days...")
    
    # Simulate historical training
    for i in range(50, len(returns_train)):
        recent_returns = returns_train.iloc[max(0, i-50):i]
        current_price = df_train['close'].iloc[i]
        current_volume = df_train['volume'].iloc[i]
        
        optimizer.update_thresholds_realtime(
            current_price, current_volume, recent_returns
        )
    
    print(f"✓ Training complete")
    
    # Real-time simulation
    print(f"\nStarting real-time simulation on {len(prices)} days...")
    
    signals = []
    thresholds_history = []
    
    for i in range(10, len(prices)):
        current_price = prices[i]
        current_volume = volumes[i]
        
        # Get recent returns
        recent_prices = prices[max(0, i-20):i]
        recent_returns = pd.Series(np.diff(recent_prices) / recent_prices[:-1])
        
        # Update thresholds in real-time
        new_thresholds = optimizer.update_thresholds_realtime(
            current_price, current_volume, recent_returns
        )
        
        # Get trading signal
        if len(recent_prices) > 1:
            current_return = (current_price - recent_prices[-2]) / recent_prices[-2]
        else:
            current_return = 0
        
        signal = optimizer.get_trading_signal(current_return)
        
        # Store results
        signals.append({
            'day': i,
            'price': current_price,
            'return': current_return,
            'signal': signal,
            'buy_threshold': new_thresholds.buy_threshold,
            'sell_threshold': new_thresholds.sell_threshold,
            'regime': new_thresholds.regime,
            'confidence': new_thresholds.confidence
        })
        
        thresholds_history.append(new_thresholds)
        
        # Print key updates
        if i % 10 == 0:
            print(f"Day {i}: Price=${current_price:.2f}, Return={current_return:.2%}, "
                  f"Signal={signal}, Buy={new_thresholds.buy_threshold:.3f}, "
                  f"Sell={new_thresholds.sell_threshold:.3f}, Regime={new_thresholds.regime}")
    
    # Analyze results
    print(f"\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)
    
    # Signal distribution
    signal_counts = {}
    for s in signals:
        signal_counts[s['signal']] = signal_counts.get(s['signal'], 0) + 1
    
    print(f"Signal Distribution:")
    for signal, count in signal_counts.items():
        percentage = count / len(signals) * 100
        print(f"  {signal}: {count} ({percentage:.1f}%)")
    
    # Threshold evolution
    buy_thresholds = [s['buy_threshold'] for s in signals]
    sell_thresholds = [s['sell_threshold'] for s in signals]
    
    print(f"\nThreshold Evolution:")
    print(f"  Buy Threshold: {buy_thresholds[0]:.4f} → {buy_thresholds[-1]:.4f}")
    print(f"  Sell Threshold: {sell_thresholds[0]:.4f} → {sell_thresholds[-1]:.4f}")
    print(f"  Buy Range: {min(buy_thresholds):.4f} - {max(buy_thresholds):.4f}")
    print(f"  Sell Range: {min(sell_thresholds):.4f} - {max(sell_thresholds):.4f}")
    
    # Regime distribution
    regime_counts = {}
    for s in signals:
        regime_counts[s['regime']] = regime_counts.get(s['regime'], 0) + 1
    
    print(f"\nRegime Distribution:")
    for regime, count in regime_counts.items():
        percentage = count / len(signals) * 100
        print(f"  {regime}: {count} ({percentage:.1f}%)")
    
    # Performance metrics
    final_metrics = optimizer.get_performance_metrics()
    print(f"\nOptimizer Performance:")
    print(f"  Total Updates: {final_metrics['total_updates']}")
    print(f"  Current Buy Threshold: {final_metrics['current_buy_threshold']:.4f}")
    print(f"  Current Sell Threshold: {final_metrics['current_sell_threshold']:.4f}")
    print(f"  Current Regime: {final_metrics['current_regime']}")
    print(f"  Confidence: {final_metrics['current_confidence']:.3f}")
    print(f"  Q-Table Size: {final_metrics['q_table_size']}")
    print(f"  Epsilon: {final_metrics['epsilon']:.4f}")
    
    # Simulate trading performance
    print(f"\n" + "=" * 50)
    print("TRADING PERFORMANCE SIMULATION")
    print("=" * 50)
    
    capital = 10000
    position = 0
    trades = []
    
    for i, s in enumerate(signals):
        if s['signal'] == 'BUY' and position == 0:
            # Buy
            position = capital / s['price']
            capital = 0
            trades.append({
                'type': 'BUY',
                'day': i,
                'price': s['price'],
                'threshold': s['buy_threshold']
            })
            print(f"Day {i}: BUY at ${s['price']:.2f} (threshold: {s['buy_threshold']:.3f})")
            
        elif s['signal'] == 'SELL' and position > 0:
            # Sell
            capital = position * s['price']
            pnl = capital - 10000
            trades.append({
                'type': 'SELL',
                'day': i,
                'price': s['price'],
                'threshold': s['sell_threshold'],
                'pnl': pnl
            })
            print(f"Day {i}: SELL at ${s['price']:.2f} (threshold: {s['sell_threshold']:.3f}) - P&L: ${pnl:.2f}")
            position = 0
    
    # Final portfolio value
    if position > 0:
        final_value = position * prices[-1]
    else:
        final_value = capital
    
    total_return = (final_value - 10000) / 10000
    
    print(f"\nFinal Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Total Trades: {len([t for t in trades if t['type'] == 'SELL'])}")
    
    if len([t for t in trades if t['type'] == 'SELL']) > 0:
        winning_trades = len([t for t in trades if t['type'] == 'SELL' and t.get('pnl', 0) > 0])
        total_trades = len([t for t in trades if t['type'] == 'SELL'])
        win_rate = winning_trades / total_trades
        print(f"Win Rate: {win_rate:.1%}")
    
    # Save model
    optimizer.save_model('realtime_adaptive_model.pkl')
    
    print(f"\n✓ Real-time adaptive model saved")
    
    return {
        'signals': signals,
        'thresholds_history': thresholds_history,
        'final_return': total_return,
        'total_trades': len([t for t in trades if t['type'] == 'SELL'])
    }

if __name__ == "__main__":
    test_realtime_system()
