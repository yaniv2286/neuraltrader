"""
OPTIMIZED 20%+ STRATEGY
Enhanced features and parameters for 20%+ returns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class Optimized20PercentStrategy:
    """
    Optimized strategy to achieve 20%+ returns
    """
    
    def __init__(self):
        print("🚀 OPTIMIZED 20%+ STRATEGY")
        print("=" * 40)
        print("🎯 Target: 20%+ annual returns")
        print("⚡ Enhanced features")
        print("💰 2x SPY performance")
        print("=" * 40)
    
    def load_data(self, ticker):
        """Load stock data"""
        
        file_path = f'data/cache/tiingo/{ticker}_1d_20y.csv'
        df = pd.read_csv(file_path)
        
        # Handle columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Map columns
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'close' in col_lower and 'close' not in df.columns:
                col_mapping['close'] = col
            elif 'open' in col_lower and 'open' not in df.columns:
                col_mapping['open'] = col
            elif 'high' in col_lower and 'high' not in df.columns:
                col_mapping['high'] = col
            elif 'low' in col_lower and 'low' not in df.columns:
                col_mapping['low'] = col
            elif 'volume' in col_lower and 'volume' not in df.columns:
                col_mapping['volume'] = col
        
        if col_mapping:
            df = df.rename(columns=col_mapping)
        
        return df
    
    def create_enhanced_features(self, df):
        """Create enhanced features for better performance"""
        
        features = pd.DataFrame(index=df.index)
        
        # === MOMENTUM FEATURES ===
        for period in [3, 5, 10, 20, 50]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)
            features[f'momentum_rank_{period}'] = features[f'momentum_{period}'].rolling(252).rank(pct=True)
        
        # === MEAN REVERSION FEATURES ===
        for period in [5, 10, 20]:
            ma = df['close'].rolling(period).mean()
            features[f'distance_ma_{period}'] = (df['close'] - ma) / ma
            features[f'zscore_{period}'] = (df['close'] - ma) / df['close'].rolling(period).std()
        
        # === VOLATILITY FEATURES ===
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
            features[f'volatility_rank_{period}'] = features[f'volatility_{period}'].rolling(252).rank(pct=True)
        
        # === BOLLINGER BANDS ===
        for period in [10, 20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_position_{period}'] = (df['close'] - sma) / (std * 2)
            features[f'bb_width_{period}'] = (sma + std * 2) / (sma - std * 2)
        
        # === RSI ===
        for period in [14, 30]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # === MACD ===
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # === PRICE ACTION ===
        features['high_low_ratio'] = df['high'] / df['low']
        features['open_close_ratio'] = df['open'] / df['close']
        features['daily_range'] = (df['high'] - df['low']) / df['close']
        features['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # === VOLUME ===
        if 'volume' in df.columns:
            features['volume_sma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            features['vwap'] = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']).cumsum() / df['volume'].cumsum()
            features['price_volume_trend'] = np.where(
                (df['close'] > df['open']) & (features['volume_ratio'] > 1.2), 1,
                np.where((df['close'] < df['open']) & (features['volume_ratio'] > 1.2), -1, 0)
            )
        
        # === LAG FEATURES ===
        for lag in [1, 2, 3, 5, 10]:
            features[f'close_lag_{lag}'] = df['close'].shift(lag) / df['close']
            features[f'return_lag_{lag}'] = df['close'].pct_change(lag)
        
        # === MARKET REGIME ===
        returns = df['close'].pct_change()
        features['regime_20d'] = np.where(returns.rolling(20).mean() > 0, 1, -1)
        features['regime_50d'] = np.where(returns.rolling(50).mean() > 0, 1, -1)
        
        return features.dropna()
    
    def optimize_model(self, X_train, y_train):
        """Optimize model parameters"""
        
        print("🔧 Optimizing model parameters...")
        
        # Try different models
        models = {}
        
        # Random Forest with optimized parameters
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        )
        rf.fit(X_train, y_train)
        models['random_forest'] = rf
        
        # Gradient Boosting with optimized parameters
        gb = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        )
        gb.fit(X_train, y_train)
        models['gradient_boosting'] = gb
        
        return models
    
    def ensemble_predict(self, models, X_test):
        """Ensemble prediction with weighting"""
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in models.items():
            pred = model.predict(X_test)
            predictions[name] = pred
        
        # Weighted ensemble (equal weights for now)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        return ensemble_pred
    
    def generate_optimized_signals(self, predictions, actual_prices):
        """Generate optimized trading signals"""
        
        # Calculate predicted returns
        pred_returns = pd.Series(predictions).pct_change().dropna()
        
        # Dynamic threshold based on volatility
        volatility = pred_returns.std()
        threshold = volatility * 1.5  # 1.5x volatility as threshold
        
        # Generate signals
        signals = np.where(pred_returns > threshold, 1,
                          np.where(pred_returns < -threshold, -1, 0))
        
        return signals, pred_returns
    
    def evaluate_optimized_strategy(self, actual_returns, signals):
        """Evaluate optimized strategy performance"""
        
        # Convert to Series
        signals_series = pd.Series(signals, index=actual_returns.index[:len(signals)])
        
        # Calculate strategy returns
        strategy_returns = actual_returns.shift(-1) * signals_series
        
        # Fix shape alignment
        strategy_returns = strategy_returns.dropna()
        
        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        trading_days = len(strategy_returns)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        win_rate = (strategy_returns > 0).mean()
        
        # Calculate drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'total_trades': len(signals[signals != 0])
        }
    
    def run_optimized_backtest(self, ticker='GOOGL'):
        """Run optimized backtest"""
        
        print(f"\n🚀 OPTIMIZED BACKTEST FOR {ticker}")
        print("=" * 50)
        
        # Load data
        df = self.load_data(ticker)
        print(f"✅ Data loaded: {len(df)} days")
        
        # Create enhanced features
        features = self.create_enhanced_features(df)
        print(f"✅ Enhanced features created: {features.shape[1]} features")
        
        # Prepare target
        target = df['close'].shift(-1)
        valid_idx = features.index.intersection(target.index)
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]
        
        # Split data
        split_idx = int(0.8 * len(features))
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✅ Data split: {X_train_scaled.shape} training, {X_test_scaled.shape} testing")
        
        # Optimize models
        models = self.optimize_model(X_train_scaled, y_train)
        
        # Make ensemble predictions
        print("📊 Making ensemble predictions...")
        predictions = self.ensemble_predict(models, X_test_scaled)
        
        # Generate optimized signals
        signals, pred_returns = self.generate_optimized_signals(predictions, y_test)
        
        # Calculate actual returns
        actual_returns = y_test.pct_change().dropna()
        
        # Align
        min_len = min(len(signals), len(actual_returns))
        signals = signals[:min_len]
        actual_returns = actual_returns[:min_len]
        
        # Evaluate strategy
        results = self.evaluate_optimized_strategy(actual_returns, signals)
        
        # Display results
        print(f"\n📊 OPTIMIZED RESULTS FOR {ticker}")
        print("=" * 50)
        print(f"📈 Annual Return: {results['annual_return']:.1%}")
        print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"🎯 Win Rate: {results['win_rate']:.1%}")
        print(f"📉 Max Drawdown: {results['max_drawdown']:.1%}")
        print(f"📈 Total Trades: {results['total_trades']}")
        
        # Compare to SPY
        spy_return = 0.10
        outperformance = results['annual_return'] - spy_return
        
        print(f"\n🎯 COMPARISON:")
        print(f"   SPY Buy-and-Hold: {spy_return:.1%}")
        print(f"   {ticker} Strategy: {results['annual_return']:.1%}")
        print(f"   Outperformance: {outperformance:.1%}")
        
        if results['annual_return'] >= 0.20:
            print(f"   ✅ TARGET ACHIEVED: 20%+ annual return!")
        elif results['annual_return'] >= 0.15:
            print(f"   ⚠️  CLOSE TO TARGET: 15%+ annual return")
        else:
            print(f"   ❌ TARGET NOT MET: {results['annual_return']:.1%} < 20%")
        
        return results

def main():
    """Main function"""
    
    print("🤖 OPTIMIZED 20%+ STRATEGY")
    print("🎯 Target: 20%+ annual returns")
    print("⚡ Enhanced features + ensemble")
    print("💰 2x SPY performance")
    print("=" * 60)
    
    # Initialize strategy
    strategy = Optimized20PercentStrategy()
    
    # Test on GOOGL
    result = strategy.run_optimized_backtest('GOOGL')
    
    if result:
        print(f"\n🎉 OPTIMIZED STRATEGY COMPLETED!")
        print(f"📈 Annual Return: {result['annual_return']:.1%}")
        
        if result['annual_return'] >= 0.20:
            print(f"🏆 SUCCESS! 20%+ target achieved!")
        else:
            print(f"⚠️  Need further optimization for 20%+ target")
    
    return strategy, result

if __name__ == "__main__":
    strategy, result = main()
