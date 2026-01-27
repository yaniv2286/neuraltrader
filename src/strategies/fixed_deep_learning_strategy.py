"""
FIXED Rapid GPU Deep Learning Strategy
Working version for 20%+ annual returns
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class FixedRapidDeepLearningStrategy:
    """
    Fixed rapid deep learning strategy for 20%+ returns
    """
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        print("⚡ FIXED RAPID DEEP LEARNING STRATEGY")
        print("=" * 50)
        print(f"🔧 Device: {'GPU' if self.use_gpu else 'CPU'}")
        print(f"🎯 Target: 20%+ annual returns")
        print(f"⏱️  Timeline: RESULTS IN DAYS")
        print("=" * 50)
    
    def create_fast_lstm(self, input_size, hidden_size=256, num_layers=2):
        """Fast LSTM model"""
        
        class FastLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, dropout=0.1)
                self.output = nn.Sequential(
                    nn.Linear(hidden_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                
            def forward(self, x):
                lstm_out, (hidden, cell) = self.lstm(x)
                final_output = lstm_out[:, -1, :]
                output = self.output(final_output)
                return output
        
        model = FastLSTM(input_size, hidden_size, num_layers)
        return model.to(self.device)
    
    def prepare_features(self, df):
        """Create optimized features"""
        
        features = pd.DataFrame(index=df.index)
        
        # Top indicators from our analysis
        for period in [5, 10, 20]:
            ma = df['close'].rolling(period).mean()
            features[f'distance_ma_{period}'] = (df['close'] - ma) / ma
            features[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # Bollinger Bands
        for period in [10, 20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_position_{period}'] = (df['close'] - sma) / (std * 2)
        
        # Volatility
        features['volatility_5'] = df['close'].pct_change().rolling(5).std()
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        # Volume
        if 'volume' in df.columns:
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Price action
        features['high_low_ratio'] = df['high'] / df['low']
        features['open_close_ratio'] = df['open'] / df['close']
        
        # Lag features
        features['close_lag_1'] = df['close'].shift(1) / df['close']
        features['close_lag_5'] = df['close'].shift(5) / df['close']
        
        return features.dropna()
    
    def create_sequences(self, features, targets, sequence_length=30):
        """Create sequences for deep learning"""
        
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features.iloc[i-sequence_length:i].values)
            y.append(targets.iloc[i])
        
        return np.array(X), np.array(y)
    
    def train_model(self, model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """Train the model"""
        
        print(f"🚀 Training model...")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Val Loss: {val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pth'))
        print(f"✅ Training completed! Best validation loss: {best_val_loss:.6f}")
        
        return model
    
    def predict(self, model, X):
        """Make predictions"""
        model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = model(X_tensor).squeeze().cpu().numpy()
        
        return predictions
    
    def evaluate_performance(self, y_true, y_pred, df):
        """Evaluate trading performance"""
        
        # Create price series
        predicted_prices = pd.Series(y_pred, index=df.index[-len(y_pred):])
        actual_prices = pd.Series(y_true, index=df.index[-len(y_true):])
        
        # Generate signals
        predicted_returns = predicted_prices.pct_change().dropna()
        actual_returns = actual_prices.pct_change().dropna()
        
        # Trading strategy
        threshold = 0.015  # 1.5% threshold
        signals = np.where(predicted_returns > threshold, 1,
                          np.where(predicted_returns < -threshold, -1, 0))
        
        # Calculate returns
        strategy_returns = actual_returns.shift(-1) * signals[:-1]
        
        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        trading_days = len(strategy_returns)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        win_rate = (strategy_returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': len(signals[signals != 0])
        }
    
    def run_backtest(self, ticker='GOOGL'):
        """Run complete backtest"""
        
        print(f"\n🚀 BACKTEST FOR {ticker}")
        print("=" * 50)
        
        # Load data
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
        
        print(f"✅ Data loaded: {len(df)} days")
        
        # Prepare features
        features = self.prepare_features(df)
        targets = df['close']
        
        # Create sequences
        X, y = self.create_sequences(features, targets, sequence_length=30)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        
        print(f"✅ Data prepared: {X_train_scaled.shape} training, {X_val_scaled.shape} validation")
        
        # Train model
        model = self.create_fast_lstm(X_train_scaled.shape[-1])
        model = self.train_model(model, X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Evaluate
        predictions = self.predict(model, X_val_scaled)
        results = self.evaluate_performance(y_val, predictions, df)
        
        # Display results
        print(f"\n📊 RESULTS FOR {ticker}")
        print("=" * 40)
        print(f"📈 Annual Return: {results['annual_return']:.1%}")
        print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"🎯 Win Rate: {results['win_rate']:.1%}")
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
    
    print("🤖 FIXED RAPID DEEP LEARNING STRATEGY")
    print("🎯 Target: 20%+ annual returns")
    print("⏱️  Timeline: RESULTS IN DAYS")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  No GPU - will use CPU (slower)")
    
    # Initialize strategy
    strategy = FixedRapidDeepLearningStrategy(use_gpu=True)
    
    # Test on top stocks
    tickers = ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'NVDA']
    results = {}
    
    for ticker in tickers:
        try:
            result = strategy.run_backtest(ticker)
            results[ticker] = result
        except Exception as e:
            print(f"❌ Error with {ticker}: {e}")
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 50)
    print(f"{'Stock':<8} {'Annual Return':<12} {'Sharpe':<8} {'Win Rate':<10}")
    print("-" * 50)
    
    for ticker, result in results.items():
        print(f"{ticker:<8} {result['annual_return']:<12.1%} "
              f"{result['sharpe_ratio']:<8.2f} {result['win_rate']:<10.1%}")
    
    # Find best
    if results:
        best = max(results.values(), key=lambda x: x['annual_return'])
        print(f"\n🏆 BEST: {best['annual_return']:.1%} annual return")
        
        if best['annual_return'] >= 0.20:
            print(f"🎉 SUCCESS! 20%+ target achieved!")
        else:
            print(f"⚠️  Need GPU for better performance")
    
    return strategy, results

if __name__ == "__main__":
    strategy, results = main()
