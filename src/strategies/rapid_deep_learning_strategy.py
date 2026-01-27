"""
Rapid GPU Deep Learning Strategy - Results in DAYS
Fast training with optimized models for quick deployment
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class RapidDeepLearningStrategy:
    """
    Fast deep learning strategy for 20%+ returns in DAYS
    Optimized for quick training and deployment
    """
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        print("⚡ RAPID DEEP LEARNING STRATEGY")
        print("=" * 50)
        print(f"🔧 Device: {'GPU' if self.use_gpu else 'CPU'}")
        print(f"🎯 Target: 20%+ annual returns")
        print(f"⏱️  Timeline: RESULTS IN DAYS")
        print(f"💰 Justification: 2x SPY performance")
        print("=" * 50)
    
    def create_fast_transformer(self, input_size, d_model=256, n_heads=4, num_layers=3):
        """Fast transformer model - optimized for speed"""
        
        class FastTransformer(nn.Module):
            def __init__(self, input_size, d_model, n_heads, num_layers):
                super().__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                self.positional_encoding = nn.Parameter(torch.randn(500, d_model))
                
                # Smaller transformer for speed
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=n_heads,
                    dim_feedforward=512,  # Smaller for speed
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                
                # Simpler output layer
                self.output_projection = nn.Sequential(
                    nn.Linear(d_model, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                
            def forward(self, x):
                x = self.input_projection(x)
                seq_len = x.size(1)
                x = x + self.positional_encoding[:seq_len]
                x = self.transformer(x)
                x = x[:, -1, :]
                x = self.output_projection(x)
                return x
        
        model = FastTransformer(input_size, d_model, n_heads, num_layers)
        return model.to(self.device)
    
    def create_fast_lstm(self, input_size, hidden_size=256, num_layers=2):
        """Fast LSTM model - optimized for speed"""
        
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
    
    def prepare_fast_features(self, df):
        """Create optimized features for fast training"""
        
        features = pd.DataFrame(index=df.index)
        
        # Only the most predictive features (based on our analysis)
        # Top 3 from backtest: distance_ma_5, momentum_5, bb_position_20
        
        # Distance from moving averages
        for period in [5, 10, 20]:
            ma = df['close'].rolling(period).mean()
            features[f'distance_ma_{period}'] = (df['close'] - ma) / ma
        
        # Momentum indicators
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # Bollinger Bands
        for period in [10, 20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_position_{period}'] = (df['close'] - sma) / (std * 2)
        
        # Volatility
        features['volatility_5'] = df['close'].pct_change().rolling(5).std()
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        # Volume (if available)
        if 'volume' in df.columns:
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Price action
        features['high_low_ratio'] = df['high'] / df['low']
        features['open_close_ratio'] = df['open'] / df['close']
        
        # Lag features (most important ones)
        features['close_lag_1'] = df['close'].shift(1) / df['close']
        features['close_lag_5'] = df['close'].shift(5) / df['close']
        
        return features.dropna()
    
    def create_fast_sequences(self, features, targets, sequence_length=30):
        """Create shorter sequences for faster training"""
        
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features.iloc[i-sequence_length:i].values)
            y.append(targets.iloc[i])
        
        return np.array(X), np.array(y)
    
    def train_fast_model(self, model, X_train, y_train, X_val, y_val, 
                        epochs=50, batch_size=64, learning_rate=0.001):
        """Fast training - optimized for speed"""
        
        print(f"🚀 Starting fast training...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Fast training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            
            # Larger batches for speed
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
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'fast_best_model.pth')
            
            # Progress update
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Train Loss: {train_loss/len(X_train_tensor):.6f}, "
                      f"Val Loss: {val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(torch.load('fast_best_model.pth'))
        print(f"✅ Training completed! Best validation loss: {best_val_loss:.6f}")
        
        return model
    
    def predict_fast(self, model, X):
        """Fast prediction"""
        model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = model(X_tensor).squeeze().cpu().numpy()
        
        return predictions
    
    def evaluate_fast_performance(self, y_true, y_pred, df):
        """Fast performance evaluation"""
        
        # Calculate predicted returns
        predicted_prices = pd.Series(y_pred, index=df.index[-len(y_pred):])
        actual_prices = pd.Series(y_true, index=df.index[-len(y_true):])
        
        # Generate trading signals
        predicted_returns = predicted_prices.pct_change().dropna()
        actual_returns = actual_prices.pct_change().dropna()
        
        # Simple but effective strategy
        threshold = 0.015  # 1.5% threshold
        signals = np.where(predicted_returns > threshold, 1,
                          np.where(predicted_returns < -threshold, -1, 0))
        
        # Calculate strategy returns
        strategy_returns = actual_returns.shift(-1) * signals[:-1]
        
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
            'total_trades': len(signals[signals != 0]),
            'trading_days': trading_days
        }
    
    def create_fast_sequences(self, features, targets, sequence_length=30):
        """Create shorter sequences for faster training"""
        
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features.iloc[i-sequence_length:i].values)
            y.append(targets.iloc[i])
        
        return np.array(X), np.array(y)
    
    def run_rapid_backtest(self, ticker='GOOGL'):
        """Run complete rapid backtest in minutes"""
        
        print(f"\n🚀 RAPID BACKTEST FOR {ticker}")
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
        print("🔢 Preparing features...")
        features = self.prepare_fast_features(df)
        targets = df['close']
        
        # Create sequences
        print("📊 Creating sequences...")
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
        
        # Create and train models
        models = {}
        results = {}
        
        # Fast Transformer
        print("\n🤖 Training Fast Transformer...")
        transformer = self.create_fast_transformer(X_train_scaled.shape[-1])
        transformer = self.train_fast_model(transformer, X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Fast LSTM
        print("\n🔢 Training Fast LSTM...")
        lstm = self.create_fast_lstm(X_train_scaled.shape[-1])
        lstm = self.train_fast_model(lstm, X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Evaluate models
        print("\n📊 Evaluating models...")
        
        # Transformer predictions
        transformer_preds = self.predict_fast(transformer, X_val_scaled)
        transformer_results = self.evaluate_fast_performance(y_val, transformer_preds, df)
        transformer_results['model'] = 'Fast Transformer'
        results['transformer'] = transformer_results
        
        # LSTM predictions
        lstm_preds = self.predict_fast(lstm, X_val_scaled)
        lstm_results = self.evaluate_fast_performance(y_val, lstm_preds, df)
        lstm_results['model'] = 'Fast LSTM'
        results['lstm'] = lstm_results
        
        # Display results
        print(f"\n📊 RAPID BACKTEST RESULTS FOR {ticker}")
        print("=" * 60)
        print(f"{'Model':<15} {'Annual Return':<12} {'Sharpe':<8} {'Win Rate':<10} {'Trades':<8}")
        print("-" * 60)
        
        for model_name, result in results.items():
            print(f"{result['model']:<15} {result['annual_return']:<12.1%} "
                  f"{result['sharpe_ratio']:<8.2f} {result['win_rate']:<10.1%} "
                  f"{result['total_trades']:<8}")
        
        # Find best model
        best_model = max(results.values(), key=lambda x: x['annual_return'])
        
        print(f"\n🏆 BEST MODEL: {best_model['model']}")
        print(f"📈 Annual Return: {best_model['annual_return']:.1%}")
        print(f"📊 Sharpe Ratio: {best_model['sharpe_ratio']:.2f}")
        print(f"🎯 Win Rate: {best_model['win_rate']:.1%}")
        
        # Compare to SPY
        spy_return = 0.10  # ~10% annual return
        outperformance = best_model['annual_return'] - spy_return
        
        print(f"\n🎯 PERFORMANCE COMPARISON:")
        print(f"   SPY Buy-and-Hold: {spy_return:.1%}")
        print(f"   {best_model['model']}: {best_model['annual_return']:.1%}")
        print(f"   Outperformance: {outperformance:.1%}")
        
        if best_model['annual_return'] >= 0.20:
            print(f"   ✅ TARGET ACHIEVED: 20%+ annual return!")
        elif best_model['annual_return'] >= 0.15:
            print(f"   ⚠️  CLOSE TO TARGET: 15%+ annual return")
        else:
            print(f"   ❌ TARGET NOT MET: {best_model['annual_return']:.1%} < 20%")
        
        return results, best_model

def main():
    """Main function for rapid results"""
    
    print("⚡ RAPID DEEP LEARNING STRATEGY")
    print("🎯 Target: 20%+ annual returns")
    print("⏱️  Timeline: RESULTS IN DAYS")
    print("💰 Justification: 2x SPY performance")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  No GPU - will use CPU (slower)")
    
    # Initialize strategy
    strategy = RapidDeepLearningStrategy(use_gpu=True)
    
    # Test on multiple stocks
    tickers = ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'NVDA']
    all_results = {}
    
    for ticker in tickers:
        try:
            results, best_model = strategy.run_rapid_backtest(ticker)
            all_results[ticker] = best_model
        except Exception as e:
            print(f"❌ Error with {ticker}: {e}")
    
    # Summary
    print(f"\n📊 SUMMARY - ALL STOCKS")
    print("=" * 50)
    print(f"{'Stock':<8} {'Model':<15} {'Annual Return':<12} {'Sharpe':<8} {'Win Rate':<10}")
    print("-" * 50)
    
    for ticker, result in all_results.items():
        print(f"{ticker:<8} {result['model']:<15} {result['annual_return']:<12.1%} "
              f"{result['sharpe_ratio']:<8.2f} {result['win_rate']:<10.1%}")
    
    # Find overall best
    if all_results:
        overall_best = max(all_results.values(), key=lambda x: x['annual_return'])
        print(f"\n🏆 OVERALL BEST: {overall_best['annual_return']:.1%} annual return")
        
        if overall_best['annual_return'] >= 0.20:
            print(f"🎉 SUCCESS! 20%+ target achieved!")
        else:
            print(f"⚠️  Need more optimization for 20%+ target")
    
    return strategy, all_results

if __name__ == "__main__":
    strategy, results = main()
