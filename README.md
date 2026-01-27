# NeuralTrader

Professional AI-powered trading system with neural network models and intelligent model selection.

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Run the system
python main.py
```

## 🎯 Features

- **Intelligent Model Selection**: Automatically chooses optimal CPU/GPU models
- **20 Years Tiingo Data**: Offline capability with cached market data
- **CPU Optimized**: XGBoost, Random Forest optimized for performance
- **GPU Ready**: Transformer, LSTM models for systems with strong GPU
- **Professional Structure**: Clean, maintainable codebase

## 📁 Project Structure

```
src/
├── data/          # Tiingo data processing
├── models/        # CPU/GPU intelligent model selection
├── strategies/    # Trading strategies
├── backtesting/   # Backtesting engine
└── trading/       # Live trading
```

## 🤖 Model Selection

The system automatically detects hardware and selects optimal models:

- **CPU Only**: XGBoost + Random Forest (optimized)
- **GPU Available**: Transformer + LSTM + CNN-LSTM

## 📊 Usage

```python
from models import create_optimal_model
from data.enhanced_preprocess import build_enhanced_model_input

# Auto-select best model for your hardware
model = create_optimal_model('stock_prediction')

# Load 20 years of enhanced data
data = build_enhanced_model_input('AAPL', validate_data=True)
```

## 📦 Dependencies

- No Yahoo Finance (uses cached Tiingo data)
- CPU-optimized ML models
- Optional GPU support for deep learning

## 🎉 Status

✅ Phase 1 Complete: Enhanced data pipeline  
✅ Phase 1.5 Complete: Intelligent model selection  
🚀 Ready for Phase 2: Basic models & validation
