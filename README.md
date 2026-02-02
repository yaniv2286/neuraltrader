# NeuralTrader Private Fund

🏛️ **High-Edge ML-Driven Private Fund**  
Target: 25% ARR with <20% drawdowns using advanced ML and risk management.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Setup data (requires Tiingo API key)
python -m src.main --api-key YOUR_TIINGO_KEY --setup-only --max-tickers 100

# Run trading simulation
python -m src.main --api-key YOUR_TIINGO_KEY --dry-run

# Paper trading (future)
python -m src.main --api-key YOUR_TIINGO_KEY --paper-trading
```

## 🎯 Architecture

### Core Components
- **Data Engine**: 50 years of market data in Parquet format
- **Regime Classifier**: SPY/VIX market regime detection
- **Feature Engineer**: 100+ ML features per ticker
- **ML Models**: XGBoost, LightGBM, PyTorch ensembles
- **Slot Manager**: 10 slots, 10% capital each, sector limits
- **Execution Engine**: Limit orders, cost modeling, risk controls

### Key Features
- **Point-in-Time Universe**: Top 1000 liquidity-filtered tickers
- **Regime-Aware Trading**: BEAR/CAUTION/BULL market adaptation
- **Volatility-Adjusted Sizing**: ATR-based position sizing
- **Sector Diversification**: Max 3 slots per sector
- **Risk Management**: 5% daily loss limit, 20% max drawdown
- **Cost Modeling**: $1 commission + 0.1% spread + 0.05% slippage

## 📁 Project Structure

```
NeuralTrader/
├── data/
│   ├── raw/          # Parquet market data
│   └── processed/    # Features, regimes, universe
├── src/
│   ├── core/
│   │   ├── data_engine.py      # Data download & management
│   │   ├── regime_classifier.py # Market regime detection
│   │   └── slot_manager.py      # Position allocation
│   ├── ml/
│   │   └── features/
│   │       └── feature_engineer.py # ML feature generation
│   ├── execution/
│   │   └── execution_engine.py   # Order execution
│   └── main.py                 # Main entry point
├── config/
│   └── fund_config.yaml        # Fund configuration
├── requirements.txt
└── .windsurfrules             # Global constraints
```

## 🤖 Trading Strategy

### Signal Generation
1. **Feature Engineering**: 100+ features per ticker
   - Price momentum & reversal
   - Volume & money flow
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Market regime integration
   - Cross-sectional ranking

2. **ML Models**: Ensemble approach
   - XGBoost (gradient boosting)
   - LightGBM (leaf-wise boosting)
   - PyTorch (deep learning)
   - Regime-aware predictions

3. **Position Allocation**
   - 10 slots, 10% capital each
   - Sector limit: max 3 slots per sector
   - Volatility-adjusted sizing
   - ATR-based stop losses

### Risk Management
- **Capital Preservation**: Priority #1
- **Daily Loss Limit**: 5% of capital
- **Maximum Drawdown**: 20% hard stop
- **Position Sizing**: Volatility-adjusted
- **Sector Limits**: Diversification enforced
- **Stop Losses**: 2x ATR below entry

## 📊 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Annualized Return | 25% | TBD |
| Max Drawdown | <20% | TBD |
| Sharpe Ratio | >1.0 | TBD |
| Win Rate | >55% | TBD |
| Trades/Year | 100-200 | TBD |

## � Market Regimes

- **BULL**: SPY > SMA(200) & VIX < 1.5x mean
  - Full position allocation
  - Normal risk parameters

- **CAUTION**: SPY > SMA(200) & VIX > 1.5x mean
  - Reduced position size
  - Tighter stops

- **BEAR**: SPY < SMA(200)
  - Minimal exposure
  - Cash preservation

## 🔧 Configuration

Edit `config/fund_config.yaml`:

```yaml
fund:
  initial_capital: 100000
  target_arr: 0.25
  max_drawdown: 0.20

trading:
  max_positions: 10
  position_size: 0.10
  sector_limit: 3

risk:
  daily_loss_limit: 0.05
  max_position_risk: 0.02
```

## 📦 Dependencies

Core requirements:
- `pandas>=2.0.0` - Data processing
- `pyarrow>=15.0.0` - Parquet support
- `xgboost>=2.0.0` - Gradient boosting
- `torch>=2.0.0` - Deep learning
- `tiingo>=0.14.0` - Market data

See `requirements.txt` for complete list.

## 🎯 Usage Examples

### Data Setup
```python
from src.core.data_engine import DataEngine

engine = DataEngine(api_key="YOUR_KEY")
top_tickers = engine.download_universe(max_tickers=100)
```

### Feature Generation
```python
from src.ml.features.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.create_features("AAPL", regime_data)
```

### Trading Simulation
```python
from src.main import NeuralTraderFund

fund = NeuralTraderFund(api_key="YOUR_KEY", capital=100000)
fund.setup_data(max_tickers=100)
fund.run_daily_cycle()
```

## � Global Constraints

See `.windsurfrules` for mandatory rules:

- ✅ Always use Parquet for data storage
- ✅ Every trade must have ATR stop loss
- ✅ Prioritize capital preservation
- ✅ Include realistic transaction costs
- ✅ No leverage (1x only)
- ✅ Maximum 10 positions
- ✅ Sector diversification enforced

## 📊 Monitoring & Reporting

- **Real-time P&L**: Live portfolio tracking
- **Risk Alerts**: Immediate breach notifications
- **Daily Reports**: Performance summary
- **Monthly Reports**: Comprehensive analysis
- **Execution Log**: Complete trade audit trail

## 🚀 Deployment

### Development
```bash
# Dry run (no actual trades)
python -m src.main --api-key YOUR_KEY --dry-run
```

### Paper Trading
```bash
# Paper trading with real data
python -m src.main --api-key YOUR_KEY --paper-trading
```

### Live Trading (Future)
```bash
# Live trading with real capital
python -m src.main --api-key YOUR_KEY --live-trading
```

## 🎉 Status

✅ **Phase 1 Complete**: Infrastructure & Data Pipeline  
✅ **Phase 2 Complete**: ML Features & Models  
✅ **Phase 3 Complete**: Risk Management & Execution  
🚀 **Phase 4**: Paper Trading & Validation  
⏳ **Phase 5**: Live Trading (Future)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/neuraltrader/neuraltrader/issues)
- **Documentation**: [Wiki](https://github.com/neuraltrader/neuraltrader/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/neuraltrader/neuraltrader/discussions)

---

⚠️ **Disclaimer**: This is a sophisticated trading system. Past performance does not guarantee future results. Trade at your own risk.
