# NeuralTrader - AI Quantitative Hedge Fund

## Overview
NeuralTrader is a machine-learning quantitative hedge fund system that uses AI ensemble models to generate trading signals and manage portfolios automatically.

## Key Features
- **AI Ensemble Models**: 91.7% precision with 68 features
- **Automated Trading**: Paper trading with TradingView integration
- **Risk Management**: Circuit breaker disabled for learning mode
- **Daily Schedule**: Automated fetch, trading, and reporting

## System Architecture
- **Core Engine**: `main_orchestrator_ist.py`
- **AI Models**: Phase 12 ensemble (XGBoost, LightGBM, RandomForest)
- **Data Pipeline**: Tiingo API integration with 2,184 tickers
- **Portfolio Management**: Real-time position tracking

## Daily Schedule (IST Timezone)

| Time            | Task         | Script / Mode                       |
|-----------------|--------------|-------------------------------------|
| 04:00 (Mon-Fri) | Data Sync    | `--mode=fetch`                      |
| 17:15 (Mon-Fri) | Execution    | `--mode=trade` or `--mode=paper`    |
| 18:00 (Mon-Fri) | Report       | `--mode=report`                     |
| Sat 04:00 AM    | Model Retrain| `--mode=saturday_retrain`           |

## Quick Start

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Daily Operations

#### Data Fetch (4:00 AM)
```bash
python main_orchestrator_ist.py --mode=fetch
```

#### Paper Trading (5:15 PM)
```bash
python main_orchestrator_ist.py --mode=paper
```

#### Report Generation (6:00 PM)
```bash
python main_orchestrator_ist.py --mode=report
```

#### Model Retraining (Saturday 4:00 AM)
```bash
python scripts/retrain_phase12.py
```

## Performance Metrics
- **Model Precision**: 91.7% (Phase 12)
- **Data Coverage**: 2,184 tickers
- **Training Data**: 14.7M rows
- **Features**: 68 total (64 technical + 4 momentum)

## Risk Management
- **Circuit Breaker**: Disabled for paper trading
- **Position Sizing**: Inverse volatility based
- **Stop Loss**: 10% Uncle Point
- **Portfolio Limit**: 20 concurrent positions

## Documentation
- [Architecture Details](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [Backtest Results](docs/BACKTEST_RESULTS.md)

## Support
- **Email**: lugassy.ai@gmail.com
- **Logs**: `logs/automation_YYYYMMDD_HHMMSS.log`
- **Reports**: `reports/dashboard_YYYYMMDD_HHMMSS.html`

## Version
- **Current**: v8.0 (Phase 12 Active)
- **Last Updated**: March 2026
