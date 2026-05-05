# NeuralTrader - AI Quantitative Hedge Fund

## Overview
NeuralTrader is a machine-learning quantitative hedge fund system that uses AI ensemble models to generate trading signals and manage portfolios automatically.

**Current Version:** Phase 16 (v16.0) - May 4, 2026

## Key Features
- **AI Ensemble Models**: 98.2% LONG precision with 64 clean features (Phase 15 TB 3-Class)
- **Tri-Model Ensemble**: XGBoost 0.40 + LightGBM 0.40 + HGB 0.20
- **Phase 16 Optimized Strategy**: +15.88% CAGR (v11) / +11.22% CAGR -13.93% DD (v12)
- **High-Turnover AI**: 100 tickers, weekly rebalancing, 5-day timeout, trailing stop exits
- **Automated Trading**: Paper trading with daily signal generation
- **Risk Management**: Brain-Gate protection with automatic rollback

## System Architecture
- **Core Engine**: `main_orchestrator_ist.py`
- **AI Models**: Phase 15 ensemble (XGBoost, LightGBM, HGB) with TB 3-class labels
- **Features**: 64 clean derived indicators (raw OHLCV excluded for stationarity)
- **Data Pipeline**: Tiingo API integration with 2,184 tickers
- **Portfolio Management**: Daily exits + weekly entries, max 18-20 long positions

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
python scripts/retrain_phase13_optimized.py
```

## Performance Metrics (Phase 16 Backtest: 2020-2025)

| Config | CAGR | Max DD | Trades | Win Rate | Final Equity |
|--------|------|--------|--------|----------|--------------|
| **v11 (Aggressive)** | **+15.88%** | -16.43% | 4,929 | 54.2% | $242,032 |
| **v12 (Conservative)** | +11.22% | **-13.93%** | 4,139 | 53.8% | $189,278 |

- **Model Precision**: 98.2% LONG prec@0.65
- **Data Coverage**: 2,184 tickers (100 used in backtest)
- **Training Data**: 14.7M rows, 64 clean features
- **Strategy**: High-turnover (5-day holds, ~985 trades/year)

## Risk Management
- **Brain-Gate Protection**: Precision @ 0.65 must be >= 55%
- **Stop Loss**: 4% fixed (cut losers fast)
- **Trailing Stop**: Activate at +2%, trail 1.2% below peak
- **Timeout**: 5-day maximum hold (fast churn)
- **Uncle Point**: DISABLED (was #1 performance killer in backtest)
- **Shorts**: DISABLED (49.8% WR = net negative)
- **Portfolio Limit**: 18-20 concurrent long positions
- **Regime-Adaptive**: AI 3-class classifier (CRISIS/BEAR/BULL)

## Documentation
- [Architecture Details](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [AI Validation Report](reports/validation/VALIDATION_SUMMARY.md)
- [Phase 13 Deployment](reports/validation/PHASE13_DEPLOYMENT_READY.md)
- [Optimization Recommendations](reports/validation/OPTIMIZATION_RECOMMENDATIONS.md)

## Support
- **Email**: lugassy.ai@gmail.com
- **Logs**: `logs/automation_YYYYMMDD_HHMMSS.log`
- **Reports**: `reports/dashboard_YYYYMMDD_HHMMSS.html`

## Phase 16 Optimization (May 4, 2026)

### Key Findings (12 iterations)
1. **Uncle Point DISABLED** - Was #1 performance killer (forced liquidation destroyed recovery)
2. **Shorts DISABLED** - 49.8% WR = net negative P&L
3. **High Turnover Strategy** - 5-day timeout + weekly rebalancing compounds thin edge faster
4. **Trailing Stop Primary Exit** - TP effectively disabled; trail locks in gains at +2%
5. **100 Tickers Optimal** - 50 too few (poor selection), 200 too slow (>1hr runtime)
6. **Lower Threshold = More CAGR** - 0.35-0.40 with top-N ranking outperforms 0.45+

### Safety Features
- Brain-Gate validation prevents bad model deployments
- 4% stop loss cuts losers fast
- Trailing stop locks in gains after +2%
- Confidence-weighted position sizing

## Version
- **Current**: v16.0 (Phase 16 Optimized)
- **Last Updated**: May 4, 2026
- **Status**: Backtest optimization complete. Paper trading validation pending.
