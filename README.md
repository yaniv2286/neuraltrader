# NeuralTrader - AI Quantitative Hedge Fund

## Overview
NeuralTrader is a machine-learning quantitative hedge fund system that uses AI ensemble models to generate trading signals and manage portfolios automatically.

**Current Version:** Phase 13 Optimized (v13.0) - March 19, 2026 (OPERATIONAL)

## Key Features
- **AI Ensemble Models**: 97.9% precision (target) with 76 features (Phase 13 Optimized)
- **Precision-Optimized Weights**: XGBoost 0.356, LightGBM 0.366, HGB 0.278
- **Regime-Adaptive Thresholds**: CRISIS 0.80, BEAR 0.72, BULL 0.65
- **Validated Performance**: Comprehensive AI validation completed March 14, 2026
- **OPERATIONAL STATUS**: Phase 13 fully deployed March 19, 2026 - Dynamic signals active
- **Automated Trading**: Paper trading with TradingView integration
- **Risk Management**: Brain-Gate protection with automatic rollback

## System Architecture
- **Core Engine**: `main_orchestrator_ist.py`
- **AI Models**: Phase 13 ensemble (XGBoost, LightGBM, HGB) with precision-optimized weights
- **Features**: 76 total (68 Phase 12 + 8 Phase 13 derivatives)
- **Data Pipeline**: Tiingo API integration with 2,184 tickers
- **Portfolio Management**: Real-time position tracking with regime-adaptive thresholds

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

## Performance Metrics
- **Model Precision**: 91.4% → 97.9% (Phase 12 → Phase 13 target)
- **Data Coverage**: 2,184 tickers
- **Training Data**: 14.9M rows
- **Features**: 76 total (68 Phase 12 + 8 Phase 13 derivatives)
  - Volatility derivatives: vol_regime_change, atr_percentile, vol_acceleration, vol_atr_ratio
  - Momentum derivatives: high_52w_momentum, breakout_strength
  - Volume derivatives: volume_trend, volume_volatility
- **Backtest CAGR**: 15.92% (2000-2026, Phase 12 baseline)
- **Expected Improvement**: +5% accuracy, +6.5% precision, -5-10% false positives

## Risk Management
- **Brain-Gate Protection**: Precision @ 0.65 must be >= 55%
- **Automatic Rollback**: Phase 12 models backed up to models/backup_phase12/
- **Position Sizing**: Inverse volatility based (1/σ weighting)
- **Stop Loss**: ATR-based (2.5x ATR20, clamped 8-20%)
- **Uncle Point**: 20% drawdown circuit breaker
- **Portfolio Limit**: 15 concurrent positions
- **Regime-Adaptive**: Different thresholds for CRISIS/BEAR/BULL markets

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

## Phase 13 Optimizations (March 15, 2026)

### What's New
1. **8 Derivative Features** - Targeting top performers (volatility, momentum, volume)
2. **Precision-Optimized Weights** - 0.356, 0.366, 0.278 (was 0.4, 0.4, 0.2)
3. **Regime-Adaptive Thresholds** - CRISIS: 0.80, BEAR: 0.72, BULL: 0.65

### Validation Evidence
- Sentiment tested and rejected: degraded performance by 14.25%
- Top features identified: vol_regime (286.41), atr_14 (250.41), high_52w_prox (241.21)
- Derivatives capture transitions and accelerations, not just levels
- Precision-based weighting outperforms arbitrary weights

### Safety Features
- Brain-Gate validation prevents bad deployments
- Phase 12 models backed up for rollback
- Incremental changes: 8 features at a time

## Version
- **Current**: v13.0 (Phase 13 Optimized)
- **Last Updated**: March 15, 2026
- **Status**: Ready for Monday market open (March 17, 2026)
