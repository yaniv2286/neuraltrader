# 🏛️ NeuralTrader Project Plan

## Project Goal

Build a **reliable, CPU-first trading platform** that:
- Generates **25%+ annualized return (ARR)** after costs
- Uses **no leverage** (1× only)
- Prioritizes **capital preservation** and consistency
- Strongly outperforms passive investing (~10%) on a risk-adjusted basis

---

## Phase Overview

| Phase | Name | Status | Duration |
|-------|------|--------|----------|
| 1 | Infrastructure & Data Pipeline | ✅ COMPLETE | 2 weeks |
| 2 | Feature Engineering | ✅ COMPLETE | 1 week |
| 3 | CPU Model Development | ✅ COMPLETE | 2 weeks |
| 4 | Trading Constitution & Backtest Engine | ✅ COMPLETE | 1 week |
| 5 | Strategy Discovery & Optimization | 🔄 CURRENT | 3-4 weeks |
| 6 | Paper Trading & Validation | ⏳ PLANNED | 2-3 weeks |
| 7 | Sentiment Analysis Integration | ⏳ PLANNED | 2 weeks |
| 8 | NLP & News Processing | ⏳ PLANNED | 3 weeks |
| 9 | GPU Models & Deep Learning | ⏳ PLANNED | 4 weeks |
| 10 | Portfolio Optimization | ⏳ PLANNED | 2 weeks |
| 11 | Broker API Integration | ⏳ FUTURE | 2 weeks |
| 12 | Live Trading Deployment | ⏳ FUTURE | 2 weeks |
| 13 | Production Scaling | ⏳ FUTURE | Ongoing |

---

## Phase 1: Infrastructure & Data Pipeline ✅

**Objective:** Establish data foundation with 150+ tickers, incremental loading, no re-downloads.

### Deliverables
- DataStore with cached CSVs (154 tickers)
- Tiingo API integration
- Adjusted price handling
- Data validation (fail loudly on errors)

### Tests
| Test | Criteria | Status |
|------|----------|--------|
| Data loading | Load 150+ tickers without errors | ✅ PASS |
| Schema validation | All OHLCV columns present | ✅ PASS |
| Date range | Min 252 days per ticker | ✅ PASS |
| No re-download | Uses cached CSVs only | ✅ PASS |
| Price policy | Adjusted prices logged | ✅ PASS |

### Key Files
- `src/core/data_store.py`
- `src/data/tiingo_loader.py`
- `src/data/cache/tiingo/*.csv`

---

## Phase 2: Feature Engineering ✅

**Objective:** Create robust technical indicators with zero redundancy.

### Deliverables
- 25+ technical indicators
- Feature selector (remove redundant)
- Momentum, volatility, trend features
- Feature importance ranking

### Tests
| Test | Criteria | Status |
|------|----------|--------|
| Feature count | 25+ features generated | ✅ PASS |
| No NaN values | All features clean | ✅ PASS |
| No redundancy | Correlation < 0.95 | ✅ PASS |
| Stationarity | Log returns used | ✅ PASS |

### Key Files
- `src/features/indicators.py`
- `src/features/momentum.py`
- `src/features/volatility.py`
- `src/features/feature_selector.py`

---

## Phase 3: CPU Model Development ✅

**Objective:** Train XGBoost, RandomForest, LightGBM models with proper regularization.

### Deliverables
- XGBoost model (primary)
- RandomForest model (backup)
- LightGBM model (fast)
- Model selector
- Hyperparameter tuning

### Tests
| Test | Criteria | Status |
|------|----------|--------|
| No overfitting | Train-Test gap < 5% | ✅ PASS |
| Direction accuracy | > 50% on test set | ✅ PASS |
| Generalization | Works on unseen data | ✅ PASS |
| Speed | < 5 sec per ticker | ✅ PASS |

### Key Files
- `src/models/cpu_models/xgboost_model.py`
- `src/models/cpu_models/random_forest_model.py`
- `src/models/cpu_models/lightgbm_model.py`
- `src/models/model_trainer.py`

---

## Phase 4: Trading Constitution & Backtest Engine ✅

**Objective:** Implement strict trading rules, veto gates, cost model, and Excel output.

### Deliverables
- `trading_constitution.json` config
- VetoGates (9 gates)
- CostModel (mandatory)
- BacktestEngine (integrated)
- Excel output (6 sheets)

### Tests
| Test | Criteria | Status |
|------|----------|--------|
| Veto gates | Block invalid trades | ✅ PASS |
| Cost model | Applied to all trades | ✅ PASS |
| Excel schema | All 6 sheets present | ✅ PASS |
| Validation | PASS/FAIL correctly | ✅ PASS |
| Blacklist | Worst performers blocked | ✅ PASS |

### Key Files
- `config/trading_constitution.json`
- `src/core/backtest_engine.py`
- `src/core/veto_gates.py`
- `src/core/cost_model.py`
- `src/core/backtest_result.py`

### Excel Output Schema (6 Sheets)
1. **How_To_Read** - Run metadata, price policy, schema version
2. **Overall_Performance** - CAGR, max DD, profit factor, win rate, PASS/FAIL
3. **All_Trades** - Complete trade log with 22 columns
4. **Stock_Summary** - Per-ticker metrics
5. **Equity_Curve** - Date, equity, drawdown
6. **Config_Snapshot** - Full config used

---

## Phase 5: Strategy Discovery & Optimization 🔄

**Objective:** Develop profitable strategy achieving 25%+ ARR with < 20% max drawdown.

### Deliverables
- Signal generation with ML predictions
- Per-ticker optimization
- Regime detection (bull/bear/sideways)
- Position sizing optimization
- Walk-forward validation

### Tests
| Test | Criteria | Target |
|------|----------|--------|
| CAGR | Annualized return | > 25% |
| Max Drawdown | Worst peak-to-trough | < 20% |
| Win Rate | Winning trades % | > 50% |
| Profit Factor | Gross profit / loss | > 1.5 |
| Sharpe Ratio | Risk-adjusted return | > 1.0 |
| Stability | Profitable years | > 60% |

### Sub-tasks
1. Integrate XGBoost predictions into BacktestEngine
2. Optimize confidence thresholds per ticker
3. Implement regime-aware trading
4. Tune trailing stop parameters
5. Walk-forward validation (rolling windows)

---

## Phase 6: Paper Trading & Validation ⏳

**Objective:** Validate strategy in real-time without risking capital.

### Deliverables
- Paper trading engine
- Real-time data feed
- Daily performance tracking
- Comparison vs backtest
- Drift detection

### Tests
| Test | Criteria | Target |
|------|----------|--------|
| Execution match | Paper vs backtest | < 5% deviation |
| Signal timing | Real-time signals | < 1 min delay |
| No look-ahead | No future data leakage | Verified |
| 30-day validation | Consistent performance | Within 2σ of backtest |
| Slippage reality | Actual vs estimated | < 0.1% difference |

---

## Phase 7: Sentiment Analysis Integration ⏳

**Objective:** Add market sentiment signals to improve predictions.

### Deliverables
- Sentiment data sources integration
- Fear & Greed index tracking
- Social media sentiment (Reddit, Twitter)
- Options flow / Put-Call ratio
- Sentiment features for models

### Tests
| Test | Criteria | Target |
|------|----------|--------|
| Data freshness | Sentiment update frequency | < 1 hour |
| Signal correlation | With price movement | > 0.3 |
| Alpha contribution | Improvement vs baseline | > 1% |
| Coverage | Tickers with sentiment | > 80% |
| Latency | Processing time | < 5 min |

### Sentiment Sources
1. **VIX** - Volatility/Fear index
2. **Put/Call Ratio** - Options sentiment
3. **Reddit/WSB** - Retail sentiment
4. **Twitter/X** - Social buzz
5. **CNN Fear & Greed** - Market sentiment

---

## Phase 8: NLP & News Processing ⏳

**Objective:** Process news and earnings for trading signals.

### Deliverables
- News API integration (Finnhub, Alpha Vantage)
- NLP sentiment model (FinBERT)
- Event detection (earnings, FDA, M&A)
- News-based features
- Real-time news alerts

### Tests
| Test | Criteria | Target |
|------|----------|--------|
| News latency | Time to process | < 5 min |
| Sentiment accuracy | vs human labels | > 80% |
| Event detection | Earnings/FDA accuracy | > 90% |
| Signal value | Adds alpha | > 1% improvement |
| Coverage | Tickers with news | > 90% |

### NLP Models
1. **FinBERT** - Financial sentiment
2. **GPT-based** - News summarization
3. **Named Entity Recognition** - Company/ticker extraction

---

## Phase 9: GPU Models & Deep Learning ⏳

**Objective:** Add GPU-accelerated models for improved predictions.

### Deliverables
- LSTM model for sequence prediction
- Transformer model for attention
- GPU training pipeline
- Model ensemble (CPU + GPU)
- A/B testing framework

### Tests
| Test | Criteria | Target |
|------|----------|--------|
| GPU detection | CUDA available | ✅ Detected |
| Training speed | vs CPU baseline | > 10x faster |
| Accuracy improvement | vs CPU models | > 2% lift |
| Memory usage | GPU memory | < 8GB |
| Inference speed | Prediction time | < 100ms |

### Models to Implement
1. **LSTM** - Sequential patterns
2. **GRU** - Faster alternative
3. **Transformer** - Attention mechanism
4. **CNN-LSTM** - Hybrid approach

---

## Phase 10: Portfolio Optimization ⏳

**Objective:** Optimize portfolio allocation across tickers.

### Deliverables
- Mean-variance optimization
- Risk parity allocation
- Sector/correlation limits
- Dynamic rebalancing
- Tax-loss harvesting

### Tests
| Test | Criteria | Target |
|------|----------|--------|
| Diversification | Max single position | < 10% |
| Sector limits | Max sector exposure | < 30% |
| Correlation | Max pairwise | < 0.7 |
| Rebalance frequency | Optimal period | Weekly/Monthly |
| Sharpe improvement | vs equal weight | > 0.2 |

---

## Phase 11: Broker API Integration ⏳

**Objective:** Connect to live broker for order execution.

### Deliverables
- Broker API wrapper (Interactive Brokers / Alpaca)
- Order management system
- Position tracking
- Risk limits enforcement
- Error handling & recovery

### Tests
| Test | Criteria | Target |
|------|----------|--------|
| Connection | API authentication | ✅ Connected |
| Order placement | Market/Limit orders | Execute correctly |
| Position sync | Match broker state | 100% accurate |
| Risk limits | Block over-sized orders | Enforced |
| Failover | Handle disconnections | Auto-reconnect |

### Broker Options
1. **Alpaca** - Free API, US stocks, paper trading
2. **Interactive Brokers** - Professional, global markets
3. **TD Ameritrade** - US stocks, options

---

## Phase 12: Live Trading Deployment ⏳

**Objective:** Deploy system for live trading with real capital.

### Deliverables
- Production deployment
- Monitoring dashboard
- Alert system
- Daily reports
- Emergency shutdown

### Tests
| Test | Criteria | Target |
|------|----------|--------|
| Capital protection | Max daily loss | < 5% |
| Order execution | Fill rate | > 95% |
| Uptime | System availability | > 99.5% |
| Latency | Signal to order | < 5 sec |
| Audit trail | All trades logged | 100% |

### Safety Checklist
- [ ] Start with 10% of intended capital
- [ ] Monitor for 30 days before scaling
- [ ] Human review on any degradation
- [ ] Kill switch accessible
- [ ] Daily P&L alerts

---

## Phase 13: Production Scaling ⏳

**Objective:** Scale system for reliability and performance.

### Deliverables
- Cloud deployment (AWS/GCP)
- Database for trade history
- Automated monitoring
- CI/CD pipeline
- Documentation

### Tests
| Test | Criteria | Target |
|------|----------|--------|
| Scalability | Handle 500+ tickers | ✅ |
| Reliability | Monthly uptime | > 99.9% |
| Recovery | Disaster recovery | < 1 hour |
| Backup | Data backup | Daily |
| Audit | Compliance ready | ✅ |

---

## Success Metrics (Overall Project)

| Metric | Target | Priority |
|--------|--------|----------|
| **Annualized Return** | > 25% | HIGH |
| **Max Drawdown** | < 20% | HIGH |
| **Sharpe Ratio** | > 1.0 | MEDIUM |
| **Win Rate** | > 50% | MEDIUM |
| **Profit Factor** | > 1.5 | MEDIUM |
| **Uptime** | > 99.5% | HIGH |
| **Capital Preservation** | No catastrophic loss | CRITICAL |

---

## Trading Constitution (Core Rules)

### Hard Veto Rules (Absolute)
A trade is **INVALID** if ANY apply:
- Low confidence (< 45%)
- Poor risk/reward (< 1.5)
- Low liquidity (< 100K volume)
- High spread (> 0.5%)
- Blacklisted ticker

### Exit Rules (ONLY allowed)
- Trailing stop
- Signal invalidation
- Manual override

❌ No fixed take-profit
❌ No time-based exits

### Risk Limits
- Max 2% risk per trade
- Max 20% portfolio drawdown
- Max 5% daily loss
- Max 20 positions

---

## Current Status

```
Phase 1-4:  ✅ COMPLETE (Infrastructure ready)
Phase 5:    🔄 IN PROGRESS (Strategy optimization)
Phase 6-13: ⏳ PLANNED (Future development)
```

---

## Next Steps

1. **Immediate:** Integrate XGBoost predictions into BacktestEngine
2. **Short-term:** Optimize for 25%+ ARR target
3. **Medium-term:** Begin paper trading validation
4. **Long-term:** Broker integration and live trading

---

*Last Updated: January 30, 2026*
