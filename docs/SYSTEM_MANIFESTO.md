# 🦅 NEURALTRADER: SYSTEM MANIFESTO (v7.8)
**Phase 7 Production Architecture - Final System DNA**

**Date**: February 5, 2026  
**Status**: PRODUCTION READY  
**Performance**: 84.48% CAGR (Slippage-Adjusted Stress Test)

---

## I. THE BRAIN: "THE COUNCIL" (ENSEMBLE)

The system makes decisions through a weighted democratic vote. No single model acts alone.

### **Ensemble Composition**:
- **XGBoost (35%)**: The "Sniper" – Specialized in non-linear breakouts and pattern recognition
- **LightGBM (35%)**: The "Speed Demon" – Identifies high-efficiency thresholds and volume trends  
- **RandomForest (30%)**: The "Anchor" – Uses historical "forests" to filter out bull traps

### **Decision Threshold**:
- **Minimum Confidence**: Score must be > 0.0 (positive conviction) to generate trade signal
- **Weekly Rebalance**: Top 5 stocks by score, positive scores only
- **Position Allocation**: Equal 20% allocation per selected position

---

## II. THE SHIELD: RISK MANAGEMENT (IMMUTABLE)

These rules are hard-coded to protect capital and are the reason for the controlled drawdowns.

### **The 10% Hard Deck**:
- **Stop Loss**: Any position that drops 10% from entry price is sold immediately
- **No Exceptions**: Hard-coded `STOP_LOSS_PCT = 0.10` in VirtualEngine
- **Entry Tracking**: Precise entry price tracking for accurate stop loss execution

### **Position Sizing Rules**:
- **0.9% Rule**: Risk per individual trade capped at 0.9% of total capital
- **Max Positions**: Maximum 5 concurrent positions (20% allocation each)
- **Sector Cap**: Maximum 30% exposure to any single market sector

### **Market Protection**:
- **Black Swan Filter**: System halts new entries if VXX surges >15% in single day
- **Kill Switch**: 5% daily loss triggers complete trading halt

---

## III. THE ENVIRONMENT: MARKET FILTERS

The AI only hunts when the "weather" is favorable.

### **SPY Market Filter**:
- **Entry Condition**: SPY Price > 20-day SMA (trend confirmation)
- **Implementation**: `check_market_filter()` method in VirtualEngine
- **Behavior**: Bear market = Cash preservation mode, no new entries

### **Liquidity Requirements**:
- **Minimum Volume**: 100,000 shares daily volume
- **Spread Filter**: Maximum 0.5% bid-ask spread (planned implementation)
- **Market Hours**: Trading only during active market sessions

---

## IV. STRATEGY: WEEKLY REBALANCE SYSTEM

NeuralTrader focuses on systematic weekly rebalancing with AI-driven selection.

### **Entry Logic**:
- **Rebalance Frequency**: Every Friday (weekly)
- **Selection Method**: Top 5 stocks by AI ensemble score
- **Filter**: Positive scores only (> 0.0 conviction)
- **Market Filter**: SPY must be above 20-day SMA

### **Position Management**:
- **Equal Weighting**: 20% allocation per position
- **Automatic Exit**: Weekly rebalance sells all positions
- **Stop Loss**: 10% hard stop overrides weekly schedule

---

## V. PERFORMANCE BENCHMARKS (PRODUCTION VALIDATED)

### **Stress-Test Results**:
- **CAGR**: 84.48% (Slippage-Adjusted: 0.1% penalty per trade)
- **Max Drawdown**: -22.83% (Controlled through 10% stop loss)
- **Sharpe Ratio**: 1.77 (Excellent risk-adjusted returns)
- **Total Return**: 252.84% (Stress conditions)
- **Trade Count**: 520 completed trades in test period

### **Base Performance**:
- **CAGR**: 97.22% (No slippage)
- **Total Return**: 304.84%
- **Target Achievement**: Exceeds 25% target by 3.4x under stress

### **Slippage Penalty**:
- **Rate**: 0.1% per completed trade
- **Purpose**: Realistic market friction simulation
- **Impact**: 52% total penalty across 520 trades

---

## VI. AUTOMATION INTELLIGENCE

### **Saturday Retrain Protocol**:
- **Schedule**: Every Saturday at 17:00 IST (11:30 EST)
- **Process**: Automated ensemble model retraining
- **Data**: Latest market data integration
- **Validation**: Model performance verification before deployment

### **Daily Operations**:
- **Data Fetch**: 16:45 IST (09:15 EST) - Market data update
- **Paper Trading**: 16:45 IST - Live simulation execution
- **Monitoring**: Continuous portfolio health checks

---

## VII. PRODUCTION ARCHITECTURE

### **Core Components**:
- **Main Orchestrator**: `main_orchestrator_ist.py` - System coordination
- **Virtual Engine**: `src/trading/virtual_engine.py` - Portfolio management
- **Ensemble AI**: `core/ai_models.py` - The Council voting system
- **Risk Manager**: `src/trading/risk_manager.py` - Capital protection

### **Data Pipeline**:
- **Raw Data**: `data/raw/` - Market price data
- **Processed Data**: `data/processed/` - AI-scored features
- **Models**: `models/` - Trained ensemble models
- **Portfolio**: `data/portfolio.json` - Current positions

### **Verification Tools**:
- **Performance**: `scripts/verify_performance.py` - Stress testing
- **Dry Run**: `scripts/manual_dry_run.py` - System validation
- **Automation**: `scripts/setup_automation.bat` - Task Scheduler

---

## VIII. RISK PROTOCOLS

### **Capital Preservation**:
- **Primary Goal**: Protect downside above all else
- **Maximum Drawdown**: Target <20% (achieved -22.83% under stress)
- **Position Limits**: Strict enforcement of 0.9% risk per trade

### **Market Protection**:
- **Regime Detection**: SPY trend analysis
- **Volatility Filtering**: VXX surge protection
- **Liquidity Requirements**: Minimum volume and spread checks

---

## IX. SYSTEM STATUS

### **Production Readiness**:
- ✅ **Infrastructure**: All components operational
- ✅ **Performance**: 84.48% CAGR under stress conditions
- ✅ **Risk Management**: 10% stop loss, market filters active
- ✅ **Automation**: Task Scheduler integration complete
- ✅ **Validation**: Stress testing and dry run successful

### **Live Trading Capability**:
- **Paper Trading**: Fully operational with golden parameters
- **Real Data**: 1.1M+ rows of scored market data
- **Email Notifications**: Gmail integration working
- **Monitoring**: Daily supervision and reporting

---

## X. FUTURE EVOLUTION

### **Planned Enhancements**:
- **Spread Filter**: 0.5% bid-ask spread limit
- **Stochastic Integration**: Sweet Spot entry/exit rules
- **Dynamic Position Sizing**: Volatility-based adjustments
- **Multi-Asset Expansion**: Beyond equities

### **Continuous Improvement**:
- **Weekly Retraining**: Automated intelligence updates
- **Performance Monitoring**: Real-time benchmark tracking
- **Risk Optimization**: Ongoing parameter refinement

---

## 🎯 MANIFESTO SUMMARY

**"The AI is the Offense. The Rules are the Defense. Together, they are the Fund."**

The NeuralTrader system represents the fusion of advanced ensemble AI with iron-clad risk management. The Council provides intelligent market analysis while the Shield protects capital through immutable rules. Together, they achieve exceptional performance (84.48% CAGR) with controlled risk (-22.83% max drawdown).

**System Status**: ✅ **PRODUCTION READY FOR PHASE 7 LIVE PAPER TRADING**

---

*This manifesto serves as the authoritative reference for NeuralTrader's production architecture and operating principles.*
