# NEURALTRADER: INSTITUTIONAL ARCHITECTURE (v16.0)
**Status:** Phase 13 Optimized ACTIVE ✅ | RAM-Cache Engine | 193s Full Scan
**Last Updated:** May 6, 2026
**Version:** v13.0_optimized — Phase 13 (76-Feature Ensemble, RAM-Injected Engine, 300-Bar Truncation)

---

## 1. VALIDATED PERFORMANCE METRICS

### Phase 16 Backtest Results (2020-2025, 6 years) — VALIDATED
Backtest period: 2020-01-01 to 2025-12-31 | Universe: 100 liquid large-caps | Capital: $100,000

#### v11 (Aggressive - Best CAGR)
| Metric              | Phase 15 (old)| **Phase 16 v11**       |
|---------------------|---------------|------------------------|
| CAGR                | +2.24%        | **+15.88%** ✅         |
| Total Return        | +14.22%       | **+142.03%**           |
| Final Value         | $114,216      | **$242,032**           |
| Max Drawdown        | -7.35%        | **-16.43%**            |
| Sharpe Ratio        | 1.56          | **1.00**               |
| Profit Factor       | 1.22          | **1.27**               |
| Win Rate            | 56.5%         | **54.2%**              |
| Total Trades        | 840           | **4,929**              |
| Avg Hold Days       | 33.7          | **6.1 days**           |
| Exit: Trailing Stop | N/A           | **737 (15.0%)**        |
| Exit: Stop Loss     | 196           | **637 (12.9%)**        |
| Exit: Timeout       | 367           | **3,533 (71.7%)**      |

#### v12 (Conservative - Best Risk-Adjusted)
| Metric              | Phase 15 (old)| **Phase 16 v12**       |
|---------------------|---------------|------------------------|
| CAGR                | +2.24%        | **+11.22%**            |
| Total Return        | +14.22%       | **+89.28%**            |
| Final Value         | $114,216      | **$189,278**           |
| Max Drawdown        | -7.35%        | **-13.93%** ✅         |
| Sharpe Ratio        | 1.56          | **0.85**               |
| Profit Factor       | 1.22          | **1.24**               |
| Win Rate            | 56.5%         | **53.8%**              |
| Total Trades        | 840           | **4,139**              |
| Avg Hold Days       | 33.7          | **6.1 days**           |

### Phase 16 Optimization Journey (12 iterations)
| Version | CAGR    | Max DD   | Trades | Key Change                          |
|---------|---------|----------|--------|-------------------------------------|
| v1-v7   | -3.6% to +3.2% | -7% to -21% | 108-742 | 50 tickers, various params  |
| v8      | +6.43%  | -13.95%  | 1,397  | 100 tickers + Uncle Point disabled  |
| v10     | +9.51%  | -13.61%  | 3,488  | 15 pos, 7d timeout, 0.40 threshold  |
| **v11** | **+15.88%** | -16.43% | 4,929 | 20 pos, 5d timeout, 0.35 threshold |
| **v12** | +11.22% | **-13.93%** | 4,139 | 18 pos, 5d timeout, 0.38 threshold |

### Key Findings in Phase 16
- **Uncle Point DISABLED:** Was #1 performance killer across all versions (forced liquidation destroyed recovery)
- **Shorts DISABLED:** 49.8% WR on shorts = net negative; long-only is superior
- **High Turnover Strategy:** 5-day timeout + weekly rebalancing compounds the thin per-trade edge faster
- **Trailing Stop as Primary Exit:** TP effectively disabled (20%); trailing stop locks in gains at +2%
- **100 Tickers Optimal:** 50 too few (poor selection), 200 too slow (>1hr runtime)
- **Lower Threshold = More Trades = More CAGR:** 0.35-0.40 threshold with top-N ranking works better than 0.45+

### Model Precision (training: 14.7M rows, 76 features, TB 3-class)
| Model       | Accuracy | LONG Prec@0.65 | SHORT Prec@0.65 | Weight (Phase 13) |
|-------------|----------|----------------|-----------------|--------------------|
| XGBoost     | 58.6%    | **98.2%**      | 53.4%           | **0.356**          |
| LightGBM    | 58.3%    | **98.1%**      | 53.2%           | **0.366**          |
| HGB         | 70.2%    | **90.0%**      | 54.6%           | **0.278**          |

**Training (May 1, 2026):**
- **Training Data:** 14,722,185 rows across 2,184 tickers
- **Features:** 64 clean derived indicators (no raw OHLCV)
- **Labels:** Triple-Barrier 3-class (LONG_WIN 83%, NEUTRAL 3.7%, SHORT_WIN 13.3%)
- **TB Params:** TP=5%, SL=1.5x ATR (3-8%), Timeout=10 days
- **Brain-Gate:** PASSED ✅
- **Training Time:** 59 minutes

**Model:** Tri-Model Ensemble (XGBoost 0.356 + LightGBM 0.366 + HGB 0.278)
**Features:** 76 clean derived indicators (64 base + 8 Phase 13 derivatives + 4 momentum)
  - Technical: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, VWAP
  - Momentum: rs_vs_spy, high_52w_prox, volume_breakout, roc_63, roc_5/10/20
  - Volatility: rolling_volatility, vol_regime, atr_ratio, vol_acceleration
  - Volume: volume_ratio, volume_log, volume_trend, volume_volatility
**Decision Logic:** 3-Class TB Prediction with confidence ranking
  - Output: BUY (P_LONG > threshold) or HOLD
  - Shorts DISABLED (49.8% WR = net negative)
  - Base model threshold: 0.30 (generates candidates)
  - Backtest entry threshold: 0.35-0.40 (top-N ranked by confidence)
**Signal Selection:** AI ranks ALL BUY signals by confidence, top 18-20 selected for execution.
**Portfolio Management:** Daily exits + weekly entries, max 18-20 positions (long only).
**Regime Gate:** AI 3-class classifier (21 features: 20 VXX+SPY + 1 CNN Fear & Greed)
  - Regime Distribution: CRISIS 4.9% | BEAR 16.2% | BULL 78.9%
  - Fallback: SPY > SMA100 = BULL
**Achieved Impact:** 54.2% win rate, +15.88% CAGR (v11), -13.93% DD (v12), 4,929 trades (v11)

### 🚀 PURE AI DECISION ENGINE - REVOLUTIONARY BREAKTHROUGH

**The Problem Solved:** Traditional quant systems use hardcoded confidence thresholds (0.44, 0.60, etc.) that interfere with AI intelligence.

**The Pure AI Solution:** 
- **No human thresholds** - AI decides purely on probability comparison
- **prob_up > prob_down = BUY** (no minimum confidence required)
- **prob_down > prob_up = SELL** (no minimum confidence required)
- **AI ranks all 2,184 tickers** by confidence daily
- **Top 20 signals selected** for portfolio execution

**Results:** This breakthrough unlocked the AI's true potential, generating 227 signals daily vs 0 signals with human interference.

### v16 Risk Parameters (Phase 16 Optimized)
| Parameter           | v15 Value | **v16 Value** | Notes                                |
|---------------------|-----------|---------------|--------------------------------------|
| STOP_LOSS_PCT       | ATR 1.5x  | **4% fixed**  | Cut losers fast                      |
| TAKE_PROFIT_PCT     | 5%        | **20%** (disabled) | Let trailing stop handle exits  |
| TRAIL_ACTIVATE_PCT  | N/A       | **2%**        | Activate trailing after +2% gain     |
| TRAIL_PCT           | N/A       | **1.2%**      | Trail 1.2% below peak               |
| TIMEOUT_DAYS        | 10        | **5 days**    | Fast churn of non-performers         |
| MAX_POSITIONS       | 15        | **18-20**     | Long only (shorts disabled)          |
| BASE_POSITION_PCT   | N/A       | **5%**        | Per position allocation              |
| LONG_THRESHOLD      | 0.35      | **0.35-0.38** | Lower = more trades = more CAGR      |
| UNCLE_POINT_DD      | 10%       | **DISABLED**  | Was #1 performance killer            |
| COOLDOWN_DAYS       | 8         | **0**         | DISABLED                             |
| REBALANCE_FREQ      | Monthly   | **Weekly (Mon)** | More entry opportunities          |
| SHORTS_ENABLED      | Yes       | **No**        | 49.8% WR = net negative              |

---

## 2. ONE-DAY PIPELINE WALKTHROUGH

### Example: Tuesday March 4, 2025 — Market Close

```
DATE: 2025-03-04 (Tuesday)
TIME: 16:05 IST  ->  DATA SYNC
TIME: 17:15 IST  ->  EXECUTION CYCLE
TIME: 18:00 IST  ->  REPORT
```

```
==========================================================================
  NEURALTRADER — FULL 1-DAY PIPELINE (2025-03-04)
==========================================================================

 STAGE 0: PRE-FLIGHT VALIDATION
 ┌─────────────────────────────────────────────────────────────┐
 │  scripts/pre_flight_check.py                                │
 │  [1] IBKR connectivity check  (port 7497)                   │
 │  [2] Email system test                                      │
 │  [3] Model integrity test  (64 features -> probability)     │
 │  [4] FRED + News API pulse                                  │
 │  Result: PASS -> proceed  |  FAIL -> sys.exit(1) + alert    │
 └──────────────────────────┬──────────────────────────────────┘
                            |
                            v
 STAGE 1: DATA SYNC  (04:00 IST)
 ┌─────────────────────────────────────────────────────────────┐
 │  scripts/data_manager.py   --mode fetch                     │
 │                                                             │
 │  Tiingo API  ──>  data/raw/AAPL.parquet                     │
 │  Tiingo API  ──>  data/raw/MSFT.parquet                     │
 │  Tiingo API  ──>  data/raw/...  (2,183 tickers)             │
 │                                                             │
 │  [RULE 3.1] If any parquet older than 24h -> [FATAL] crash  │
 │  Columns stored: date, open, high, low, close, volume,      │
 │                  adjClose, adjHigh, adjLow, adjOpen         │
 └──────────────────────────┬──────────────────────────────────┘
                            |
                            v
 STAGE 2: FEATURE ENGINEERING  (per ticker, vectorized)
 ┌─────────────────────────────────────────────────────────────┐
 │  core/feature_engineer.py  (FeatureEngineer)                │
 │  core/indicators.py                                         │
 │                                                             │
 │  Input : AAPL.parquet  (last 300 rows — truncated for perf) │
 │  Output: 76-feature row vector (one row per ticker/day)     │
 │                                                             │
 │  64 technical features:                                     │
 │    Price_SMA_Ratio   (close / SMA50)   <- #1 predictor      │
 │    RSI_14            (momentum)                             │
 │    ATR_14            (volatility)                           │
 │    MACD / Signal     (trend)                                │
 │    Bollinger_%B      (mean reversion)                       │
 │    Volume_Ratio      (vol / 20d avg)                        │
 │    ... 58 more features                                     │
 │                                                             │
 │  4 Phase 12 momentum features (NEW):                        │
 │    rs_vs_spy       (ticker 20d ret - SPY 20d ret)           │
 │    high_52w_prox   (close / 52-week high)                   │
 │    volume_breakout (volume / 50d avg volume)                │
 │    roc_63          (63-day rate of change)                  │
 │                                                             │
 │  [RULE] If ticker has < 60 rows  -> drop + [WARNING]        │
 │  [RULE] If ATR fails             -> drop + [WARNING]        │
 └──────────────────────────┬──────────────────────────────────┘
                            |
                            v
 STAGE 3: AI REGIME CHECK  (market condition gate)
 ┌─────────────────────────────────────────────────────────────┐
 │  scripts/run_full_backtest.py  (_get_regime)                │
 │  models/regime_classifier.pkl  (GradientBoosting 3-class)   │
 │  models/regime_scaler.pkl                                   │
 │                                                             │
 │  Features: SPY + VXX (20 regime features)                   │
 │    Output: 0=CRISIS -> block all entries                    │
 │            1=BEAR   -> allow at threshold 0.45              │
 │            2=BULL   -> allow at threshold 0.35              │
 │                                                             │
 │  Validation accuracy: 100% on 2022+ holdout                 │
 │  Crisis days (VXX BB trigger): 4.8% of trading days         │
 │  Fallback if no model: SPY > SMA100 = BULL                  │
 └──────────────────────────┬──────────────────────────────────┘
                            |
                            v
 STAGE 4: AI ENSEMBLE SCORING  (RAM-Injected Council Decision)
 ┌─────────────────────────────────────────────────────────────┐
 │  core/ai_models.py  (EnsemblePredictor — RAM-CACHED)        │
 │                                                             │
 │  Models loaded ONCE at startup into self.ai_model:          │
 │    xgboost_model.pkl    weight = 0.356 (200 trees, hist)    │
 │    lightgbm_model.pkl   weight = 0.366 (200 trees)          │
 │    rf_model.pkl         weight = 0.278 (HGB, 100 iter)      │
 │                                                             │
 │  Scaler: feature_scaler.pkl  (StandardScaler)               │
 │  Strict Integrity: NO try/except — [FATAL] crash on fail   │
 │                                                             │
 │  For each ticker (RAM-cached, ~80ms/call):                  │
 │    X_scaled = scaler.transform(X_76features.float32)        │
 │    p_xgb  = xgb.predict_proba(X_scaled)[:, 1]  = 0.47      │
 │    p_lgb  = lgb.predict_proba(X_scaled)[:, 1]  = 0.45      │
 │    p_hgb  = hgb.predict_proba(X_scaled)[:, 1]  = 0.44      │
 │                                                             │
 │    confidence = (0.47*0.356 + 0.45*0.366 + 0.44*0.278)     │
 │               = 0.454  <- COUNCIL VERDICT: BUY SIGNAL       │
 │                                                             │
 │  BULL regime:   confidence >= 0.65  -> CANDIDATE            │
 │  BEAR regime:   confidence >= 0.72  -> CANDIDATE            │
 │  CRISIS regime: confidence >= 0.80  (or ALL blocked)        │
 │  CONTRARIAN:    CNN < 20 + conf > 0.60 -> force BUY         │
 └──────────────────────────┬──────────────────────────────────┘
                            |
                            v
 STAGE 5: LIQUIDITY FILTERS  (entry guard)
 ┌─────────────────────────────────────────────────────────────┐
 │  scripts/run_full_backtest.py  /  scripts/run_simulation.py │
 │                                                             │
 │  [FILTER 1] Price gate                                      │
 │    AAPL close = $213.50  >= MIN_PRICE ($10.00) -> PASS      │
 │                                                             │
 │  [FILTER 2] Liquidity gate                                  │
 │    AAPL 20d avg volume = 58.4M  >= 500,000     -> PASS      │
 │                                                             │
 │  [FILTER 3] Duplication gate  (Rule 2.3)                    │
 │    AAPL already in positions?  NO              -> PASS      │
 └──────────────────────────┬──────────────────────────────────┘
                            |
                            v
 STAGE 6: RISK MACHINE  (position sizing + circuit breakers)
 ┌─────────────────────────────────────────────────────────────┐
 │  core/strategy.py  +  scripts/sector_rotation.py            │
 │                                                             │
 │  [RULE 2.1] Uncle Point check                               │
 │    Portfolio value = $127,400 | Peak = $129,200             │
 │    Drawdown = (129,200 - 127,400) / 129,200 = 1.4%         │
 │    1.4% < 20%  -> NO COOLDOWN  -> proceed                   │
 │                                                             │
 │  [RULE 2.2] Inverse Volatility Sizing (v5)                  │
 │    AAPL 20d returns std = 0.0116  (annualized: 18.46%)      │
 │    Dollar risk  = portfolio * 2%  = $2,548                  │
 │    Position $   = dollar_risk / annualized_vol              │
 │                 = $2,548 / 0.1846 = $13,800                 │
 │    Floor check  = max($13,800, 5% * $127,400) = $13,800     │
 │    Shares       = $13,800 / $213.50 = 64 shares             │
 │                                                             │
 │  [RULE 2.4] Sector Authority check                          │
 │    AAPL -> Technology sector                                │
 │    Tech momentum rank = #2 of 11 sectors  -> PASS           │
 │    (Bottom 3 sectors get 15% confidence tax)                │
 │                                                             │
 │  [RULE 2.4] Global Volatility Gate                          │
 │    VXX vs Bollinger Upper Band  -> BELOW  -> entries OK     │
 └──────────────────────────┬──────────────────────────────────┘
                            |
                            v
 STAGE 7: ORDER EXECUTION
 ┌─────────────────────────────────────────────────────────────┐
 │                                                             │
 │  BACKTEST / SIMULATION mode:                                │
 │    scripts/run_full_backtest.py  (UnifiedBacktest)          │
 │    [BUY] AAPL x32 @ $213.50  conf=0.677                     │
 │    Cash -= $6,832 | Position added to portfolio dict        │
 │                                                             │
 │  PAPER / LIVE mode (IBKR):                                  │
 │    core/ibkr_engine.py  (IbkrEngine)                        │
 │    Market order -> TWS port 7497                            │
 │    Confirmation received -> portfolio.json updated           │
 └──────────────────────────┬──────────────────────────────────┘
                            |
                            v
 STAGE 8: EXIT MONITORING  (runs every day for open positions)
 ┌─────────────────────────────────────────────────────────────┐
 │  For each open position (e.g. MSFT bought 5 days ago):      │
 │                                                             │
 │  [EXIT 1] ATR Stop-Loss     1.5x ATR20, clamped 3%-8%       │
 │  [EXIT 2] Take-Profit       close rises 5% from entry       │
 │  [EXIT 3] Timeout           held 10 days -> force close     │
 │  [EXIT 4] Uncle Point       portfolio DD > 10% -> liquidate │
 │                                                             │
 │  Exit Priority:  Uncle Point > ATR Stop > Trail Stop        │
 │                  > Take-Profit > Timeout                    │
 └──────────────────────────┬──────────────────────────────────┘
                            |
                            v
 STAGE 9: REPORTING  (18:00 IST)
 ┌─────────────────────────────────────────────────────────────┐
 │  scripts/report_generator.py  (HTMLDashboardGenerator)      │
 │  scripts/send_html_report.py                                │
 │  core/utils/notifier.py                                     │
 │                                                             │
 │  Output:  reports/backtest_metrics.json   (metrics)         │
 │           reports/equity_curve.csv        (daily P&L)       │
 │           reports/trade_log.csv           (all trades)      │
 │           Email -> Architect              (HTML dashboard)  │
 └─────────────────────────────────────────────────────────────┘

==========================================================================
```

---

## 3. IMMUTABLE RISK LAWS (ARCHITECTURE.md enforces these — no overrides)

### Rule 2.1 — Uncle Point (Circuit Breaker)
- Portfolio drawdown > **20%** from peak triggers immediate liquidation of ALL positions
- **10-day cooldown** — zero new entries allowed
- Peak value resets after liquidation to prevent re-triggering during cooldown

### Rule 2.2 — Inverse Volatility Sizing (Phase 12 v5)
```
dollar_risk      = portfolio_value * 0.02         (2% risk per trade)
annualized_vol   = std(20d_returns) * sqrt(252)
position_dollars = dollar_risk / annualized_vol * conf_mult
position_dollars = clamp(position_dollars, 5% capital, 10% capital)
shares           = floor(position_dollars / price)
```
- Flat percentage sizing is **STRICTLY PROHIBITED**
- No fallback defaults — if volatility cannot be computed, ticker is **dropped**
- Minimum position floor: **5% of portfolio** (ensures compounding)
- Maximum position cap: **10% of portfolio**

### Rule 2.3 — Anti-Whipsaw (Hysteresis)
- A held position is only replaced if the new candidate's confidence score is **>15% higher**
- No duplicate buys — if ticker already in positions, skip regardless of signal

### Rule 2.4 — Exit Priority (Phase 12 v5)
```
Uncle Point  >  ATR Stop-Loss (8-20%)  >  Trailing Stop (12% from peak, activates at +5%)
             >  Take-Profit (40%)  >  Timeout (25d)
```

### Data Freshness (Rule 3.1)
- Any parquet file older than **24 hours** (accounting for weekends/holidays) causes **[FATAL] crash**
- Missing ticker data: drop ticker + log **[WARNING]**

---

## 4. DIRECTORY STRUCTURE (v8.0 — Lean Pipeline)

```
NeuralTrader/
|
+-- main_orchestrator_ist.py          <- SINGLE ENTRY POINT (all modes)
|
+-- core/                             <- Core engine (DO NOT MODIFY without [UNLOCK])
|   +-- ai_models.py                  <- EnsemblePredictor (loads 68-feat models)
|   +-- feature_engineer.py           <- 68-feature vector generation
|   +-- indicators.py                 <- RSI, ATR, MACD, Bollinger etc.
|   +-- strategy.py                   <- TradingStrategy (signal + sizing logic)
|   +-- ibkr_engine.py                <- IBKR TWS integration (ib_insync)
|   +-- integrity.py                  <- verify_system_integrity() startup check
|   +-- execution/                    <- Order execution helpers
|   +-- sentiment/                    <- Sentiment feature engineering (optional)
|   +-- utils/                        <- notifier.py, logging helpers
|
+-- scripts/                          <- Execution scripts (one job each)
|   +-- run_full_backtest.py          <- Backtest engine (UnifiedBacktest + AI regime)
|   +-- run_simulation.py             <- Simulation mode (wraps UnifiedBacktest)
|   +-- retrain_phase12.py            <- Phase 12 retraining (24h cache, Brain-Gate)
|   +-- train_regime_classifier.py    <- Train AI 3-class regime model
|   +-- retrain_ensemble.py           <- Legacy retrain (kept for compatibility)
|   +-- data_manager.py               <- Tiingo data fetch + update
|   +-- report_generator.py           <- HTML dashboard generator
|   +-- send_html_report.py           <- Email dispatch
|   +-- pre_flight_check.py           <- Tier 0 pre-flight validation
|
+-- models/                           <- ALL model files (flat - no subfolders)
|   +-- xgboost_model.pkl             <- XGBoost  (68 features, weight 0.40)
|   +-- lightgbm_model.pkl            <- LightGBM (68 features, weight 0.40)
|   +-- rf_model.pkl                  <- HGB      (68 features, weight 0.20)
|   +-- regime_classifier.pkl         <- AI regime: CRISIS/BEAR/BULL
|   +-- regime_scaler.pkl             <- StandardScaler for regime features
|   +-- feature_names.pkl             <- 68 feature name list
|   +-- feature_scaler.pkl            <- StandardScaler (fitted on 14.7M rows)
|   +-- ensemble_metadata.pkl/json    <- weights + training metadata
|   +-- backup/                       <- Brain-Gate auto-backup before retrain
|
+-- data/
|   +-- raw/                          <- 2,184 tickers as *.parquet (Tiingo)
|   +-- cache/                        <- train_X_phase12.npy + train_y_phase12.npy
|   +-- portfolio_backtest.json       <- Backtest portfolio state
|
+-- config/
|   +-- sector_map.json               <- Ticker -> sector mapping (11 sectors)
|   +-- tickers.txt                   <- Active universe list
+-- docs/
|   +-- ARCHITECTURE.md               <- This file
|   +-- ROADMAP.md                    <- Phase roadmap
+-- logs/                             <- Execution logs (ASCII only)
+-- reports/                          <- backtest_metrics.json, trade_log.csv
+-- temp/                             <- Temp scripts: ws_*.py prefix only
+-- src/                              <- ARCHIVED - do not use
```

---

## 5. EXECUTION MODES (Single Entry Point)

All modes run through `python main_orchestrator_ist.py --mode=<MODE>`

| Mode               | Handler                                  | Output                          |
|--------------------|------------------------------------------|---------------------------------|
| `fetch`            | `orchestrator.run_fetch_mode()`          | Updated parquet files           |
| `trade`            | `orchestrator.run_trade_mode()`          | Live IBKR orders                |
| `paper`            | `orchestrator.run_trade_mode()` [PAPER]  | IBKR paper orders (port 7497)   |
| `report`           | `orchestrator.run_report_mode()`         | HTML dashboard email            |
| `auto`             | `orchestrator.run_auto_mode()`           | Full daily sequence             |
| `saturday_retrain` | `scripts/retrain_phase12.py`             | New models (Brain-Gate gated)   |
| `backtest`         | `scripts/run_full_backtest.py`           | `reports/backtest_metrics.json` |
| `simulation`       | `scripts/run_simulation.py`              | `reports/simulation_metrics.json` |

---

## 6. AI ENSEMBLE (Phase 13 Optimized — RAM-Injected Council)

```
            SPY + VXX + CNN Fear & Greed (21 regime features)
                      |
          regime_classifier.pkl
                      |
          0=CRISIS  1=BEAR       2=BULL
          (0.80)    (thr=0.72)   (thr=0.65)
                      |
                      v
             76 features (64 base + 8 derivatives + 4 momentum)
             Input truncated to last 300 bars (perf optimization)
                              |
              +---------------+---------------+
              |               |               |
      XGBoost (0.356) LightGBM (0.366) HGB (0.278)
              |               |               |
          p = 0.47        p = 0.45        p = 0.44
              |               |               |
              +---------------+---------------+
                              |
              weighted_avg = (0.47*0.356 + 0.45*0.366 + 0.44*0.278)
                           = 0.454
                              |
                   BULL:    >= 0.65  -> CANDIDATE
                   BEAR:    >= 0.72  -> CANDIDATE
                   CRISIS:  >= 0.80  (effectively blocked)
                   CONTRARIAN: CNN < 20 + conf > 0.60 -> BUY

  STRICT INTEGRITY: NO try/except around predict_proba()
  If any model fails or returns NaN -> [FATAL] sys.exit(1)
  Models loaded ONCE at startup -> injected into MockVirtualEngine
```

**Models location:** `models/` (flat, no subfolders)
**Load logic:** `EnsemblePredictor._load_ensemble()` in `core/ai_models.py`
**RAM-Cache:** `self.ai_model` injected into `self.virtual_engine.ai_model` at line 1925
**Training:** `scripts/retrain_phase12.py` | `scripts/train_regime_classifier.py`
**Training data:** 14,678,159 rows | 2,184 tickers | float32 numpy cache (24h TTL)

### Performance (May 6, 2026 Audit)
| Metric | Pre-Optimization | Post-Optimization |
|--------|-----------------|-------------------|
| Scan Duration | 503.78s | **193.27s** |
| Model Disk Loads | 6,552/scan | **3/scan** |
| Memory (RSS) | Unbounded | **242MB stable** |
| Per-ticker latency | ~230ms | **~80ms** |

---

## 7. INSTITUTIONAL SCHEDULE (IST Timezone)

| Time            | Task         | Script / Mode                       |
|-----------------|--------------|-------------------------------------|
| 04:00 (Mon-Fri) | Data Sync    | `--mode=fetch`                      |
| 17:15 (Mon-Fri) | Execution    | `--mode=trade` or `--mode=paper`    |
| 18:00 (Mon-Fri) | Report       | `--mode=report`                     |
| Sat 04:00 AM    | Model Retrain| `--mode=saturday_retrain`           |

---

## 8. SAFETY PROTOCOLS

### Tier 0 — Pre-Flight Checklist (`scripts/pre_flight_check.py`)
Runs before every trading session. Any failure blocks execution with `sys.exit(1)`.

1. **IBKR connectivity** — test connection to TWS port 7497
2. **Email system** — test delivery with attachment
3. **Model integrity** — load models, run test prediction (must return valid probability)
4. **API pulse** — FRED + News API connectivity (200 OK required)

### Brain-Gate — Model Retraining (`scripts/retrain_phase12.py`)
Protects Monday trading from degraded models:
- Backs up current models to `models/backup/` before training
- Validates ensemble precision@0.65 must exceed 55% or rollback
- Uses 24h disk cache (data/cache/) for fast reruns (~8 min vs ~15 min full rebuild)
- Run: `python scripts/retrain_phase12.py` (cached) or `--rebuild` (full rescan)
- Auto-rollback on any validation failure + emergency email alert

### No Silent Failures (Sentinel Protocol)
- Every `try/except` block **must** log full traceback
- Stale data (>24h) causes `[FATAL]` crash — never feeds stale data to AI
- ASCII-only logs — no Unicode/emoji in logger output

---

## 9. RISK MANAGEMENT LAWS (IMMUTABLE — require [UNLOCK] to modify)

| Law                        | Value                        | Enforced In                          |
|----------------------------|------------------------------|--------------------------------------|
| Uncle Point DD threshold   | **20%**                      | `run_full_backtest.py`, `risk_manager.py` |
| Cooldown after Uncle Point | **10 days**                  | `run_full_backtest.py`, `risk_manager.py` |
| Risk per trade             | **2% of portfolio**          | `strategy.py`, `risk_manager.py`     |
| Position floor             | **5% of portfolio**          | `strategy.py`                        |
| Position cap               | **10% of portfolio**         | `strategy.py`                        |
| Position sizing method     | Inverse volatility (1/σ)     | `strategy.py`                        |
| Volatility window          | 20-day returns, √252 ann.    | `strategy.py`                        |
| Confidence threshold (BULL)| **0.65**                     | `run_full_backtest.py`, `strategy.py`|
| Confidence threshold (BEAR)| **0.72**                     | `run_full_backtest.py`, `strategy.py`|
| Confidence threshold (CRISIS)| **0.80**                   | `run_full_backtest.py`, `strategy.py`|
| Regime gate                | AI 3-class classifier        | `run_full_backtest.py`               |
| Stop-loss                  | **ATR 2.5x, clamped 8-20%** | `run_full_backtest.py`, `strategy.py`|
| Trailing stop              | **12% from peak (+5% gate)** | `run_full_backtest.py`, `strategy.py`|
| Take-profit                | **40%**                      | `run_full_backtest.py`, `strategy.py`|
| Max hold period            | **25 days**                  | `run_full_backtest.py`, `strategy.py`|
| Min entry price            | **$10.00**                   | `run_full_backtest.py`, `strategy.py`|
| Min avg daily volume       | **500,000 shares**           | `run_full_backtest.py`, `strategy.py`|
| Max concurrent positions   | **15**                       | `run_full_backtest.py`, `strategy.py`|
| Hysteresis premium         | 15% confidence delta         | `strategy.py`                        |
| Sector tax (bottom 3)      | 15% confidence reduction     | `sector_rotation.py`                 |

---

## 10. LOGGING CONVENTIONS

All log tags are ASCII-only:

| Tag          | Meaning                                    |
|--------------|--------------------------------------------|
| `[OK]`       | Component loaded / check passed            |
| `[PASS]`     | Validation test passed                     |
| `[FAIL]`     | Validation test failed                     |
| `[WARN]`     | Non-fatal warning                          |
| `[FATAL]`    | System must crash (stale data, bad state)  |
| `[SHIELD]`   | Risk circuit breaker triggered             |
| `[SECTOR]`   | Sector authority action                    |
| `[SENTIMENT]`| Sentiment signal logged                    |
| `[BUY]`      | Position opened                            |
| `[SELL]`     | Position closed                            |
| `[DONE]`     | Execution complete                         |
| `[REPORT]`   | Report saved                               |
| `[EMERGENCY]`| Conflicting rules — temporary fix applied  |

---

---

## 11. PHASE 13 OPTIMIZATIONS (May 6, 2026)

### RAM-Injected Model Engine
- `EnsemblePredictor` instantiated ONCE at startup in `TradingOrchestrator.__init__()`
- Injected into `MockVirtualEngine` via `self.virtual_engine.ai_model = self.ai_model`
- Signal loop calls `self.ai_model.predict_from_ohlcv(ticker_data)` directly
- Eliminates 6,552 pickle loads per scan (3 models × 2,184 tickers)
- Location: `main_orchestrator_ist.py` line 1925 (injection), line 1144 (usage)

### 300-Bar Data Truncation
- Feature engineering only needs ~252 bars (200d SMA + buffer)
- All ticker DataFrames truncated to `.iloc[-300:]` before prediction
- Reduces pandas rolling computation by ~80% for tickers with long history
- Location: `main_orchestrator_ist.py` line 1140-1141

### Strict Model Integrity (No Silent Failures)
- `predict_proba()` is NOT wrapped in try/except
- Any model failure or NaN probability triggers `[FATAL]` + `sys.exit(1)`
- Silent "neutral" dilution is permanently prohibited
- Location: `core/ai_models.py` `EnsemblePredictor.predict()` and `predict_batch()`

### sklearn Warning Suppression
- `warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')`
- Applied before scan loop to eliminate 4,368 string-format overhead calls
- Location: `main_orchestrator_ist.py` line 1118-1120

### Heartbeat Monitoring
- Logs RSS memory usage every 100 tickers: `[HEARTBEAT] Processed N/2184 | Memory: XMB`
- Uses `psutil.Process().memory_info().rss`
- Location: `main_orchestrator_ist.py` line 1130-1133

---

*"The AI is the Pilot. The Constitution is the Law. The Alpha is the Mission."*
**Last Updated: May 6, 2026 | v13.0_optimized | Phase 13 — RAM-Cache Engine ACTIVE**
