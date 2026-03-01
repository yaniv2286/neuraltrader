# NEURALTRADER: INSTITUTIONAL ARCHITECTURE (v8.0)
**Status:** Phase 12 Active | Pure AI Entry/Exit | IBKR Paper Integration Available
**Last Updated:** February 28, 2026
**Version:** v8.0 — 68-Feature Phase 12 Models + AI Regime Classifier (VXX-Powered)

---

## 1. VALIDATED PERFORMANCE METRICS

### Phase 12 Results (2000-2026, 26.2 years)
Backtest period: 2000-01-03 to 2026-02-27 | Universe: 2,184 tickers | Capital: $100,000

| Metric              | Phase 11.1   | Phase 12       | Delta     |
|---------------------|--------------|----------------|-----------|
| CAGR                | 6.67%        | **7.87%**      | +1.20%    |
| Total Return        | 440.99%      | **625.59%**    | +184.6%   |
| Max Drawdown        | 11.02%       | **10.62%**     | -0.40%    |
| Sharpe Ratio        | 1.01         | **1.00**       | flat      |
| Win Rate            | 56.5%        | 47.5%          | -9.0%     |
| Profit Factor       | 1.42         | 1.30           | -0.12     |
| Total Trades        | 11,888       | 21,622         | +1.8x     |
| 2000-02 Dot-Com     | +14.86%      | **+21.77%**    | +6.91%    |
| 2008 Crisis         | +20.28%      | +1.58%         | -18.70%   |
| 2020 COVID Crash    | +5.26%       | **+18.59%**    | +13.33%   |
| 2022 Bear Market    | -6.96%       | **+4.27%**     | +11.23%   |

### Phase 12 Model Precision (training: 14.7M rows, 68 features, 5-day fwd >2%)
| Model       | prec@0.44 | prec@0.65(train) | AUC   | Weight |
|-------------|-----------|------------------|-------|--------|
| XGBoost     | -         | 87.9%            | 0.601 | 0.40   |
| LightGBM    | -         | 87.9%            | 0.601 | 0.40   |
| HGB         | -         | 92.4%            | 0.599 | 0.20   |
| **Ensemble**| -         | **91.4%** [PASS] | 0.601 | -      |

**Model:** Tri-Model Ensemble (XGBoost 0.40 + LightGBM 0.40 + HGB 0.20)
**Features:** 68 features (64 technical + 4 Phase 12 momentum: rs_vs_spy, high_52w_prox, volume_breakout, roc_63)
**Confidence threshold:** 0.44 BULL / 0.46 BEAR / blocked CRISIS  (prob range 0.30-0.52, base rate 31.9%)
**Regime Gate:** AI 3-class classifier (20 VXX+SPY features) — pre-2009: SPY SMA100 fallback
**Regime dist:** CRISIS 4.8% | BEAR 16.1% | BULL 79.1%
**Filters:** MIN_PRICE = $5.00 | MIN_AVG_VOLUME = 100,000 shares/day

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
 STAGE 1: DATA SYNC  (16:05 IST)
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
 │  Input : AAPL.parquet  (last 252 rows of OHLCV)             │
 │  Output: 68-feature row vector (one row per ticker/day)     │
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
 │            1=BEAR   -> allow at threshold 0.72              │
 │            2=BULL   -> allow at threshold 0.65              │
 │                                                             │
 │  Validation accuracy: 100% on 2022+ holdout                 │
 │  Crisis days (VXX BB trigger): 4.8% of trading days         │
 │  Fallback if no model: SPY > SMA100 = BULL                  │
 └──────────────────────────┬──────────────────────────────────┘
                            |
                            v
 STAGE 4: AI ENSEMBLE SCORING  (Council Decision)
 ┌─────────────────────────────────────────────────────────────┐
 │  core/ai_models.py  (EnsemblePredictor)                     │
 │                                                             │
 │  Models loaded from  models/                                │
 │    xgboost_model.pkl    weight = 0.40  (200 trees, hist)    │
 │    lightgbm_model.pkl   weight = 0.40  (200 trees)          │
 │    rf_model.pkl         weight = 0.20  (HGB, 100 iter)      │
 │                                                             │
 │  Scaler: feature_scaler.pkl  (StandardScaler)               │
 │                                                             │
 │  For each ticker (vectorized batch):                        │
 │    X_scaled = scaler.transform(X_68features)                │
 │    p_xgb  = xgb.predict_proba(X_scaled)[:, 1]  = 0.71      │
 │    p_lgb  = lgb.predict_proba(X_scaled)[:, 1]  = 0.68      │
 │    p_hgb  = hgb.predict_proba(X_scaled)[:, 1]  = 0.63      │
 │                                                             │
 │    confidence = (0.71*0.40 + 0.68*0.40 + 0.63*0.20)        │
 │               = 0.682  <- COUNCIL VERDICT: BUY SIGNAL       │
 │                                                             │
 │  BULL regime:   confidence >= 0.65  -> CANDIDATE            │
 │  BEAR regime:   confidence >= 0.72  -> CANDIDATE            │
 │  CRISIS regime: ALL entries blocked                         │
 └──────────────────────────┬──────────────────────────────────┘
                            |
                            v
 STAGE 5: LIQUIDITY FILTERS  (entry guard)
 ┌─────────────────────────────────────────────────────────────┐
 │  scripts/run_full_backtest.py  /  scripts/run_simulation.py │
 │                                                             │
 │  [FILTER 1] Price gate                                      │
 │    AAPL close = $213.50  >= MIN_PRICE ($5.00)  -> PASS      │
 │                                                             │
 │  [FILTER 2] Liquidity gate                                  │
 │    AAPL 20d avg volume = 58.4M  >= 100,000     -> PASS      │
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
 │    1.4% < 10%  -> NO COOLDOWN  -> proceed                   │
 │                                                             │
 │  [RULE 2.2] Inverse Volatility Sizing                       │
 │    AAPL 20d returns std = 0.0116  (annualized: 18.46%)      │
 │    Dollar risk  = portfolio * 1%  = $1,274                  │
 │    Position $   = dollar_risk / annualized_vol              │
 │                 = $1,274 / 0.1846 = $6,900                  │
 │    Shares       = $6,900 / $213.50 = 32 shares              │
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
 │  [EXIT 1] Stop-Loss         close drops 10% from entry      │
 │  [EXIT 2] Take-Profit       close rises 30% from entry      │
 │  [EXIT 3] Timeout           held 20 days -> force sell      │
 │  [EXIT 4] Uncle Point       portfolio DD > 10% -> liquidate │
 │                                                             │
 │  Exit Priority:  Uncle Point > Stop-Loss > Take-Profit      │
 │                  > Timeout                                  │
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
- Portfolio drawdown > **10%** from peak triggers immediate liquidation of ALL positions
- **8-day cooldown** — zero new entries allowed
- Peak value resets after liquidation to prevent re-triggering during cooldown

### Rule 2.2 — Inverse Volatility Sizing
```
dollar_risk     = portfolio_value * 0.01          (1% risk per trade)
annualized_vol  = std(20d_returns) * sqrt(252)
position_dollars = dollar_risk / annualized_vol
shares           = floor(position_dollars / price)
```
- Flat percentage sizing is **STRICTLY PROHIBITED**
- No fallback defaults — if volatility cannot be computed, ticker is **dropped**

### Rule 2.3 — Anti-Whipsaw (Hysteresis)
- A held position is only replaced if the new candidate's confidence score is **>15% higher**
- No duplicate buys — if ticker already in positions, skip regardless of signal

### Rule 2.4 — Exit Priority
```
Uncle Point  >  Stop-Loss (10%)  >  Take-Profit (30%)  >  Timeout (20d)
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

## 6. AI ENSEMBLE (Phase 12 — Council Decision Architecture)

```
            SPY + VXX (20 regime features)
                      |
          regime_classifier.pkl
                      |
          0=CRISIS  1=BEAR    2=BULL
          (block)   (thr=0.72) (thr=0.65)
                      |
                      v
             68 features (technical + momentum)
                              |
              +---------------+---------------+
              |               |               |
      XGBoost (0.40)  LightGBM (0.40)  HGB (0.20)
      prec@0.65=89.6% prec@0.65=89.5%
              |               |               |
          p = 0.71        p = 0.68        p = 0.63
              |               |               |
              +---------------+---------------+
                              |
              weighted_avg = (0.71*0.40 + 0.68*0.40 + 0.63*0.20)
                           = 0.682
                              |
                   BULL:   >= 0.65  -> CANDIDATE
                   BEAR:   >= 0.72  -> CANDIDATE
                   CRISIS: blocked
```

**Models location:** `models/` (flat, no subfolders)
**Load logic:** `EnsemblePredictor._load_ensemble()` in `core/ai_models.py`
**Training:** `scripts/retrain_phase12.py` | `scripts/train_regime_classifier.py`
**Training data:** 14,678,159 rows | 2,184 tickers | float32 numpy cache (24h TTL)

---

## 7. INSTITUTIONAL SCHEDULE (IST Timezone)

| Time            | Task         | Script / Mode                       |
|-----------------|--------------|-------------------------------------|
| 16:05 (Mon-Fri) | Data Sync    | `--mode=fetch`                      |
| 17:15 (Mon-Fri) | Execution    | `--mode=trade` or `--mode=paper`    |
| 18:00 (Mon-Fri) | Report       | `--mode=report`                     |
| Sat 10:00 AM    | Model Retrain| `--mode=saturday_retrain`           |

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

| Law                       | Value                     | Enforced In                    |
|---------------------------|---------------------------|--------------------------------|
| Uncle Point DD threshold  | 10%                       | `run_full_backtest.py`         |
| Cooldown after Uncle Point| 8 days                    | `run_full_backtest.py`         |
| Risk per trade            | 1% of portfolio           | `strategy.py`                  |
| Position sizing method    | Inverse volatility (1/σ)  | `strategy.py`                  |
| Volatility window         | 20-day returns, √252 ann. | `strategy.py`                  |
| Confidence threshold (BULL)| 0.65                     | `run_full_backtest.py`         |
| Confidence threshold (BEAR)| 0.72                     | `run_full_backtest.py`         |
| Regime gate               | AI 3-class classifier     | `run_full_backtest.py`         |
| Stop-loss                 | 10%                       | `run_full_backtest.py`         |
| Take-profit               | 30%                       | `run_full_backtest.py`         |
| Max hold period           | 20 days                   | `run_full_backtest.py`         |
| Min entry price           | $5.00                     | `run_full_backtest.py`         |
| Min avg daily volume      | 100,000 shares            | `run_full_backtest.py`         |
| Hysteresis premium        | 15% confidence delta      | `strategy.py`                  |
| Sector tax (bottom 3)     | 15% confidence reduction  | `sector_rotation.py`           |

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

*"The AI is the Pilot. The Constitution is the Law. The Alpha is the Mission."*
**Last Updated: February 28, 2026 | v8.0**
