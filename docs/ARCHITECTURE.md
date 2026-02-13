# 🏛️ NEURALTRADER: INSTITUTIONAL ARCHITECTURE (v5.1)
**Status:** PRODUCTION READY - Institutional Risk Machine Integration Complete
**Last Updated:** February 13, 2026

## 🧠 1. CORE EXECUTION ENGINE
* **Model:** Tri-Model AI Ensemble (XGBoost, LightGBM, Random Forest).
* **Threshold:** High-Conviction Alpha (Entry > 0.60).
* **Universe:** Modern Era Universe (2,183 Tickers, 2000-Present).

## 🛡️ 2. CORE RISK MODULES (v5.1)
* **Sector Authority:** Dynamic sector momentum analysis with 15% tax on bottom 3 sectors.
* **Global Volatility Gate:** VXX Bollinger Band shield (entries blocked if VXX > Upper BB).
* **Primary Regime Filter:** Weekly Breakdown Shield (Exit if Close < previous 5-day minimum low).
* **Emergency Stop Loss:** Fixed ATR 5.0 (Volatility-Adjusted Floor).
* **Take Profit:** Trend Extension Mode (No static TP; winners run until Shield/ATR trigger).
* **Fail-Fast Guard:** Data Freshness Sentinel mandatory check before every execution.

## 🚨 3. INSTITUTIONAL RISK MACHINE (Phase 9.5)
* **Portfolio Circuit Breaker (Uncle Point):** Trading HALT at 12% drawdown with 8-day cooldown.
* **Inverse Volatility Sizing:** Mathematical position allocation (more capital to safe stocks, less to volatile).
* **Hysteresis Rule (Anti-Whipsaw):** 15% AI score premium required for position swaps at max capacity.
* **Peak Portfolio Tracking:** Continuous monitoring of portfolio peak value for drawdown calculations.
* **Risk-Based Position Sizing:** 1% risk per trade with volatility-weighted allocations.
* **Concentration Analysis:** Real-time monitoring of position concentration risk.

## 📊 4. PERFORMANCE METRICS (v5.1)
* **CAGR:** 34.3% (Sector Tax + Volatility Gate Optimized)
* **Max Drawdown:** -20.8% (Within 20% Risk Policy)
* **Universe:** 217 Tickers | 1.28M Rows (Modern Era Data)
* **Strategy:** Sector Authority Active (15% Tax on Bottom 3 Sectors)
* **Sector Tax:** 15% penalty applied to tickers in bottom 3 weakest sectors
* **Volatility Shield:** VXX Bollinger Band logic prevents entries during extreme volatility
* **Risk Management:** Circuit Breaker at 12% DD, Hysteresis at 15% premium, Inverse Vol sizing

## 📅 5. INSTITUTIONAL SCHEDULE (IST)
* **16:05 (4:05 PM):** P1_DataSync - Daily candle capture.
* **17:15 (5:15 PM):** P3_DailyExecution - Sector Analysis -> Volatility Check -> AI Scoring -> Risk Check -> Execution.
* **18:00 (6:00 PM):** P4_DailyReport - Scorecard delivery.
* **Sat 10:00 AM:** P2_WeeklyRetrain - Weekend model validation.

## 📜 6. CO-FOUNDER PROTOCOLS
* **No Silent Failures:** System must crash if data is stale or ATR fails to calculate.
* **Sanitized Logs:** Standardized text tags [PASS], [FAIL], [SHIELD] for Unicode stability.
* **Sector Discipline:** All trades must pass sector momentum filter before execution.
* **Volatility Discipline:** Global volatility gate overrides all signals during market stress.
* **Risk Discipline:** Circuit Breaker and Hysteresis rules are LAWS of the system - no overrides.
* **Position Discipline:** Inverse volatility sizing is mandatory for all new positions.

## 🔐 7. RISK MANAGEMENT LAWS (Active)
* **Circuit Breaker Law:** Trading HALTS immediately at 12% portfolio drawdown. No exceptions.
* **Hysteresis Law:** Position swaps require 15% AI score improvement over current holdings.
* **Volatility Sizing Law:** New positions allocated by inverse volatility (safe stocks get more capital).
* **Peak Tracking Law:** Portfolio peak value continuously monitored and persisted.
* **Risk Per Trade Law:** Maximum 1% portfolio risk per trade enforced automatically.
* **Concentration Law:** Position concentration monitored and logged in real-time.

## 📁 8. SYSTEM ARCHITECTURE
* **Core Engine:** `src/core/` - AI models, strategy logic, backtesting
* **Risk Machine:** `src/execution/risk_manager.py` - Quant Risk Engine
* **Portfolio Manager:** `main_orchestrator_ist.py` - MockVirtualEngine with persistence
* **Data Pipeline:** `scripts/data_manager.py` - Unified data management
* **Persistence:** `data/portfolio.json` - Portfolio state with risk tracking
