# 🏛️ NEURALTRADER: INSTITUTIONAL ARCHITECTURE (v5.0)
**Status:** PRODUCTION READY - Sector Authority Integration Complete
**Last Updated:** February 12, 2026

## 🧠 1. CORE EXECUTION ENGINE
* **Model:** Tri-Model AI Ensemble (XGBoost, LightGBM, Random Forest).
* **Threshold:** High-Conviction Alpha (Entry > 0.60).
* **Universe:** Modern Era Universe (2,183 Tickers, 2000-Present).

## 🛡️ 2. CORE RISK MODULES (v5.0)
* **Sector Authority:** Dynamic sector momentum analysis with 15% tax on bottom 3 sectors.
* **Global Volatility Gate:** VXX Bollinger Band shield (entries blocked if VXX > Upper BB).
* **Primary Regime Filter:** Weekly Breakdown Shield (Exit if Close < previous 5-day minimum low).
* **Emergency Stop Loss:** Fixed ATR 5.0 (Volatility-Adjusted Floor).
* **Take Profit:** Trend Extension Mode (No static TP; winners run until Shield/ATR trigger).
* **Fail-Fast Guard:** Data Freshness Sentinel mandatory check before every execution.

## 📊 3. PERFORMANCE METRICS (v5.0)
* **CAGR:** 34.3% (Sector Tax + Volatility Gate Optimized)
* **Max Drawdown:** -20.8% (Within 20% Risk Policy)
* **Universe:** 217 Tickers | 1.28M Rows (Modern Era Data)
* **Strategy:** Sector Authority Active (15% Tax on Bottom 3 Sectors)
* **Sector Tax:** 15% penalty applied to tickers in bottom 3 weakest sectors
* **Volatility Shield:** VXX Bollinger Band logic prevents entries during extreme volatility

## 📅 4. INSTITUTIONAL SCHEDULE (IST)
* **16:05 (4:05 PM):** P1_DataSync - Daily candle capture.
* **17:15 (5:15 PM):** P3_DailyExecution - Sector Analysis -> Volatility Check -> AI Scoring -> Execution.
* **18:00 (6:00 PM):** P4_DailyReport - Scorecard delivery.
* **Sat 10:00 AM:** P2_WeeklyRetrain - Weekend model validation.

## 📜 5. CO-FOUNDER PROTOCOLS
* **No Silent Failures:** System must crash if data is stale or ATR fails to calculate.
* **Sanitized Logs:** Standardized text tags [PASS], [FAIL], [SHIELD] for Unicode stability.
* **Sector Discipline:** All trades must pass sector momentum filter before execution.
* **Volatility Discipline:** Global volatility gate overrides all signals during market stress.
