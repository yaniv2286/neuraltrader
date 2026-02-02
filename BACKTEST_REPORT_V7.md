# NeuralTrader 2.0 - Backtest Report V7

**Generated:** 2026-02-02 17:20:09

## Strategy V7: The Institutional Architect

### Sniper Logic with Market Regime 2.0

- **VXX Shield:** Volatility surge detection (>20% in 5 days, Black Swan only)
- **Tightened Chandelier Exit:** 2.2x ATR trailing stop (lock gains faster)
- **Super-Alpha (>0.75):** 12.5% position size, ignores all filters (Power Tier)
- **Strong Alpha (>0.40):** 10% position size (Power Tier)
- **Normal Alpha (<0.40):** 7.5% position size
- **Sector Caps:** Maximum 2 stocks per sector (diversification)
- **SPY RSI Filter:** Block trades if SPY RSI > 70 (Market Regime 2.0)
- **Aggressive Sizing:** 8% minimum, 18% maximum position sizes
- **Volatility-Adjusted Sizing:** Final Size = Tier Size × (3% / ATR Pct)
- **S&R Shield:** Bypass VXX cash trigger when SPY near 200-day low

### Deep History Verification

- **Total Tickers:** 5
- **Deep Tickers (>10k):** 5
- **Audit Status:** ✅ PASSED
- **Data Quality:** Excellent

## V1 vs V2 vs V3 vs V4 vs V5 vs V6 vs V7 Performance Comparison

| Metric | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|--------|----|----|----|----|----|----|----|
| CAGR | 344.69% | 5.13% | 7.88% | 25.47% | 12.69% | 11.12% | **30.83%** |
| Max Drawdown | -99.47% | -15.12% | -10.46% | -19.08% | -9.06% | -21.42% | **-18.94%** |
| Win Rate | 49.70% | 48.25% | 51.17% | 52.28% | 51.98% | 50.99% | **57.50%** |
| Total Trades | 1010 | 342 | 342 | 1010 | 1010 | 1010 | **840** |

## Executive Summary

- **Final CAGR:** 30.83%
- **Max Drawdown:** -18.94%
- **Win Rate:** 57.50%
- **Total Trades:** 840
- **Final Portfolio:** $154,039.2105519865

### Performance vs Goals

- **CAGR Target:** 35.0%
- **CAGR Achieved:** 30.83% (❌ FAILED)
- **Max DD Target:** <12.0%
- **Max DD Achieved:** 18.94% (❌ FAILED)

## Risk Metrics

- **Sharpe Ratio:** 1.54
- **Sortino Ratio:** 2.69
- **Average P&L:** 0.00%

## Portfolio Performance

- **Initial Capital:** $100,000
- **Final Capital:** $154,039.2105519865
- **Total Return:** 54.04%

## Trade Log

Last 10 simulated trades:

| Ticker | Entry | Exit | P&L | Tier | Mode | Reason |
|--------|-------|------|-----|------|--------|--------|
| AAPL | 248.04 | 259.48 | 4.61% | strong_alpha | aggressive | week_end |
| MSFT | 465.95 | 430.29 | -7.65% | strong_alpha | aggressive | week_end |
| NFLX | 86.12 | 95.00 | 10.31% | strong_alpha | aggressive | chandelier_exit |
| META | 658.76 | 716.50 | 8.76% | strong_alpha | aggressive | week_end |
| NVDA | 187.67 | 191.13 | 1.84% | normal_alpha | aggressive | week_end |
| TSLA | 449.06 | 430.41 | -4.15% | normal_alpha | aggressive | week_end |
| AMZN | 239.16 | 239.30 | 0.06% | normal_alpha | aggressive | week_end |
| UNH | 356.26 | 286.93 | -19.46% | normal_alpha | aggressive | week_end |
| AMD | 259.68 | 236.73 | -8.84% | normal_alpha | aggressive | week_end |
| GOOGL | 327.93 | 338.00 | 3.07% | normal_alpha | aggressive | week_end |

## Alpha Tier Analysis

- **Super_Alpha Tier:** 2 trades (0.2%)
- **Strong_Alpha Tier:** 142 trades (16.9%)
- **Normal_Alpha Tier:** 696 trades (82.9%)

## Position Size Analysis

- **Average Position Size:** 7.2%
- **Position Range:** 0.5% - 25.0%

## Exit Reason Analysis

- **Week End:** 795 trades (94.6%)
- **Portfolio Stop:** 28 trades (3.3%)
- **Chandelier Exit:** 17 trades (2.0%)

## SPY Buy & Hold Comparison

| Metric | NeuralTrader V7 | SPY | Outperformance |
|--------|----------------|-----|----------------|
| CAGR | 30.83% | 0.00% | 30.83% |
| Max Drawdown | -18.94% | 0.00% | 18.94% |

## Strategy V7 Details

- **Period:** 2024-2026
- **Frequency:** Weekly rebalancing
- **Selection:** Top 10 stocks using XGBoost Ranker (with sector caps)
- **Exit:** Tightened Chandelier Exit (2.2x ATR trailing) or Friday close
- **Position Size:** Dynamic based on alpha tier and volatility
- **Market Regime 2.0:** SPY RSI filter to avoid melt-ups

## Risk Management V7

- **Weekly Portfolio Stop Loss:** 5.0%
- **ATR Multiplier:** 2.0x (Tightened Chandelier Exit)
- **Market Filter:** Enabled
- **VXX Shield:** Enabled
- **Structural Filter:** Enabled
- **Sector Caps:** Enabled (max 2 per sector)
- **SPY RSI Filter:** Enabled (threshold: 80)
