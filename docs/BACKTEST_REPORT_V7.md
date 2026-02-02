# NeuralTrader 2.0 - Backtest Report V7

**Generated:** 2026-02-02 15:43:26

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
| CAGR | 344.69% | 5.13% | 7.88% | 25.47% | 12.69% | 11.12% | **-0.52%** |
| Max Drawdown | -99.47% | -15.12% | -10.46% | -19.08% | -9.06% | -21.42% | **-4.60%** |
| Win Rate | 49.70% | 48.25% | 51.17% | 52.28% | 51.98% | 50.99% | **33.93%** |
| Total Trades | 1010 | 342 | 342 | 1010 | 1010 | 1010 | **168** |

## Executive Summary

- **Final CAGR:** -0.52%
- **Max Drawdown:** -4.60%
- **Win Rate:** 33.93%
- **Total Trades:** 168
- **Final Portfolio:** $99,171.5957609425

### Performance vs Goals

- **CAGR Target:** 35.0%
- **CAGR Achieved:** -0.52% (❌ FAILED)
- **Max DD Target:** <12.0%
- **Max DD Achieved:** 4.60% (✅ PASSED)

## Risk Metrics

- **Sharpe Ratio:** -0.19
- **Sortino Ratio:** -0.27
- **Average P&L:** 0.00%

## Portfolio Performance

- **Initial Capital:** $100,000
- **Final Capital:** $99,171.5957609425
- **Total Return:** -0.83%

## Trade Log

Last 10 simulated trades:

| Ticker | Entry | Exit | P&L | Tier | Mode | Reason |
|--------|-------|------|-----|------|--------|--------|
| nan | 273.40 | 271.01 | -0.87% | normal_alpha | aggressive | week_end |
| SPY | 231.66 | 231.66 | 0.00% | normal_alpha | aggressive | week_end |
| nan | 271.01 | 259.37 | -4.30% | normal_alpha | aggressive | week_end |
| SPY | 231.66 | 231.66 | 0.00% | normal_alpha | aggressive | week_end |
| nan | 259.37 | 255.53 | -1.48% | strong_alpha | aggressive | week_end |
| SPY | 231.66 | 231.66 | 0.00% | normal_alpha | aggressive | week_end |
| nan | 255.53 | 248.04 | -2.93% | strong_alpha | aggressive | week_end |
| SPY | 231.66 | 231.66 | 0.00% | normal_alpha | aggressive | week_end |
| nan | 248.04 | 259.48 | 4.61% | strong_alpha | aggressive | week_end |
| SPY | 231.66 | 231.66 | 0.00% | normal_alpha | aggressive | week_end |

## Alpha Tier Analysis

- **Strong_Alpha Tier:** 16 trades (9.5%)
- **Normal_Alpha Tier:** 152 trades (90.5%)

## Position Size Analysis

- **Average Position Size:** 7.7%
- **Position Range:** 7.5% - 10.0%

## Exit Reason Analysis

- **Week End:** 168 trades (100.0%)

## SPY Buy & Hold Comparison

| Metric | NeuralTrader V7 | SPY | Outperformance |
|--------|----------------|-----|----------------|
| CAGR | -0.52% | 0.00% | -0.52% |
| Max Drawdown | -4.60% | 0.00% | 4.60% |

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
