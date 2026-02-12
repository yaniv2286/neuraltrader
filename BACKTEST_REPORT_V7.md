# NeuralTrader 2.0 - Backtest Report V7

**Generated:** 2026-02-12 18:24:00

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
| CAGR | 344.69% | 5.13% | 7.88% | 25.47% | 12.69% | 11.12% | **3952746752049724069511168.00%** |
| Max Drawdown | -99.47% | -15.12% | -10.46% | -19.08% | -9.06% | -21.42% | **-0.10%** |
| Win Rate | 49.70% | 48.25% | 51.17% | 52.28% | 51.98% | 50.99% | **90.80%** |
| Total Trades | 1010 | 342 | 342 | 1010 | 1010 | 1010 | **174** |

## Executive Summary

- **Final CAGR:** 3952746752049724069511168.00%
- **Max Drawdown:** -0.10%
- **Win Rate:** 90.80%
- **Total Trades:** 174
- **Final Portfolio:** $2.128281152523314e+41

### Performance vs Goals

- **CAGR Target:** 35.0%
- **CAGR Achieved:** 3952746752049724069511168.00% (✅ PASSED)
- **Max DD Target:** <12.0%
- **Max DD Achieved:** 0.10% (✅ PASSED)

## Risk Metrics

- **Sharpe Ratio:** 3.24
- **Sortino Ratio:** 0.00
- **Average P&L:** 0.00%

## Portfolio Performance

- **Initial Capital:** $100,000
- **Final Capital:** $2.128281152523314e+41
- **Total Return:** 212828115252331380833595969041690787840.00%

## Trade Log

Last 10 simulated trades:

| Ticker | Entry | Exit | P&L | Tier | Mode | Reason |
|--------|-------|------|-----|------|--------|--------|
| WATT | 3.76 | 95.00 | 2429.96% | super_alpha | aggressive | chandelier_exit |
| FWDI | 7.03 | 95.00 | 1251.35% | super_alpha | aggressive | chandelier_exit |
| EXPI | 9.10 | 95.00 | 943.96% | super_alpha | aggressive | chandelier_exit |
| ARCT | 6.25 | 95.00 | 1420.00% | super_alpha | aggressive | chandelier_exit |
| HPP | 9.72 | 95.00 | 877.37% | super_alpha | aggressive | chandelier_exit |
| QURE | 25.32 | 95.00 | 275.20% | strong_alpha | aggressive | chandelier_exit |
| CABO | 86.15 | 95.00 | 10.27% | super_alpha | aggressive | chandelier_exit |
| GWRE | 158.99 | 160.03 | 0.65% | super_alpha | aggressive | week_end |
| CABO | 80.65 | 95.00 | 17.79% | super_alpha | aggressive | chandelier_exit |
| ATRA | 5.22 | 95.00 | 1719.92% | super_alpha | aggressive | chandelier_exit |

## Alpha Tier Analysis

- **Super_Alpha Tier:** 149 trades (85.6%)
- **Strong_Alpha Tier:** 25 trades (14.4%)

## Position Size Analysis

- **Average Position Size:** 11.2%
- **Position Range:** 0.5% - 25.0%

## Exit Reason Analysis

- **Chandelier Exit:** 145 trades (83.3%)
- **Week End:** 29 trades (16.7%)

## SPY Buy & Hold Comparison

| Metric | NeuralTrader V7 | SPY | Outperformance |
|--------|----------------|-----|----------------|
| CAGR | 3952746752049724069511168.00% | 0.00% | 3952746752049724069511168.00% |
| Max Drawdown | -0.10% | 0.00% | 0.10% |

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
