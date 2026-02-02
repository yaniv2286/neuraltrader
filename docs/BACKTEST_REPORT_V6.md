# NeuralTrader 2.0 - Backtest Report V6

**Generated:** 2026-01-31 23:23:42

## Strategy V6: The Unleashed Hunter

### Aggressive Tier Stretching with Chandelier Exit

- **VXX Shield:** Volatility surge detection (>20% in 5 days, Black Swan only)
- **Chandelier Exit:** 3.0x ATR trailing stop (follows price up)
- **Super-Alpha (>0.80):** 12.5% position size, ignores all filters
- **Strong Alpha (>0.50):** 10% position size (lowered from 0.60)
- **Normal Alpha (<0.50):** 7.5% position size
- **Aggressive Sizing:** 8% minimum, 18% maximum position sizes
- **Volatility-Adjusted Sizing:** Final Size = Tier Size × (3% / ATR Pct)
- **S&R Shield:** Bypass VXX cash trigger when SPY near 200-day low

### Deep History Verification

- **Total Tickers:** 0
- **Deep Tickers (>10k):** 0
- **Audit Status:** ❌ FAILED
- **Data Quality:** Limited

## V1 vs V2 vs V3 vs V4 vs V5 vs V6 Performance Comparison

| Metric | Strategy V1 | Strategy V2 | Strategy V3 | Strategy V4 | Strategy V5 | Strategy V6 |
|--------|-------------|-------------|-------------|-------------|-------------|-------------|
| CAGR | 344.69% | 5.13% | 7.88% | 25.47% | 12.69% | **11.12%** |
| Max Drawdown | -99.47% | -15.12% | -10.46% | -19.08% | -9.06% | **-21.42%** |
| Win Rate | 49.70% | 48.25% | 51.17% | 52.28% | 51.98% | **50.99%** |
| Total Trades | 1010 | 342 | 342 | 1010 | 1010 | **1010** |

## Executive Summary

- **Final CAGR:** 11.12%
- **Max Drawdown:** -21.42%
- **Win Rate:** 50.99%
- **Total Trades:** 1010
- **Final Portfolio:** $123,592.63433856682

### Performance vs Goals

- **CAGR Target:** 35.0%
- **CAGR Achieved:** 11.12% (❌ FAILED)
- **Max DD Target:** <15.0%
- **Max DD Achieved:** 21.42% (❌ FAILED)

## Risk Metrics

- **Sharpe Ratio:** 0.71
- **Sortino Ratio:** 1.46
- **Average P&L:** 0.00%

## Portfolio Performance

- **Initial Capital:** $100,000
- **Final Capital:** $123,592.63433856682
- **Total Return:** 23.59%

## Trade Log

Last 10 simulated trades:

| Ticker | Entry | Exit | P&L | Tier | Mode | Reason |
|--------|-------|------|-----|------|--------|--------|
| abt | 107.42 | 109.30 | 1.75% | strong_alpha | N/A | week_end |
| adbe | 301.07 | 293.25 | -2.60% | strong_alpha | N/A | week_end |
| AAPL | 248.04 | 259.48 | 4.61% | normal_alpha | N/A | week_end |
| aeye | 9.49 | 9.42 | -0.74% | normal_alpha | N/A | week_end |
| aat | 17.98 | 18.06 | 0.44% | normal_alpha | N/A | week_end |
| adma | 16.73 | 17.30 | 3.41% | normal_alpha | N/A | week_end |
| acm | 97.08 | 96.43 | -0.67% | normal_alpha | N/A | week_end |
| adp | 257.87 | 246.04 | -4.59% | normal_alpha | N/A | chandelier_exit |
| adsk | 270.00 | 249.20 | -7.70% | normal_alpha | N/A | chandelier_exit |
| abr | 7.74 | 7.70 | -0.52% | normal_alpha | N/A | week_end |

## Alpha Tier Analysis

- **Super_Alpha Tier:** 11 trades (1.1%)
- **Strong_Alpha Tier:** 109 trades (10.8%)
- **Normal_Alpha Tier:** 890 trades (88.1%)

## Position Size Analysis

- **Average Position Size:** 8.5%
- **Position Range:** 8.0% - 18.0%

## Exit Reason Analysis

- **Week End:** 851 trades (84.3%)
- **Chandelier Exit:** 157 trades (15.5%)
- **Portfolio Stop:** 2 trades (0.2%)

## SPY Buy & Hold Comparison

| Metric | NeuralTrader V6 | SPY | Outperformance |
|--------|----------------|-----|----------------|
| CAGR | 11.12% | 0.00% | 11.12% |
| Max Drawdown | -21.42% | 0.00% | 21.42% |

## Strategy V6 Details

- **Period:** 2024-2026
- **Frequency:** Weekly rebalancing
- **Selection:** Top 10 stocks using XGBoost Ranker
- **Exit:** Chandelier Exit (3.0x ATR trailing) or Friday close
- **Position Size:** Dynamic based on alpha tier and volatility

## Risk Management V6

- **Weekly Portfolio Stop Loss:** 5.0%
- **ATR Multiplier:** 3.0x (Chandelier Exit)
- **Market Filter:** Enabled
- **VXX Shield:** Enabled
- **Structural Filter:** Enabled
