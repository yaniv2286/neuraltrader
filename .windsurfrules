# NeuralTrader 2.0 - Private Family Fund
# =======================================
# Mission: Target 25% ARR. Max Drawdown <20%. Use Parquet & Float32. Priority #1: Capital Preservation.
# Goal: 25% ARR for private family fund. Capital preservation is priority #1.

## Data Policy
- Use Parquet for all storage. Float32 for memory efficiency.
- All raw data in data/raw/ with Snappy compression
- Processed data in data/processed/ with proper versioning
- Never use adjusted prices for real-time trading (only for backtesting)

## Risk Policy
- Every trade must have an ATR-based stop loss
- Maximum position size: 10% of capital per position
- Maximum portfolio drawdown: 20% (hard stop)
- Daily loss limit: 5% of capital
- No leverage allowed (1x only)
- Capital preservation is priority #1

## Execution Policy
- Assume 0.15% round-trip transaction costs
- All trades must respect one-bar delay (signal T, execute T+1)
- No trading in illiquid assets (<$1M daily volume)
- All orders must be limit orders

## Model Development Rules
- All ML models must use cross-validation with time-series splits
- No lookahead bias in feature engineering
- Feature importance must be tracked and explained
- Models must be validated on out-of-sample data
- Ensemble methods preferred over single models
- All models must include regime-aware features

## Code Quality Rules
- All new code must include comprehensive error handling
- All functions must have proper type hints
- All modules must include logging
- No hardcoded API keys or secrets
- All configuration must be externalized
- Code must be production-ready with proper testing

## Performance Rules
- Target ARR: 25% with strictly controlled drawdowns (<20%)
- Maximum Sharpe ratio degradation: 0.5
- Maximum win rate decline: 10%
- All strategies must beat SPY on risk-adjusted basis
- Backtest period: Minimum 10 years, preferably 20+ years

## Documentation Rules
- All strategies must have clear entry/exit rules
- All risk parameters must be documented
- All performance metrics must be transparent
- All failures must be logged and analyzed
- Monthly performance reports mandatory

## Compliance Rules
- All trades must be audit-able
- All decisions must be reproducible
- No manual overrides without proper documentation
- All model changes must be version-controlled
- Regulatory compliance checks mandatory

## Technology Rules
- Use PyArrow for data processing (not pandas for large datasets)
- All ML models must use GPU acceleration where available
- All computations must be deterministic
- No random seeds allowed in production
- All timestamps must be UTC

## Capital Rules
- Initial capital: $100,000 (scalable to larger amounts)
- Maximum 10 positions at any time
- Rebalancing: Weekly or on signal changes
- No short selling (long-only for now)
- No options, futures, or derivatives (equities only)

## Monitoring Rules
- Real-time P&L monitoring mandatory
- Position limits must be enforced automatically
- Risk alerts must trigger immediately
- Daily reconciliation required
- Monthly strategy review mandatory

## Security Rules
- All API keys must be encrypted
- No credentials in code
- All data must be encrypted at rest
- Network communications must be secure
- Access logs must be maintained
- dont delete ROADMAP.md and PROJECT_PLAN.md

## Execution Rules
- All orders must be limit orders
- No market orders except for emergency stops
- Execution quality must be monitored
- Slippage must be tracked and minimized
- Trade timing must be optimized

## Backtesting Rules
- Walk-forward analysis required
- Multiple market cycles must be tested
- Stress testing mandatory
- Monte Carlo simulations for validation
- No survivorship bias allowed

## Portfolio Rules
- Diversification across sectors mandatory
- Correlation limits must be enforced
- Concentration limits: 10% per ticker
- Rebalancing must be tax-efficient
- Turnover must be minimized

## Research Rules
- All hypotheses must be tested
- No data mining without proper validation
- Sample size must be statistically significant
- Multiple comparison corrections required
- Results must be reproducible

## Deployment Rules
- All deployments must be gradual
- A/B testing required for changes
- Rollback plans mandatory
- Monitoring must be comprehensive
- Documentation must be up-to-date

## Emergency Rules
- Circuit breakers at 10% daily loss
- Automatic position liquidation at 20% drawdown
- Manual override procedures documented
- Emergency contact list maintained
- Disaster recovery plan tested quarterly
