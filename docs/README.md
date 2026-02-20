🦅 NeuralTrader: AI Quantitative Fund (v5.2 - Phase 10 Active)
The Architectural Goal: 25%+ CAGR | < 20% Max Drawdown | 45-Year Universal Alpha.

🏛️ The Deep Time Pipeline
The system is now a universal quantitative engine designed to exploit 45 years of market physics (1980–2026) across 2,000+ instruments.

1. Data Fuel (Deep Time Archives)
Exclusive Fuel: data/raw/*.parquet (14.8M rows of history).

Universal Scope: 2,183 tickers spanning the 1980s tech birth to the 2026 AI era.

Integrity: Type-aligned datetime64[ns] precision with unified string-key lookups.

2. The Brain (Ensemble Council)
Tri-Model Council: XGBoost (35%), LightGBM (35%), RandomForest (30%).

Language: Speaks exclusively in Normalized Ratios. Blinds itself to raw prices to find universal market mechanics.

Brain-Gate: Automatic validation ensuring no model degradation during retraining.

3. The Shield (Quad-Core Regime)
Multi-Timeframe Logic: Daily Trend (200 SMA) + Weekly Momentum (20 EMA) + Breadth (RSP) + Volatility.

The Kill-Switch: Daily SPY/GSPC drops > 3% trigger an immediate "Force Cash" protocol.

Sniper Entry: Dynamic AI Thresholds (0.35–0.41) calibrated for high-capacity trade volume.

4. 🎯 Phase 10: Dynamic Sector-Based Exits (NEW)
Intelligent Exit Logic: Sector-aware adaptive thresholds based on sector performance.

Real-Time Analysis: 11 sector ETFs ranked by 20-day ROC with dynamic exit rules.

Smart Exits: 4-tier system (Weakest 25%, Bottom 50%, Top 25%, Negative Momentum).

Coverage: All 220 tickers mapped to sectors with intelligent position management.

⚡ Quick Start (Zen Mode)
Bash
# 1. Environment Lockdown
pip install -r requirements.txt

# 2. Re-Educate the Brain (Retrain on 14.8M rows)
python scripts/retrain_ensemble.py

# 3. Launch the Titan (Universal Optimization)
python scripts/optimize_strategy.py
🛡️ Risk Management (Institutional Grade)
Aggressor Position Sizing: 2.5% risk per trade for CAGR breakout.

Max Capacity: 15 concurrent positions to maximize diversification.

Outlier Protection: Indicators clipped to [-10, 10]; Raw prices remain untouched.

Silent Failure Guard: System crashes (Fail Fast) if features are missing or log-zeros are detected.

📂 Repository Topology (Post-Sanitation)
core/ — The Kernel. Immutable logic (AI, Indicators, Strategy).

scripts/ — The Pillars. Optimization, Retraining, Data Management.

models/production/ — The Soul. Single Source of Truth for model weights.

data/raw/ — The Fuel. 45-year Parquet archives (1980–2026).

reports/ — The Scorecard. Performance audits and optimization CSVs.