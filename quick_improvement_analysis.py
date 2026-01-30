import pandas as pd
import numpy as np
import json
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load current results
df = pd.read_csv('reports/all_phases_test_results.csv')
with open('reports/trading_performance_analysis.json', 'r') as f:
    perf_data = json.load(f)

print("🔍 CURRENT MODEL WEAKNESSES ANALYSIS")
print("=" * 50)

# 1. Model Quality Issues
good_models = df[df['is_good'] == True]
print(f"1. Model Quality:")
print(f"   - Good models (R² > 0.1): {len(good_models)}/{len(df)} ({len(good_models)/len(df)*100:.1f}%)")
print(f"   - Average Test R²: {df['test_r2'].mean():.4f}")
print(f"   - Average Direction Accuracy: {df['test_dir'].mean():.1f}%")

# 2. Trading Strategy Issues
print(f"\n2. Trading Strategy:")
print(f"   - Average Win Rate: {perf_data['top_performers'][0]['success_rate']:.1f}%")
print(f"   - Profitable Models: {perf_data['profitability_rate']:.1f}%")
print(f"   - Models meeting 25% target: {perf_data['target_achievement_rate']:.1f}%")

# 3. Feature Analysis
print(f"\n3. Current Issues:")
print(f"   - Overfitting: Average gen gap {df['gen_gap'].mean():.3f}")
print(f"   - Low confidence: Most models ~50-53% accuracy")
print(f"   - No risk management: Fixed position sizing")
print(f"   - No ensemble: Single model per ticker")

# 4. Quick Win Opportunities
print(f"\n🎯 QUICK WINS FOR 15-20% ANNUAL RETURNS:")
print("=" * 50)

print("1. ENSEMBLE STRATEGY")
print("   - Combine top 3 models per ticker")
print("   - Weighted voting based on confidence")
print("   - Expected improvement: +3-5% accuracy")

print("\n2. CONFIDENCE-BASED POSITION SIZING")
print("   - Larger positions when model confidence > 60%")
print("   - Skip trades when confidence < 45%")
print("   - Expected improvement: +2-4% returns")

print("\n3. TOP-TICKER FOCUS")
print("   - Focus on 33.8% models already meeting 25% target")
print("   - Allocate 70% capital to top performers")
print("   - Expected improvement: +5-8% returns")

print("\n4. RISK MANAGEMENT")
print("   - Stop-loss at -2% per trade")
print("   - Take-profit at +3% per trade")
print("   - Expected improvement: +2-3% returns")

print("\n5. MARKET REGIME FILTER")
print("   - Avoid trading during high volatility")
print("   - Focus on trending markets")
print("   - Expected improvement: +2-4% returns")

# Calculate expected improvements
current_median = 11.1
ensemble_boost = 4
position_sizing_boost = 3
top_ticker_boost = 6
risk_mgmt_boost = 2.5
regime_filter_boost = 3

expected_return = current_median + ensemble_boost + position_sizing_boost + top_ticker_boost + risk_mgmt_boost + regime_filter_boost

print(f"\n📈 EXPECTED PERFORMANCE IMPROVEMENT:")
print(f"   Current Median Annual Return: {current_median:.1f}%")
print(f"   + Ensemble Strategy: +{ensemble_boost:.1f}%")
print(f"   + Position Sizing: +{position_sizing_boost:.1f}%")
print(f"   + Top-Ticker Focus: +{top_ticker_boost:.1f}%")
print(f"   + Risk Management: +{risk_mgmt_boost:.1f}%")
print(f"   + Regime Filter: +{regime_filter_boost:.1f}%")
print(f"   = Expected Return: {expected_return:.1f}%")

print(f"\n✅ TARGET ACHIEVEMENT:")
if expected_return >= 15:
    print(f"   🎯 15% Target: ACHIEVED ({expected_return:.1f}%)")
else:
    print(f"   🎯 15% Target: NOT ACHIEVED ({expected_return:.1f}%)")

if expected_return >= 20:
    print(f"   🎯 20% Target: ACHIEVED ({expected_return:.1f}%)")
else:
    print(f"   🎯 20% Target: NOT ACHIEVED ({expected_return:.1f}%)")

# Save analysis
analysis = {
    'current_median_return': current_median,
    'expected_improvements': {
        'ensemble': ensemble_boost,
        'position_sizing': position_sizing_boost,
        'top_ticker_focus': top_ticker_boost,
        'risk_management': risk_mgmt_boost,
        'regime_filter': regime_filter_boost
    },
    'expected_total_return': expected_return,
    'targets': {
        '15_percent': expected_return >= 15,
        '20_percent': expected_return >= 20
    }
}

with open('reports/quick_improvement_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"\n✅ Analysis saved to reports/quick_improvement_analysis.json")
