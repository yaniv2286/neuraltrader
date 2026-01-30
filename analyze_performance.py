import pandas as pd
import numpy as np
import json

# Read the results
df = pd.read_csv('reports/all_phases_test_results.csv')

# Extract trading performance
trading_performances = []
for _, row in df.iterrows():
    if pd.notna(row['trading_results']) and row['trading_results'] != '':
        try:
            trading_data = eval(row['trading_results'])
            trading_performances.append({
                'ticker': row['ticker'],
                'total_return_pct': float(trading_data['total_return_pct']),
                'success_rate': float(trading_data['success_rate']),
                'total_trades': int(trading_data['total_trades'])
            })
        except:
            pass

# Convert to DataFrame
perf_df = pd.DataFrame(trading_performances)

# Calculate statistics
avg_return = perf_df['total_return_pct'].mean()
median_return = perf_df['total_return_pct'].median()
positive_returns = len(perf_df[perf_df['total_return_pct'] > 0])
total_models = len(perf_df)

print(f'📊 TRADING PERFORMANCE ANALYSIS:')
print(f'   Total Models: {total_models}')
print(f'   Average Return: {avg_return:.2f}%')
print(f'   Median Return: {median_return:.2f}%')
print(f'   Profitable Models: {positive_returns}/{total_models} ({positive_returns/total_models*100:.1f}%)')

# Top performers
top_10 = perf_df.nlargest(10, 'total_return_pct')[['ticker', 'total_return_pct', 'success_rate']]
print(f'\n🏆 TOP 10 PERFORMERS:')
for _, row in top_10.iterrows():
    print(f'   {row["ticker"]}: {row["total_return_pct"]:.1f}% return, {row["success_rate"]:.1f}% win rate')

# Bottom performers
bottom_10 = perf_df.nsmallest(10, 'total_return_pct')[['ticker', 'total_return_pct', 'success_rate']]
print(f'\n📉 WORST 10 PERFORMERS:')
for _, row in bottom_10.iterrows():
    print(f'   {row["ticker"]}: {row["total_return_pct"]:.1f}% return, {row["success_rate"]:.1f}% win rate')

# Annual return analysis (assuming ~20 years of data)
print(f'\n📈 ANNUALIZED RETURNS:')
print(f'   Average Annual Return: {(avg_return/20):.1f}%')
print(f'   Median Annual Return: {(median_return/20):.1f}%')

# 25% target analysis
target_25_annual = 25.0
required_total_return = target_25_annual * 20
models_meeting_target = len(perf_df[perf_df['total_return_pct'] >= required_total_return])

print(f'\n🎯 25% ANNUAL TARGET ANALYSIS:')
print(f'   Required 20-year return: {required_total_return:.0f}%')
print(f'   Models meeting 25% annual target: {models_meeting_target}/{total_models} ({models_meeting_target/total_models*100:.1f}%)')

# Save analysis
analysis = {
    'total_models': total_models,
    'avg_total_return_pct': avg_return,
    'median_total_return_pct': median_return,
    'avg_annual_return_pct': avg_return/20,
    'median_annual_return_pct': median_return/20,
    'target_25_annual': target_25_annual,
    'models_meeting_target': models_meeting_target,
    'target_achievement_rate': models_meeting_target/total_models*100,
    'profitable_models': positive_returns,
    'profitability_rate': positive_returns/total_models*100,
    'top_performers': top_10.to_dict('records'),
    'worst_performers': bottom_10.to_dict('records')
}

with open('reports/trading_performance_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f'\n✅ Analysis saved to reports/trading_performance_analysis.json')
