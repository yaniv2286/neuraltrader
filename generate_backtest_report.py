"""Generate comprehensive Excel report from backtest results."""
import pandas as pd
import numpy as np
from datetime import datetime
import json

def generate_report():
    """Generate Excel report with multiple sheets."""
    print("📊 Generating Comprehensive Backtest Report...")
    
    # Load results
    df = pd.read_csv('reports/real_backtest_results.csv')
    print(f"Loaded {len(df)} ticker results")
    
    # Create Excel writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'reports/backtest_analysis_{timestamp}.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: Executive Summary
        summary_data = {
            'Metric': [
                'Total Tickers Tested',
                'Successful Tests',
                'Failed Tests',
                'Average Test Direction Accuracy',
                'Average Win Rate',
                'Average Return',
                'Profitable Tickers',
                'Profitable Rate',
                'Best Performer (Direction)',
                'Best Performer (Return)',
                'Total Trades Executed'
            ],
            'Value': [
                len(df),
                len(df),
                0,
                f"{df['test_dir'].mean():.1f}%",
                f"{df['win_rate'].mean():.1f}%",
                f"{df['total_return'].mean():.1f}%",
                len(df[df['total_return'] > 0]),
                f"{len(df[df['total_return'] > 0]) / len(df) * 100:.0f}%",
                df.loc[df['test_dir'].idxmax(), 'ticker'],
                df.loc[df['total_return'].idxmax(), 'ticker'],
                df['trades'].sum()
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Sheet 2: All Results
        df_sorted = df.sort_values('test_dir', ascending=False)
        df_sorted.to_excel(writer, sheet_name='All Results', index=False)
        
        # Sheet 3: Top Performers
        top_performers = df.nlargest(20, 'test_dir')
        top_performers.to_excel(writer, sheet_name='Top Performers', index=False)
        
        # Sheet 4: Most Profitable
        most_profitable = df.nlargest(20, 'total_return')
        most_profitable.to_excel(writer, sheet_name='Most Profitable', index=False)
        
        # Sheet 5: Underperformers (for blacklist consideration)
        underperformers = df.nsmallest(20, 'test_dir')
        underperformers.to_excel(writer, sheet_name='Underperformers', index=False)
        
        # Sheet 6: Trading Statistics
        trading_stats = pd.DataFrame({
            'Category': ['High Direction (>52%)', 'Medium Direction (50-52%)', 'Low Direction (<50%)',
                        'Profitable', 'Unprofitable', 'High Win Rate (>70%)', 'Low Win Rate (<50%)'],
            'Count': [
                len(df[df['test_dir'] > 52]),
                len(df[(df['test_dir'] >= 50) & (df['test_dir'] <= 52)]),
                len(df[df['test_dir'] < 50]),
                len(df[df['total_return'] > 0]),
                len(df[df['total_return'] <= 0]),
                len(df[df['win_rate'] > 70]),
                len(df[df['win_rate'] < 50])
            ],
            'Percentage': [
                f"{len(df[df['test_dir'] > 52]) / len(df) * 100:.1f}%",
                f"{len(df[(df['test_dir'] >= 50) & (df['test_dir'] <= 52)]) / len(df) * 100:.1f}%",
                f"{len(df[df['test_dir'] < 50]) / len(df) * 100:.1f}%",
                f"{len(df[df['total_return'] > 0]) / len(df) * 100:.1f}%",
                f"{len(df[df['total_return'] <= 0]) / len(df) * 100:.1f}%",
                f"{len(df[df['win_rate'] > 70]) / len(df) * 100:.1f}%",
                f"{len(df[df['win_rate'] < 50]) / len(df) * 100:.1f}%"
            ]
        })
        trading_stats.to_excel(writer, sheet_name='Trading Statistics', index=False)
    
    print(f"✅ Report saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 BACKTEST ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n🎯 Direction Accuracy Distribution:")
    print(f"   >52% (Good): {len(df[df['test_dir'] > 52])} tickers ({len(df[df['test_dir'] > 52])/len(df)*100:.0f}%)")
    print(f"   50-52% (Neutral): {len(df[(df['test_dir'] >= 50) & (df['test_dir'] <= 52)])} tickers")
    print(f"   <50% (Poor): {len(df[df['test_dir'] < 50])} tickers")
    
    print(f"\n💰 Profitability:")
    print(f"   Profitable: {len(df[df['total_return'] > 0])} tickers ({len(df[df['total_return'] > 0])/len(df)*100:.0f}%)")
    print(f"   Avg Return: {df['total_return'].mean():.1f}%")
    print(f"   Max Return: {df['total_return'].max():.1f}% ({df.loc[df['total_return'].idxmax(), 'ticker']})")
    
    print(f"\n🏆 Top 10 by Direction Accuracy:")
    for _, row in df.nlargest(10, 'test_dir').iterrows():
        print(f"   {row['ticker']}: {row['test_dir']:.1f}% dir, {row['total_return']:.1f}% return")
    
    # Update high confidence models in trading engine config
    high_conf_tickers = df[df['test_dir'] > 54]['ticker'].tolist()
    print(f"\n⭐ High Confidence Models (>54% accuracy): {len(high_conf_tickers)}")
    print(f"   {', '.join(high_conf_tickers[:10])}...")
    
    # Save updated trading config
    config = {
        'high_confidence_models': high_conf_tickers,
        'blacklist_candidates': df[df['test_dir'] < 48]['ticker'].tolist(),
        'generated_at': datetime.now().isoformat(),
        'total_tickers': len(df),
        'avg_direction': float(df['test_dir'].mean()),
        'avg_return': float(df['total_return'].mean())
    }
    
    with open('reports/trading_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n📁 Trading config saved to reports/trading_config.json")
    
    return df

if __name__ == "__main__":
    df = generate_report()
