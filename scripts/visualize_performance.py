#!/usr/bin/env python3
"""
Deep Time Performance Visualizer
================================

Generates comprehensive equity curve visualizations with historical era shading
to identify the 'Dilution Effect' and validate alpha concentration.

Usage:
    python scripts/visualize_performance.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PerformanceVisualizer')


class PerformanceVisualizer:
    """Visualizer for deep time performance analysis"""
    
    def __init__(self):
        self.reports_dir = PROJECT_ROOT / 'reports' / 'backtests'
        self.output_dir = PROJECT_ROOT / 'reports' / 'visualizations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Historical crisis periods for shading
        self.crisis_eras = {
            '1987 Crash': {
                'start': '1987-10-19',
                'end': '1987-12-31',
                'color': 'red',
                'alpha': 0.2,
                'label': 'Black Monday (1987)'
            },
            'Dot Com Bubble': {
                'start': '2000-03-10',
                'end': '2002-10-09',
                'color': 'orange',
                'alpha': 0.2,
                'label': 'Dot Com Crash (2000-2002)'
            },
            '2008 GFC': {
                'start': '2008-09-15',
                'end': '2009-03-09',
                'color': 'darkred',
                'alpha': 0.2,
                'label': 'Global Financial Crisis (2008-2009)'
            },
            '2020 COVID': {
                'start': '2020-02-20',
                'end': '2020-03-23',
                'color': 'purple',
                'alpha': 0.2,
                'label': 'COVID Crash (2020)'
            }
        }
        
        logger.info("Performance Visualizer initialized")
    
    def find_latest_backtest(self) -> Path:
        """Find the most recent backtest CSV file"""
        csv_files = list(self.reports_dir.glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError("No backtest CSV files found")
        
        # Sort by modification time
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Found latest backtest: {latest_file.name}")
        return latest_file
    
    def load_backtest_data(self, file_path: Path) -> pd.DataFrame:
        """Load and validate backtest data"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded backtest data: {len(df)} rows")
            
            # Check required columns
            required_cols = ['date', 'equity']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate cumulative returns
            initial_equity = df['equity'].iloc[0]
            df['cumulative_return'] = (df['equity'] / initial_equity - 1) * 100
            
            logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            logger.info(f"Total return: {df['cumulative_return'].iloc[-1]:.2f}%")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load backtest data: {e}")
            raise
    
    def create_equity_curve(self, df: pd.DataFrame) -> plt.Figure:
        """Create comprehensive equity curve with era shading"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot equity curve
        ax.plot(df['date'], df['equity'], linewidth=2, color='blue', label='Portfolio Equity')
        
        # Add crisis era shading
        for era_name, era_config in self.crisis_eras.items():
            start_date = pd.to_datetime(era_config['start'])
            end_date = pd.to_datetime(era_config['end'])
            
            # Only shade if within data range
            if start_date <= df['date'].max() and end_date >= df['date'].min():
                ax.axvspan(start_date, end_date, 
                          color=era_config['color'], 
                          alpha=era_config['alpha'],
                          label=era_config['label'])
        
        # Formatting
        ax.set_title('Deep Time Equity Curve (1970-2026)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Equity ($)', fontsize=12)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.YearLocator(5))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Add statistics text box
        stats_text = self._calculate_statistics(df)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_returns_analysis(self, df: pd.DataFrame) -> plt.Figure:
        """Create returns analysis dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Calculate daily returns
        df['daily_return'] = df['equity'].pct_change()
        df['cumulative_return'] = (df['equity'] / df['equity'].iloc[0] - 1) * 100
        
        # 1. Cumulative Returns
        axes[0, 0].plot(df['date'], df['cumulative_return'], linewidth=2, color='green')
        axes[0, 0].set_title('Cumulative Returns (%)', fontweight='bold')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add crisis shading to cumulative returns
        for era_config in self.crisis_eras.values():
            start_date = pd.to_datetime(era_config['start'])
            end_date = pd.to_datetime(era_config['end'])
            if start_date <= df['date'].max() and end_date >= df['date'].min():
                axes[0, 0].axvspan(start_date, end_date, 
                                   color=era_config['color'], 
                                   alpha=era_config['alpha'])
        
        # 2. Drawdown Analysis
        peak = df['equity'].expanding().max()
        drawdown = (df['equity'] - peak) / peak * 100
        axes[0, 1].fill_between(df['date'], drawdown, 0, color='red', alpha=0.3)
        axes[0, 1].plot(df['date'], drawdown, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown Analysis (%)', fontweight='bold')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Daily Returns Distribution
        daily_returns = df['daily_return'].dropna() * 100
        axes[1, 0].hist(daily_returns, bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Daily Returns Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Daily Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Rolling Returns
        rolling_1yr = df['equity'].pct_change(252) * 100  # Approx 252 trading days
        axes[1, 1].plot(df['date'], rolling_1yr, linewidth=1, color='purple')
        axes[1, 1].set_title('Rolling 1-Year Returns (%)', fontweight='bold')
        axes[1, 1].set_ylabel('Return (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def _calculate_statistics(self, df: pd.DataFrame) -> str:
        """Calculate key performance statistics"""
        try:
            # Basic stats
            total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100
            days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
            years = days / 365.25
            cagr = (df['equity'].iloc[-1] / df['equity'].iloc[0]) ** (1/years) - 1
            
            # Drawdown
            peak = df['equity'].expanding().max()
            drawdown = (df['equity'] - peak) / peak
            max_drawdown = drawdown.min() * 100
            
            # Volatility
            daily_returns = df['equity'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
            
            stats = f"""Performance Statistics:
Total Return: {total_return:.2f}%
CAGR: {cagr * 100:.2f}%
Max Drawdown: {max_drawdown:.2f}%
Volatility: {volatility:.2f}%
Sharpe Ratio: {sharpe:.2f}
Period: {years:.1f} years"""
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
            return "Statistics calculation failed"
    
    def detect_dilution_effect(self, df: pd.DataFrame) -> bool:
        """Detect if the equity curve shows dilution effect (flat performance)"""
        try:
            # Calculate annual returns
            df['year'] = df['date'].dt.year
            annual_returns = df.groupby('year')['equity'].apply(
                lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 1 else 0
            )
            
            # Check if majority of years have flat performance (< 5% return)
            flat_years = (annual_returns.abs() < 5).sum()
            total_years = len(annual_returns)
            flat_ratio = flat_years / total_years
            
            # Also check overall CAGR
            total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100
            days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
            years = days / 365.25
            cagr = (df['equity'].iloc[-1] / df['equity'].iloc[0]) ** (1/years) - 1
            
            logger.info(f"Flat years ratio: {flat_ratio:.2f} ({flat_years}/{total_years})")
            logger.info(f"Overall CAGR: {cagr * 100:.2f}%")
            
            # Dilution effect detected if:
            # 1. More than 60% of years have flat performance, OR
            # 2. Overall CAGR is less than 2%
            is_diluted = flat_ratio > 0.6 or (cagr * 100) < 2.0
            
            if is_diluted:
                logger.warning("🚨 DILUTION EFFECT DETECTED: Performance is flat across most periods")
            else:
                logger.info("✅ No significant dilution effect detected")
            
            return is_diluted
            
        except Exception as e:
            logger.error(f"Failed to detect dilution effect: {e}")
            return False
    
    def generate_visualizations(self):
        """Generate all visualizations"""
        try:
            # Find and load latest backtest
            latest_file = self.find_latest_backtest()
            df = self.load_backtest_data(latest_file)
            
            # Detect dilution effect
            is_diluted = self.detect_dilution_effect(df)
            
            # Create equity curve
            logger.info("Creating equity curve visualization...")
            equity_fig = self.create_equity_curve(df)
            
            # Create returns analysis
            logger.info("Creating returns analysis dashboard...")
            returns_fig = self.create_returns_analysis(df)
            
            # Save visualizations
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            equity_file = self.output_dir / f'equity_curve_{timestamp}.png'
            returns_file = self.output_dir / f'returns_analysis_{timestamp}.png'
            
            equity_fig.savefig(equity_file, dpi=300, bbox_inches='tight')
            returns_fig.savefig(returns_file, dpi=300, bbox_inches='tight')
            
            logger.info(f"✅ Equity curve saved: {equity_file}")
            logger.info(f"✅ Returns analysis saved: {returns_file}")
            
            # Generate summary report
            self._generate_summary_report(df, is_diluted, timestamp)
            
            plt.close('all')
            logger.info("🎯 Deep Time Performance Visualization Complete!")
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            raise
    
    def _generate_summary_report(self, df: pd.DataFrame, is_diluted: bool, timestamp: str):
        """Generate summary report of the analysis"""
        try:
            report_file = self.output_dir / f'performance_summary_{timestamp}.txt'
            
            with open(report_file, 'w') as f:
                f.write("🏛️ NEURALTRADER - DEEP TIME PERFORMANCE ANALYSIS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data Range: {df['date'].min().date()} to {df['date'].max().date()}\n")
                f.write(f"Total Days: {len(df)}\n\n")
                
                f.write("📊 PERFORMANCE METRICS\n")
                f.write("-" * 30 + "\n")
                stats = self._calculate_statistics(df)
                f.write(stats + "\n\n")
                
                f.write("🔍 DILUTION EFFECT ANALYSIS\n")
                f.write("-" * 30 + "\n")
                if is_diluted:
                    f.write("🚨 DILUTION EFFECT DETECTED!\n")
                    f.write("The strategy shows flat performance across most periods.\n")
                    f.write("RECOMMENDATION: Implement Top-K Sniper filtering to concentrate alpha.\n")
                else:
                    f.write("✅ No significant dilution effect detected.\n")
                    f.write("The strategy shows meaningful performance across periods.\n")
                
                f.write("\n📈 HISTORICAL CRISIS PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                for era_name, era_config in self.crisis_eras.items():
                    f.write(f"• {era_config['label']}\n")
                
                f.write(f"\n📁 FILES GENERATED\n")
                f.write("-" * 30 + "\n")
                f.write(f"• Equity Curve: equity_curve_{timestamp}.png\n")
                f.write(f"• Returns Analysis: returns_analysis_{timestamp}.png\n")
                f.write(f"• Summary Report: performance_summary_{timestamp}.txt\n")
            
            logger.info(f"✅ Summary report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")


def main():
    """Main execution function"""
    try:
        logger.info("🎯 Starting Deep Time Performance Visualization...")
        
        visualizer = PerformanceVisualizer()
        visualizer.generate_visualizations()
        
        logger.info("🚀 Mission Complete! Check reports/visualizations/ for results.")
        
    except Exception as e:
        logger.error(f"❌ Mission Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
