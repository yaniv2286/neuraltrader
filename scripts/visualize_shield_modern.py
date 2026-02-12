#!/usr/bin/env python3
"""
Modern Shield Audit - Regime Filter Diagnostic Tool (Modern Era)
==============================================================

Visualizes and audits the Regime Filter (The Shield) performance
using Modern Institutional Era data (2000-01-01 to Present).

Usage:
    python scripts/visualize_shield.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
import logging
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.regime import RegimeFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModernShieldAuditor')


class ModernShieldAuditor:
    """Auditor for Regime Filter performance with Modern Era data"""
    
    def __init__(self):
        """Initialize Modern Shield Auditor"""
        self.reports_dir = PROJECT_ROOT / 'reports'
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Modern Era parameters
        self.start_date = '2000-01-01'
        
        # Initialize Regime Filter
        self.regime_filter = RegimeFilter()
        
        logger.info("🛡️ Modern Shield Auditor initialized")
        logger.info(f"   Era: Modern Institutional ({self.start_date} to Present)")
    
    def load_spy_data(self) -> pd.DataFrame:
        """Load SPY benchmark data with modern era filtering"""
        try:
            # Load from raw data directory
            raw_data_path = PROJECT_ROOT / 'data' / 'raw' / 'SPY.parquet'
            
            if not raw_data_path.exists():
                logger.error(f"SPY parquet not found at {raw_data_path}")
                return pd.DataFrame()
            
            logger.info(f"Loading SPY data from: {raw_data_path}")
            spy_data = pd.read_parquet(raw_data_path)
            
            # Standardize column names
            column_mapping = {
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'open': 'Open',
                'volume': 'Volume',
                'adjclose': 'Adj Close'
            }
            
            spy_data = spy_data.rename(columns={k: v for k, v in column_mapping.items() if k in spy_data.columns})
            
            # Process date column
            if 'date' in spy_data.columns:
                spy_data['date'] = pd.to_datetime(spy_data['date'])
                spy_data = spy_data.set_index('date')
            elif not isinstance(spy_data.index, pd.DatetimeIndex):
                logger.error("Invalid date format in SPY data")
                return pd.DataFrame()
            
            # Sort by date
            spy_data = spy_data.sort_index()
            
            # 🦅 MODERN ERA FILTER - Apply 2000-01-01 start date
            start_date = pd.to_datetime(self.start_date)
            original_length = len(spy_data)
            spy_data = spy_data[spy_data.index >= start_date]
            
            filtered_length = len(spy_data)
            logger.info(f"✅ SPY data loaded: {filtered_length:,} days from {spy_data.index.min().date()} to {spy_data.index.max().date()}")
            logger.info(f"   Modern Era filter: {original_length:,} → {filtered_length:,} days ({(filtered_length/original_length)*100:.1f}% retained)")
            
            return spy_data
            
        except Exception as e:
            logger.error(f"Failed to load SPY data: {e}")
            return pd.DataFrame()
    
    def calculate_regime_states(self, spy_data: pd.DataFrame) -> pd.Series:
        """Calculate regime states using the exact RegimeFilter logic"""
        logger.info("🛡️ Calculating regime states for Modern Era...")
        
        try:
            regime_series = self.regime_filter.calculate_regime_state(spy_data)
            
            # Validate the results
            validation = self.regime_filter.validate_regime_data(regime_series)
            
            if not validation.get('valid_states', False):
                logger.error("❌ Invalid regime states detected")
                return pd.Series()
            
            logger.info("✅ Regime calculation complete")
            return regime_series
            
        except Exception as e:
            logger.error(f"❌ Regime calculation failed: {e}")
            return pd.Series()
    
    def create_modern_shield_visualization(self, spy_data: pd.DataFrame, regime_series: pd.Series) -> plt.Figure:
        """Create 3-subplot visualization with regime shading"""
        try:
            # Create figure with 3 subplots sharing X-axis
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
            fig.suptitle('🛡️ NeuralTrader Shield Audit - Modern Era (2000-Present)', fontsize=16, fontweight='bold')
            
            # Calculate indicators for visualization
            indicators = self._calculate_indicators(spy_data)
            
            # PLOT 1: SPY Price (Log Scale) + 200 SMA with Regime Shading
            self._plot_price_with_regime(ax1, indicators, regime_series)
            
            # PLOT 2: Weekly 20 EMA vs Weekly Price
            self._plot_weekly_ema(ax2, indicators, regime_series)
            
            # PLOT 3: Volatility Analysis
            self._plot_volatility(ax3, indicators, regime_series)
            
            # Format shared X-axis
            self._format_x_axis(ax3)
            
            # Add regime legend
            self._add_regime_legend(fig)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            return plt.figure()
    
    def _calculate_indicators(self, spy_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all necessary indicators for visualization"""
        try:
            indicators = spy_data.copy()
            
            # Daily indicators
            indicators['sma_200'] = indicators['Close'].rolling(window=200).mean()
            indicators['returns'] = indicators['Close'].pct_change()
            indicators['volatility'] = indicators['returns'].rolling(window=20).std() * np.sqrt(252) * 100
            
            # Weekly indicators
            weekly_close = indicators['Close'].resample('W').last()
            weekly_ema = weekly_close.ewm(span=20).mean()
            
            # Map weekly EMA back to daily dates
            indicators['weekly_ema'] = weekly_ema.reindex(indicators.index, method='ffill')
            indicators['weekly_close'] = weekly_close.reindex(indicators.index, method='ffill')
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return pd.DataFrame()
    
    def _plot_price_with_regime(self, ax, indicators: pd.DataFrame, regime_series: pd.Series):
        """Plot SPY Price (Log Scale) + 200 SMA with regime shading"""
        try:
            # Add regime shading
            self._add_regime_shading(ax, regime_series, indicators.index)
            
            # Plot price on log scale
            ax.semilogy(indicators.index, indicators['Close'], label='SPY Close (Log Scale)', linewidth=1.5, color='blue')
            ax.semilogy(indicators.index, indicators['sma_200'], label='200-Day SMA', linewidth=2, color='orange')
            
            ax.set_title('SPY Price (Log Scale) vs 200-Day SMA', fontweight='bold')
            ax.set_ylabel('Price ($) - Log Scale')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Add horizontal grid lines for log scale
            ax.yaxis.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Failed to plot price with regime: {e}")
    
    def _plot_weekly_ema(self, ax, indicators: pd.DataFrame, regime_series: pd.Series):
        """Plot Weekly 20 EMA vs Weekly Price"""
        try:
            # Add regime shading
            self._add_regime_shading(ax, regime_series, indicators.index)
            
            # Plot weekly data
            ax.plot(indicators.index, indicators['weekly_close'], label='Weekly Close', linewidth=1.5, color='purple', alpha=0.8)
            ax.plot(indicators.index, indicators['weekly_ema'], label='Weekly 20 EMA', linewidth=2, color='red')
            
            ax.set_title('Weekly 20 EMA vs Weekly Price', fontweight='bold')
            ax.set_ylabel('Price ($)')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Failed to plot weekly EMA: {e}")
    
    def _plot_volatility(self, ax, indicators: pd.DataFrame, regime_series: pd.Series):
        """Plot Volatility Analysis"""
        try:
            # Add regime shading
            self._add_regime_shading(ax, regime_series, indicators.index)
            
            # Plot volatility
            ax.plot(indicators.index, indicators['volatility'], label='20-Day Volatility (%)', linewidth=1.5, color='darkred')
            ax.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Volatility Threshold (20%)')
            
            ax.set_title('Volatility Analysis (20-Day Rolling)', fontweight='bold')
            ax.set_ylabel('Volatility (%)')
            ax.set_xlabel('Date')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Failed to plot volatility: {e}")
    
    def _add_regime_shading(self, ax, regime_series: pd.Series, date_index):
        """Add background shading based on regime states"""
        try:
            # Ensure regime series aligns with date index
            if len(regime_series) != len(date_index):
                logger.warning("Regime series length mismatch with date index")
                return
            
            # Convert regime series to numpy array for faster processing
            regime_array = regime_series.values
            
            # Find contiguous blocks of the same regime
            current_regime = regime_array[0]
            start_idx = 0
            
            for i in range(1, len(regime_array)):
                if regime_array[i] != current_regime:
                    # End of current regime block
                    end_idx = i - 1
                    self._shade_regime_block(ax, date_index[start_idx], date_index[end_idx], current_regime)
                    
                    # Start new block
                    current_regime = regime_array[i]
                    start_idx = i
            
            # Shade the last block
            self._shade_regime_block(ax, date_index[start_idx], date_index[-1], current_regime)
            
        except Exception as e:
            logger.error(f"Failed to add regime shading: {e}")
    
    def _shade_regime_block(self, ax, start_date, end_date, regime_state):
        """Shade a block of dates with the appropriate regime color"""
        try:
            color = self.regime_filter.get_regime_color(regime_state)
            alpha = 0.2  # Lighter shading for better visibility
            
            ax.axvspan(start_date, end_date, color=color, alpha=alpha, zorder=0)
            
        except Exception as e:
            logger.error(f"Failed to shade regime block: {e}")
    
    def _format_x_axis(self, ax):
        """Format the shared X-axis"""
        try:
            # Major ticks every 5 years
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
            # Minor ticks every year
            ax.xaxis.set_minor_locator(mdates.YearLocator(1))
            
            # Rotate labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        except Exception as e:
            logger.error(f"Failed to format X-axis: {e}")
    
    def _add_regime_legend(self, fig):
        """Add regime state legend to the figure"""
        try:
            from matplotlib.patches import Patch
            
            legend_elements = [
                Patch(facecolor='lightgreen', alpha=0.5, label='GREEN (Aggressive)'),
                Patch(facecolor='lightyellow', alpha=0.5, label='YELLOW (Caution)'),
                Patch(facecolor='lightcoral', alpha=0.5, label='RED (Defensive/Cash)')
            ]
            
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
            
        except Exception as e:
            logger.error(f"Failed to add regime legend: {e}")
    
    def print_modern_era_summary(self, regime_series: pd.Series):
        """Print detailed Modern Era regime summary to console"""
        try:
            total_days = len(regime_series)
            green_days = (regime_series == 0).sum()
            yellow_days = (regime_series == 1).sum()
            red_days = (regime_series == 2).sum()
            
            green_pct = (green_days / total_days) * 100
            yellow_pct = (yellow_days / total_days) * 100
            red_pct = (red_days / total_days) * 100
            
            print("\n" + "="*70)
            print("🛡️ NEURALTRADER MODERN ERA SHIELD AUDIT RESULTS")
            print("="*70)
            print(f"Analysis Period: {self.start_date} to {regime_series.index.max().date()}")
            print(f"Total Trading Days: {total_days:,}")
            print("")
            print("📊 MODERN ERA REGIME DISTRIBUTION:")
            print(f"🟢 GREEN (Aggressive): {green_days:,} days ({green_pct:.1f}%)")
            print(f"🟡 YELLOW (Caution): {yellow_days:,} days ({yellow_pct:.1f}%)")
            print(f"🔴 RED (Defensive): {red_days:,} days ({red_pct:.1f}%)")
            print("")
            print("🛡️ MODERN ERA SHIELD PROTECTION:")
            print(f"• Days in CASH (RED regime): {red_days:,} ({red_pct:.1f}%)")
            print(f"• Days with FULL deployment (GREEN): {green_days:,} ({green_pct:.1f}%)")
            print(f"• Days with CAUTIOUS deployment (YELLOW): {yellow_days:,} ({yellow_pct:.1f}%)")
            print("")
            print("📈 MODERN ERA MARKET INSIGHTS:")
            print(f"• Shield protects capital during {red_pct:.1f}% of trading days")
            print(f"• Aggressive deployment only during {green_pct:.1f}% of days")
            print(f"• Cautious approach during {yellow_pct:.1f}% of days")
            print("")
            print("✅ Modern Era Shield audit complete. Check reports/shield_audit_modern.png")
            print("="*70)
            
        except Exception as e:
            logger.error(f"Failed to print modern era summary: {e}")
    
    def run_modern_audit(self):
        """Run complete Modern Era shield audit"""
        try:
            logger.info("🛡️ Starting Modern Era Shield Audit...")
            
            # Load SPY data with modern era filter
            spy_data = self.load_spy_data()
            if spy_data.empty:
                logger.error("❌ No SPY data available")
                return False
            
            # Calculate regime states
            regime_series = self.calculate_regime_states(spy_data)
            if regime_series.empty:
                logger.error("❌ Failed to calculate regime states")
                return False
            
            # Create visualization
            logger.info("🎨 Creating Modern Era shield visualization...")
            fig = self.create_modern_shield_visualization(spy_data, regime_series)
            
            # Save visualization
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.reports_dir / f'shield_audit_modern_{timestamp}.png'
            
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Modern Era shield audit saved: {output_file}")
            
            # Print summary
            self.print_modern_era_summary(regime_series)
            
            plt.close(fig)
            return True
            
        except Exception as e:
            logger.error(f"❌ Modern Era shield audit failed: {e}")
            return False


def main():
    """Main execution function"""
    try:
        auditor = ModernShieldAuditor()
        success = auditor.run_modern_audit()
        
        if success:
            logger.info("🚀 Modern Era Shield Audit Complete!")
        else:
            logger.error("❌ Modern Era Shield Audit Failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
