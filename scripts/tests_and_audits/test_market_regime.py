#!/usr/bin/env python3
"""
Test Market Regime Classifier
=============================

Trains and tests the HMM-based market regime classifier
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.market_regime import MarketBreadth, RegimeClassifier
from scripts.data_manager import DataManager

def main():
    """Main test routine"""
    print("=" * 80)
    print("TESTING MARKET REGIME CLASSIFIER")
    print("=" * 80)
    
    # Initialize components
    dm = DataManager()
    breadth_calc = MarketBreadth()
    regime_classifier = RegimeClassifier()
    
    # Load data for top 20 tickers
    tickers = dm.get_existing_tickers()[:20]
    print(f"Loading data for {len(tickers)} tickers...")
    
    universe_data = {}
    for ticker in tickers:
        try:
            import glob
            pattern = Path(dm.cache_dir) / f"{ticker}_sp100_emergency_*.csv"
            files = glob.glob(str(pattern))
            
            if files:
                df = pd.read_csv(files[0])
                df['Date'] = pd.to_datetime(df['Date'])
                universe_data[ticker] = df
        except Exception as e:
            print(f"Failed to load {ticker}: {e}")
    
    print(f"Loaded data for {len(universe_data)} tickers")
    
    # Calculate market breadth
    print("\nCalculating market breadth...")
    mmtw_data = breadth_calc.calculate_mmtw(universe_data)
    
    # Load SPY and VIX data
    print("\nLoading SPY and VIX data...")
    spy_data = None
    vix_data = None
    
    # Try to load SPY (use V as proxy if SPY not available)
    spy_pattern = Path(dm.cache_dir) / "V_sp100_emergency_*.csv"
    spy_files = glob.glob(str(spy_pattern))
    if spy_files:
        spy_data = pd.read_csv(spy_files[0])
        spy_data['Date'] = pd.to_datetime(spy_data['Date'])
        print("Loaded V data as SPY proxy")
    else:
        # Try SPY
        spy_pattern = Path(dm.cache_dir) / "SPY_sp100_emergency_*.csv"
        spy_files = glob.glob(str(spy_pattern))
        if spy_files:
            spy_data = pd.read_csv(spy_files[0])
            spy_data['Date'] = pd.to_datetime(spy_data['Date'])
            print("Loaded SPY data")
    
    # Calculate VIX from SPY/QQQ returns (20-day rolling volatility)
    if spy_data is not None:
        spy_data['Returns'] = spy_data['Close'].pct_change()
        spy_data['Volatility'] = spy_data['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
        vix_data = spy_data[['Date', 'Volatility']].copy()
        vix_data.columns = ['Date', 'Close']
        print("Calculated VIX from SPY/QQQ volatility")
    
    if spy_data is None or vix_data is None:
        print("ERROR: Could not load or calculate SPY/VIX data")
        return
    
    # Align data by date
    print("\nAligning data by date...")
    date_range = set(spy_data['Date']).intersection(set(vix_data['Date'])).intersection(set(mmtw_data.index))
    
    spy_data = spy_data[spy_data['Date'].isin(date_range)].sort_values('Date')
    vix_data = vix_data[vix_data['Date'].isin(date_range)].sort_values('Date')
    mmtw_data = mmtw_data[mmtw_data.index.isin(date_range)].sort_index()
    
    print(f"Aligned data for {len(date_range)} trading days")
    
    # Train regime classifier
    print("\nTraining HMM regime classifier...")
    results = regime_classifier.fit(spy_data, vix_data, mmtw_data)
    
    print("\nTraining Results:")
    print(f"  Model saved to: {results['model_path']}")
    print(f"  Training samples: {results['training_samples']}")
    print(f"  State mapping: {results['state_mapping']}")
    
    print("\nState Statistics:")
    for regime, stats in results['state_stats'].items():
        regime_name = regime_classifier.get_regime_name(regime)
        print(f"  {regime_name}:")
        print(f"    Days: {stats['count']} ({stats['percentage']:.1f}%)")
        print(f"    Avg Return: {stats['avg_return']*100:.2f}%")
        print(f"    Avg VIX: {stats['avg_vix']:.2f}")
        print(f"    Avg MMTW: {stats['avg_mmtw']:.3f}")
    
    # Test prediction
    print("\nTesting regime prediction...")
    
    # Get latest data point
    latest_spy_return = spy_data['Close'].pct_change().iloc[-1]
    latest_vix = vix_data['Close'].iloc[-1]
    latest_mmtw = mmtw_data.iloc[-1]
    
    current_state = regime_classifier.predict_state((latest_spy_return, latest_vix, latest_mmtw))
    
    print(f"Current market state: {regime_classifier.get_regime_name(current_state)}")
    print(f"  Latest SPY return: {latest_spy_return*100:.2f}%")
    print(f"  Latest VIX: {latest_vix:.2f}")
    print(f"  Latest MMTW: {latest_mmtw:.3f}")
    
    # Analyze regime history
    print("\nAnalyzing regime history...")
    history = regime_classifier.analyze_regime_history(spy_data, vix_data, mmtw_data)
    
    if not history.empty:
        print(f"Analyzed {len(history)} days of regime history")
        
        # Save results
        output_dir = PROJECT_ROOT / 'reports'
        output_dir.mkdir(exist_ok=True)
        
        # Save regime history
        history_file = output_dir / 'regime_history.csv'
        history.to_csv(history_file, index=False)
        print(f"Regime history saved to: {history_file}")
        
        # Create visualization
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Regime timeline
            plt.subplot(3, 1, 1)
            plt.plot(history['Date'], history['Regime'], 'o-', markersize=2)
            plt.ylabel('Regime')
            plt.title('Market Regime Timeline')
            plt.grid(True)
            
            # Plot 2: VIX with regime colors
            plt.subplot(3, 1, 2)
            colors = {0: 'green', 1: 'orange', 2: 'red'}
            for regime in [0, 1, 2]:
                mask = history['Regime'] == regime
                plt.plot(history[mask]['Date'], history[mask]['VIX'], 
                        'o-', color=colors[regime], label=regime_classifier.get_regime_name(regime), 
                        markersize=2, alpha=0.7)
            plt.ylabel('VIX')
            plt.title('VIX by Market Regime')
            plt.legend()
            plt.grid(True)
            
            # Plot 3: MMTW with regime colors
            plt.subplot(3, 1, 3)
            for regime in [0, 1, 2]:
                mask = history['Regime'] == regime
                plt.plot(history[mask]['Date'], history[mask]['MMTW'], 
                        'o-', color=colors[regime], label=regime_classifier.get_regime_name(regime), 
                        markersize=2, alpha=0.7)
            plt.ylabel('MMTW')
            plt.title('Market Breadth by Regime')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / 'regime_analysis.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"Regime analysis plot saved to: {plot_file}")
            
        except Exception as e:
            print(f"Failed to create visualization: {e}")
    
    print("\n" + "=" * 80)
    print("MARKET REGIME CLASSIFIER TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
