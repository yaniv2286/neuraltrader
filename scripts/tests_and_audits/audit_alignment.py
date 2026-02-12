#!/usr/bin/env python3
"""
NeuralTrader Architectural Alignment Audit
"""

import pandas as pd
import os
from pathlib import Path
import json

print('🦅 NEURALTRADER ARCHITECTURAL ALIGNMENT AUDIT')
print('=' * 60)

# TASK 1: Data Inventory Audit
print('\n📊 TASK 1: DATA INVENTORY AUDIT')
cache_dir = Path('d:/GitHub/NeuralTrader/data/cache/tiingo')
if cache_dir.exists():
    files = list(cache_dir.glob('*.csv'))
    print(f'Found {len(files)} CSV files in cache')
    
    ticker_data = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            ticker = f.stem.split('_')[0]  # Extract ticker from filename
            
            # Get date range and count
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                earliest = df['Date'].min().date()
                latest = df['Date'].max().date()
                count = len(df)
                
                # Check for key columns
                has_adj_close = 'Adj Close' in df.columns
                has_price = 'Price' in df.columns
                
                ticker_data[ticker] = {
                    'earliest': earliest,
                    'latest': latest,
                    'count': count,
                    'has_adj_close': has_adj_close,
                    'has_price': has_price
                }
        except Exception as e:
            print(f'  Error reading {f.name}: {e}')
    
    # Find earliest and latest across all tickers
    if ticker_data:
        earliest_overall = min(data['earliest'] for data in ticker_data.values())
        latest_overall = max(data['latest'] for data in ticker_data.values())
        
        print(f'\n📅 Deep Time Coverage: {earliest_overall} to {latest_overall}')
        print(f'   Total Years: {(latest_overall - earliest_overall).days / 365.25:.1f} years')
        
        # Check for specific tickers
        print('\n🔍 KEY TICKER ANALYSIS:')
        for ticker in ['SPY', 'GSPC']:
            if ticker in ticker_data:
                data = ticker_data[ticker]
                print(f'  {ticker}: {data["earliest"]} to {data["latest"]} ({data["count"]} rows)')
                print(f'    Adj Close: {data["has_adj_close"]}, Price: {data["has_price"]}')
            else:
                print(f'  {ticker}: NOT FOUND')
        
        # Show first 10 tickers
        print('\n📋 FIRST 10 TICKERS:')
        for i, (ticker, data) in enumerate(sorted(ticker_data.items())[:10]):
            print(f'  {i+1:2d}. {ticker:6s}: {data["earliest"]} ({data["count"]:5d} rows)')
    else:
        print('No valid ticker data found')
else:
    print('Cache directory not found')

# TASK 2: Logic Path Verification
print('\n🔍 TASK 2: LOGIC PATH VERIFICATION')

# Check ai_models.py for BUY signal logic
print('\n📝 Checking BUY signal logic in core/ai_models.py...')
ai_models_path = Path('d:/GitHub/NeuralTrader/core/ai_models.py')
if ai_models_path.exists():
    with open(ai_models_path, 'r') as f:
        content = f.read()
        if 'threshold: Optional[float]' in content:
            print('✅ Dynamic threshold parameter found')
        if 'weighted_prob_up > threshold' in content:
            print('✅ Dynamic threshold comparison found')
        if 'weighted_prob_up > 0.70' in content:
            print('❌ OLD 0.70 threshold still present')
        else:
            print('✅ No hard-coded 0.70 threshold found')
else:
    print('❌ ai_models.py not found')

# Check optimize_strategy.py for regime filter
print('\n📝 Checking Regime Filter in scripts/optimize_strategy.py...')
optimize_path = Path('d:/GitHub/NeuralTrader/scripts/optimize_strategy.py')
if optimize_path.exists():
    with open(optimize_path, 'r') as f:
        content = f.read()
        if 'regime_dict.get(date_str' in content:
            print('✅ Regime dictionary lookup with string keys found')
        if 'market_state == 2' in content:
            print('✅ RED state hard stop found')
        if 'effective_threshold = 1.0' in content:
            print('✅ Force cash logic found')
else:
    print('❌ optimize_strategy.py not found')

# TASK 3: Feature Scaler Alignment
print('\n🔧 TASK 3: FEATURE SCALER ALIGNMENT')

# Check ensemble metadata
metadata_path = Path('d:/GitHub/NeuralTrader/models/ensemble_metadata.json')
if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        print(f'✅ Ensemble metadata found')
        print(f'   Features: {len(metadata.get("feature_names", []))}')
        print(f'   Training samples: {metadata.get("n_samples", "N/A")}')
        
        # Check for raw price features
        feature_names = metadata.get('feature_names', [])
        raw_price_features = [f for f in feature_names if f in ['close', 'open', 'high', 'low', 'adjClose']]
        if raw_price_features:
            print(f'❌ Raw price features found: {raw_price_features}')
        else:
            print('✅ No raw price features found')
        
        # Check for normalized features
        normalized_features = [f for f in feature_names if 'ratio' in f or 'return' in f or 'pct' in f]
        print(f'✅ Normalized features found: {len(normalized_features)}')
else:
    print('❌ Ensemble metadata not found')

# Check feature engineer
feature_engineer_path = Path('d:/GitHub/NeuralTrader/core/feature_engineer.py')
if feature_engineer_path.exists():
    with open(feature_engineer_path, 'r') as f:
        content = f.read()
        if 'FEATURE NORMALIZATION' in content:
            print('✅ Feature normalization implemented')
        if 'pct_change' in content:
            print('✅ Percentage returns implemented')
        if 'ratio' in content:
            print('✅ Price ratios implemented')
else:
    print('❌ feature_engineer.py not found')

print('\n' + '=' * 60)
print('📋 SUMMARY REPORT')
print('=' * 60)

# Create summary
summary = {
    'deep_time_start': earliest_overall if ticker_data else None,
    'deep_time_end': latest_overall if ticker_data else None,
    'total_tickers': len(ticker_data) if ticker_data else 0,
    'spy_available': 'SPY' in ticker_data,
    'gspc_available': 'GSPC' in ticker_data,
    'dynamic_threshold': 'threshold: Optional[float]' in content if optimize_path.exists() else False,
    'regime_filter': 'regime_dict.get' in content if optimize_path.exists() else False,
    'feature_normalization': 'FEATURE NORMALIZATION' in content if feature_engineer_path.exists() else False
}

print(f"Deep Time Start: {summary['deep_time_start']}")
print(f"Deep Time End: {summary['deep_time_end']}")
print(f"Total Tickers: {summary['total_tickers']}")
print(f"SPY Available: {summary['spy_available']}")
print(f"GSPC Available: {summary['gspc_available']}")
print(f"Dynamic Threshold: {summary['dynamic_threshold']}")
print(f"Regime Filter: {summary['regime_filter']}")
print(f"Feature Normalization: {summary['feature_normalization']}")

# Save to docs/STATUS_SYNC.md
status_path = Path('d:/GitHub/NeuralTrader/docs/STATUS_SYNC.md')
with open(status_path, 'w') as f:
    f.write('# NeuralTrader Status Sync\n\n')
    f.write('## Architectural Alignment Audit Results\n\n')
    f.write(f'- **Deep Time Coverage**: {summary["deep_time_start"]} to {summary["deep_time_end"]}\n')
    f.write(f'- **Total Tickers**: {summary["total_tickers"]}\n')
    f.write(f'- **SPY Available**: {summary["spy_available"]}\n')
    f.write(f'- **GSPC Available**: {summary["gspc_available"]}\n')
    f.write(f'- **Dynamic Threshold**: {summary["dynamic_threshold"]}\n')
    f.write(f'- **Regime Filter**: {summary["regime_filter"]}\n')
    f.write(f'- **Feature Normalization**: {summary["feature_normalization"]}\n')

print(f'\n📄 Report saved to: {status_path}')
