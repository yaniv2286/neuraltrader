#!/usr/bin/env python3
"""
Data Landscape Audit - Find the true 60-year historical data
"""

import pandas as pd
from pathlib import Path

print('🦅 DATA LANDSCAPE AUDIT')
print('=' * 60)

# Define search locations
locations = [
    'd:/GitHub/NeuralTrader/data/raw',
    'd:/GitHub/NeuralTrader/data/processed', 
    'd:/GitHub/NeuralTrader/data/cache/tiingo'
]

# Target ticker for deep time analysis
target_ticker = 'AAPL'

results = []

for location in locations:
    print(f'\n📁 Scanning: {location}')
    path = Path(location)
    
    if path.exists():
        # Find AAPL files
        aapl_files = list(path.glob('*AAPL*')) + list(path.glob('*aapl*'))
        print(f'  Found {len(aapl_files)} AAPL files')
        
        for f in aapl_files:
            try:
                df = pd.read_csv(f)
                
                # Find date column
                date_col = None
                for col in ['Date', 'date', 'DATE']:
                    if col in df.columns:
                        date_col = col
                        break
                
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    earliest = df[date_col].min().date()
                    latest = df[date_col].max().date()
                    years = (latest - earliest).days / 365.25
                    
                    results.append({
                        'ticker': target_ticker,
                        'folder': location,
                        'file': f.name,
                        'years': years,
                        'rows': len(df),
                        'earliest': earliest,
                        'latest': latest,
                        'columns': list(df.columns)
                    })
                    
                    print(f'    {f.name}: {earliest} to {latest} ({years:.1f} years, {len(df)} rows)')
                    print(f'      Columns: {list(df.columns)[:5]}...')
                else:
                    print(f'    {f.name}: No date column found')
                    
            except Exception as e:
                print(f'    Error reading {f.name}: {e}')
    else:
        print(f'  Location does not exist')

# Summary table
print('\n' + '=' * 60)
print('📊 SUMMARY TABLE')
print('=' * 60)

if results:
    print(f'{"TICKER":<10} {"FOLDER":<25} {"YEARS":<8} {"ROWS":<8}')
    print('-' * 60)
    
    for result in sorted(results, key=lambda x: x['years'], reverse=True):
        folder_short = result['folder'].split('/')[-1]
        print(f'{result["ticker"]:<10} {folder_short:<25} {result["years"]:>7.1f} {result["rows"]:>7}')
    
    # Find the deepest time data
    deepest = max(results, key=lambda x: x['years'])
    print(f'\n🎯 DEEPEST TIME DATA: {deepest["folder"]}')
    print(f'   File: {deepest["file"]}')
    print(f'   Coverage: {deepest["earliest"]} to {deepest["latest"]}')
    print(f'   Duration: {deepest["years"]:.1f} years')
    
    # Check for 60-year data
    if deepest['years'] > 50:
        print(f'\n✅ FOUND 60-YEAR DATA!')
        print(f'   Location: {deepest["folder"]}')
        print(f'   This should be the primary data source')
    else:
        print(f'\n❌ NO 60-YEAR DATA FOUND')
        print(f'   Deepest coverage: {deepest["years"]:.1f} years')
else:
    print('❌ No AAPL data found')

print('\n' + '=' * 60)
