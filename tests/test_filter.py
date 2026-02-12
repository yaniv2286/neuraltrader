#!/usr/bin/env python3

import pandas as pd
from core.data_loader import ModernEraDataLoader

# Test with a single ticker to verify filter
print("Testing Modern Era Filter on AAON...")
df = pd.read_parquet('data/raw/AAON.parquet')
print(f'Original AAON data: {len(df)} rows, date range: {df["date"].min()} to {df["date"].max()}')

# Apply filter manually
df['date'] = pd.to_datetime(df['date'])
filtered = df[df['date'] >= '2000-01-01']
print(f'Filtered AAON data: {len(filtered)} rows, date range: {filtered["date"].min()} to {filtered["date"].max()}')
print(f'Reduction: {((len(df) - len(filtered)) / len(df)) * 100:.1f}%')

# Test with the data loader
print("\nTesting with ModernEraDataLoader...")
loader = ModernEraDataLoader()
single_data = loader._load_raw_parquet_files()
if 'AAON' in single_data:
    aaon_filtered = loader._filter_single_ticker(single_data['AAON'], 'AAON')
    print(f'DataLoader AAON result: {len(aaon_filtered)} rows')
else:
    print("AAON not found in raw data")
