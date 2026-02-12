#!/usr/bin/env python3
"""
Build Universe - Dynamically consolidate all CSV data into production parquet file
Creates the modern_era_universe.parquet file from ALL cached Tiingo CSV data
"""

import os
import pandas as pd
import glob
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def extract_ticker_from_filename(filename: str) -> str:
    """Extract ticker symbol from filename"""
    # Handle patterns like: AAPL_1d_full_20260212.csv, AAPL_sp100_emergency_20260212.csv
    basename = os.path.basename(filename)
    
    # Split on first underscore to get ticker
    parts = basename.split('_')
    if parts:
        return parts[0]
    else:
        # Fallback: remove extension and return full name
        return basename.replace('.csv', '')

def load_csv_with_ticker(file_path: str, ticker: str) -> pd.DataFrame:
    """Load CSV file and add ticker column"""
    try:
        df = pd.read_csv(file_path)
        
        # Standardize date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.rename(columns={'date': 'Date'}, inplace=True)
        else:
            # Try to use index as date
            df.index = pd.to_datetime(df.index)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Date'}, inplace=True)
        
        # Add ticker column
        df['ticker'] = ticker
        
        logger.info(f"Loaded {ticker}: {len(df)} records from {os.path.basename(file_path)}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise

def build_universe():
    """Build the universe parquet file from ALL cached CSV data"""
    logger.info("Building universe from cached CSV data...")
    
    # Define cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache', 'tiingo')
    
    # Verify cache directory exists
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
    
    # Scan for ALL CSV files in cache directory
    csv_files = glob.glob(os.path.join(cache_dir, '*.csv'))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {cache_dir}")
    
    logger.info(f"Found {len(csv_files)} CSV files in cache directory")
    
    # Load all ticker data dynamically
    all_data = []
    ticker_stats = {}
    
    for csv_file in csv_files:
        try:
            # Extract ticker from filename
            ticker = extract_ticker_from_filename(csv_file)
            
            # Skip backup files
            if ticker.endswith('.bak'):
                logger.debug(f"Skipping backup file: {csv_file}")
                continue
            
            # Load and process
            df = load_csv_with_ticker(csv_file, ticker)
            all_data.append(df)
            
            # Track statistics
            ticker_stats[ticker] = len(df)
            
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data files found to build universe")
    
    # Concatenate all data
    logger.info("Concatenating data...")
    universe_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date and ticker
    universe_df = universe_df.sort_values(['Date', 'ticker']).reset_index(drop=True)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to parquet
    output_file = os.path.join(output_dir, 'modern_era_universe.parquet')
    universe_df.to_parquet(output_file, index=False)
    
    # Log success
    logger.info(f"[PASS] Universe built with {len(universe_df)} records")
    logger.info(f"Output saved to: {output_file}")
    
    # Final Report
    logger.info("=" * 50)
    logger.info("FINAL REPORT")
    logger.info("=" * 50)
    logger.info(f"Total tickers found: {len(ticker_stats)}")
    logger.info(f"Total rows in universe: {len(universe_df):,}")
    
    # Top 10 tickers by record count
    sorted_tickers = sorted(ticker_stats.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top 10 tickers by record count:")
    for i, (ticker, count) in enumerate(sorted_tickers[:10], 1):
        logger.info(f"  {i:2d}. {ticker}: {count:,} records")
    
    # Date range
    date_range = f"{universe_df['Date'].min().strftime('%Y-%m-%d')} to {universe_df['Date'].max().strftime('%Y-%m-%d')}"
    logger.info(f"Date range: {date_range}")
    logger.info("=" * 50)
    
    return output_file

def main():
    """Main entry point"""
    try:
        output_file = build_universe()
        print(f"✅ Universe successfully built: {output_file}")
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to build universe: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
