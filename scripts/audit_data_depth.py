#!/usr/bin/env python3
"""
NeuralTrader 2.0 - Data Depth Audit
=====================================

Comprehensive audit of historical data depth and continuity.
Verifies if we truly have 50+ years of history across tickers.

Features:
- Sample Audit: Top 10 tickers by row count
- Range Check: Start/End dates and total rows for every ticker
- 50-Year Flag: Highlights tickers with data before 1980-01-01
- Gap Detection: Finds time jumps >5 days (excluding weekends/holidays)
- Summary Table: Counts by data length categories (>10, >30, >50 years)

Usage:
    python scripts/audit_data_depth.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralTrader.DataDepthAudit")

class DataDepthAuditor:
    """
    Comprehensive auditor for historical data depth and continuity.
    """
    
    def __init__(self, raw_path: str = "data/raw"):
        self.raw_path = Path(raw_path)
        
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_path}")
        
        logger.info(f"DataDepthAuditor initialized")
        logger.info(f"Raw data path: {self.raw_path}")
    
    def load_all_ticker_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load data for all tickers in data/raw/.
        
        Returns:
            Dictionary mapping ticker names to DataFrames
        """
        logger.info("Loading all ticker data...")
        
        # Get all parquet files
        parquet_files = list(self.raw_path.glob("*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError("No parquet files found in data/raw/")
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        # Load all data
        ticker_data = {}
        
        for file_path in parquet_files:
            try:
                ticker = file_path.stem
                df = pd.read_parquet(file_path)
                
                # Convert date column and sort
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                ticker_data[ticker] = df
                logger.debug(f"Loaded {ticker}: {len(df)} rows")
                
            except Exception as e:
                logger.warning(f"Error loading {file_path.name}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(ticker_data)} tickers")
        return ticker_data
    
    def sample_audit(self, ticker_data: Dict[str, pd.DataFrame], top_n: int = 10) -> pd.DataFrame:
        """
        Find the top N tickers with the most rows.
        
        Args:
            ticker_data: Dictionary of ticker DataFrames
            top_n: Number of top tickers to return
            
        Returns:
            DataFrame with top N tickers by row count
        """
        logger.info(f"Performing sample audit - Top {top_n} tickers by row count...")
        
        # Count rows for each ticker
        row_counts = []
        for ticker, df in ticker_data.items():
            row_counts.append({
                'ticker': ticker,
                'rows': len(df),
                'start_date': df['date'].min(),
                'end_date': df['date'].max()
            })
        
        # Sort by row count and get top N
        df_counts = pd.DataFrame(row_counts)
        df_counts = df_counts.sort_values('rows', ascending=False).head(top_n)
        
        logger.info(f"Top {top_n} tickers by row count:")
        for _, row in df_counts.iterrows():
            years = (row['end_date'] - row['start_date']).days / 365.25
            logger.info(f"  {row['ticker']}: {row['rows']:,} rows ({years:.1f} years)")
        
        return df_counts
    
    def range_check(self, ticker_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Check date range and row count for every ticker.
        
        Args:
            ticker_data: Dictionary of ticker DataFrames
            
        Returns:
            DataFrame with range information for all tickers
        """
        logger.info("Performing range check for all tickers...")
        
        range_data = []
        
        for ticker, df in ticker_data.items():
            start_date = df['date'].min()
            end_date = df['date'].max()
            total_rows = len(df)
            years = (end_date - start_date).days / 365.25
            
            # Check if ticker has 50+ years of data
            is_50_year = start_date < pd.Timestamp('1980-01-01')
            
            range_data.append({
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'total_rows': total_rows,
                'years': years,
                'is_50_year': is_50_year
            })
        
        df_range = pd.DataFrame(range_data)
        df_range = df_range.sort_values('years', ascending=False)
        
        logger.info(f"Range check completed for {len(df_range)} tickers")
        return df_range
    
    def detect_gaps(self, df: pd.DataFrame, ticker: str) -> List[Dict]:
        """
        Detect gaps in time series greater than 5 days.
        
        Args:
            df: DataFrame with date column
            ticker: Ticker name for logging
            
        Returns:
            List of gap dictionaries
        """
        gaps = []
        
        # Sort by date
        df_sorted = df.sort_values('date')
        
        for i in range(1, len(df_sorted)):
            prev_date = df_sorted.iloc[i-1]['date']
            curr_date = df_sorted.iloc[i]['date']
            
            # Calculate days difference
            days_diff = (curr_date - prev_date).days
            
            # Check if gap > 5 days
            if days_diff > 5:
                # Check if it's a weekend/holiday gap
                # Weekends typically have 1-3 day gaps
                # Holidays can have 1-4 day gaps
                # Gaps > 5 days are suspicious
                
                # Count weekends in the gap
                weekend_days = 0
                check_date = prev_date + timedelta(days=1)
                while check_date < curr_date:
                    if check_date.weekday() >= 5:  # Saturday (5) or Sunday (6)
                        weekend_days += 1
                    check_date += timedelta(days=1)
                
                # If gap is much larger than expected, flag it
                if days_diff > weekend_days + 2:  # Allow 2 extra days for holidays
                    gaps.append({
                        'ticker': ticker,
                        'gap_start': prev_date,
                        'gap_end': curr_date,
                        'gap_days': days_diff,
                        'weekend_days': weekend_days,
                        'suspicious': True
                    })
        
        return gaps
    
    def gap_detection(self, ticker_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Perform gap detection for all tickers.
        
        Args:
            ticker_data: Dictionary of ticker DataFrames
            
        Returns:
            DataFrame with all detected gaps
        """
        logger.info("Performing gap detection for all tickers...")
        
        all_gaps = []
        
        for ticker, df in ticker_data.items():
            gaps = self.detect_gaps(df, ticker)
            all_gaps.extend(gaps)
        
        if all_gaps:
            df_gaps = pd.DataFrame(all_gaps)
            df_gaps = df_gaps.sort_values('gap_days', ascending=False)
            
            logger.info(f"Found {len(df_gaps)} suspicious gaps across all tickers")
            
            # Log top 10 largest gaps
            logger.info("Top 10 largest gaps:")
            for _, gap in df_gaps.head(10).iterrows():
                logger.info(f"  {gap['ticker']}: {gap['gap_days']} days "
                          f"({gap['gap_start'].date()} to {gap['gap_end'].date()})")
        else:
            df_gaps = pd.DataFrame()
            logger.info("No suspicious gaps detected")
        
        return df_gaps
    
    def generate_summary_table(self, df_range: pd.DataFrame) -> Dict[str, int]:
        """
        Generate summary counts by data length categories.
        
        Args:
            df_range: DataFrame with range information
            
        Returns:
            Dictionary with summary counts
        """
        logger.info("Generating summary table...")
        
        # Count by data length categories
        summary = {
            'total_tickers': len(df_range),
            '>10_years': len(df_range[df_range['years'] > 10]),
            '>30_years': len(df_range[df_range['years'] > 30]),
            '>50_years': len(df_range[df_range['years'] > 50]),
            '50_year_flag': len(df_range[df_range['is_50_year'] == True])
        }
        
        # Calculate percentages
        total = summary['total_tickers']
        if total > 0:
            summary['>10_years_pct'] = (summary['>10_years'] / total) * 100
            summary['>30_years_pct'] = (summary['>30_years'] / total) * 100
            summary['>50_years_pct'] = (summary['>50_years'] / total) * 100
            summary['50_year_flag_pct'] = (summary['50_year_flag'] / total) * 100
        else:
            summary['>10_years_pct'] = 0
            summary['>30_years_pct'] = 0
            summary['>50_years_pct'] = 0
            summary['50_year_flag_pct'] = 0
        
        return summary
    
    def run_audit(self) -> None:
        """Run the complete data depth audit."""
        logger.info("Starting NeuralTrader 2.0 Data Depth Audit...")
        logger.info("=" * 60)
        
        try:
            # Load all ticker data
            ticker_data = self.load_all_ticker_data()
            
            # Sample Audit: Top 10 tickers by row count
            logger.info("\n" + "=" * 60)
            logger.info("SAMPLE AUDIT: Top 10 Tickers by Row Count")
            logger.info("=" * 60)
            sample_audit = self.sample_audit(ticker_data, top_n=10)
            
            # Range Check: Start/End dates and row counts
            logger.info("\n" + "=" * 60)
            logger.info("RANGE CHECK: All Tickers")
            logger.info("=" * 60)
            range_check = self.range_check(ticker_data)
            
            # Print range check results
            logger.info(f"\nRange Check Results ({len(range_check)} tickers):")
            logger.info("-" * 80)
            logger.info(f"{'Ticker':<10} {'Start Date':<12} {'End Date':<12} {'Rows':<8} {'Years':<8} {'50+ Year':<8}")
            logger.info("-" * 80)
            
            for _, row in range_check.iterrows():
                fifty_year_flag = "YES" if row['is_50_year'] else "NO"
                logger.info(f"{row['ticker']:<10} {row['start_date'].strftime('%Y-%m-%d'):<12} "
                          f"{row['end_date'].strftime('%Y-%m-%d'):<12} {row['total_rows']:<8} "
                          f"{row['years']:.1f}{'':<7} {fifty_year_flag:<8}")
            
            # 50-Year Flag: Highlight tickers with data before 1980
            fifty_year_tickers = range_check[range_check['is_50_year'] == True]
            if not fifty_year_tickers.empty:
                logger.info("\n" + "=" * 60)
                logger.info("50-YEAR FLAG: Tickers with Data Before 1980-01-01")
                logger.info("=" * 60)
                logger.info(f"Found {len(fifty_year_tickers)} tickers with 50+ years of data:")
                for _, row in fifty_year_tickers.iterrows():
                    years = row['years']
                    logger.info(f"  {row['ticker']}: {years:.1f} years "
                              f"({row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')})")
            else:
                logger.info("\n" + "=" * 60)
                logger.info("50-YEAR FLAG: No tickers with data before 1980-01-01")
                logger.info("=" * 60)
            
            # Gap Detection: Find time jumps >5 days
            logger.info("\n" + "=" * 60)
            logger.info("GAP DETECTION: Time Jumps >5 Days")
            logger.info("=" * 60)
            gap_detection = self.gap_detection(ticker_data)
            
            # Summary Table: Counts by data length
            logger.info("\n" + "=" * 60)
            logger.info("SUMMARY TABLE: Data Length Categories")
            logger.info("=" * 60)
            summary = self.generate_summary_table(range_check)
            
            logger.info(f"Total Tickers: {summary['total_tickers']}")
            logger.info(f">10 Years: {summary['>10_years']} ({summary['>10_years_pct']:.1f}%)")
            logger.info(f">30 Years: {summary['>30_years']} ({summary['>30_years_pct']:.1f}%)")
            logger.info(f">50 Years: {summary['>50_years']} ({summary['>50_years_pct']:.1f}%)")
            logger.info(f"50-Year Flag: {summary['50_year_flag']} ({summary['50_year_flag_pct']:.1f}%)")
            
            # Overall assessment
            logger.info("\n" + "=" * 60)
            logger.info("OVERALL ASSESSMENT")
            logger.info("=" * 60)
            
            if summary['>50_years'] > 0:
                logger.info("✅ SUCCESS: Found tickers with 50+ years of historical data")
            else:
                logger.info("⚠️  WARNING: No tickers with 50+ years of data found")
            
            if summary['>30_years'] >= summary['total_tickers'] * 0.5:
                logger.info("✅ GOOD: Majority of tickers have 30+ years of data")
            else:
                logger.info("⚠️  LIMITED: Less than 50% of tickers have 30+ years of data")
            
            if summary['>10_years'] >= summary['total_tickers'] * 0.8:
                logger.info("✅ EXCELLENT: Most tickers have 10+ years of data")
            else:
                logger.info("⚠️  CONCERN: Less than 80% of tickers have 10+ years of data")
            
            if len(gap_detection) == 0:
                logger.info("✅ CLEAN: No suspicious gaps detected in time series")
            else:
                logger.info(f"⚠️  GAPS: Found {len(gap_detection)} suspicious gaps in data")
            
            logger.info("\n" + "=" * 60)
            logger.info("DATA DEPTH AUDIT COMPLETED")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            raise

def main():
    """Main entry point."""
    auditor = DataDepthAuditor()
    auditor.run_audit()

if __name__ == "__main__":
    main()
