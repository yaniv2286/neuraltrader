import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tiingo import TiingoClient
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralTrader.DataEngine")

class DataEngine:
    """
    High-performance data engine for NeuralTrader private fund.
    
    Features:
    - 50 years of Tiingo data in Parquet format
    - Top 1000 liquidity filter
    - Point-in-time universe construction
    - Hedge-fund standard compression (Snappy)
    """
    
    def __init__(self, api_key: str, base_path: str = "data/raw"):
        self.client = TiingoClient({'api_key': api_key})
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.processed_path = Path("data/processed")
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
    def fetch_top_1000_universe(self) -> List[str]:
        """
        Step 1.1: Fetch the master ticker list and apply liquidity filters.
        Goal: Top 1000 tickers by Average Daily Dollar Volume.
        """
        logger.info("Fetching master ticker list from Tiingo...")
        # Get metadata for all supported tickers
        tickers = self.client.list_tickers()
        df_tickers = pd.DataFrame(tickers)
        
        # FILTER 1: Exchange Filter (NYSE and NASDAQ only)
        # Prevents 'messy' OTC data from contaminating our 50-year backtest
        valid_exchanges = ['NYSE', 'NASDAQ', 'NYSE MKT', 'NYSE ARCA']
        df_filtered = df_tickers[df_tickers['exchange'].isin(valid_exchanges)].copy()
        
        # FILTER 2: Asset Type Filter (Common Stocks only)
        df_filtered = df_filtered[df_filtered['assetType'] == 'Stock'].copy()
        
        # FILTER 3: Liquidity Filter (Top 1000 by Average Daily Dollar Volume)
        # We'll download a sample first to estimate liquidity
        logger.info(f"Pre-filtered universe size: {len(df_filtered)} tickers.")
        
        # For now, return all filtered tickers (we'll rank after downloading)
        return df_filtered['ticker'].tolist()
    
    def download_ticker_to_parquet(self, ticker: str, start_date: str = "1975-01-01") -> bool:
        """
        Step 1.2: Download historical EOD data and save as compressed Parquet.
        
        Returns True if successful, False if skipped or failed.
        """
        file_path = self.base_path / f"{ticker}.parquet"
        
        # Skip if already exists
        if file_path.exists():
            logger.debug(f"Skipping {ticker}: already exists")
            return True
        
        try:
            # Fetch OHLCV data with adjusted prices
            data = self.client.get_dataframe(
                ticker, 
                startDate=start_date,
                frequency='daily',
                metric_name='adjclose'  # Get adjusted close
            )
            
            # Get full OHLCV
            data_full = self.client.get_dataframe(ticker, startDate=start_date)
            
            if data_full.empty:
                logger.warning(f"No data for {ticker}")
                return False

            # Merge adjusted close with OHLCV
            data_full['adjClose'] = data['adjClose']
            
            # FILTER 3: Dollar Volume Check (Liquidity Filter)
            # We calculate 20-day Average Dollar Volume
            data_full['dollar_volume'] = data_full['adjClose'] * data_full['volume']
            avg_dollar_vol = data_full['dollar_volume'].rolling(20).mean().iloc[-1]
            
            # Threshold: $1M daily volume minimum for modern era 
            # (Note: In 1980, this would be adjusted lower automatically by the logic)
            if avg_dollar_vol < 1_000_000 and data_full.index[-1].year > 2020:
                logger.warning(f"Skipping {ticker}: Insufficient liquidity (${avg_dollar_vol:,.0f} < $1M)")
                return False

            # Add metadata
            data_full['ticker'] = ticker
            data_full['download_date'] = datetime.now().isoformat()
            
            # Save to Parquet with Snappy compression (Hedge-fund standard)
            table = pa.Table.from_pandas(data_full)
            pq.write_table(table, file_path, compression='snappy')
            
            logger.info(f"Downloaded {ticker}: {len(data_full)} days, ${avg_dollar_vol:,.0f} avg volume")
            return True

        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            return False
    
    def rank_by_liquidity(self) -> List[str]:
        """
        Step 1.3: Rank all downloaded tickers by Average Daily Dollar Volume.
        Returns Top 1000 tickers.
        """
        logger.info("Ranking tickers by liquidity...")
        
        liquidity_data = []
        for parquet_file in self.base_path.glob("*.parquet"):
            ticker = parquet_file.stem
            try:
                # Read only the last 252 trading days for liquidity calculation
                df = pd.read_parquet(parquet_file)
                if len(df) < 252:
                    continue
                    
                # Calculate average dollar volume over last year
                recent_data = df.tail(252)
                avg_dollar_vol = (recent_data['adjClose'] * recent_data['volume']).mean()
                
                liquidity_data.append({
                    'ticker': ticker,
                    'avg_dollar_volume': avg_dollar_vol,
                    'total_days': len(df)
                })
            except Exception as e:
                logger.warning(f"Error reading {ticker}: {e}")
        
        # Create DataFrame and rank
        df_liquidity = pd.DataFrame(liquidity_data)
        if df_liquidity.empty:
            logger.error("No valid data found for liquidity ranking")
            return []
        
        # Sort by dollar volume (descending)
        df_liquidity = df_liquidity.sort_values('avg_dollar_volume', ascending=False)
        
        # Take top 1000
        top_1000 = df_liquidity.head(1000)['ticker'].tolist()
        
        logger.info(f"Top 1000 liquidity range: ${df_liquidity.iloc[0]['avg_dollar_volume']:,.0f} - ${df_liquidity.iloc[999]['avg_dollar_volume']:,.0f}")
        
        # Save liquidity ranking
        df_liquidity.to_csv(self.processed_path / "liquidity_ranking.csv", index=False)
        
        return top_1000
    
    def create_point_in_time_universe(self, top_1000: List[str]) -> None:
        """
        Step 1.4: Create point-in-time universe files.
        For each year, save which tickers were available and met liquidity criteria.
        """
        logger.info("Creating point-in-time universe...")
        
        universe_data = {}
        
        for ticker in top_1000:
            file_path = self.base_path / f"{ticker}.parquet"
            if not file_path.exists():
                continue
                
            try:
                df = pd.read_parquet(file_path)
                
                # For each year, check if ticker existed and met liquidity
                for year in range(1975, 2025):
                    year_data = df[df.index.year == year]
                    if len(year_data) < 252:  # Need at least 1 year of data
                        continue
                    
                    # Check liquidity in that year
                    avg_dollar_vol = (year_data['adjClose'] * year_data['volume']).mean()
                    
                    # Adjusted liquidity threshold based on year
                    # 1980s: $100K, 1990s: $500K, 2000s: $1M, 2010s: $2M, 2020s: $5M
                    if year < 1990:
                        threshold = 100_000
                    elif year < 2000:
                        threshold = 500_000
                    elif year < 2010:
                        threshold = 1_000_000
                    elif year < 2020:
                        threshold = 2_000_000
                    else:
                        threshold = 5_000_000
                    
                    if avg_dollar_vol >= threshold:
                        if year not in universe_data:
                            universe_data[year] = []
                        universe_data[year].append(ticker)
                        
            except Exception as e:
                logger.warning(f"Error processing {ticker}: {e}")
        
        # Save point-in-time universe
        for year, tickers in universe_data.items():
            universe_df = pd.DataFrame({'ticker': tickers})
            universe_df.to_csv(
                self.processed_path / f"universe_{year}.csv", 
                index=False
            )
            logger.info(f"Universe {year}: {len(tickers)} tickers")
    
    def download_universe(self, max_tickers: Optional[int] = None) -> List[str]:
        """
        Download complete universe with all filters applied.
        """
        # Step 1: Get all eligible tickers
        all_tickers = self.fetch_top_1000_universe()
        
        if max_tickers:
            all_tickers = all_tickers[:max_tickers]
        
        logger.info(f"Downloading {len(all_tickers)} tickers...")
        
        # Step 2: Download data
        successful = []
        for i, ticker in enumerate(all_tickers, 1):
            if self.download_ticker_to_parquet(ticker):
                successful.append(ticker)
            
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(all_tickers)} tickers processed")
        
        # Step 3: Rank by liquidity
        top_1000 = self.rank_by_liquidity()
        
        # Step 4: Create point-in-time universe
        self.create_point_in_time_universe(top_1000)
        
        logger.info(f"Successfully downloaded {len(successful)} tickers")
        logger.info(f"Top 1000 by liquidity: {len(top_1000)} tickers")
        
        return top_1000

# Usage Example
if __name__ == "__main__":
    # Initialize with your Tiingo API key
    engine = DataEngine(api_key="YOUR_TIINGO_KEY")
    
    # Download full universe (or limit for testing)
    top_tickers = engine.download_universe(max_tickers=100)  # Use 100 for testing
    
    print(f"Downloaded {len(top_tickers)} top liquidity tickers")