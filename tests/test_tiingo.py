import os
import requests
import pandas as pd
from datetime import datetime, timedelta

def test_tiingo_direct():
    """Test Tiingo API directly"""
    api_key = '72e14af10f4c32db4a7631275929617481aed281'
    
    ticker = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20*365)
    
    print(f"Testing Tiingo API for {ticker}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    # Test API endpoint
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {api_key}'
    }
    
    params = {
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'resampleFreq': 'daily',
        'columns': 'open,high,low,close,volume,adjClose,adjHigh,adjLow,adjOpen,adjVolume'
    }
    
    try:
        print("Making API request...")
        response = requests.get(url, headers=headers, params=params)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Success! Retrieved {len(data)} data points")
            
            if data:
                # Convert to DataFrame
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Standardize columns
                df.columns = [col.replace('adj', '') for col in df.columns]
                
                print(f"\nData Summary:")
                print(f"  Date Range: {df.index.min()} to {df.index.max()}")
                print(f"  Total Days: {len(df)}")
                print(f"  Years: {(df.index.max() - df.index.min()).days / 365.25:.1f}")
                print(f"  Price Range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
                print(f"  Mean Price: ${df['close'].mean():.2f}")
                print(f"  Mean Volume: {df['volume'].mean():,.0f}")
                
                # Save to cache
                cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache', 'tiingo')
                os.makedirs(cache_dir, exist_ok=True)
                
                filename = f"{ticker}_1d_20y.csv"
                filepath = os.path.join(cache_dir, filename)
                
                df.to_csv(filepath)
                print(f"\n✓ Saved to: {filepath}")
                
                # Show first few rows
                print(f"\nFirst 5 rows:")
                print(df.head())
                
                return df
            else:
                print("✗ No data returned")
                return None
        else:
            print(f"✗ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

if __name__ == "__main__":
    test_tiingo_direct()
