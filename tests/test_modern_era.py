#!/usr/bin/env python3

from core.data_loader import ModernEraDataLoader

def test_modern_era():
    print("🦅 TESTING MODERN ERA DATA LOADER...")
    
    loader = ModernEraDataLoader()
    data = loader.load_modern_era_universe()
    
    print(f"✅ Modern Era Data Loader Test Results:")
    print(f"   Total tickers: {len(data)}")
    
    if data:
        total_rows = sum(len(df) for df in data.values())
        print(f"   Total rows: {total_rows:,}")
        
        # Test compliance
        compliance = loader.validate_modern_era_compliance(data)
        print(f"   Time Filter: {compliance.get('time_filter', False)}")
        print(f"   Liquidity Gate: {compliance.get('liquidity_gate', False)}")
        print(f"   No Survivorship Bias: {compliance.get('no_survivorship_bias', False)}")
        print(f"   Data Quality: {compliance.get('data_quality', False)}")
        
        # Test a few tickers
        sample_tickers = list(data.keys())[:3]
        for ticker in sample_tickers:
            df = data[ticker]
            if not df.empty:
                print(f"   {ticker}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
        
        print("✅ MODERN ERA PROTOCOL WORKING!")
    else:
        print("❌ No data loaded")

if __name__ == "__main__":
    test_modern_era()
