#!/usr/bin/env python3
"""
Create Sector Map - Generate ticker-to-sector mapping file
Creates config/sector_map.json with 11 Sector ETFs and their holdings
"""

import json
import os

def create_sector_map():
    """Create sector mapping file with ETF tickers and their holdings"""
    
    # Define sector ETFs and their holdings
    sector_map = {
        "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CSCO", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "ORCL", "IBM", "NOW", "AMAT", "MU", "LRCX", "ADI", "PANW", "KLAC"],
        "XLF": ["JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "BLK", "C", "AXP", "SPGI", "PGR", "CB", "MMC", "USB", "PNC", "TFC", "COF", "HIG", "BK"],
        "XLV": ["LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "AMGN", "PFE", "ISRG", "DHR", "BMY", "GILD", "VRTX", "REGN", "SYK", "ZTS", "BSX", "CVS", "CI", "BDX"],
        "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX", "CMG", "MAR", "HLT", "F", "GM", "ORLY", "ROST", "YUM", "LULU", "TSCO", "EXPE"],
        "XLC": ["GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR", "EA", "TTWO", "WBD", "OMC", "IPG", "LYV", "PARA", "FOXA", "FOX", "NWSA"],
        "XLP": ["PG", "COST", "WMT", "KO", "PEP", "PM", "MO", "CL", "TGT", "MDLZ", "EL", "GIS", "KMB", "SYY", "STZ", "DG", "ADM", "HSY", "K", "CLX"],
        "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HES", "HAL", "BKR", "KMI", "DVN", "WMB", "TRGP", "FANG", "CTRA", "MRO", "APA"],
        "XLI": ["CAT", "GE", "UNP", "HON", "UPS", "RTX", "LMT", "DE", "ADP", "BA", "ETN", "MMM", "CSX", "NSC", "GD", "ITW", "EMR", "PH", "FDX", "NOC"],
        "XLB": ["LIN", "SHW", "FCX", "APD", "ECL", "NEM", "CTVA", "DOW", "DD", "MLM", "VMC", "PPG", "ALB", "FMC", "CE", "LYB", "MOS", "CF", "EMN", "IP"],
        "XLRE": ["PLD", "AMT", "EQIX", "PSA", "CCI", "O", "DLR", "SPG", "VICI", "WELL", "CBRE", "AVB", "EQR", "EXR", "INVH", "MAA", "ESS", "ARE", "UDR", "HST"],
        "XLU": ["NEE", "SO", "DUK", "SRE", "AEP", "D", "PEG", "EXC", "XEL", "ED", "WEC", "ES", "EIX", "DTE", "FE", "PPL", "AEE", "ETR", "CMS", "CNP"]
    }
    
    # Create config directory if it doesn't exist
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(config_dir, 'sector_map.json')
    
    # Write sector map to JSON file
    with open(output_file, 'w') as f:
        json.dump(sector_map, f, indent=4)
    
    # Print success message
    print("[PASS] Sector Map created at config/sector_map.json")
    
    # Print summary statistics
    total_tickers = sum(len(tickers) for tickers in sector_map.values())
    print(f"[INFO] Created mapping for {len(sector_map)} sectors with {total_tickers} total tickers")
    
    # Print sector breakdown
    print("[INFO] Sector breakdown:")
    for etf, tickers in sector_map.items():
        print(f"  {etf}: {len(tickers)} tickers")
    
    return output_file

def main():
    """Main entry point"""
    try:
        output_file = create_sector_map()
        print(f"[SUCCESS] Sector map file created: {output_file}")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to create sector map: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
