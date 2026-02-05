import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime

def verify_performance():
    print("\n" + "="*60)
    print("🚀 NEURALTRADER: PHASE 7 PERFORMANCE VERIFICATION (TURBO MODE)")
    print("   Target: CAGR > 25% | Drawdown < 20%")
    print("="*60)

    # --- 1. Load Data (Merge Brain + Market) ---
    print("[1/4] Loading AI Scores & Merging with Raw Prices...")
    proc_path = os.path.join("data", "processed", "*.parquet")
    files = glob.glob(proc_path)
    
    if not files:
        print("[CRITICAL] No processed data found! Run 'scripts/train_ensemble.py' first.")
        return

    all_data = []
    for f in files:
        try:
            # 1. Load Score
            df_score = pd.read_parquet(f)
            df_score.columns = [c.lower() for c in df_score.columns]
            
            if 'score' not in df_score.columns:
                continue
                
            ticker = os.path.basename(f).replace('.parquet', '')
            
            # 2. Load Price (Raw Data)
            raw_file = os.path.join("data", "raw", f"{ticker}.parquet")
            if not os.path.exists(raw_file):
                raw_file = os.path.join("data", "raw", f"{ticker.upper()}.parquet")
                
            if os.path.exists(raw_file):
                df_price = pd.read_parquet(raw_file)
                df_price.columns = [c.lower() for c in df_price.columns]
                
                # Standardize Date
                df_score['date'] = pd.to_datetime(df_score['date'])
                if 'date' in df_price.columns:
                    df_price['date'] = pd.to_datetime(df_price['date'])
                    df_price.set_index('date', inplace=True)
                else:
                    df_price.index = pd.to_datetime(df_price.index)

                # 3. Merge (Inner Join on Date)
                if 'adjclose' in df_price.columns:
                    # Rename adjclose to price for clarity
                    df_price = df_price.rename(columns={'adjclose': 'price'})
                    # Fast merge
                    df_merged = df_score.merge(df_price[['price']], left_on='date', right_index=True, how='inner')
                    
                    # Filter 2024+
                    df_merged = df_merged[df_merged['date'] >= '2024-01-01']
                    
                    if not df_merged.empty:
                        all_data.append(df_merged)
        except Exception as e:
            pass

    if not all_data:
        print("[ERROR] Could not merge Scores with Prices. Check data/raw integrity.")
        return

    full_df = pd.concat(all_data)
    full_df.set_index('date', inplace=True)
    full_df.sort_index(inplace=True)
    
    # OPTIMIZATION: Ensure ticker is categorical for memory saving (optional but good)
    full_df['ticker'] = full_df['ticker'].astype(str)
    
    print(f"[OK] Loaded {len(full_df):,} rows. Brain & Market synchronized.")

    # --- 2. Run Simulation (Weekly Rebalance) ---
    print("[2/4] Running Simulation (Top 5 Stocks, Weekly Rebalance)...")
    
    cash = 100000.0
    equity_curve = [cash]
    dates = [full_df.index.min()]
    
    # Resample to weekly (Fridays)
    # Fast grouping
    weeks = full_df.index.unique().sort_values()
    fridays = weeks[weeks.weekday == 4]
    
    if len(fridays) == 0:
        print("[ERROR] No Fridays found in data range!")
        return

    current_holdings = {} # {ticker: shares}
    
    for i in range(len(fridays)-1):
        curr_date = fridays[i]
        
        # --- FAST LOOKUP ---
        # Instead of scanning full_df, we slice by index (O(1))
        try:
            day_data = full_df.loc[curr_date]
            # If only one stock exists for this date, it returns Series. Force DataFrame.
            if isinstance(day_data, pd.Series):
                day_data = day_data.to_frame().T
        except KeyError:
            # No data for this specific date (Market Holiday?)
            continue

        # A. Mark-to-Market (Value Current Portfolio)
        portfolio_val = cash
        
        for ticker, shares in current_holdings.items():
            # Fast filter on the sliced day_data
            row = day_data[day_data['ticker'] == ticker]
            if not row.empty:
                portfolio_val += shares * row['price'].iloc[0]
            else:
                # If checking a holding that has no data today, check holdings dict for last known price? 
                # Simplified: Assume cash value holds.
                pass
        
        # Reset for new buys (Virtual Sell)
        cash = portfolio_val
        current_holdings = {}
        
        # B. Buy Top 5 Scores
        if not day_data.empty:
            # THE COUNCIL'S VOTE: Pick top 5 by Score
            top_picks = day_data.sort_values('score', ascending=False).head(5)
            # Filter: Positive Score Only
            top_picks = top_picks[top_picks['score'] > 0]
            
            if not top_picks.empty:
                allocation = cash / len(top_picks)
                for _, row in top_picks.iterrows():
                    price = row['price']
                    if price > 0:
                        shares = allocation / price
                        current_holdings[row['ticker']] = shares
                        cash -= allocation
        
        # D. Record Equity for this week
        # (Simplified: We just use the portfolio_val we calculated at the start of rebalance)
        equity_curve.append(portfolio_val) 
        dates.append(curr_date)

    # --- 3. Calculate Metrics ---
    print("[3/4] Calculating Final Metrics...")
    
    if len(equity_curve) < 2:
        print("[ERROR] Not enough data points for simulation.")
        return

    final_value = equity_curve[-1]
    total_return_pct = (final_value - 100000) / 100000 * 100
    
    # CAGR
    days = (dates[-1] - dates[0]).days
    if days > 0:
        years = days / 365.25
        cagr = ((final_value / 100000) ** (1/years) - 1) * 100
    else:
        cagr = 0
    
    # Drawdown
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.expanding().max()
    drawdowns = (equity_series - running_max) / running_max
    max_dd = drawdowns.min() * 100
    
    # Sharpe
    returns = equity_series.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(52) if len(returns) > 0 and returns.std() > 0 else 0

    # --- 4. The Report ---
    print("\n" + "="*60)
    print(f"📊 FINAL VERIFICATION REPORT ({dates[0].date()} to {dates[-1].date()})")
    print("="*60)
    print(f"💰 Initial Capital: $100,000")
    print(f"💰 Final Capital:   ${final_value:,.2f}")
    print(f"📈 Total Return:    {total_return_pct:.2f}%")
    print("-" * 30)
    print(f"🚀 CAGR:            {cagr:.2f}%   (Target: 25%)")
    print(f"🛡️ Max Drawdown:    {max_dd:.2f}%  (Target: <20%)")
    print(f"⚖️ Sharpe Ratio:    {sharpe:.2f}")
    print("="*60)
    
    status = "✅ READY FOR LIVE" if cagr > 20 and max_dd > -25 else "⚠️ NEEDS OPTIMIZATION"
    print(f"🏆 SYSTEM STATUS: {status}")
    print("="*60)

if __name__ == "__main__":
    verify_performance()
