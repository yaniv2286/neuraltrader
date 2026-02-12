import os
import sys
import pandas as pd
import numpy as np
import logging

# --- 1. CONFIGURATION ---
TRAILING_STOP_PCT = 0.03  # 3.0% Trailing Stop
# Force console logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger("DebugExits")

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 2. IMPORTS ---
try:
    from core.data_loader import ModernEraDataLoader as DataLoader
    from core.feature_engineer import FeatureEngineer 
    from core.ai_models import EnsemblePredictor
except ImportError as e:
    logger.error(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

def run_trade_autopsy():
    logger.info("🩺 STARTING TRADE AUTOPSY (Micro-Backtest)")
    logger.info(f"⚙️  Config: Trailing Stop = {TRAILING_STOP_PCT*100}% | Shield = Weekly Breakdown")
    
    # Init
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    ensemble = EnsemblePredictor()
    
    # Load Tickers (Limit to 3 for deep dive)
    tickers = ['a', 'aal', 'aaon'] 
    
    for ticker in tickers:
        logger.info(f"\n{'='*60}")
        logger.info(f"🦅 ANALYZING TICKER: {ticker.upper()}")
        logger.info(f"{'='*60}")
        
        # A. Load & Prep
        try:
            df = data_loader.load_single_ticker(ticker)
            if df is None or df.empty: continue
            
            # Standard cleanup
            df.reset_index(inplace=True)
            df.columns = df.columns.str.lower()
            if 'index' in df.columns: df.rename(columns={'index': 'date'}, inplace=True)
            cols_to_drop = ['dividend', 'split', 'divcash', 'splitfactor']
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            df.set_index('date', inplace=True)
            
            # B. Features & AI
            features = feature_engineer.create_features(df)
            if isinstance(features, tuple): features = features[0]
            
            preds = ensemble.predict_batch(features)
            # Handle varied output formats
            if isinstance(preds, tuple): preds = preds[0]
            if isinstance(preds, pd.DataFrame): preds = preds.iloc[:, 0].values
            
            # Align Data
            df = df.loc[features.index].copy()
            df['ai_score'] = preds
            
            # C. Shield Indicators (Weekly Breakdown)
            # Logic: Green if Close > Lowest Low of last 5 days (shifted 1)
            df['week_low_5d'] = df['low'].rolling(5).min().shift(1)
            df['shield_green'] = (df['close'] > df['week_low_5d'])
            
            # D. Identify Entries
            # Entry: AI > 0.60 AND Shield is Green
            entry_signals = (df['ai_score'] > 0.60) & (df['shield_green'])
            entry_dates = df.index[entry_signals].tolist()
            
            logger.info(f"📍 Found {len(entry_dates)} Potential Entries")
            
            # E. SIMULATE TRADES
            in_trade = False
            entry_price = 0.0
            highest_price = 0.0
            stop_price = 0.0
            trades_log = []
            
            dates = df.index.tolist()
            closes = df['close'].values
            lows = df['low'].values
            highs = df['high'].values
            shield_green = df['shield_green'].values
            
            # Map dates to array indices for speed
            date_map = {d: i for i, d in enumerate(dates)}
            
            for date in entry_dates:
                idx = date_map[date]
                
                # 1. Skip if we are already in a trade
                if in_trade:
                    continue
                
                # 2. ENTER TRADE
                in_trade = True
                entry_price = closes[idx]
                entry_date = date
                highest_price = entry_price
                stop_price = entry_price * (1 - TRAILING_STOP_PCT)
                
                # 3. FAST FORWARD (Trade Lifecycle)
                for fwd_idx in range(idx + 1, len(dates)):
                    curr_date = dates[fwd_idx]
                    curr_close = closes[fwd_idx]
                    curr_low = lows[fwd_idx]
                    curr_high = highs[fwd_idx]
                    curr_shield_green = shield_green[fwd_idx]
                    
                    exit_reason = None
                    exit_price = 0.0
                    
                    # --- CHECK EXITS ---
                    
                    # A. SHIELD EXIT (End of Day)
                    # If Close < Week Low (Shield turns Red)
                    if not curr_shield_green:
                        exit_reason = "🛡️ SHIELD (Red Regime)"
                        exit_price = curr_close # We exit at close
                    
                    # B. TRAILING STOP (Intraday)
                    # Update High Water Mark
                    if curr_high > highest_price:
                        highest_price = curr_high
                        stop_price = highest_price * (1 - TRAILING_STOP_PCT)
                    
                    # Check if Low hit the stop
                    if curr_low < stop_price:
                        exit_reason = "🛑 TRAILING STOP (-3%)"
                        exit_price = stop_price # We exit at the stop price
                    
                    # --- EXECUTE EXIT ---
                    if exit_reason:
                        pnl_pct = (exit_price - entry_price) / entry_price
                        trades_log.append({
                            'Entry': entry_date.strftime('%Y-%m-%d'),
                            'Exit': curr_date.strftime('%Y-%m-%d'),
                            'Reason': exit_reason,
                            'P&L': f"{pnl_pct:.2%}",
                            'Days': (fwd_idx - idx)
                        })
                        in_trade = False
                        break # Stop the forward loop, look for next entry
            
            # F. PRINT RESULTS
            if not trades_log:
                logger.warning("   ⚠️ No completed trades (Might be holding till end)")
            else:
                logger.info(f"   📊 COMPLETED TRADES: {len(trades_log)}")
                logger.info("   📝 RECENT TRADE LOG (Last 5):")
                for t in trades_log[-5:]:
                    pnl_clean = t['P&L'].strip('%')
                    try:
                        val = float(pnl_clean)
                        pnl_color = "🟢" if val > 0 else "🔴"
                        if val == 0: pnl_color = "⚪"
                    except:
                        pnl_color = "⚪"
                    
                    logger.info(f"      {t['Entry']} ➔ {t['Exit']} | {pnl_color} {t['P&L']} | {t['Reason']} ({t['Days']}d)")
                    
        except Exception as e:
            logger.error(f"❌ Error processing {ticker}: {e}", exc_info=True)

if __name__ == "__main__":
    run_trade_autopsy()