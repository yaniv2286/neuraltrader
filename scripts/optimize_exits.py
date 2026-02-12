import os
import sys
import pandas as pd
import numpy as np
import logging
import itertools
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- ROBUST IMPORTS ---
try:
    from core.data_loader import ModernEraDataLoader as DataLoader
except ImportError:
    from core.data_loader import DataLoader

try:
    from core.feature_engineer import FeatureEngineer
except ImportError:
    from core.feature_engineer import FeatureEngineer 

try:
    from core.ai_models import EnsemblePredictor
except ImportError:
    from core.ensemble import EnsemblePredictor

# --- CONFIGURATION ---
LOG_DIR = "logs"
REPORT_DIR = "reports"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Logger Setup (Force Console & File)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "exit_optimization.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger("ExitOptimizer")

# Suppress Warnings
import warnings
warnings.filterwarnings("ignore")

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def simulate_trade_path(prices, highs, lows, atrs, shield_signals, entry_idx, stop_type, stop_param, tp_param):
    """
    Simulates a single trade with SHIELD enforcement.
    """
    entry_price = prices[entry_idx]
    entry_atr = atrs[entry_idx]
    
    # Initialize Stop Price
    if stop_type == 'Fixed_ATR':
        stop_price = entry_price - (entry_atr * stop_param)
    elif stop_type == 'Trailing_ATR':
        stop_price = entry_price - (entry_atr * stop_param)
    elif stop_type == 'Trailing_Pct':
        stop_price = entry_price * (1 - stop_param/100.0)
    
    # Take Profit Price (0 means no TP)
    tp_price = 0
    if tp_param > 0:
        tp_price = entry_price + (entry_atr * tp_param)
    
    highest_price = entry_price
    
    # Walk forward
    for i in range(entry_idx + 1, len(prices)):
        curr_low = lows[i]
        curr_high = highs[i]
        curr_close = prices[i]
        curr_shield = shield_signals[i]
        
        # 0. CHECK SHIELD (Iron Law: Red Shield = Cash)
        if curr_shield == 0:
            ret = (curr_close - entry_price) / entry_price
            return ret # Exit at Close
            
        # 1. Check Stop Loss (Hit on Low)
        if curr_low <= stop_price:
            ret = (stop_price - entry_price) / entry_price
            return ret
            
        # 2. Check Take Profit (Hit on High)
        if tp_price > 0 and curr_high >= tp_price:
            ret = (tp_price - entry_price) / entry_price
            return ret
            
        # 3. Update Trailing Stops
        if stop_type == 'Trailing_ATR':
            new_stop = curr_close - (atrs[i] * stop_param)
            if new_stop > stop_price:
                stop_price = new_stop
        elif stop_type == 'Trailing_Pct':
            if curr_close > highest_price:
                highest_price = curr_close
                new_stop = highest_price * (1 - stop_param/100.0)
                if new_stop > stop_price:
                    stop_price = new_stop
                    
    # End of data - Force Close
    ret = (prices[len(prices)-1] - entry_price) / entry_price
    return ret

def run_exit_optimization():
    logger.info("🚀 STARTING PHASE 2: EXIT OPTIMIZATION (Shield + Stops)")
    
    # 1. Initialize
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    try:
        ensemble = EnsemblePredictor()
        logger.info("✅ AI Council Loaded")
    except:
        logger.error("❌ AI Models missing.")
        return

    # 2. Load Tickers
    try:
        full_df = pd.read_parquet('data/processed/modern_era_universe.parquet', columns=['ticker'])
        tickers = full_df['ticker'].unique().tolist()
        logger.info(f"🦅 Analyzing {len(tickers)} Tickers")
    except:
        logger.error("❌ Ticker list not found.")
        return

    # 3. Define Grid (Reduced for Speed, Expanded for Impact)
    # Focusing on Trailing Pct as it performed best in debug
    stop_types = ['Trailing_Pct', 'Fixed_ATR'] 
    stop_params = [2.0, 3.0, 4.0, 5.0] 
    tp_params = [0] # Pure trend following (Let winners run)
    
    combinations = list(itertools.product(stop_types, stop_params, tp_params))
    logger.info(f"⚔️  Testing {len(combinations)} Exit Strategies per trade...")

    results = []

    # 4. MAIN LOOP
    # We use tqdm for the progress bar
    pbar = tqdm(tickers, desc="Optimizing")
    
    valid_tickers = 0
    
    for ticker in pbar:
        try:
            # A. Load
            try:
                df = data_loader.load_single_ticker(ticker)
            except:
                continue
                
            if df is None or df.empty: continue
            
            # B. Prep
            df.reset_index(inplace=True)
            df.columns = df.columns.str.lower()
            if 'index' in df.columns: df.rename(columns={'index': 'date'}, inplace=True)
            cols_to_drop = ['dividend', 'split', 'divcash', 'splitfactor']
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            df.set_index('date', inplace=True)
            
            df_prices = df.copy() # KEEP PRICES SEPARATE

            # C. Features
            features_output = feature_engineer.create_features(df)
            if isinstance(features_output, tuple): features_df = features_output[0]
            else: features_df = features_output

            if features_df is None or features_df.empty: continue

            # D. AI
            preds = ensemble.predict_batch(features_df)
            if isinstance(preds, tuple): preds = preds[0]
            if isinstance(preds, pd.DataFrame): preds = preds.iloc[:, 0].values
            
            # Align
            pred_series = pd.Series(preds, index=features_df.index)
            df_prices = df_prices.loc[pred_series.index] # Align dates
            df_prices['ai_score'] = pred_series
            df_prices['ai_score'].fillna(0.5, inplace=True)

            # E. Indicators
            df_prices['atr'] = calculate_atr(df_prices)
            
            # SHIELD: Weekly Breakdown
            df_prices['week_low_5d'] = df_prices['low'].rolling(5).min().shift(1)
            # 1 = Green, 0 = Red
            df_prices['shield_signal'] = (df_prices['close'] > df_prices['week_low_5d']).astype(int)

            # Entry Signals
            entry_mask = (df_prices['ai_score'] > 0.60) & (df_prices['shield_signal'] == 1)
            entry_indices = np.where(entry_mask)[0]
            
            if len(entry_indices) == 0: continue
            valid_tickers += 1

            # Arrays for Speed
            prices_np = df_prices['close'].values
            highs_np = df_prices['high'].values
            lows_np = df_prices['low'].values
            atrs_np = df_prices['atr'].values
            shield_np = df_prices['shield_signal'].values
            
            # F. SIMULATE
            for stop_type, stop_val, tp_val in combinations:
                trades_ret = []
                for entry_idx in entry_indices:
                    if entry_idx >= len(prices_np) - 1: continue
                    if np.isnan(atrs_np[entry_idx]): continue

                    ret = simulate_trade_path(
                        prices_np, highs_np, lows_np, atrs_np, shield_np,
                        entry_idx, stop_type, stop_val, tp_val
                    )
                    trades_ret.append(ret)
                
                if not trades_ret: continue
                
                avg_ret = np.mean(trades_ret)
                win_rate = np.sum(np.array(trades_ret) > 0) / len(trades_ret)
                
                results.append({
                    'Ticker': ticker,
                    'Stop_Type': stop_type,
                    'Stop_Param': stop_val,
                    'TP_Param': tp_val,
                    'WinRate': win_rate,
                    'AvgReturn': avg_ret,
                    'TradeCount': len(trades_ret)
                })
                
        except Exception as e:
            # logger.error(f"Error {ticker}: {e}") # Keep silent in loop to avoid spam, debug via pbar
            continue

    # 5. Report
    if not results:
        logger.error("❌ No results generated.")
        return

    res_df = pd.DataFrame(results)
    
    # Aggregation
    leaderboard = res_df.groupby(['Stop_Type', 'Stop_Param', 'TP_Param']).agg({
        'WinRate': 'mean',
        'AvgReturn': 'mean',
        'TradeCount': 'sum'
    })
    
    # Calculate weighted metrics or Sharpe approximation
    leaderboard['Score'] = leaderboard['AvgReturn'] * leaderboard['WinRate'] # Simple score
    leaderboard = leaderboard.sort_values('AvgReturn', ascending=False)
    
    logger.info(f"\n✅ Optimization Complete on {valid_tickers} valid tickers.")
    print("\n🏆 EXIT OPTIMIZATION LEADERBOARD 🏆")
    print(leaderboard.head(10))
    
    leaderboard.to_csv(os.path.join(REPORT_DIR, "exit_optimization.csv"))

if __name__ == "__main__":
    run_exit_optimization()