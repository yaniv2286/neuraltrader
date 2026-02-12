import os
import sys
import pandas as pd
import numpy as np
import logging
import itertools
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORT FIXES (Modern Era Architecture) ---
try:
    from core.data_loader import ModernEraDataLoader as DataLoader
except ImportError:
    from core.data_loader import DataLoader

from core.feature_engineer import FeatureEngineer

# FIX: EnsemblePredictor is now in 'ai_models', not 'ensemble'
try:
    from core.ai_models import EnsemblePredictor
except ImportError:
    from core.ensemble import EnsemblePredictor

# --- CONFIGURATION ---
LOG_DIR = "logs"
REPORT_DIR = "reports"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Logger Setup (Windows Safe)
logger = logging.getLogger("StrategyTournament")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = logging.FileHandler(os.path.join(LOG_DIR, "tournament.log"), encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

def calculate_drawdown(equity_curve):
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def run_strategy_tournament():
    logger.info("[START] NEURALTRADER STRATEGY TOURNAMENT")
    
    # 1. Initialize Core Components
    try:
        data_loader = DataLoader()
        logger.info("[OK] Data Loader Initialized")
    except Exception as e:
        logger.error(f"[ERROR] Data Loader Init Failed: {e}")
        return

    feature_engineer = FeatureEngineer()
    
    # Load Models
    try:
        ensemble = EnsemblePredictor()
        logger.info("[OK] AI Council Loaded Successfully")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load AI Models: {e}")
        return

    # 2. Get Universe
    logger.info("[INFO] Loading Modern Era Ticker List...")
    try:
        full_df = pd.read_parquet('data/processed/modern_era_universe.parquet', columns=['ticker'])
        tickers = full_df['ticker'].unique()  # Process ALL tickers for full tournament
        logger.info(f"[OK] Found {len(tickers)} unique tickers (FULL TOURNAMENT).")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load ticker list: {e}")
        return

    # 3. Define the Tournament Bracket
    params_shield = ['SMA_200', 'SMA_50', 'Donchian_4W', 'Weekly_Breakdown']
    params_stop = ['Fixed_ATR', 'Trailing_ATR', 'Trailing_Pct']
    params_stop_val = [2.0, 3.0] 

    combinations = list(itertools.product(params_shield, params_stop, params_stop_val))
    logger.info(f"[INFO] Battling {len(combinations)} Strategy Combinations...")

    results = []

    # 4. THE MAIN LOOP
    for ticker in tqdm(tickers, desc="Processing Tickers"):
        try:
            # --- A. DATA LOADING ---
            try:
                df = data_loader.load_single_ticker(ticker)
            except AttributeError:
                df = data_loader.load_ticker_data(ticker)

            if df is None or df.empty:
                continue

            # Reset Index & Normalize
            df.reset_index(inplace=True)
            df.columns = df.columns.str.lower()
            if 'index' in df.columns: df.rename(columns={'index': 'date'}, inplace=True)
            
            # Poison Pill Removal
            cols_to_drop = ['dividend', 'split', 'divcash', 'splitfactor']
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

            # Sort & Fill
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            df.set_index('date', inplace=True)

            # --- B. FEATURE ENGINEERING ---
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Preserve OHLC columns before feature engineering (map to expected names)
                ohlc_mapping = {}
                for col in df.columns:
                    if col.lower() == 'close':
                        ohlc_mapping['Price'] = df[col].copy()  # Map close to Price
                    elif col.lower() in ['high', 'low', 'open']:
                        ohlc_mapping[col.lower()] = df[col].copy()
                
                try:
                    features, target = feature_engineer.create_features(df, target_type='direction')
                    if features is not None and len(features) > 0:
                        df = features
                        # Add back OHLC columns with correct names for models
                        for model_col_name, col_data in ohlc_mapping.items():
                            df[model_col_name] = col_data.reindex(df.index, method='ffill')
                    else:
                        continue
                except AttributeError:
                    df = feature_engineer.generate_features(df)

            if df.empty or 'rsi' not in df.columns:
                logger.warning(f"[WARNING] {ticker}: Feature engineering failed or missing RSI")
                continue

            # --- C. AI PREDICTION ---
            # Use all columns for prediction (models expect OHLC + features)
            preds = ensemble.predict_batch(df)
            if isinstance(preds, tuple):
                preds = preds[0]
            
            prediction_series = pd.Series(preds, index=df.index[-len(preds):])
            df['ai_score'] = prediction_series
            df['ai_score'].fillna(0.5, inplace=True)

            # --- D. STRATEGY SIMULATION ---
            # Pre-calculate Shield Indicators
            # Use the correct column names that models expect
            close_col = 'Price'  # Models expect 'Price' for close price
            low_col = 'low'      # Models expect 'low'
            
            df['sma_200'] = df[close_col].rolling(200).mean()
            df['sma_50'] = df[close_col].rolling(50).mean()
            df['donchian_low'] = df[low_col].rolling(20).min()
            df['week_low_5d'] = df[low_col].rolling(5).min()

            # Run Combinations
            for shield_type, stop_type, stop_val in combinations:
                
                # 1. Calculate Shield Status
                if shield_type == 'SMA_200':
                    mask_shield = (df[close_col] > df['sma_200']).astype(int)
                elif shield_type == 'SMA_50':
                    mask_shield = (df[close_col] > df['sma_50']).astype(int)
                elif shield_type == 'Donchian_4W':
                    mask_shield = (df[close_col] > df[low_col].shift(1)).astype(int)
                elif shield_type == 'Weekly_Breakdown':
                    mask_shield = (df[close_col] > df['week_low_5d'].shift(1)).astype(int)
                
                # 2. Simulate Trades
                df['ret'] = df[close_col].pct_change()
                signal = ((df['ai_score'] > 0.60) & (mask_shield == 1)).astype(int)
                position = signal.shift(1).fillna(0)
                strat_ret = df['ret'] * position
                
                # Metrics
                cum_ret = (1 + strat_ret).cumprod()
                if cum_ret.empty:
                    total_return = 0
                else:
                    total_return = cum_ret.iloc[-1] - 1
                
                if strat_ret.std() == 0:
                    sharpe = 0
                else:
                    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252)
                    
                drawdown = calculate_drawdown(cum_ret)
                
                results.append({
                    'Ticker': ticker,
                    'Shield': shield_type,
                    'Stop_Type': stop_type,
                    'Param': stop_val,
                    'Return': total_return,
                    'Sharpe': sharpe,
                    'MaxDD': drawdown
                })

        except Exception as e:
            # logger.error(f"[ERROR] Processing {ticker}: {e}") # Reduce spam for minor errors
            continue

    # 5. Compile & Save Results
    logger.info("[FINISH] Tournament Complete. Compiling Results...")
    if not results:
        logger.error("[ERROR] No results generated!")
        return

    results_df = pd.DataFrame(results)
    
    leaderboard = results_df.groupby(['Shield', 'Stop_Type', 'Param']).agg({
        'Return': 'mean',
        'Sharpe': 'mean',
        'MaxDD': 'mean'
    }).sort_values('Sharpe', ascending=False)
    
    print("\n[TOURNAMENT RESULTS (TOP 5)]")
    print(leaderboard.head(5))
    
    leaderboard.to_csv(os.path.join(REPORT_DIR, "optimization_tournament.csv"))
    logger.info(f"[OK] Results saved to {os.path.join(REPORT_DIR, 'optimization_tournament.csv')}")

if __name__ == "__main__":
    run_strategy_tournament()