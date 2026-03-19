#!/usr/bin/env python3
"""
Historical Prediction Backtest - Part 4
60-day rolling validation of AI predictions
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('HistoricalBacktest')

MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'validation'
CHARTS_DIR = OUTPUT_DIR / 'charts'

# Backtest parameters
END_DATE = datetime(2026, 3, 7)  # 5 trading days before March 13
START_DATE = END_DATE - timedelta(days=60)
FORWARD_DAYS = 5
SAMPLE_SIZE = 100  # Sample 100 tickers per day for speed


def load_models():
    """Load ensemble models"""
    logger.info("[LOAD] Loading models...")
    
    models = {}
    
    with open(MODELS_DIR / 'xgboost_model.pkl', 'rb') as f:
        models['xgboost'] = pickle.load(f)
    
    with open(MODELS_DIR / 'lightgbm_model.pkl', 'rb') as f:
        models['lightgbm'] = pickle.load(f)
    
    with open(MODELS_DIR / 'rf_model.pkl', 'rb') as f:
        models['hgb'] = pickle.load(f)
    
    with open(MODELS_DIR / 'feature_scaler.pkl', 'rb') as f:
        models['scaler'] = pickle.load(f)
    
    with open(MODELS_DIR / 'feature_names.pkl', 'rb') as f:
        models['feature_names'] = pickle.load(f)
    
    logger.info(f"[LOAD] Models loaded: {len(models['feature_names'])} features")
    return models


def get_ticker_list():
    """Get list of tickers to test"""
    parquet_files = list(DATA_DIR.glob('*.parquet'))
    tickers = [pf.stem.upper() for pf in parquet_files]
    
    # Sample for speed
    if len(tickers) > SAMPLE_SIZE:
        np.random.seed(42)
        tickers = np.random.choice(tickers, SAMPLE_SIZE, replace=False).tolist()
    
    logger.info(f"[TICKERS] Testing {len(tickers)} tickers")
    return tickers


def generate_prediction(models, ticker, pred_date, spy_close):
    """Generate prediction for a ticker on a specific date"""
    try:
        ticker_file = DATA_DIR / f'{ticker.lower()}.parquet'
        if not ticker_file.exists():
            return None
        
        df = pd.read_parquet(ticker_file)
        if 'date' in df.columns:
            df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        
        # Get data up to prediction date
        df_subset = df[df.index <= pred_date]
        if len(df_subset) < 252:
            return None
        
        # Generate features
        fe = FeatureEngineer(use_advanced_features=True, verbose=False)
        features, _ = fe.create_features(df_subset, target_type='direction')
        if features.empty:
            return None
        
        # Add Phase 12 momentum features
        close = df_subset['adjClose'] if 'adjClose' in df_subset.columns else df_subset['close']
        close = close.reindex(features.index)
        
        # rs_vs_spy
        ticker_ret20 = close.pct_change(20)
        spy_aligned = spy_close.reindex(features.index, method='ffill')
        spy_ret20 = spy_aligned.pct_change(20)
        features['rs_vs_spy'] = (ticker_ret20 - spy_ret20).fillna(0).clip(-2, 2)
        
        # high_52w_prox
        high_52w = close.rolling(252, min_periods=60).max()
        features['high_52w_prox'] = (close / high_52w.replace(0, np.nan)).fillna(0).clip(0, 1.5)
        
        # volume_breakout
        if 'volume' in df_subset.columns:
            vol = df_subset['volume'].reindex(features.index)
            vol_avg50 = vol.rolling(50, min_periods=20).mean()
            features['volume_breakout'] = (vol / vol_avg50.replace(0, np.nan)).fillna(1.0).clip(0, 10)
        else:
            features['volume_breakout'] = 1.0
        
        # roc_63
        features['roc_63'] = close.pct_change(63).fillna(0).clip(-2, 2)
        
        features = features.dropna()
        if features.empty:
            return None
        
        # Get last row
        X = features.iloc[[-1]][models['feature_names']]
        X_scaled = models['scaler'].transform(X)
        
        # Ensemble prediction
        xgb_prob = models['xgboost'].predict_proba(X_scaled)[0, 1]
        lgb_prob = models['lightgbm'].predict_proba(X_scaled)[0, 1]
        hgb_prob = models['hgb'].predict_proba(X_scaled)[0, 1]
        
        weighted_prob = 0.4 * xgb_prob + 0.4 * lgb_prob + 0.2 * hgb_prob
        
        # Calculate actual forward return
        future_df = df[df.index > pred_date]
        if len(future_df) < FORWARD_DAYS:
            return None
        
        current_price = close.iloc[-1]
        future_price = future_df['adjClose'].iloc[FORWARD_DAYS-1] if 'adjClose' in future_df.columns else future_df['close'].iloc[FORWARD_DAYS-1]
        actual_return = (future_price - current_price) / current_price
        
        return {
            'ticker': ticker,
            'date': pred_date,
            'prob_up': weighted_prob,
            'predicted': 1 if weighted_prob > 0.5 else 0,
            'actual_return': actual_return,
            'actual': 1 if actual_return > 0.02 else 0,  # 2% threshold
            'correct': (1 if weighted_prob > 0.5 else 0) == (1 if actual_return > 0.02 else 0)
        }
        
    except Exception as e:
        logger.debug(f"[SKIP] {ticker} on {pred_date}: {e}")
        return None


def run_backtest(models, tickers, start_date, end_date):
    """Run 60-day backtest"""
    logger.info(f"[BACKTEST] Running from {start_date.date()} to {end_date.date()}...")
    
    # Load SPY
    spy_df = pd.read_parquet(DATA_DIR / 'SPY.parquet')
    if 'date' in spy_df.columns:
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        spy_df = spy_df.set_index('date')
    else:
        spy_df.index = pd.to_datetime(spy_df.index)
    
    spy_close = spy_df['adjClose'] if 'adjClose' in spy_df.columns else spy_df['close']
    
    # Get trading days
    trading_days = spy_close[(spy_close.index >= start_date) & (spy_close.index <= end_date)].index
    logger.info(f"[BACKTEST] {len(trading_days)} trading days")
    
    all_predictions = []
    
    for i, pred_date in enumerate(trading_days, 1):
        logger.info(f"[BACKTEST] Day {i}/{len(trading_days)}: {pred_date.date()}")
        
        day_predictions = []
        for ticker in tickers:
            pred = generate_prediction(models, ticker, pred_date, spy_close)
            if pred:
                day_predictions.append(pred)
        
        all_predictions.extend(day_predictions)
        logger.info(f"[BACKTEST]   Generated {len(day_predictions)} predictions")
    
    return pd.DataFrame(all_predictions)


def analyze_results(predictions_df):
    """Analyze backtest results"""
    logger.info("[ANALYZE] Computing metrics...")
    
    # Overall metrics
    total = len(predictions_df)
    correct = predictions_df['correct'].sum()
    accuracy = correct / total if total > 0 else 0
    
    # Precision at different thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    precision_results = []
    
    for thresh in thresholds:
        signals = predictions_df[predictions_df['prob_up'] > thresh]
        if len(signals) > 0:
            prec = signals['correct'].sum() / len(signals)
            precision_results.append({
                'threshold': thresh,
                'signals': len(signals),
                'precision': prec
            })
    
    # Daily accuracy
    daily_accuracy = predictions_df.groupby('date')['correct'].agg(['sum', 'count'])
    daily_accuracy['accuracy'] = daily_accuracy['sum'] / daily_accuracy['count']
    
    # Average returns
    buy_signals = predictions_df[predictions_df['predicted'] == 1]
    sell_signals = predictions_df[predictions_df['predicted'] == 0]
    
    avg_buy_return = buy_signals['actual_return'].mean() if len(buy_signals) > 0 else 0
    avg_sell_return = sell_signals['actual_return'].mean() if len(sell_signals) > 0 else 0
    
    metrics = {
        'total_predictions': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'buy_signals': len(buy_signals),
        'sell_signals': len(sell_signals),
        'avg_buy_return': avg_buy_return,
        'avg_sell_return': avg_sell_return,
        'precision_by_threshold': precision_results,
        'daily_accuracy': daily_accuracy
    }
    
    return metrics


def create_charts(predictions_df, metrics, output_dir):
    """Create visualization charts"""
    logger.info("[CHARTS] Creating visualizations...")
    
    # Chart 1: Daily accuracy over time
    plt.figure(figsize=(12, 6))
    daily_acc = metrics['daily_accuracy']
    plt.plot(daily_acc.index, daily_acc['accuracy'], marker='o', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random (50%)')
    plt.axhline(y=metrics['accuracy'], color='g', linestyle='--', label=f'Average ({metrics["accuracy"]:.1%})')
    plt.xlabel('Date')
    plt.ylabel('Accuracy')
    plt.title('Daily Prediction Accuracy (60-Day Backtest)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'historical_accuracy_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Precision by threshold
    plt.figure(figsize=(10, 6))
    prec_df = pd.DataFrame(metrics['precision_by_threshold'])
    plt.bar(prec_df['threshold'].astype(str), prec_df['precision'])
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random (50%)')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.title('Precision by Confidence Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_by_threshold.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("[CHARTS] Charts saved")


def create_excel_report(predictions_df, metrics, output_path):
    """Create Excel report"""
    logger.info(f"[EXCEL] Creating report at {output_path}")
    
    wb = Workbook()
    wb.remove(wb.active)
    
    # Sheet 1: Summary
    ws1 = wb.create_sheet("Summary")
    ws1.append(['Metric', 'Value'])
    
    summary_rows = [
        ['Backtest Period', f"{predictions_df['date'].min().date()} to {predictions_df['date'].max().date()}"],
        ['Total Predictions', metrics['total_predictions']],
        ['Correct Predictions', metrics['correct_predictions']],
        ['Overall Accuracy', f"{metrics['accuracy']:.2%}"],
        ['BUY Signals', metrics['buy_signals']],
        ['SELL Signals', metrics['sell_signals']],
        ['Avg BUY Return', f"{metrics['avg_buy_return']:.2%}"],
        ['Avg SELL Return', f"{metrics['avg_sell_return']:.2%}"],
        ['Edge over Random', f"{(metrics['accuracy'] - 0.5):.2%}"],
    ]
    
    for row in summary_rows:
        ws1.append(row)
    
    for cell in ws1[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        cell.font = Font(bold=True, color='FFFFFF')
    
    # Sheet 2: Daily Metrics
    ws2 = wb.create_sheet("Daily Accuracy")
    ws2.append(['Date', 'Predictions', 'Correct', 'Accuracy'])
    
    for date, row in metrics['daily_accuracy'].iterrows():
        ws2.append([
            date.strftime('%Y-%m-%d'),
            int(row['count']),
            int(row['sum']),
            f"{row['accuracy']:.2%}"
        ])
    
    for cell in ws2[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        cell.font = Font(bold=True, color='FFFFFF')
    
    # Sheet 3: Precision by Threshold
    ws3 = wb.create_sheet("Precision Analysis")
    ws3.append(['Threshold', 'Signals', 'Precision'])
    
    for item in metrics['precision_by_threshold']:
        ws3.append([
            item['threshold'],
            item['signals'],
            f"{item['precision']:.2%}"
        ])
    
    for cell in ws3[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        cell.font = Font(bold=True, color='FFFFFF')
    
    # Adjust column widths
    for ws in wb.worksheets:
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    wb.save(output_path)
    logger.info("[EXCEL] Report saved")


def main():
    logger.info("=" * 80)
    logger.info("HISTORICAL PREDICTION BACKTEST - Part 4")
    logger.info("=" * 80)
    
    # Load models
    models = load_models()
    
    # Get ticker list
    tickers = get_ticker_list()
    
    # Run backtest
    predictions_df = run_backtest(models, tickers, START_DATE, END_DATE)
    
    if predictions_df.empty:
        logger.error("[ERROR] No predictions generated")
        return 1
    
    # Analyze results
    metrics = analyze_results(predictions_df)
    
    # Create charts
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    create_charts(predictions_df, metrics, CHARTS_DIR)
    
    # Create Excel report
    output_path = OUTPUT_DIR / 'historical_backtest.xlsx'
    create_excel_report(predictions_df, metrics, output_path)
    
    # Summary
    logger.info("=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Period: {predictions_df['date'].min().date()} to {predictions_df['date'].max().date()}")
    logger.info(f"Total Predictions: {metrics['total_predictions']:,}")
    logger.info(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    logger.info(f"Edge over Random: {(metrics['accuracy'] - 0.5):.2%}")
    logger.info(f"BUY Signals: {metrics['buy_signals']} (avg return: {metrics['avg_buy_return']:.2%})")
    logger.info(f"SELL Signals: {metrics['sell_signals']} (avg return: {metrics['avg_sell_return']:.2%})")
    logger.info(f"\nPrecision by Threshold:")
    for item in metrics['precision_by_threshold']:
        logger.info(f"  {item['threshold']:.2f}: {item['precision']:.2%} ({item['signals']} signals)")
    logger.info(f"\nReport: {output_path}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
