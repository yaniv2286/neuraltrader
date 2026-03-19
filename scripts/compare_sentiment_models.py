#!/usr/bin/env python3
"""
Sentiment Model Comparison - Part 3
Empirically test if sentiment features degrade or improve performance
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('SentimentComparison')

MODELS_DIR = PROJECT_ROOT / 'models'
ARCHIVE_DIR = PROJECT_ROOT / 'archive' / 'models_sentiment'
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'validation'


def load_current_models():
    """Load current Phase 12 models (68 features)"""
    logger.info("[LOAD] Loading current models (68 features)...")
    
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
    
    logger.info(f"[LOAD] Current models loaded: {len(models['feature_names'])} features")
    return models


def load_sentiment_models():
    """Load archived sentiment models (116 features)"""
    logger.info("[LOAD] Loading sentiment models (116 features)...")
    
    if not ARCHIVE_DIR.exists():
        logger.error(f"[ERROR] Sentiment models not found at {ARCHIVE_DIR}")
        return None
    
    models = {}
    
    try:
        with open(ARCHIVE_DIR / 'xgboost_sentiment_model.pkl', 'rb') as f:
            models['xgboost'] = pickle.load(f)
        
        with open(ARCHIVE_DIR / 'lightgbm_sentiment_model.pkl', 'rb') as f:
            models['lightgbm'] = pickle.load(f)
        
        with open(ARCHIVE_DIR / 'random_forest_sentiment_model.pkl', 'rb') as f:
            models['rf'] = pickle.load(f)
        
        with open(ARCHIVE_DIR / 'sentiment_feature_scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)
        
        with open(ARCHIVE_DIR / 'sentiment_feature_names.json', 'r') as f:
            models['feature_names'] = json.load(f)
        
        logger.info(f"[LOAD] Sentiment models loaded: {len(models['feature_names'])} features")
        return models
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to load sentiment models: {e}")
        return None


def generate_test_predictions_current(current_models, test_tickers, test_date):
    """Generate predictions using current models (68 features)"""
    logger.info(f"[PREDICT] Generating current model predictions for {test_date}...")
    
    from core.feature_engineer import FeatureEngineer
    
    fe = FeatureEngineer(use_advanced_features=True, verbose=False)
    predictions = []
    
    # Load SPY for rs_vs_spy feature
    spy_df = pd.read_parquet(DATA_DIR / 'SPY.parquet')
    spy_close = spy_df['adjClose'] if 'adjClose' in spy_df.columns else spy_df['close']
    spy_close.index = pd.to_datetime(spy_close.index)
    
    for ticker in test_tickers:
        try:
            ticker_file = DATA_DIR / f'{ticker.lower()}.parquet'
            if not ticker_file.exists():
                continue
            
            df = pd.read_parquet(ticker_file)
            if 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            
            # Get data up to test_date
            df_subset = df[df.index <= test_date]
            if len(df_subset) < 252:
                continue
            
            # Generate features
            features, _ = fe.create_features(df_subset, target_type='direction')
            if features.empty:
                continue
            
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
                continue
            
            # Get last row (most recent)
            X = features.iloc[[-1]][current_models['feature_names']]
            X_scaled = current_models['scaler'].transform(X)
            
            # Ensemble prediction
            xgb_prob = current_models['xgboost'].predict_proba(X_scaled)[0, 1]
            lgb_prob = current_models['lightgbm'].predict_proba(X_scaled)[0, 1]
            hgb_prob = current_models['hgb'].predict_proba(X_scaled)[0, 1]
            
            weighted_prob = 0.4 * xgb_prob + 0.4 * lgb_prob + 0.2 * hgb_prob
            
            predictions.append({
                'ticker': ticker,
                'date': test_date,
                'prob_up': weighted_prob,
                'signal': 'BUY' if weighted_prob > 0.5 else 'SELL'
            })
            
        except Exception as e:
            logger.debug(f"[SKIP] {ticker}: {e}")
            continue
    
    logger.info(f"[PREDICT] Generated {len(predictions)} current model predictions")
    return pd.DataFrame(predictions)


def compare_metadata():
    """Compare metadata from both model sets"""
    logger.info("[COMPARE] Comparing model metadata...")
    
    # Current metadata
    with open(MODELS_DIR / 'ensemble_metadata.json', 'r') as f:
        current_meta = json.load(f)
    
    # Sentiment metadata
    sentiment_meta = None
    if (ARCHIVE_DIR / 'final_metadata.json').exists():
        with open(ARCHIVE_DIR / 'final_metadata.json', 'r') as f:
            sentiment_meta = json.load(f)
    
    comparison = {
        'current': {
            'features': current_meta.get('n_features', 0),
            'train_samples': current_meta.get('n_train', 0),
            'precision_at_65': current_meta.get('ensemble', {}).get('precision_at_65', 0),
            'accuracy_at_50': current_meta.get('ensemble', {}).get('accuracy_at_50', 0),
            'auc': current_meta.get('ensemble', {}).get('auc', 0),
        }
    }
    
    if sentiment_meta:
        comparison['sentiment'] = {
            'features': sentiment_meta.get('n_features', 0),
            'train_samples': sentiment_meta.get('train_samples', 0),
            'precision_at_60': sentiment_meta.get('ensemble', {}).get('acc60', 0),
            'accuracy_at_50': sentiment_meta.get('ensemble', {}).get('acc', 0),
            'auc': sentiment_meta.get('ensemble', {}).get('auc', 0),
        }
    
    return comparison


def create_excel_report(metadata_comparison, output_path):
    """Create Excel comparison report"""
    logger.info(f"[EXCEL] Creating report at {output_path}")
    
    wb = Workbook()
    wb.remove(wb.active)
    
    # Sheet 1: Performance Comparison
    ws1 = wb.create_sheet("Performance Comparison")
    ws1.append(['Metric', 'Current (68 features)', 'Sentiment (116 features)', 'Difference'])
    
    current = metadata_comparison.get('current', {})
    sentiment = metadata_comparison.get('sentiment', {})
    
    rows = [
        ['Features', current.get('features', 0), sentiment.get('features', 0), 
         sentiment.get('features', 0) - current.get('features', 0)],
        ['Training Samples', current.get('train_samples', 0), sentiment.get('train_samples', 0),
         sentiment.get('train_samples', 0) - current.get('train_samples', 0)],
        ['Accuracy @ 0.50', f"{current.get('accuracy_at_50', 0):.4f}", 
         f"{sentiment.get('accuracy_at_50', 0):.4f}",
         f"{sentiment.get('accuracy_at_50', 0) - current.get('accuracy_at_50', 0):.4f}"],
        ['Precision @ 0.60-0.65', f"{current.get('precision_at_65', 0):.4f}",
         f"{sentiment.get('precision_at_60', 0):.4f}",
         f"{sentiment.get('precision_at_60', 0) - current.get('precision_at_65', 0):.4f}"],
        ['AUC', f"{current.get('auc', 0):.4f}", f"{sentiment.get('auc', 0):.4f}",
         f"{sentiment.get('auc', 0) - current.get('auc', 0):.4f}"],
    ]
    
    for row in rows:
        ws1.append(row)
    
    # Header formatting
    for cell in ws1[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        cell.font = Font(bold=True, color='FFFFFF')
    
    # Sheet 2: Analysis
    ws2 = wb.create_sheet("Analysis")
    ws2.append(['Finding', 'Value'])
    
    acc_diff = sentiment.get('accuracy_at_50', 0) - current.get('accuracy_at_50', 0)
    prec_diff = sentiment.get('precision_at_60', 0) - current.get('precision_at_65', 0)
    
    findings = [
        ['Accuracy Difference', f"{acc_diff:.4f} ({acc_diff*100:.2f}%)"],
        ['Precision Difference', f"{prec_diff:.4f} ({prec_diff*100:.2f}%)"],
        ['Sentiment Impact', 'NEGATIVE' if acc_diff < 0 else 'POSITIVE'],
        ['Recommendation', 'Keep current models (68 features)' if acc_diff < 0 else 'Consider sentiment models (116 features)'],
        ['Reason', 'Sentiment features degraded performance' if acc_diff < 0 else 'Sentiment features improved performance'],
    ]
    
    for finding in findings:
        ws2.append(finding)
    
    for cell in ws2[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        cell.font = Font(bold=True, color='FFFFFF')
    
    # Adjust column widths
    for ws in [ws1, ws2]:
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
    logger.info(f"[EXCEL] Report saved")


def main():
    logger.info("=" * 80)
    logger.info("SENTIMENT MODEL COMPARISON - Part 3")
    logger.info("=" * 80)
    
    # Load models
    current_models = load_current_models()
    sentiment_models = load_sentiment_models()
    
    if sentiment_models is None:
        logger.warning("[WARN] Sentiment models not available - using metadata comparison only")
    
    # Compare metadata
    metadata_comparison = compare_metadata()
    
    # Create Excel report
    output_path = OUTPUT_DIR / 'sentiment_comparison.xlsx'
    create_excel_report(metadata_comparison, output_path)
    
    # Summary
    logger.info("=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    
    current = metadata_comparison.get('current', {})
    sentiment = metadata_comparison.get('sentiment', {})
    
    logger.info(f"Current Models (68 features):")
    logger.info(f"  Accuracy @ 0.50: {current.get('accuracy_at_50', 0):.2%}")
    logger.info(f"  Precision @ 0.65: {current.get('precision_at_65', 0):.2%}")
    logger.info(f"  AUC: {current.get('auc', 0):.4f}")
    
    logger.info(f"\nSentiment Models (116 features):")
    logger.info(f"  Accuracy @ 0.50: {sentiment.get('accuracy_at_50', 0):.2%}")
    logger.info(f"  Precision @ 0.60: {sentiment.get('precision_at_60', 0):.2%}")
    logger.info(f"  AUC: {sentiment.get('auc', 0):.4f}")
    
    acc_diff = sentiment.get('accuracy_at_50', 0) - current.get('accuracy_at_50', 0)
    logger.info(f"\nAccuracy Difference: {acc_diff:.4f} ({acc_diff*100:.2f}%)")
    
    if acc_diff < 0:
        logger.info("CONCLUSION: Sentiment features DEGRADED performance")
        logger.info("RECOMMENDATION: Keep current models (68 features)")
    else:
        logger.info("CONCLUSION: Sentiment features IMPROVED performance")
        logger.info("RECOMMENDATION: Consider retraining with sentiment")
    
    logger.info(f"\nReport: {output_path}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
