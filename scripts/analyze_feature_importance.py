#!/usr/bin/env python3
"""
Feature Importance Analysis - Part 2
Extract and compare feature importance from current and sentiment models
"""

import os
import sys
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('FeatureImportance')

MODELS_DIR = PROJECT_ROOT / 'models'
ARCHIVE_DIR = PROJECT_ROOT / 'archive' / 'models_sentiment'
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'validation'
CHARTS_DIR = OUTPUT_DIR / 'charts'


def extract_current_importance():
    """Extract feature importance from current models (68 features)"""
    logger.info("[EXTRACT] Current model feature importance...")
    
    with open(MODELS_DIR / 'xgboost_model.pkl', 'rb') as f:
        xgb = pickle.load(f)
    
    with open(MODELS_DIR / 'lightgbm_model.pkl', 'rb') as f:
        lgb = pickle.load(f)
    
    with open(MODELS_DIR / 'rf_model.pkl', 'rb') as f:
        hgb = pickle.load(f)
    
    with open(MODELS_DIR / 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Get importance from each model
    xgb_importance = xgb.feature_importances_
    lgb_importance = lgb.feature_importances_
    
    # HGB may not have feature_importances_ attribute - compute from trees if needed
    if hasattr(hgb, 'feature_importances_'):
        hgb_importance = hgb.feature_importances_
    else:
        # Use permutation importance or set to uniform
        logger.warning("[WARN] HGB does not have feature_importances_, using uniform weights")
        hgb_importance = np.ones(len(feature_names)) / len(feature_names)
    
    # Weighted ensemble importance (0.4, 0.4, 0.2)
    ensemble_importance = 0.4 * xgb_importance + 0.4 * lgb_importance + 0.2 * hgb_importance
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'xgboost': xgb_importance,
        'lightgbm': lgb_importance,
        'hgb': hgb_importance,
        'ensemble': ensemble_importance
    })
    
    importance_df = importance_df.sort_values('ensemble', ascending=False)
    
    logger.info(f"[EXTRACT] Extracted importance for {len(importance_df)} features")
    return importance_df


def extract_sentiment_importance():
    """Extract feature importance from sentiment models (116 features)"""
    logger.info("[EXTRACT] Sentiment model feature importance...")
    
    if not ARCHIVE_DIR.exists():
        logger.warning("[WARN] Sentiment models not found")
        return None
    
    try:
        with open(ARCHIVE_DIR / 'xgboost_sentiment_model.pkl', 'rb') as f:
            xgb = pickle.load(f)
        
        with open(ARCHIVE_DIR / 'lightgbm_sentiment_model.pkl', 'rb') as f:
            lgb = pickle.load(f)
        
        with open(ARCHIVE_DIR / 'random_forest_sentiment_model.pkl', 'rb') as f:
            rf = pickle.load(f)
        
        import json
        with open(ARCHIVE_DIR / 'sentiment_feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        # Get importance
        xgb_importance = xgb.feature_importances_
        lgb_importance = lgb.feature_importances_
        rf_importance = rf.feature_importances_
        
        # Weighted ensemble (0.4, 0.4, 0.2)
        ensemble_importance = 0.4 * xgb_importance + 0.4 * lgb_importance + 0.2 * rf_importance
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'xgboost': xgb_importance,
            'lightgbm': lgb_importance,
            'rf': rf_importance,
            'ensemble': ensemble_importance
        })
        
        importance_df = importance_df.sort_values('ensemble', ascending=False)
        
        logger.info(f"[EXTRACT] Extracted importance for {len(importance_df)} features")
        return importance_df
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to extract sentiment importance: {e}")
        return None


def categorize_features(importance_df):
    """Categorize features by type"""
    categories = []
    
    for feature in importance_df['feature']:
        if feature in ['Price', 'high', 'low', 'open', 'volume', 'Adj Close', 'Adj High', 'Adj Low', 'Adj Open', 'Adj Volume', 'Dividend', 'Split Factor']:
            categories.append('Raw OHLCV')
        elif 'sentiment' in feature.lower():
            categories.append('Sentiment')
        elif feature in ['rs_vs_spy', 'roc_63', 'high_52w_prox', 'volume_breakout']:
            categories.append('Momentum')
        else:
            categories.append('Technical')
    
    importance_df['category'] = categories
    return importance_df


def create_importance_chart(importance_df, title, output_path):
    """Create feature importance bar chart"""
    logger.info(f"[CHART] Creating {title}...")
    
    top_20 = importance_df.head(20).copy()
    
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4' if cat == 'Technical' else '#ff7f0e' if cat == 'Momentum' else '#2ca02c' if cat == 'Raw OHLCV' else '#d62728' 
              for cat in top_20['category']]
    
    plt.barh(range(len(top_20)), top_20['ensemble'], color=colors)
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"[CHART] Saved to {output_path}")


def create_excel_report(current_importance, sentiment_importance, output_path):
    """Create comprehensive Excel report"""
    logger.info(f"[EXCEL] Creating report at {output_path}")
    
    wb = Workbook()
    wb.remove(wb.active)
    
    # Sheet 1: Current Model Top 20
    ws1 = wb.create_sheet("Current Top 20")
    ws1.append(['Rank', 'Feature', 'Category', 'Ensemble', 'XGBoost', 'LightGBM', 'HGB'])
    
    for i, row in current_importance.head(20).iterrows():
        ws1.append([
            i + 1,
            row['feature'],
            row['category'],
            round(row['ensemble'], 6),
            round(row['xgboost'], 6),
            round(row['lightgbm'], 6),
            round(row['hgb'], 6)
        ])
    
    for cell in ws1[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        cell.font = Font(bold=True, color='FFFFFF')
    
    # Sheet 2: Current Model All Features
    ws2 = wb.create_sheet("Current All Features")
    ws2.append(['Rank', 'Feature', 'Category', 'Ensemble', 'XGBoost', 'LightGBM', 'HGB'])
    
    for i, row in current_importance.iterrows():
        ws2.append([
            i + 1,
            row['feature'],
            row['category'],
            round(row['ensemble'], 6),
            round(row['xgboost'], 6),
            round(row['lightgbm'], 6),
            round(row['hgb'], 6)
        ])
    
    for cell in ws2[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        cell.font = Font(bold=True, color='FFFFFF')
    
    # Sheet 3: Category Summary
    ws3 = wb.create_sheet("Category Summary")
    ws3.append(['Category', 'Count', 'Avg Importance', 'Total Importance'])
    
    category_summary = current_importance.groupby('category').agg({
        'feature': 'count',
        'ensemble': ['mean', 'sum']
    }).round(6)
    
    for cat in category_summary.index:
        ws3.append([
            cat,
            int(category_summary.loc[cat, ('feature', 'count')]),
            round(category_summary.loc[cat, ('ensemble', 'mean')], 6),
            round(category_summary.loc[cat, ('ensemble', 'sum')], 6)
        ])
    
    for cell in ws3[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        cell.font = Font(bold=True, color='FFFFFF')
    
    # Sheet 4: Sentiment Comparison (if available)
    if sentiment_importance is not None:
        ws4 = wb.create_sheet("Sentiment Top 20")
        ws4.append(['Rank', 'Feature', 'Category', 'Ensemble', 'XGBoost', 'LightGBM', 'RF'])
        
        for i, row in sentiment_importance.head(20).iterrows():
            ws4.append([
                i + 1,
                row['feature'],
                row['category'],
                round(row['ensemble'], 6),
                round(row['xgboost'], 6),
                round(row['lightgbm'], 6),
                round(row['rf'], 6)
            ])
        
        for cell in ws4[1]:
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
    logger.info(f"[EXCEL] Report saved")


def main():
    logger.info("=" * 80)
    logger.info("FEATURE IMPORTANCE ANALYSIS - Part 2")
    logger.info("=" * 80)
    
    # Extract current model importance
    current_importance = extract_current_importance()
    current_importance = categorize_features(current_importance)
    
    # Extract sentiment model importance
    sentiment_importance = extract_sentiment_importance()
    if sentiment_importance is not None:
        sentiment_importance = categorize_features(sentiment_importance)
    
    # Create charts
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    
    chart_path = CHARTS_DIR / 'feature_importance_top20.png'
    create_importance_chart(current_importance, 'Top 20 Features - Current Models (68 features)', chart_path)
    
    if sentiment_importance is not None:
        chart_path_sent = CHARTS_DIR / 'feature_importance_sentiment_top20.png'
        create_importance_chart(sentiment_importance, 'Top 20 Features - Sentiment Models (116 features)', chart_path_sent)
    
    # Create Excel report
    output_path = OUTPUT_DIR / 'feature_importance_analysis.xlsx'
    create_excel_report(current_importance, sentiment_importance, output_path)
    
    # Summary
    logger.info("=" * 80)
    logger.info("FEATURE IMPORTANCE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Current Models (68 features):")
    logger.info(f"  Top 5 features:")
    for i, row in current_importance.head(5).iterrows():
        logger.info(f"    {i+1}. {row['feature']} ({row['category']}): {row['ensemble']:.4f}")
    
    logger.info(f"\nCategory breakdown:")
    for cat, count in current_importance['category'].value_counts().items():
        logger.info(f"  {cat}: {count} features")
    
    if sentiment_importance is not None:
        logger.info(f"\nSentiment Models (116 features):")
        logger.info(f"  Top 5 features:")
        for i, row in sentiment_importance.head(5).iterrows():
            logger.info(f"    {i+1}. {row['feature']} ({row['category']}): {row['ensemble']:.4f}")
    
    logger.info(f"\nReport: {output_path}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
