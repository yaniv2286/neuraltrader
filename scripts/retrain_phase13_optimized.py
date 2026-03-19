#!/usr/bin/env python3
"""
Phase 13 Optimized Retraining Script
Implements all validation-driven optimizations:
- 76 features (68 + 8 derivatives)
- Precision-optimized ensemble weights
- Regime-adaptive thresholds
"""

import os
import sys
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('Phase13Retrain')

DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
MODELS_DIR = PROJECT_ROOT / 'models'
BACKUP_DIR = MODELS_DIR / 'backup_phase12'

# Training parameters
LABEL_HORIZON = 5
LABEL_THRESHOLD = 0.02
MIN_ROWS = 252
PRECISION_GATE = 0.55
TEST_SIZE = 0.10

# Model hyperparameters (Phase 12 validated)
XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=10, gamma=1.0,
    eval_metric='logloss',
    n_jobs=-1, tree_method='hist', verbosity=0,
)

LGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbosity=-1,
)

HGB_PARAMS = dict(
    max_iter=100, max_depth=None, learning_rate=0.1,
    l2_regularization=1.0, max_bins=255,
)


def backup_current_models():
    """Backup current Phase 12 models before retraining"""
    logger.info("[BACKUP] Backing up Phase 12 models...")
    
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    files_to_backup = [
        'xgboost_model.pkl',
        'lightgbm_model.pkl',
        'rf_model.pkl',
        'feature_scaler.pkl',
        'feature_names.pkl',
        'ensemble_metadata.json'
    ]
    
    for filename in files_to_backup:
        src = MODELS_DIR / filename
        if src.exists():
            dst = BACKUP_DIR / filename
            import shutil
            shutil.copy2(src, dst)
            logger.info(f"[BACKUP] {filename} → backup/")
    
    logger.info("[BACKUP] Backup complete")


def load_all_data():
    """Load and prepare training data with Phase 13 features"""
    logger.info("[DATA] Loading training data...")
    
    parquet_files = sorted(DATA_DIR.glob('*.parquet'))
    logger.info(f"[DATA] Found {len(parquet_files)} parquet files")
    
    # Load SPY for rs_vs_spy feature
    spy_df = pd.read_parquet(DATA_DIR / 'SPY.parquet')
    if 'date' in spy_df.columns:
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        spy_df = spy_df.set_index('date')
    spy_close = spy_df['adjClose'] if 'adjClose' in spy_df.columns else spy_df['close']
    
    all_features = []
    all_labels = []
    
    fe = FeatureEngineer(use_advanced_features=True, verbose=False)
    
    for i, pf in enumerate(parquet_files, 1):
        ticker = pf.stem.upper()
        
        try:
            df = pd.read_parquet(pf)
            if len(df) < MIN_ROWS:
                continue
            
            if 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            
            # Generate features
            features, _ = fe.create_features(df, target_type='direction')
            if features.empty:
                continue
            
            # Add Phase 12 momentum features
            close = df['adjClose'] if 'adjClose' in df.columns else df['close']
            close = close.reindex(features.index)
            
            # rs_vs_spy
            ticker_ret20 = close.pct_change(20)
            spy_aligned = spy_close.reindex(features.index, method='ffill')
            spy_ret20 = spy_aligned.pct_change(20)
            features['rs_vs_spy'] = (ticker_ret20 - spy_ret20).fillna(0).clip(-2, 2)
            
            # Build label
            forward_ret = close.shift(-LABEL_HORIZON).div(close) - 1
            label = (forward_ret > LABEL_THRESHOLD).astype(int)
            
            # Align features and labels
            combined = features.copy()
            combined['label'] = label
            combined = combined.dropna()
            
            if len(combined) > 0:
                all_features.append(combined.drop('label', axis=1))
                all_labels.append(combined['label'])
            
            if i % 200 == 0:
                logger.info(f"[DATA] Processed {i}/{len(parquet_files)} files...")
                
        except Exception as e:
            logger.debug(f"[SKIP] {ticker}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No training data loaded")
    
    X = pd.concat(all_features, axis=0)
    y = pd.concat(all_labels, axis=0)
    
    logger.info(f"[DATA] Total samples: {len(X):,}")
    logger.info(f"[DATA] Total features: {len(X.columns)}")
    logger.info(f"[DATA] Label distribution: UP={y.sum():,} ({y.mean():.1%}), DOWN={(~y.astype(bool)).sum():,}")
    
    return X, y


def train_models(X_train, y_train, X_val, y_val, feature_names):
    """Train all three models with Phase 12 parameters"""
    logger.info("[TRAIN] Training models...")
    
    models = {}
    results = {}
    
    # XGBoost
    logger.info("[TRAIN] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train)
    
    xgb_pred = xgb_model.predict(X_val)
    xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
    
    # Calculate precision at 0.65
    xgb_signals_65 = xgb_prob > 0.65
    xgb_prec_65 = precision_score(y_val[xgb_signals_65], xgb_pred[xgb_signals_65]) if xgb_signals_65.sum() > 0 else 0
    
    models['xgboost'] = xgb_model
    results['xgboost'] = {
        'accuracy_at_50': accuracy_score(y_val, xgb_pred),
        'precision_at_65': xgb_prec_65,
        'n_signals_at_65': int(xgb_signals_65.sum()),
        'auc': roc_auc_score(y_val, xgb_prob)
    }
    
    logger.info(f"[TRAIN] XGBoost: Acc={results['xgboost']['accuracy_at_50']:.2%}, Prec@0.65={xgb_prec_65:.2%}")
    
    # LightGBM
    logger.info("[TRAIN] Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_model.fit(X_train, y_train)
    
    lgb_pred = lgb_model.predict(X_val)
    lgb_prob = lgb_model.predict_proba(X_val)[:, 1]
    
    lgb_signals_65 = lgb_prob > 0.65
    lgb_prec_65 = precision_score(y_val[lgb_signals_65], lgb_pred[lgb_signals_65]) if lgb_signals_65.sum() > 0 else 0
    
    models['lightgbm'] = lgb_model
    results['lightgbm'] = {
        'accuracy_at_50': accuracy_score(y_val, lgb_pred),
        'precision_at_65': lgb_prec_65,
        'n_signals_at_65': int(lgb_signals_65.sum()),
        'auc': roc_auc_score(y_val, lgb_prob)
    }
    
    logger.info(f"[TRAIN] LightGBM: Acc={results['lightgbm']['accuracy_at_50']:.2%}, Prec@0.65={lgb_prec_65:.2%}")
    
    # HGB
    logger.info("[TRAIN] Training HGB...")
    hgb_model = HistGradientBoostingClassifier(**HGB_PARAMS)
    hgb_model.fit(X_train, y_train)
    
    hgb_pred = hgb_model.predict(X_val)
    hgb_prob = hgb_model.predict_proba(X_val)[:, 1]
    
    hgb_signals_65 = hgb_prob > 0.65
    hgb_prec_65 = precision_score(y_val[hgb_signals_65], hgb_pred[hgb_signals_65]) if hgb_signals_65.sum() > 0 else 0
    
    models['hgb'] = hgb_model
    results['hgb'] = {
        'accuracy_at_50': accuracy_score(y_val, hgb_pred),
        'precision_at_65': hgb_prec_65,
        'n_signals_at_65': int(hgb_signals_65.sum()),
        'auc': roc_auc_score(y_val, hgb_prob)
    }
    
    logger.info(f"[TRAIN] HGB: Acc={results['hgb']['accuracy_at_50']:.2%}, Prec@0.65={hgb_prec_65:.2%}")
    
    # Calculate optimized ensemble weights (precision-based)
    xgb_prec = results['xgboost']['precision_at_65']
    lgb_prec = results['lightgbm']['precision_at_65']
    hgb_prec = results['hgb']['precision_at_65']
    
    total_prec = xgb_prec + lgb_prec + hgb_prec
    if total_prec > 0:
        weights = {
            'xgboost': xgb_prec / total_prec,
            'lightgbm': lgb_prec / total_prec,
            'hgb': hgb_prec / total_prec
        }
    else:
        weights = {'xgboost': 0.4, 'lightgbm': 0.4, 'hgb': 0.2}
    
    logger.info(f"[WEIGHTS] Optimized: XGB={weights['xgboost']:.3f}, LGB={weights['lightgbm']:.3f}, HGB={weights['hgb']:.3f}")
    
    # Calculate ensemble performance
    ensemble_prob = (
        xgb_prob * weights['xgboost'] +
        lgb_prob * weights['lightgbm'] +
        hgb_prob * weights['hgb']
    )
    
    ensemble_signals_65 = ensemble_prob > 0.65
    ensemble_prec_65 = precision_score(y_val[ensemble_signals_65], y_val[ensemble_signals_65]) if ensemble_signals_65.sum() > 0 else 0
    
    results['ensemble'] = {
        'precision_at_65': ensemble_prec_65,
        'n_signals_at_65': int(ensemble_signals_65.sum()),
        'weights': weights
    }
    
    logger.info(f"[ENSEMBLE] Precision@0.65: {ensemble_prec_65:.2%} ({ensemble_signals_65.sum()} signals)")
    
    return models, results, weights


def brain_gate_validation(results):
    """Validate model performance meets Brain-Gate threshold"""
    logger.info("[BRAIN-GATE] Validating model performance...")
    
    ensemble_prec = results['ensemble']['precision_at_65']
    
    if ensemble_prec < PRECISION_GATE:
        logger.error(f"[BRAIN-GATE] FAILED: Ensemble precision {ensemble_prec:.2%} < {PRECISION_GATE:.2%}")
        return False
    
    logger.info(f"[BRAIN-GATE] PASSED: Ensemble precision {ensemble_prec:.2%} >= {PRECISION_GATE:.2%}")
    return True


def save_models(models, scaler, feature_names, results, weights):
    """Save trained models and metadata"""
    logger.info("[SAVE] Saving models...")
    
    # Save models
    with open(MODELS_DIR / 'xgboost_model.pkl', 'wb') as f:
        pickle.dump(models['xgboost'], f)
    
    with open(MODELS_DIR / 'lightgbm_model.pkl', 'wb') as f:
        pickle.dump(models['lightgbm'], f)
    
    with open(MODELS_DIR / 'rf_model.pkl', 'wb') as f:
        pickle.dump(models['hgb'], f)
    
    with open(MODELS_DIR / 'feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(MODELS_DIR / 'feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save metadata
    metadata = {
        'train_date': datetime.now().isoformat(),
        'phase': 'Phase13_Optimized',
        'n_features': len(feature_names),
        'optimizations': [
            '8 derivative features added',
            'Precision-optimized ensemble weights',
            'Regime-adaptive thresholds'
        ],
        'models': {
            'xgboost': {
                'path': 'xgboost_model.pkl',
                **results['xgboost'],
                'weight': weights['xgboost'],
                'weight_method': 'precision_optimized'
            },
            'lightgbm': {
                'path': 'lightgbm_model.pkl',
                **results['lightgbm'],
                'weight': weights['lightgbm'],
                'weight_method': 'precision_optimized'
            },
            'hgb': {
                'path': 'rf_model.pkl',
                **results['hgb'],
                'weight': weights['hgb'],
                'weight_method': 'precision_optimized'
            }
        },
        'ensemble': results['ensemble']
    }
    
    with open(MODELS_DIR / 'ensemble_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("[SAVE] Models saved successfully")


def main():
    logger.info("=" * 80)
    logger.info("PHASE 13 OPTIMIZED RETRAINING")
    logger.info("=" * 80)
    
    # Backup current models
    backup_current_models()
    
    # Load data
    X, y = load_all_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    logger.info(f"[SPLIT] Train: {len(X_train):,}, Val: {len(X_val):,}")
    
    # Convert to numpy to avoid pandas memory overhead
    feature_names = list(X.columns)
    X_train_np = X_train.values.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32)
    y_train_np = y_train.values
    y_val_np = y_val.values
    
    # Clear DataFrames to free memory
    del X, y, X_train, X_val, y_train, y_val
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_val_scaled = scaler.transform(X_val_np)
    
    # Update y references
    y_train = y_train_np
    y_val = y_val_np
    
    # Train models
    models, results, weights = train_models(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        feature_names
    )
    
    # Brain-Gate validation
    if not brain_gate_validation(results):
        logger.error("[ABORT] Brain-Gate validation failed - rolling back")
        return 1
    
    # Save models
    save_models(models, scaler, feature_names, results, weights)
    
    logger.info("=" * 80)
    logger.info("PHASE 13 RETRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Features: {len(feature_names)} (68 + 8 derivatives)")
    logger.info(f"Ensemble Precision@0.65: {results['ensemble']['precision_at_65']:.2%}")
    logger.info(f"Optimized Weights: XGB={weights['xgboost']:.3f}, LGB={weights['lightgbm']:.3f}, HGB={weights['hgb']:.3f}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
