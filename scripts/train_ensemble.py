"""
NeuralTrader Ensemble Model Training - The Council
Train XGBoost, LightGBM, and RandomForest models for ensemble voting
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engineer import FeatureEngineer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_raw_data(data_dir: Path, max_tickers: int = 50):
    """Load raw OHLCV data from parquet files"""
    print(f"[DATA] Loading raw data from {data_dir}")
    
    parquet_files = list(data_dir.glob("*.parquet"))
    
    # Filter out non-ticker files
    parquet_files = [f for f in parquet_files if f.stem not in ['feature_metadata', 'master_feature_matrix', 'scored_data']]
    
    if max_tickers:
        parquet_files = parquet_files[:max_tickers]
    
    print(f"[DATA] Found {len(parquet_files)} ticker files")
    
    all_data = []
    successful = 0
    
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                continue
            
            # Add ticker column
            df['ticker'] = file.stem.upper()
            all_data.append(df)
            successful += 1
            
            if successful % 50 == 0:
                print(f"[DATA] Loaded {successful} tickers...")
                
        except Exception as e:
            continue
    
    print(f"[OK] Successfully loaded {successful} tickers")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"[DATA] Total rows: {len(combined_df):,}")
    
    return combined_df

def create_target_variable(df: pd.DataFrame) -> pd.Series:
    """Create binary target: 1 if next day return > 0, else 0"""
    df = df.sort_values(['ticker', 'date'] if 'date' in df.columns else ['ticker', df.index])
    df['next_day_return'] = df.groupby('ticker')['close'].pct_change().shift(-1)
    target = (df['next_day_return'] > 0).astype(int)
    return target

def train_model(model_name: str, model, X_train, X_test, y_train, y_test):
    """Train a single model and return metrics"""
    print(f"\n[TRAINING] {model_name}...")
    
    # Train model (LightGBM doesn't accept verbose in fit)
    if hasattr(model, 'verbose'):
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"[OK] {model_name} trained")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    return model, train_acc, test_acc

def main():
    """Main ensemble training pipeline"""
    print("=" * 80)
    print("NeuralTrader Ensemble Training - The Council")
    print("=" * 80)
    print(f"[TIME] Started: {datetime.now()}")
    
    # Paths
    data_dir = PROJECT_ROOT / "data" / "raw"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading raw data...")
    df = load_raw_data(data_dir, max_tickers=50)
    
    # Step 2: Generate features
    print("\n[STEP 2] Generating technical indicators...")
    feature_engineer = FeatureEngineer(use_advanced_features=True)
    
    try:
        features, target = feature_engineer.create_features(df, target_type='direction')
        print(f"[OK] Generated {len(features.columns)} features")
        
    except Exception as e:
        print(f"[ERROR] Feature generation failed: {e}")
        return
    
    print(f"[DATA] Training samples: {len(features):,}")
    print(f"[DATA] Target distribution: {target.value_counts().to_dict()}")
    
    # Remove non-numeric columns
    numeric_features = features.select_dtypes(include=[np.number])
    print(f"[DATA] Numeric features: {len(numeric_features.columns)}")
    
    # Step 3: Train/test split
    print("\n[STEP 3] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        numeric_features, target, test_size=0.2, random_state=42, stratify=target
    )
    print(f"[OK] Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Step 4: Scale features
    print("\n[STEP 4] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[OK] Features scaled")
    
    # Step 5: Train ensemble models
    print("\n[STEP 5] Training ensemble models...")
    
    models = {}
    metrics = {}
    
    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    models['xgboost'], train_acc, test_acc = train_model(
        'XGBoost', xgb_model, X_train_scaled, X_test_scaled, y_train, y_test
    )
    metrics['xgboost'] = {'train_acc': train_acc, 'test_acc': test_acc}
    
    # LightGBM
    lgbm_model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    models['lightgbm'], train_acc, test_acc = train_model(
        'LightGBM', lgbm_model, X_train_scaled, X_test_scaled, y_train, y_test
    )
    metrics['lightgbm'] = {'train_acc': train_acc, 'test_acc': test_acc}
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    models['rf'], train_acc, test_acc = train_model(
        'RandomForest', rf_model, X_train_scaled, X_test_scaled, y_train, y_test
    )
    metrics['rf'] = {'train_acc': train_acc, 'test_acc': test_acc}
    
    # Step 6: Ensemble evaluation
    print("\n[STEP 6] Evaluating ensemble...")
    
    # Get probabilities from each model
    xgb_probs = models['xgboost'].predict_proba(X_test_scaled)[:, 1]
    lgbm_probs = models['lightgbm'].predict_proba(X_test_scaled)[:, 1]
    rf_probs = models['rf'].predict_proba(X_test_scaled)[:, 1]
    
    # Weighted average (equal weights for now)
    ensemble_probs = (xgb_probs * 0.35 + lgbm_probs * 0.35 + rf_probs * 0.30)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    print(f"[ENSEMBLE] Weighted Average Accuracy: {ensemble_acc:.4f}")
    
    # Step 7: Save models
    print("\n[STEP 7] Saving models...")
    
    # Save XGBoost
    xgb_path = models_dir / "xgboost_model.pkl"
    with open(xgb_path, 'wb') as f:
        pickle.dump(models['xgboost'], f)
    print(f"[OK] XGBoost saved: {xgb_path}")
    
    # Save LightGBM
    lgbm_path = models_dir / "lightgbm_model.pkl"
    with open(lgbm_path, 'wb') as f:
        pickle.dump(models['lightgbm'], f)
    print(f"[OK] LightGBM saved: {lgbm_path}")
    
    # Save RandomForest
    rf_path = models_dir / "rf_model.pkl"
    with open(rf_path, 'wb') as f:
        pickle.dump(models['rf'], f)
    print(f"[OK] RandomForest saved: {rf_path}")
    
    # Save scaler
    scaler_path = models_dir / "feature_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[OK] Scaler saved: {scaler_path}")
    
    # Save feature names
    feature_names_path = models_dir / "feature_names.pkl"
    with open(feature_names_path, 'wb') as f:
        pickle.dump(list(numeric_features.columns), f)
    print(f"[OK] Feature names saved: {feature_names_path}")
    
    # Step 8: Check for neural ranker
    print("\n[STEP 8] Checking for neural ranker...")
    neural_ranker_path = models_dir / "neural_ranker_v1.json"
    has_neural_ranker = neural_ranker_path.exists()
    
    if has_neural_ranker:
        print(f"[OK] Neural Ranker found: {neural_ranker_path}")
    else:
        print("[INFO] Neural Ranker not found (optional)")
    
    # Step 9: Save ensemble metadata
    print("\n[STEP 9] Saving ensemble metadata...")
    
    ensemble_metadata = {
        'train_date': datetime.now().isoformat(),
        'n_features': len(numeric_features.columns),
        'n_samples': len(numeric_features),
        'feature_names': list(numeric_features.columns),
        'models': {
            'xgboost': {
                'path': 'xgboost_model.pkl',
                'train_acc': float(metrics['xgboost']['train_acc']),
                'test_acc': float(metrics['xgboost']['test_acc']),
                'weight': 0.35
            },
            'lightgbm': {
                'path': 'lightgbm_model.pkl',
                'train_acc': float(metrics['lightgbm']['train_acc']),
                'test_acc': float(metrics['lightgbm']['test_acc']),
                'weight': 0.35
            },
            'rf': {
                'path': 'rf_model.pkl',
                'train_acc': float(metrics['rf']['train_acc']),
                'test_acc': float(metrics['rf']['test_acc']),
                'weight': 0.30
            }
        },
        'ensemble': {
            'test_acc': float(ensemble_acc),
            'has_neural_ranker': has_neural_ranker
        }
    }
    
    metadata_path = models_dir / "ensemble_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)
    print(f"[OK] Metadata saved: {metadata_path}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("[SUCCESS] Ensemble Training Complete - The Council is Ready")
    print("=" * 80)
    print(f"[TIME] Finished: {datetime.now()}")
    print(f"\n[SUMMARY]")
    print(f"  XGBoost:      Test Acc = {metrics['xgboost']['test_acc']:.2%}")
    print(f"  LightGBM:     Test Acc = {metrics['lightgbm']['test_acc']:.2%}")
    print(f"  RandomForest: Test Acc = {metrics['rf']['test_acc']:.2%}")
    print(f"  Ensemble:     Test Acc = {ensemble_acc:.2%}")
    print(f"\n[MODELS SAVED]")
    print(f"  {xgb_path}")
    print(f"  {lgbm_path}")
    print(f"  {rf_path}")
    print(f"  {scaler_path}")
    print(f"  {metadata_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
