"""
XGBoost Model Training Script
Train a binary classifier for next-day return prediction using S&P 100 data
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engineer import FeatureEngineer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_processed_data(data_dir: Path, max_tickers: int = None):
    """Load all processed parquet files"""
    print(f"[DATA] Loading processed data from {data_dir}")
    
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
            
            if successful % 100 == 0:
                print(f"[DATA] Loaded {successful} tickers...")
                
        except Exception as e:
            print(f"[WARN] Failed to load {file.stem}: {e}")
            continue
    
    print(f"[OK] Successfully loaded {successful} tickers")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"[DATA] Total rows: {len(combined_df):,}")
    
    return combined_df

def create_target_variable(df: pd.DataFrame) -> pd.Series:
    """Create binary target: 1 if next day return > 0, else 0"""
    # Calculate next day return
    df = df.sort_values(['ticker', 'date'] if 'date' in df.columns else ['ticker', df.index])
    df['next_day_return'] = df.groupby('ticker')['close'].pct_change().shift(-1)
    
    # Binary classification: 1 if positive return, 0 otherwise
    target = (df['next_day_return'] > 0).astype(int)
    
    return target

def main():
    """Main training pipeline"""
    print("=" * 80)
    print("NeuralTrader XGBoost Model Training")
    print("=" * 80)
    print(f"[TIME] Started: {datetime.now()}")
    
    # Paths
    data_dir = PROJECT_ROOT / "data" / "raw"  # Use raw data with OHLCV
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading raw data...")
    df = load_processed_data(data_dir, max_tickers=50)  # Use 50 tickers for faster training
    
    # Step 2: Generate features
    print("\n[STEP 2] Generating technical indicators...")
    feature_engineer = FeatureEngineer(use_advanced_features=True)
    
    try:
        features, target = feature_engineer.create_features(df, target_type='direction')
        print(f"[OK] Generated {len(features.columns)} features")
        print(f"[OK] Feature names: {list(features.columns[:10])}...")  # Show first 10
        
    except Exception as e:
        print(f"[ERROR] Feature generation failed: {e}")
        print("[INFO] Falling back to manual feature creation...")
        
        # Manual feature creation as fallback
        df = df.copy()
        
        # Basic features
        df['returns'] = df.groupby('ticker')['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df.groupby('ticker')['close'].shift(1))
        
        # Simple moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window).mean())
        
        # RSI
        delta = df.groupby('ticker')['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features
        df['volume_sma_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20).mean())
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Create target
        target = create_target_variable(df)
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in ['ticker', 'date', 'next_day_return', 'open', 'high', 'low', 'close', 'volume']]
        features = df[feature_cols]
        
        # Drop NaNs
        valid_idx = features.notna().all(axis=1) & target.notna()
        features = features[valid_idx]
        target = target[valid_idx]
        
        print(f"[OK] Created {len(features.columns)} features manually")
    
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
    
    # Step 5: Train XGBoost
    print("\n[STEP 5] Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(X_train_scaled, y_train, verbose=False)
    print("[OK] Model trained")
    
    # Step 6: Evaluate
    print("\n[STEP 6] Evaluating model...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"[METRICS] Train Accuracy: {train_acc:.4f}")
    print(f"[METRICS] Test Accuracy: {test_acc:.4f}")
    
    print("\n[REPORT] Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=['Down', 'Up']))
    
    # Feature importance
    print("\n[IMPORTANCE] Top 10 Features:")
    feature_importance = pd.DataFrame({
        'feature': numeric_features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Step 7: Save models
    print("\n[STEP 7] Saving models...")
    model_path = models_dir / "xgboost_model.pkl"
    scaler_path = models_dir / "feature_scaler.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[OK] Model saved: {model_path}")
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[OK] Scaler saved: {scaler_path}")
    
    # Save feature names for inference
    feature_names_path = models_dir / "feature_names.pkl"
    with open(feature_names_path, 'wb') as f:
        pickle.dump(list(numeric_features.columns), f)
    print(f"[OK] Feature names saved: {feature_names_path}")
    
    # Save metadata
    metadata = {
        'train_date': datetime.now().isoformat(),
        'n_features': len(numeric_features.columns),
        'n_samples': len(numeric_features),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'feature_names': list(numeric_features.columns)
    }
    
    metadata_path = models_dir / "model_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata saved: {metadata_path}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] XGBoost Model Training Complete!")
    print("=" * 80)
    print(f"[TIME] Finished: {datetime.now()}")
    print(f"\n[SUMMARY]")
    print(f"  Model: {model_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  Features: {len(numeric_features.columns)}")
    print(f"  Test Accuracy: {test_acc:.2%}")
    print("=" * 80)

if __name__ == "__main__":
    main()
