"""
NeuralTrader Ensemble Training - Tiingo Integration Demo
=========================================================

Demonstration of Tiingo-based training using existing data.
Shows the pipeline without requiring Tiingo API access.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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

class TiingoDemoTrainer:
    """
    Demo trainer using existing cached data to simulate Tiingo integration
    """
    
    def __init__(self):
        """Initialize the demo trainer"""
        self.feature_engineer = FeatureEngineer()
        
        # Production models directory
        self.models_dir = PROJECT_ROOT / 'models' / 'production'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Data directory (use existing parquet files)
        self.data_dir = PROJECT_ROOT / 'data' / 'raw'
        
        print("🚀 Tiingo Demo Trainer initialized")
        print(f"📁 Models directory: {self.models_dir}")
        print(f"📊 Data directory: {self.data_dir}")
        print(f"🔄 Simulating Tiingo 50-stock batch processing")
        print()
    
    def load_training_data(self, max_tickers: int = 50) -> pd.DataFrame:
        """
        Load training data from existing parquet files (simulating Tiingo data)
        
        Args:
            max_tickers: Maximum number of tickers to use
            
        Returns:
            Combined DataFrame with all features
        """
        print(f"[STEP 1] Loading data (simulating Tiingo fetch)...")
        
        # Get parquet files
        parquet_files = list(self.data_dir.glob("*.parquet"))
        
        # Filter out non-ticker files
        parquet_files = [f for f in parquet_files if f.stem not in ['feature_metadata', 'master_feature_matrix', 'scored_data']]
        
        if max_tickers:
            parquet_files = parquet_files[:max_tickers]
        
        print(f"[DATA] Found {len(parquet_files)} ticker files")
        
        all_data = []
        successful_tickers = 0
        
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                
                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    continue
                
                # Simulate adjClose (use close for demo)
                df['adjClose'] = df['close']
                
                # Add ticker column
                ticker = file.stem
                df['ticker'] = ticker
                
                # Calculate returns using adjClose
                df['returns'] = df['adjClose'].pct_change()
                
                # Create target variable (1 if next day return > 0, else 0)
                df['target'] = (df['returns'].shift(-1) > 0).astype(int)
                
                # Remove last row (no target) and first row (NaN returns)
                df = df.iloc[1:-1].copy()
                
                if len(df) < 200:  # Minimum after processing
                    continue
                
                all_data.append(df)
                successful_tickers += 1
                
            except Exception as e:
                print(f"[ERROR] {file.stem}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid data processed")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"[OK] Successfully processed {successful_tickers} tickers")
        print(f"[DATA] Total rows: {len(combined_df):,}")
        print(f"[DATA] Target distribution: {combined_df['target'].value_counts().to_dict()}")
        
        return combined_df
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical features
        
        Args:
            df: Raw OHLCV data
            
        Returns:
            DataFrame with features
        """
        print(f"[STEP 2] Generating technical indicators...")
        
        feature_dfs = []
        
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].copy()
            
            try:
                # Generate features using feature engineer
                features_df, _ = self.feature_engineer.create_features(ticker_df)
                
                if features_df is not None and len(features_df) > 0:
                    # Align target with features (they may have different lengths due to processing)
                    # Find common indices
                    common_indices = ticker_df.index.intersection(features_df.index)
                    if len(common_indices) == 0:
                        continue
                    
                    # Get target values for common indices
                    target_values = ticker_df.loc[common_indices, 'target'].values
                    
                    # Add target and ticker
                    features_df = features_df.loc[common_indices].copy()
                    features_df['target'] = target_values
                    features_df['ticker'] = ticker
                    
                    feature_dfs.append(features_df)
                
            except Exception as e:
                print(f"[ERROR] Features for {ticker}: {e}")
                continue
        
        if not feature_dfs:
            raise ValueError("No features generated")
        
        # Combine all features
        combined_features = pd.concat(feature_dfs, ignore_index=True)
        
        # Remove rows with NaN values
        combined_features = combined_features.dropna()
        
        print(f"[OK] Generated {len(combined_features.columns)-3} features")  # -3 for target, ticker, returns
        print(f"[DATA] Training samples: {len(combined_features):,}")
        print(f"[DATA] Target distribution: {combined_features['target'].value_counts().to_dict()}")
        
        return combined_features
    
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            X_train, X_test, y_train, y_test, scaler
        """
        print(f"[STEP 3] Preparing training data...")
        
        # Remove non-feature columns
        feature_cols = [col for col in df.columns if col not in ['target', 'ticker', 'returns']]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"[OK] Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"[OK] Features scaled")
        print(f"[DATA] Numeric features: {len(feature_cols)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols
    
    def train_models(self, X_train, X_test, y_train, y_test) -> Dict:
        """
        Train ensemble models
        
        Args:
            X_train, X_test, y_train, y_test: Training data
            
        Returns:
            Dictionary of trained models
        """
        print(f"[STEP 4] Training ensemble models...")
        
        models = {}
        
        # XGBoost
        print(f"[TRAINING] XGBoost...")
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        
        train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
        test_acc = accuracy_score(y_test, xgb_model.predict(X_test))
        print(f"[OK] XGBoost trained")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        # LightGBM
        print(f"[TRAINING] LightGBM...")
        lgb_model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model
        
        train_acc = accuracy_score(y_train, lgb_model.predict(X_train))
        test_acc = accuracy_score(y_test, lgb_model.predict(X_test))
        print(f"[OK] LightGBM trained")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        # RandomForest
        print(f"[TRAINING] RandomForest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['randomforest'] = rf_model
        
        train_acc = accuracy_score(y_train, rf_model.predict(X_train))
        test_acc = accuracy_score(y_test, rf_model.predict(X_test))
        print(f"[OK] RandomForest trained")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        return models
    
    def evaluate_ensemble(self, models: Dict, X_test, y_test) -> float:
        """
        Evaluate ensemble performance
        
        Args:
            models: Dictionary of trained models
            X_test, y_test: Test data
            
        Returns:
            Ensemble accuracy
        """
        print(f"[STEP 5] Evaluating ensemble...")
        
        # Weighted voting
        weights = {'xgboost': 0.35, 'lightgbm': 0.35, 'randomforest': 0.30}
        
        predictions = []
        for name, model in models.items():
            pred = model.predict_proba(X_test)[:, 1]  # Probability of class 1
            predictions.append(pred * weights[name])
        
        ensemble_pred = np.sum(predictions, axis=0)
        ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
        
        ensemble_acc = accuracy_score(y_test, ensemble_pred_binary)
        
        print(f"[ENSEMBLE] Weighted Average Accuracy: {ensemble_acc:.4f}")
        
        return ensemble_acc
    
    def save_models(self, models: Dict, scaler, feature_cols: List[str]) -> None:
        """
        Save models to production directory
        
        Args:
            models: Dictionary of trained models
            scaler: Feature scaler
            feature_cols: List of feature names
        """
        print(f"[STEP 6] Saving models...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save models
        model_files = {
            'xgboost': self.models_dir / f'xgboost_model_{timestamp}.pkl',
            'lightgbm': self.models_dir / f'lightgbm_model_{timestamp}.pkl',
            'randomforest': self.models_dir / f'rf_model_{timestamp}.pkl',
            'scaler': self.models_dir / f'feature_scaler_{timestamp}.pkl',
            'features': self.models_dir / f'feature_names_{timestamp}.pkl',
            'metadata': self.models_dir / f'ensemble_metadata_{timestamp}.json'
        }
        
        # Save individual models
        for name, model in models.items():
            with open(model_files[name], 'wb') as f:
                pickle.dump(model, f)
            print(f"[OK] {name.capitalize()} saved: {model_files[name]}")
        
        # Save scaler
        with open(model_files['scaler'], 'wb') as f:
            pickle.dump(scaler, f)
        print(f"[OK] Scaler saved: {model_files['scaler']}")
        
        # Save feature names
        with open(model_files['features'], 'wb') as f:
            pickle.dump(feature_cols, f)
        print(f"[OK] Feature names saved: {model_files['features']}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'models': list(models.keys()),
            'feature_count': len(feature_cols),
            'training_date': datetime.now().isoformat(),
            'data_source': 'Tiingo (Demo)',
            'batch_size': 50,
            'universe_size': 50,
            'note': 'Demo using existing parquet data to simulate Tiingo integration'
        }
        
        with open(model_files['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Metadata saved: {model_files['metadata']}")
        
        # Also save to main models directory for compatibility
        main_models_dir = PROJECT_ROOT / 'models'
        for name, model in models.items():
            main_file = main_models_dir / f'{name}_model.pkl'
            with open(main_file, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler and features to main directory
        with open(main_models_dir / 'feature_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open(main_models_dir / 'feature_names.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        
        print(f"[OK] All models saved to {self.models_dir}")
        print(f"[OK] Models also saved to main models directory for compatibility")
    
    def run_training(self, max_tickers: int = 50) -> bool:
        """
        Run complete training pipeline
        
        Args:
            max_tickers: Maximum number of tickers to use
            
        Returns:
            True if successful
        """
        try:
            print("=" * 80)
            print("NeuralTrader Tiingo Ensemble Training (Demo)")
            print("=" * 80)
            print(f"[TIME] Started: {datetime.now()}")
            print()
            
            # Load data
            raw_data = self.load_training_data(max_tickers)
            
            # Generate features
            features_df = self.generate_features(raw_data)
            
            # Prepare training data
            X_train, X_test, y_train, y_test, scaler, feature_cols = self.prepare_training_data(features_df)
            
            # Train models
            models = self.train_models(X_train, X_test, y_train, y_test)
            
            # Evaluate ensemble
            ensemble_acc = self.evaluate_ensemble(models, X_test, y_test)
            
            # Save models
            self.save_models(models, scaler, feature_cols)
            
            print()
            print("=" * 80)
            print("[SUCCESS] Tiingo Ensemble Training Complete (Demo)")
            print("=" * 80)
            print(f"[TIME] Finished: {datetime.now()}")
            print()
            print("[SUMMARY]")
            print(f"  XGBoost:      Test Acc = {accuracy_score(y_test, models['xgboost'].predict(X_test)):.4f}")
            print(f"  LightGBM:     Test Acc = {accuracy_score(y_test, models['lightgbm'].predict(X_test)):.4f}")
            print(f"  RandomForest: Test Acc = {accuracy_score(y_test, models['randomforest'].predict(X_test)):.4f}")
            print(f"  Ensemble:     Test Acc = {ensemble_acc:.4f}")
            print()
            print("[MODELS SAVED]")
            print(f"  {self.models_dir}")
            print(f"  {PROJECT_ROOT / 'models'} (for compatibility)")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main training function"""
    trainer = TiingoDemoTrainer()
    success = trainer.run_training(max_tickers=50)
    
    if success:
        print("\n✅ DATA UPDATED | MODELS TRAINED | SIMULATION READY")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
