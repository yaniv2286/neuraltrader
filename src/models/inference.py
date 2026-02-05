"""
XGBoost Inference Engine
Loads trained model and generates trading signals with confidence scores
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

class XGBoostInference:
    """XGBoost model inference for trading signals"""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the inference engine
        
        Args:
            models_dir: Path to models directory (default: PROJECT_ROOT/models)
        """
        self.logger = logging.getLogger('XGBoostInference')
        
        # Set models directory
        if models_dir is None:
            project_root = Path(__file__).parent.parent.parent
            models_dir = project_root / "models"
        
        self.models_dir = Path(models_dir)
        
        # Model components
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load XGBoost model, scaler, and feature names"""
        try:
            # Load model
            model_path = self.models_dir / "xgboost_model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.logger.info(f"[OK] XGBoost model loaded from {model_path}")
            
            # Load scaler
            scaler_path = self.models_dir / "feature_scaler.pkl"
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.logger.info(f"[OK] Feature scaler loaded from {scaler_path}")
            
            # Load feature names
            feature_names_path = self.models_dir / "feature_names.pkl"
            if not feature_names_path.exists():
                raise FileNotFoundError(f"Feature names not found: {feature_names_path}")
            
            with open(feature_names_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            self.logger.info(f"[OK] Feature names loaded: {len(self.feature_names)} features")
            
            # Load metadata
            metadata_path = self.models_dir / "model_metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.logger.info(f"[OK] Model metadata loaded (Test Accuracy: {self.metadata.get('test_accuracy', 0):.2%})")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load models: {e}")
            raise
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from OHLCV data
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with features matching training
        """
        try:
            from src.features.feature_engineer import FeatureEngineer
            
            # Create feature engineer
            feature_engineer = FeatureEngineer(use_advanced_features=True)
            
            # Generate features
            features, _ = feature_engineer.create_features(data, target_type='direction')
            
            # Select only numeric features
            numeric_features = features.select_dtypes(include=[np.number])
            
            # Ensure we have all required features
            missing_features = set(self.feature_names) - set(numeric_features.columns)
            if missing_features:
                self.logger.warning(f"[WARN] Missing features: {missing_features}")
                # Add missing features with zeros
                for feat in missing_features:
                    numeric_features[feat] = 0
            
            # Select features in correct order
            ordered_features = numeric_features[self.feature_names]
            
            return ordered_features
            
        except Exception as e:
            self.logger.error(f"[ERROR] Feature generation failed: {e}")
            raise
    
    def predict(self, features: pd.DataFrame) -> Tuple[str, float, Dict]:
        """
        Generate trading signal from features
        
        Args:
            features: DataFrame with features (must match training features)
            
        Returns:
            Tuple of (signal, confidence, details)
            - signal: 'BUY', 'SELL', or 'HOLD'
            - confidence: 0.0 to 1.0
            - details: Dict with additional info (probabilities, top features, etc.)
        """
        try:
            # Get latest row (most recent data)
            if len(features) == 0:
                return 'HOLD', 0.0, {'error': 'No features provided'}
            
            latest_features = features.iloc[[-1]]  # Keep as DataFrame
            
            # Check for NaN values
            if latest_features.isnull().any().any():
                self.logger.warning("[WARN] NaN values detected in features, filling with 0")
                latest_features = latest_features.fillna(0)
            
            # Scale features
            features_scaled = self.scaler.transform(latest_features)
            
            # Get predictions
            probabilities = self.model.predict_proba(features_scaled)[0]
            prediction = self.model.predict(features_scaled)[0]
            
            # probabilities[0] = probability of DOWN (class 0)
            # probabilities[1] = probability of UP (class 1)
            prob_down = probabilities[0]
            prob_up = probabilities[1]
            
            # Determine signal based on confidence thresholds
            if prob_up > 0.60:  # Strong BUY signal
                signal = 'BUY'
                confidence = prob_up
            elif prob_down > 0.60:  # Strong SELL signal
                signal = 'SELL'
                confidence = prob_down
            else:  # Weak signal - HOLD
                signal = 'HOLD'
                confidence = max(prob_up, prob_down)
            
            # Get feature importance for explainability
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features = feature_importance.head(3).to_dict('records')
            
            # Prepare details
            details = {
                'prob_up': float(prob_up),
                'prob_down': float(prob_down),
                'prediction': int(prediction),
                'top_features': top_features,
                'model_accuracy': self.metadata.get('test_accuracy', 0) if self.metadata else 0
            }
            
            self.logger.info(f"[ML] Signal: {signal}, Confidence: {confidence:.2%}")
            self.logger.info(f"[ML] Probabilities - UP: {prob_up:.2%}, DOWN: {prob_down:.2%}")
            
            return signal, confidence, details
            
        except Exception as e:
            self.logger.error(f"[ERROR] Prediction failed: {e}")
            return 'HOLD', 0.0, {'error': str(e)}
    
    def predict_from_ohlcv(self, data: pd.DataFrame) -> Tuple[str, float, Dict]:
        """
        Generate trading signal directly from OHLCV data
        
        Args:
            data: DataFrame with OHLCV columns (open, high, low, close, volume)
            
        Returns:
            Tuple of (signal, confidence, details)
        """
        try:
            # Generate features
            features = self.generate_features(data)
            
            # Make prediction
            signal, confidence, details = self.predict(features)
            
            return signal, confidence, details
            
        except Exception as e:
            self.logger.error(f"[ERROR] OHLCV prediction failed: {e}")
            return 'HOLD', 0.0, {'error': str(e)}
    
    def get_model_info(self) -> Dict:
        """Get model metadata and information"""
        return {
            'model_type': 'XGBoostClassifier',
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }

# Convenience function for quick inference
def get_trading_signal(data: pd.DataFrame, models_dir: Optional[Path] = None) -> Tuple[str, float, Dict]:
    """
    Quick function to get trading signal from OHLCV data
    
    Args:
        data: DataFrame with OHLCV columns
        models_dir: Optional path to models directory
        
    Returns:
        Tuple of (signal, confidence, details)
    """
    inference_engine = XGBoostInference(models_dir=models_dir)
    return inference_engine.predict_from_ohlcv(data)
