"""
Core AI Models - The Machine Learning Brain
Protected module containing all ML inference engines for trading signals
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
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
            project_root = Path(__file__).parent.parent
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
            from core.feature_engineer import FeatureEngineer
            
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
            
            # 🦅 BRAIN TRANSPLANT - Tree-based models don't need scaling
            if self.scaler is not None:
                features_scaled = self.scaler.transform(latest_features)
            else:
                features_scaled = latest_features
            
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


class EnsemblePredictor:
    """Ensemble predictor with weighted voting from multiple models"""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the ensemble predictor
        
        Args:
            models_dir: Path to models directory (default: PROJECT_ROOT/models)
        """
        self.logger = logging.getLogger('EnsemblePredictor')
        
        # Set models directory
        if models_dir is None:
            project_root = Path(__file__).parent.parent
            models_dir = project_root / "models"
        
        self.models_dir = Path(models_dir)
        
        # Model components
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.weights = {}
        
        # Load models
        self._load_ensemble()
    
    def _load_ensemble(self):
        """Load all ensemble models and metadata"""
        try:
            # 🦅 BRAIN TRANSPLANT - Load from production models with new nomenclature
            production_dir = self.models_dir / "production"
            
            # Load metadata from production
            metadata_path = production_dir / "ensemble_metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                self.feature_names = self.metadata.get('feature_names', None)
                self.logger.info(f"[OK] Production ensemble metadata loaded")
            else:
                # Fallback to old metadata
                metadata_path = self.models_dir / "ensemble_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    self.logger.info(f"[OK] Legacy ensemble metadata loaded")
            
            # 🦅 BRAIN TRANSPLANT - Load new production models with STANDARD names
            model_files = {
                'xgboost': 'xgboost_model.pkl',
                'lightgbm': 'lightgbm_model.pkl', 
                'randomforest': 'rf_model.pkl'
            }
            
            # Default weights for production models
            default_weights = {
                'xgboost': 0.35,
                'lightgbm': 0.35,
                'randomforest': 0.30
            }
            
            for model_name, filename in model_files.items():
                model_path = production_dir / filename
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    self.weights[model_name] = default_weights[model_name]
                    self.logger.info(f"[OK] {model_name} model loaded from production (weight: {default_weights[model_name]:.2f})")
                else:
                    self.logger.warning(f"[WARN] {model_name} model not found: {model_path}")
            
            # Load feature names and scaler from production directory
            feature_names_path = production_dir / 'feature_names.pkl'
            scaler_path = production_dir / 'feature_scaler.pkl'
            
            if feature_names_path.exists():
                with open(feature_names_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                self.logger.info(f"[OK] Feature names loaded from {feature_names_path}")
            else:
                self.logger.warning("[WARN] Feature names not found, will be generated during training")
            
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"[OK] Scaler loaded from {scaler_path}")
            else:
                self.logger.warning("[WARN] Scaler not found, will be created during training")
            
            # No scaler needed for tree-based models
            
            if self.models:
                self.logger.info(f"[OK] Loaded {len(self.models)} production models")
            else:
                self.logger.error("[ERROR] No production models found")
                
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load ensemble: {e}")
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
            from core.feature_engineer import FeatureEngineer
            
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
    
    def predict_batch(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble probabilities for all rows in features
        
        Args:
            features: DataFrame with features (must match training features)
            
        Returns:
            numpy.ndarray: Array of probabilities for class 1 (UP)
        """
        try:
            if len(features) == 0:
                return np.array([])
            
            # Check for NaN values
            if features.isnull().any().any():
                self.logger.warning("[WARN] NaN values detected in features, filling with 0")
                features = features.fillna(0)
            
            # 🦅 BRAIN TRANSPLANT - Tree-based models don't need scaling
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Get predictions from each model for all rows
            all_probs = []
            
            for model_name, model in self.models.items():
                try:
                    # Get probability of UP (class 1) for all rows
                    probs = model.predict_proba(features_scaled)[:, 1]  # All rows, class 1
                    all_probs.append(probs)
                    
                except Exception as e:
                    self.logger.error(f"[ERROR] {model_name} batch prediction failed: {e}")
                    # Add zeros for failed model
                    all_probs.append(np.zeros(len(features)))
            
            if not all_probs:
                return np.array([])
            
            # Weighted average of all model probabilities
            weighted_probs = np.zeros(len(features))
            total_weight = 0
            
            for i, (model_name, model) in enumerate(self.models.items()):
                if model_name in self.weights:
                    weight = self.weights[model_name]
                    weighted_probs += all_probs[i] * weight
                    total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                weighted_probs /= total_weight
            
            return weighted_probs
            
        except Exception as e:
            self.logger.error(f"[ERROR] Batch prediction failed: {e}")
            return np.array([])
    
    def predict(self, features: pd.DataFrame, threshold: Optional[float] = 0.5) -> Tuple[str, float, Dict]:
        """
        Generate ensemble trading signal from features with dynamic threshold
        
        Args:
            features: DataFrame with features (must match training features)
            threshold: Dynamic threshold for BUY decision (None = return raw score only)
            
        Returns:
            Tuple of (signal, confidence, details)
            - signal: 'BUY' or 'HOLD' (based on threshold if provided)
            - confidence: 0.0 to 1.0 (weighted average probability)
            - details: Dict with vote breakdown and individual model predictions
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
            
            # 🦅 BRAIN TRANSPLANT - Tree-based models don't need scaling
            if self.scaler is not None:
                features_scaled = self.scaler.transform(latest_features)
            else:
                features_scaled = latest_features
            
            # Get predictions from each model
            model_votes = {}
            model_probs = {}
            
            for model_name, model in self.models.items():
                try:
                    # Get probability of UP (class 1)
                    probs = model.predict_proba(features_scaled)[0]
                    prob_up = probs[1]
                    
                    model_probs[model_name] = prob_up
                    model_votes[model_name] = {
                        'prob_up': float(prob_up),
                        'prob_down': float(probs[0]),
                        'weight': self.weights[model_name]
                    }
                    
                except Exception as e:
                    self.logger.error(f"[ERROR] {model_name} prediction failed: {e}")
                    model_probs[model_name] = 0.5  # Neutral if model fails
                    model_votes[model_name] = {
                        'prob_up': 0.5,
                        'prob_down': 0.5,
                        'weight': self.weights[model_name],
                        'error': str(e)
                    }
            
            # Calculate weighted average
            weighted_prob_up = sum(
                model_probs[name] * self.weights[name]
                for name in model_probs
            )
            
            # Normalize by total weight (in case some models failed)
            total_weight = sum(self.weights[name] for name in model_probs)
            if total_weight > 0:
                weighted_prob_up /= total_weight
            
            # DYNAMIC THRESHOLD - Free the Sniper!
            if threshold is None:
                # Return raw score without threshold decision
                signal = 'HOLD'  # Default, will be overridden by backtester
                confidence = weighted_prob_up
                threshold_used = None
            elif weighted_prob_up > threshold:
                signal = 'BUY'
                confidence = weighted_prob_up
                threshold_used = threshold
            else:
                signal = 'HOLD'
                confidence = weighted_prob_up
                threshold_used = threshold
            
            # Prepare details
            details = {
                'ensemble_prob_up': float(weighted_prob_up),
                'ensemble_prob_down': float(1.0 - weighted_prob_up),
                'model_votes': model_votes,
                'n_models': len(self.models),
                'ensemble_accuracy': self.metadata.get('ensemble', {}).get('test_acc', 0) if self.metadata else 0,
                'threshold_used': threshold_used
            }
            
            # Log ensemble vote - show raw score
            vote_str = " | ".join([
                f"{name.upper()}:{model_probs[name]:.2f}"
                for name in sorted(model_probs.keys())
            ])
            if threshold is None:
                self.logger.info(f"[AI] Ensemble Raw Score: {weighted_prob_up:.3f} | {vote_str}")
            else:
                self.logger.info(f"[AI] Ensemble Vote: {weighted_prob_up:.2f} ({signal}) | {vote_str} | Threshold: {threshold}")
            
            return signal, confidence, details
            
        except Exception as e:
            self.logger.error(f"[ERROR] Ensemble prediction failed: {e}")
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
    
    def get_ensemble_info(self) -> Dict:
        """Get ensemble metadata and information"""
        return {
            'n_models': len(self.models),
            'models': list(self.models.keys()),
            'weights': self.weights,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'metadata': self.metadata
        }


# Convenience functions for quick inference
def get_trading_signal(data: pd.DataFrame, models_dir: Optional[Path] = None) -> Tuple[str, float, Dict]:
    """
    Quick function to get trading signal from OHLCV data using XGBoost
    
    Args:
        data: DataFrame with OHLCV columns
        models_dir: Optional path to models directory
        
    Returns:
        Tuple of (signal, confidence, details)
    """
    inference_engine = XGBoostInference(models_dir=models_dir)
    return inference_engine.predict_from_ohlcv(data)


def get_ensemble_signal(data: pd.DataFrame, models_dir: Optional[Path] = None) -> Tuple[str, float, Dict]:
    """
    Quick function to get ensemble trading signal from OHLCV data
    
    Args:
        data: DataFrame with OHLCV columns
        models_dir: Optional path to models directory
        
    Returns:
        Tuple of (signal, confidence, details)
    """
    predictor = EnsemblePredictor(models_dir=models_dir)
    return predictor.predict_from_ohlcv(data)
