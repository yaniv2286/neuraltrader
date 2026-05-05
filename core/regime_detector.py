"""
Regime Detector - Market regime classification for adaptive thresholds
Phase 13 - AI-driven regime detection using SPY and VXX features
"""

import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np

# CNN Fear & Greed Index integration
try:
    from fear_and_greed import get as get_fear_greed
    FEAR_GREED_AVAILABLE = True
except ImportError:
    FEAR_GREED_AVAILABLE = False


class RegimeDetector:
    """Detect market regime (CRISIS/BEAR/BULL) for adaptive confidence thresholds"""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize regime detector
        
        Args:
            models_dir: Path to models directory (default: PROJECT_ROOT/models)
        """
        self.logger = logging.getLogger('RegimeDetector')
        
        if models_dir is None:
            project_root = Path(__file__).parent.parent
            models_dir = project_root / "models"
        
        self.models_dir = Path(models_dir)
        self.classifier = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        
        # Regime labels and thresholds
        self.regime_labels = {0: 'CRISIS', 1: 'BEAR', 2: 'BULL'}
        self.regime_thresholds = {
            0: None,    # CRISIS - no new entries
            1: 0.45,    # BEAR - stricter threshold
            2: 0.35,    # BULL - calibrated for Phase 15 (5% TP, 64 clean features)
        }
        
        # CNN Fear & Greed sentiment categories
        self.sentiment_categories = {
            (0, 25): 'Extreme Fear',
            (25, 45): 'Fear',
            (45, 55): 'Neutral',
            (55, 75): 'Greed',
            (75, 101): 'Extreme Greed'
        }
        
        self._load_classifier()
    
    def _load_classifier(self):
        """Load regime classifier and scaler"""
        try:
            clf_path = self.models_dir / 'regime_classifier.pkl'
            scaler_path = self.models_dir / 'regime_scaler.pkl'
            meta_path = self.models_dir / 'regime_classifier_meta.json'
            
            if not clf_path.exists():
                self.logger.warning("[WARN] Regime classifier not found - using default BULL threshold (0.55)")
                return
            
            # Load classifier
            with open(clf_path, 'rb') as f:
                self.classifier = pickle.load(f)
            
            # Load scaler
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load metadata
            if meta_path.exists():
                import json
                with open(meta_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.feature_names = self.metadata.get('feature_names', None)
                    self.regime_thresholds = self.metadata.get('regime_thresholds', self.regime_thresholds)
            
            self.logger.info(f"[OK] Regime Classifier Loaded | Features: {len(self.feature_names) if self.feature_names else 'unknown'}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load regime classifier: {e}")
            self.classifier = None
    
    def get_cnn_sentiment(self) -> Tuple[float, str]:
        """
        Fetch CNN Fear & Greed Index
        
        Returns:
            Tuple of (sentiment_value, category)
            - sentiment_value: 0-100 (50 = neutral default on failure)
            - category: 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'
        """
        try:
            if not FEAR_GREED_AVAILABLE:
                self.logger.warning("[WARN] fear-and-greed library not available - using neutral sentiment (50)")
                return 50.0, 'Neutral'
            
            # Fetch current Fear & Greed Index
            result = get_fear_greed()
            sentiment_value = float(result.value)
            
            # Determine category
            category = 'Neutral'
            for (low, high), cat in self.sentiment_categories.items():
                if low <= sentiment_value < high:
                    category = cat
                    break
            
            self.logger.info(f"[SENTIMENT] CNN Fear & Greed: {sentiment_value:.1f} ({category})")
            return sentiment_value, category
            
        except Exception as e:
            # NO SILENT FAILURES - Log error and return neutral default
            self.logger.warning(f"[WARN] Failed to fetch CNN sentiment: {e} - using neutral default (50)")
            return 50.0, 'Neutral'
    
    def build_regime_features(self, spy_data: pd.DataFrame, vxx_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build regime features from SPY, VXX, and CNN Fear & Greed data
        
        Args:
            spy_data: SPY OHLCV DataFrame
            vxx_data: VXX OHLCV DataFrame (optional)
            
        Returns:
            DataFrame with regime features (21 features including CNN sentiment)
        """
        close_spy = spy_data['adjClose'] if 'adjClose' in spy_data.columns else spy_data['close']
        close_spy = close_spy.sort_index()
        
        features = pd.DataFrame(index=close_spy.index)
        
        # SPY trend features
        for w in [20, 50, 100, 200]:
            sma = close_spy.rolling(w, min_periods=w).mean()
            features[f'spy_sma{w}_ratio'] = (close_spy / sma).fillna(1.0)
        
        # SPY momentum
        for w in [5, 10, 20, 60]:
            features[f'spy_roc_{w}'] = close_spy.pct_change(w).fillna(0)
        
        # SPY volatility
        spy_ret = close_spy.pct_change()
        features['spy_vol_20'] = spy_ret.rolling(20).std().fillna(0) * np.sqrt(252)
        
        # SPY vol regime
        vol_60 = spy_ret.rolling(60).std()
        features['spy_vol_regime'] = (features['spy_vol_20'] / (vol_60 * np.sqrt(252))).fillna(1.0)
        
        # SPY drawdown
        high_52w = close_spy.rolling(252, min_periods=60).max()
        features['spy_dd_52w'] = ((close_spy - high_52w) / high_52w).fillna(0)
        
        # SPY above/below SMAs
        for w in [50, 100, 200]:
            sma = close_spy.rolling(w, min_periods=w).mean()
            features[f'spy_above_sma{w}'] = (close_spy > sma).astype(float)
        
        # SPY streak
        spy_up = (spy_ret > 0).astype(int)
        features['spy_streak'] = spy_up.rolling(10).sum() / 10.0
        
        # VXX features
        if vxx_data is not None and not vxx_data.empty:
            close_vxx = vxx_data['adjClose'] if 'adjClose' in vxx_data.columns else vxx_data['close']
            close_vxx = close_vxx.reindex(close_spy.index, method='ffill')
            
            features['vxx_level'] = close_vxx.ffill()
            features['vxx_roc_5'] = close_vxx.pct_change(5).fillna(0)
            features['vxx_roc_20'] = close_vxx.pct_change(20).fillna(0)
            
            # VXX Bollinger Band
            vxx_sma20 = close_vxx.rolling(20).mean()
            vxx_std20 = close_vxx.rolling(20).std()
            bb_upper = vxx_sma20 + 2 * vxx_std20
            features['vxx_bb_pct'] = ((close_vxx - vxx_sma20) / (vxx_std20 + 1e-9)).fillna(0)
            features['vxx_above_bb'] = (close_vxx > bb_upper).astype(float)
        else:
            for col in ['vxx_level', 'vxx_roc_5', 'vxx_roc_20', 'vxx_bb_pct', 'vxx_above_bb']:
                features[col] = 0.0
        
        # 🚀 CNN Fear & Greed Index (21st feature)
        sentiment_value, _ = self.get_cnn_sentiment()
        features['cnn_fear_greed'] = sentiment_value / 100.0  # Normalize to 0-1 range
        
        return features.dropna()
    
    def apply_contrarian_override(self, regime_code: int, threshold: float, ai_confidence: float) -> Tuple[int, float, bool]:
        """
        Apply contrarian buy override based on CNN Fear & Greed Index
        
        Rule: If CNN Sentiment < 20 (Extreme Fear) AND AI confidence > 0.50,
              allow BULL entry even if primary threshold is 0.55
        
        Args:
            regime_code: Current regime (0=CRISIS, 1=BEAR, 2=BULL)
            threshold: Current threshold
            ai_confidence: AI model confidence score
            
        Returns:
            Tuple of (adjusted_regime, adjusted_threshold, override_applied)
        """
        sentiment_value, sentiment_category = self.get_cnn_sentiment()
        
        # Contrarian Buy Override: Extreme Fear + Good AI Confidence
        if sentiment_value < 20 and ai_confidence > 0.60:
            self.logger.info(f"[CONTRARIAN] Extreme Fear ({sentiment_value:.1f}) + AI Confidence ({ai_confidence:.3f}) > 0.60 - Lowering threshold to 0.60")
            return 2, 0.60, True  # Force BULL regime with 0.60 threshold
        
        return regime_code, threshold, False
    
    def detect_regime(self, spy_data: pd.DataFrame, vxx_data: Optional[pd.DataFrame] = None, ai_confidence: Optional[float] = None) -> Tuple[int, str, float]:
        """
        Detect current market regime with contrarian override support
        
        Args:
            spy_data: SPY OHLCV DataFrame
            vxx_data: VXX OHLCV DataFrame (optional)
            ai_confidence: AI model confidence (optional, for contrarian override)
            
        Returns:
            Tuple of (regime_code, regime_name, confidence_threshold)
            - regime_code: 0=CRISIS, 1=BEAR, 2=BULL
            - regime_name: 'CRISIS', 'BEAR', or 'BULL'
            - confidence_threshold: Recommended threshold for this regime
        """
        if self.classifier is None:
            # Fallback: use SPY > SMA200 as BULL indicator
            close_spy = spy_data['adjClose'] if 'adjClose' in spy_data.columns else spy_data['close']
            sma200 = close_spy.rolling(200, min_periods=100).mean()
            is_bull = close_spy.iloc[-1] > sma200.iloc[-1] if len(sma200) > 0 else True
            
            regime_code = 2 if is_bull else 1  # BULL or BEAR
            regime_name = self.regime_labels[regime_code]
            threshold = self.regime_thresholds[regime_code]
            
            self.logger.info(f"[REGIME] Fallback SPY>SMA200: {regime_name} | Threshold: {threshold}")
            return regime_code, regime_name, threshold
        
        try:
            # Build features
            features = self.build_regime_features(spy_data, vxx_data)
            
            if len(features) == 0:
                self.logger.warning("[WARN] No regime features - using default BULL")
                return 2, 'BULL', 0.55
            
            # Get latest features
            latest = features.iloc[[-1]]
            
            # Ensure correct feature order
            if self.feature_names:
                latest = latest[self.feature_names]
            
            # Scale features
            if self.scaler:
                latest_scaled = self.scaler.transform(latest)
            else:
                latest_scaled = latest.values
            
            # Predict regime
            regime_code = int(self.classifier.predict(latest_scaled)[0])
            regime_name = self.regime_labels.get(regime_code, 'UNKNOWN')
            # Handle both string and int keys from JSON metadata
            threshold = self.regime_thresholds.get(regime_code, self.regime_thresholds.get(str(regime_code), 0.55))
            
            # Apply contrarian override if AI confidence provided
            if ai_confidence is not None:
                regime_code, threshold, override_applied = self.apply_contrarian_override(regime_code, threshold, ai_confidence)
                if override_applied:
                    regime_name = self.regime_labels.get(regime_code, 'UNKNOWN')
            
            # Get prediction probabilities for logging
            probs = self.classifier.predict_proba(latest_scaled)[0]
            prob_str = " | ".join([f"{self.regime_labels[i]}:{probs[i]:.2f}" for i in range(len(probs))])
            
            self.logger.info(f"[OK] Regime Classifier Loaded")
            self.logger.info(f"[REGIME] Current: {regime_name} ({regime_code}) | Threshold: {threshold} | Probs: {prob_str}")
            
            return regime_code, regime_name, threshold
            
        except Exception as e:
            self.logger.error(f"[ERROR] Regime detection failed: {e}")
            return 2, 'BULL', 0.55  # Default to BULL on error
    
    def get_threshold_for_regime(self, regime_code: int) -> Optional[float]:
        """Get confidence threshold for a given regime"""
        return self.regime_thresholds.get(regime_code, 0.55)
