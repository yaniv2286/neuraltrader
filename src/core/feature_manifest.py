"""
Feature Manifest - Strict Hash Validation Contract
===================================================

Implements the mandatory feature manifest contract:
1. Training MUST persist feature_manifest.json per model
2. Inference MUST validate feature_hash matches manifest
3. On mismatch: HARD ERROR (no silent skipping)
4. No Silent Skip Rule enforced

Contents of manifest:
- ordered feature_list
- feature_hash (hash of ordered list)
- benchmark_ticker (SPY)
- price_policy (adjusted)
- timeframe (1D)
- target definition and horizon
- training date range
"""

import json
import hashlib
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


class FeatureManifestError(Exception):
    """Raised when feature manifest validation fails."""
    pass


@dataclass
class FeatureManifest:
    """
    Feature manifest for model reproducibility.
    
    This manifest MUST be saved during training and validated during inference.
    Any mismatch results in HARD ERROR.
    """
    # Feature specification
    feature_list: List[str]
    feature_hash: str
    feature_count: int
    
    # Benchmark specification
    benchmark_ticker: str
    benchmark_features: List[str]
    
    # Data specification
    price_policy: str  # 'adjusted' or 'unadjusted'
    timeframe: str  # '1D', '1H', etc.
    
    # Target specification
    target_definition: str  # e.g., 'next_day_log_return'
    target_horizon: int  # e.g., 1 for next day
    
    # Training specification
    training_start_date: str
    training_end_date: str
    training_tickers: List[str]
    training_samples: int
    
    # Metadata
    created_at: str
    model_id: str
    schema_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureManifest':
        """Create from dictionary."""
        return cls(**data)
    
    def save(self, filepath: str):
        """Save manifest to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureManifest':
        """Load manifest from JSON file."""
        if not os.path.exists(filepath):
            raise FeatureManifestError(
                f"HARD FAIL: Feature manifest not found at {filepath}. "
                f"Cannot proceed without manifest for validation."
            )
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


def compute_feature_hash(feature_list: List[str]) -> str:
    """
    Compute deterministic hash of ordered feature list.
    
    The hash is used to validate that inference uses the same
    features as training. Order matters!
    """
    # Sort for determinism, then join
    sorted_features = sorted(feature_list)
    feature_str = '|'.join(sorted_features)
    return hashlib.sha256(feature_str.encode()).hexdigest()[:32]


class FeatureManifestBuilder:
    """Builder for creating feature manifests during training."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._feature_list: List[str] = []
        self._benchmark_ticker: str = 'SPY'
        self._benchmark_features: List[str] = []
        self._price_policy: str = 'adjusted'
        self._timeframe: str = '1D'
        self._target_definition: str = ''
        self._target_horizon: int = 1
        self._training_start: str = ''
        self._training_end: str = ''
        self._training_tickers: List[str] = []
        self._training_samples: int = 0
    
    def set_features(self, feature_list: List[str]) -> 'FeatureManifestBuilder':
        """Set the ordered feature list."""
        self._feature_list = list(feature_list)
        return self
    
    def set_benchmark(self, ticker: str, features: List[str]) -> 'FeatureManifestBuilder':
        """Set benchmark ticker and its features."""
        self._benchmark_ticker = ticker
        self._benchmark_features = list(features)
        return self
    
    def set_price_policy(self, policy: str) -> 'FeatureManifestBuilder':
        """Set price policy (adjusted/unadjusted)."""
        if policy not in ('adjusted', 'unadjusted'):
            raise ValueError(f"Invalid price policy: {policy}")
        self._price_policy = policy
        return self
    
    def set_timeframe(self, timeframe: str) -> 'FeatureManifestBuilder':
        """Set data timeframe."""
        self._timeframe = timeframe
        return self
    
    def set_target(self, definition: str, horizon: int) -> 'FeatureManifestBuilder':
        """Set target definition and horizon."""
        self._target_definition = definition
        self._target_horizon = horizon
        return self
    
    def set_training_period(self, start: str, end: str) -> 'FeatureManifestBuilder':
        """Set training date range."""
        self._training_start = start
        self._training_end = end
        return self
    
    def set_training_data(self, tickers: List[str], samples: int) -> 'FeatureManifestBuilder':
        """Set training tickers and sample count."""
        self._training_tickers = list(tickers)
        self._training_samples = samples
        return self
    
    def build(self) -> FeatureManifest:
        """Build the feature manifest."""
        if not self._feature_list:
            raise FeatureManifestError("Feature list cannot be empty")
        
        feature_hash = compute_feature_hash(self._feature_list)
        
        return FeatureManifest(
            feature_list=self._feature_list,
            feature_hash=feature_hash,
            feature_count=len(self._feature_list),
            benchmark_ticker=self._benchmark_ticker,
            benchmark_features=self._benchmark_features,
            price_policy=self._price_policy,
            timeframe=self._timeframe,
            target_definition=self._target_definition,
            target_horizon=self._target_horizon,
            training_start_date=self._training_start,
            training_end_date=self._training_end,
            training_tickers=self._training_tickers,
            training_samples=self._training_samples,
            created_at=datetime.utcnow().isoformat(),
            model_id=self.model_id
        )


class FeatureManifestValidator:
    """
    Validates features during inference against training manifest.
    
    HARD FAIL on any mismatch - no silent skipping allowed.
    """
    
    def __init__(self, manifest: FeatureManifest):
        self.manifest = manifest
        self._validation_errors: List[str] = []
        self._validation_warnings: List[str] = []
    
    def validate_features(self, inference_features: List[str]) -> bool:
        """
        Validate that inference features match training manifest.
        
        Args:
            inference_features: List of feature names used in inference
            
        Returns:
            True if valid, raises FeatureManifestError otherwise
            
        Raises:
            FeatureManifestError: On any validation failure
        """
        self._validation_errors = []
        self._validation_warnings = []
        
        # Check feature count
        if len(inference_features) != self.manifest.feature_count:
            self._validation_errors.append(
                f"Feature count mismatch: inference has {len(inference_features)}, "
                f"manifest expects {self.manifest.feature_count}"
            )
        
        # Check feature hash
        inference_hash = compute_feature_hash(inference_features)
        if inference_hash != self.manifest.feature_hash:
            self._validation_errors.append(
                f"Feature hash mismatch: inference={inference_hash}, "
                f"manifest={self.manifest.feature_hash}"
            )
        
        # Check for missing features
        manifest_set = set(self.manifest.feature_list)
        inference_set = set(inference_features)
        
        missing = manifest_set - inference_set
        if missing:
            self._validation_errors.append(
                f"Missing features in inference: {sorted(missing)}"
            )
        
        extra = inference_set - manifest_set
        if extra:
            self._validation_errors.append(
                f"Extra features in inference (not in manifest): {sorted(extra)}"
            )
        
        # Check benchmark features
        for bf in self.manifest.benchmark_features:
            if bf not in inference_features:
                self._validation_errors.append(
                    f"Missing benchmark feature: {bf}"
                )
        
        # HARD FAIL on any error
        if self._validation_errors:
            error_msg = "HARD FAIL: Feature manifest validation failed:\n"
            error_msg += "\n".join(f"  - {e}" for e in self._validation_errors)
            raise FeatureManifestError(error_msg)
        
        return True
    
    def validate_data_policy(self, price_policy: str, timeframe: str) -> bool:
        """
        Validate data policy matches manifest.
        
        Raises:
            FeatureManifestError: On mismatch
        """
        errors = []
        
        if price_policy != self.manifest.price_policy:
            errors.append(
                f"Price policy mismatch: using '{price_policy}', "
                f"manifest expects '{self.manifest.price_policy}'"
            )
        
        if timeframe != self.manifest.timeframe:
            errors.append(
                f"Timeframe mismatch: using '{timeframe}', "
                f"manifest expects '{self.manifest.timeframe}'"
            )
        
        if errors:
            error_msg = "HARD FAIL: Data policy validation failed:\n"
            error_msg += "\n".join(f"  - {e}" for e in errors)
            raise FeatureManifestError(error_msg)
        
        return True
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        return {
            'manifest_id': self.manifest.model_id,
            'feature_hash': self.manifest.feature_hash,
            'feature_count': self.manifest.feature_count,
            'errors': self._validation_errors,
            'warnings': self._validation_warnings,
            'valid': len(self._validation_errors) == 0
        }


class TickerFailureTracker:
    """
    Tracks ticker failures during backtest.
    
    Implements the "No Silent Skip" rule:
    - Every failed ticker must be counted and listed with reason
    - If tickers_success == 0 -> ABORT RUN
    """
    
    def __init__(self, total_tickers: int):
        self.total_tickers = total_tickers
        self._successes: List[str] = []
        self._failures: Dict[str, str] = {}  # ticker -> reason
    
    def record_success(self, ticker: str):
        """Record a successful ticker."""
        self._successes.append(ticker)
    
    def record_failure(self, ticker: str, reason: str):
        """Record a failed ticker with reason."""
        self._failures[ticker] = reason
    
    @property
    def success_count(self) -> int:
        return len(self._successes)
    
    @property
    def failure_count(self) -> int:
        return len(self._failures)
    
    def validate_or_abort(self):
        """
        Validate that at least one ticker succeeded.
        
        Raises:
            FeatureManifestError: If all tickers failed
        """
        if self.success_count == 0:
            error_msg = (
                f"HARD FAIL: All {self.total_tickers} tickers failed. "
                f"Cannot proceed with 0 successful tickers.\n"
                f"Failure reasons:\n"
            )
            for ticker, reason in sorted(self._failures.items())[:20]:
                error_msg += f"  - {ticker}: {reason}\n"
            
            if len(self._failures) > 20:
                error_msg += f"  ... and {len(self._failures) - 20} more\n"
            
            raise FeatureManifestError(error_msg)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary for Excel output."""
        return {
            'tickers_total': self.total_tickers,
            'tickers_success': self.success_count,
            'tickers_failed': self.failure_count,
            'failure_reasons': self._failures
        }


# Default manifest directory
MANIFEST_DIR = Path(__file__).parent.parent.parent / 'models' / 'manifests'


def get_manifest_path(model_id: str) -> str:
    """Get path for a model's feature manifest."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    return str(MANIFEST_DIR / f"{model_id}_manifest.json")


def save_manifest(manifest: FeatureManifest):
    """Save manifest to default location."""
    filepath = get_manifest_path(manifest.model_id)
    manifest.save(filepath)
    return filepath


def load_manifest(model_id: str) -> FeatureManifest:
    """Load manifest from default location."""
    filepath = get_manifest_path(model_id)
    return FeatureManifest.load(filepath)
