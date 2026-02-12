"""
Model Cache - Save/Load Trained Models
=======================================

Caches trained ML models to disk to avoid retraining.
Saves hours of computation time on repeated backtests.
"""

import pickle
import os
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class ModelCache:
    """
    Cache for trained ML models.
    
    Models are saved to disk with metadata about training data.
    Cache key is based on: tickers, date range, features used.
    """
    
    def __init__(self, cache_dir: str = "models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_cache_key(
        self,
        tickers: List[str],
        train_start: str,
        train_end: str,
        n_features: int
    ) -> str:
        """Generate unique cache key for model configuration."""
        # Sort tickers for consistency
        tickers_sorted = sorted(tickers)
        
        # Create hash of configuration
        config_str = f"{','.join(tickers_sorted)}_{train_start}_{train_end}_{n_features}"
        cache_key = hashlib.md5(config_str.encode()).hexdigest()
        
        return cache_key
    
    def save_models(
        self,
        models: Dict,
        tickers: List[str],
        train_start: str,
        train_end: str,
        n_features: int,
        feature_names: List[str],
        train_samples: int,
        direction_accuracy: float
    ):
        """
        Save trained models to cache.
        
        Args:
            models: Dictionary of trained models (xgb, rf, lgbm)
            tickers: List of tickers used in training
            train_start: Training start date
            train_end: Training end date
            n_features: Number of features used
            feature_names: Names of features used
            train_samples: Number of training samples
            direction_accuracy: Training direction accuracy
        """
        cache_key = self._generate_cache_key(tickers, train_start, train_end, n_features)
        
        # Create cache entry
        cache_entry = {
            'models': models,
            'metadata': {
                'cache_key': cache_key,
                'tickers': sorted(tickers),
                'train_start': train_start,
                'train_end': train_end,
                'n_features': n_features,
                'feature_names': feature_names,
                'train_samples': train_samples,
                'direction_accuracy': direction_accuracy,
                'cached_at': datetime.now().isoformat()
            }
        }
        
        # Save to disk
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_entry, f)
        
        # Save metadata separately for easy inspection
        metadata_path = self.cache_dir / f"{cache_key}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(cache_entry['metadata'], f, indent=2)
        
        print(f"   💾 Models cached: {cache_key}")
        print(f"   📁 Cache path: {cache_path}")
    
    def load_models(
        self,
        tickers: List[str],
        train_start: str,
        train_end: str,
        n_features: int
    ) -> Optional[Dict]:
        """
        Load cached models if available.
        
        Args:
            tickers: List of tickers
            train_start: Training start date
            train_end: Training end date
            n_features: Number of features
        
        Returns:
            Dictionary with models and metadata, or None if not cached
        """
        cache_key = self._generate_cache_key(tickers, train_start, train_end, n_features)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_entry = pickle.load(f)
            
            print(f"   ✅ Loaded cached models: {cache_key}")
            print(f"   📊 Cached at: {cache_entry['metadata']['cached_at']}")
            print(f"   📊 Training samples: {cache_entry['metadata']['train_samples']:,}")
            print(f"   📊 Direction accuracy: {cache_entry['metadata']['direction_accuracy']:.1f}%")
            
            return cache_entry
            
        except Exception as e:
            print(f"   ⚠️ Error loading cache: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached models."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        for metadata_file in self.cache_dir.glob("*_metadata.json"):
            metadata_file.unlink()
        print(f"   🗑️ Cache cleared: {self.cache_dir}")
    
    def list_cached_models(self):
        """List all cached models."""
        cached_models = []
        
        for metadata_file in self.cache_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                cached_models.append(metadata)
            except:
                continue
        
        return cached_models


# Global cache instance
_model_cache = None

def get_model_cache() -> ModelCache:
    """Get global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache
