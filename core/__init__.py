"""
CPU Models Library
Models that run efficiently on CPU without GPU requirements
Optimized for performance on standard hardware
"""

import logging

# Lazy imports to avoid sklearn dependency issues
def _lazy_import():
    """Lazy import CPU models only when needed"""
    try:
        from .random_forest_model import RandomForestModel
        from .xgboost_model import XGBoostModel
        from .lightgbm_model import LightGBMModel
        from .base_cpu_model import BaseCPUModel
        return {
            'RandomForestModel': RandomForestModel,
            'XGBoostModel': XGBoostModel,
            'LightGBMModel': LightGBMModel,
            'BaseCPUModel': BaseCPUModel
        }
    except ImportError as e:
        logging.getLogger(__name__).warning(f"[WARN] CPU models not available: {e}")
        return {}

__all__ = [
    'RandomForestModel',
    'XGBoostModel', 
    'LightGBMModel',
    'BaseCPUModel'
]
