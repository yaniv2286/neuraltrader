"""
NeuralTrader - Models Package
Intelligent model selection with CPU/GPU optimization
"""

from .model_selector import ModelSelector, get_best_models, create_optimal_model
from .cpu_models import RandomForestModel, XGBoostModel

# GPU models are imported conditionally based on availability
try:
    from .gpu_models import (
        GPU_AVAILABLE, GPU_MODELS_AVAILABLE
    )
except ImportError:
    GPU_AVAILABLE = False
    GPU_MODELS_AVAILABLE = False

__all__ = [
    # Core functionality
    'ModelSelector',
    'get_best_models', 
    'create_optimal_model',
    
    # CPU models (always available)
    'RandomForestModel',
    'XGBoostModel',
    
    # GPU models (conditional)
    'GPU_AVAILABLE',
    'GPU_MODELS_AVAILABLE'
]

def get_available_models():
    """Get list of available models based on hardware"""
    available = {
        'cpu_models': ['RandomForestModel', 'XGBoostModel'],
        'gpu_models': []
    }
    
    if GPU_MODELS_AVAILABLE and GPU_AVAILABLE:
        available['gpu_models'] = [
            'TransformerModel', 'LSTMModel', 'CNNLSTMModel',
            'DeepARModel', 'InformerModel', 'NBEATSModel',
            'PatchTSTModel', 'TFTModel'
        ]
    
    return available

def print_model_status():
    """Print current model availability status"""
    print("🤖 NeuralTrader - Model Status")
    print("=" * 50)
    
    available = get_available_models()
    
    print(f"✅ CPU Models Available: {len(available['cpu_models'])}")
    for model in available['cpu_models']:
        print(f"   • {model}")
    
    if available['gpu_models']:
        print(f"🎮 GPU Models Available: {len(available['gpu_models'])}")
        for model in available['gpu_models']:
            print(f"   • {model}")
    else:
        print("⚠️ GPU Models Not Available (No suitable GPU detected)")
    
    print(f"\n💡 Recommendation: Use {'GPU' if available['gpu_models'] else 'CPU'} models for optimal performance")
