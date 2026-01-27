"""
GPU Models Library
Models that require strong GPU for optimal performance
Deep learning models with large parameter counts
"""

# GPU availability check
import torch

def check_gpu_availability():
    """Check if GPU is available and suitable for training"""
    if not torch.cuda.is_available():
        return False, "No CUDA GPU available"
    
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Check if GPU has sufficient memory (recommend at least 8GB)
    if gpu_memory < 8:
        return False, f"GPU memory insufficient: {gpu_memory:.1f}GB (recommended: 8GB+)"
    
    return True, f"GPU Available: {gpu_name} ({gpu_memory:.1f}GB)"

# Auto-check on import
GPU_AVAILABLE, GPU_INFO = check_gpu_availability()
if not GPU_AVAILABLE:
    print(f"⚠️ GPU Models Warning: {GPU_INFO}")
    print("💡 Use CPU models for better performance on this hardware")
else:
    print(f"✅ GPU Models Ready: {GPU_INFO}")

# Import models only if GPU is available
GPU_MODELS_AVAILABLE = False
if GPU_AVAILABLE:
    try:
        from .transformer_model import TransformerModel
        from .lstm_model import LSTMModel
        from .cnn_lstm_model import CNNLSTMModel
        from .deepar_model import DeepARModel
        from .informer_model import InformerModel
        from .nbeats_model import NBEATSModel
        from .patchtst_model import PatchTSTModel
        from .tft_model import TFTModel
        from .elite_models import EliteHedgeFundModels
        
        GPU_MODELS_AVAILABLE = True
        
        __all__ = [
            'TransformerModel',
            'LSTMModel', 
            'CNNLSTMModel',
            'DeepARModel',
            'InformerModel',
            'NBEATSModel',
            'PatchTSTModel',
            'TFTModel',
            'EliteHedgeFundModels'
        ]
        
    except ImportError as e:
        print(f"⚠️ GPU models import failed: {e}")
        print("💡 Install required dependencies for GPU models")
        GPU_MODELS_AVAILABLE = False
else:
    print("💡 GPU not available - using CPU models only")
    
    __all__ = []
