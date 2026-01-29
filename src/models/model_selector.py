"""
Intelligent Model Selector
Automatically selects best models based on available hardware
"""

from .cpu_models import RandomForestModel, XGBoostModel
import warnings
warnings.filterwarnings('ignore')

# Since we removed GPU models, we'll use CPU models only
GPU_AVAILABLE = False

class ModelSelector:
    """
    Intelligent model selector that chooses the best models
    based on available hardware and requirements
    """
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        self.gpu_info = "CPU models only"
        
    def get_recommended_models(self, task_type: str = 'stock_prediction'):
        """
        Get recommended models based on hardware availability
        
        Args:
            task_type: Type of task ('stock_prediction', 'volatility', 'regime_detection')
            
        Returns:
            Dictionary with recommended models and configurations
        """
        
        if self.gpu_available:
            return self._get_gpu_recommended_models(task_type)
        else:
            return self._get_cpu_recommended_models(task_type)
    
    def _get_cpu_recommended_models(self, task_type: str):
        """Get CPU-optimized model recommendations"""
        
        if task_type == 'stock_prediction':
            return {
                'primary': {
                    'model': XGBoostModel,
                    'params': {
                        'n_estimators': 300,
                        'max_depth': 4,
                        'learning_rate': 0.03,
                        'tree_method': 'hist'
                    },
                    'reason': 'Best performance on CPU for stock prediction'
                },
                'secondary': {
                    'model': RandomForestModel,
                    'params': {
                        'n_estimators': 200,
                        'max_depth': 10,
                        'n_jobs': -1
                    },
                    'reason': 'Good baseline model with built-in cross-validation'
                },
                'ensemble': {
                    'model': XGBoostModel,
                    'params': {
                        'n_estimators': 400,
                        'max_depth': 3,
                        'learning_rate': 0.02
                    },
                    'reason': 'Optimized XGBoost with conservative parameters'
                }
            }
        
        elif task_type == 'volatility':
            return {
                'primary': {
                    'model': XGBoostModel,
                    'params': {
                        'n_estimators': 200,
                        'max_depth': 3,
                        'learning_rate': 0.05
                    },
                    'reason': 'Good for volatility forecasting'
                }
            }
        
        elif task_type == 'regime_detection':
            return {
                'primary': {
                    'model': RandomForestModel,
                    'params': {
                        'n_estimators': 150,
                        'max_depth': 8,
                        'class_weight': 'balanced'
                    },
                    'reason': 'Good for classification tasks'
                }
            }
        
        else:
            return self._get_cpu_recommended_models('stock_prediction')
    
    def _get_gpu_recommended_models(self, task_type: str):
        """Get GPU-optimized model recommendations"""
        
        if task_type == 'stock_prediction':
            return {
                'primary': {
                    'model': 'TransformerModel',
                    'params': {
                        'd_model': 256,
                        'n_heads': 8,
                        'num_layers': 6,
                        'dropout': 0.1
                    },
                    'reason': 'Best performance with GPU for sequence modeling'
                },
                'secondary': {
                    'model': 'LSTMModel',
                    'params': {
                        'hidden_size': 256,
                        'num_layers': 3,
                        'dropout': 0.2
                    },
                    'reason': 'Good alternative to Transformer'
                },
                'hybrid': {
                    'model': 'CNNLSTMModel',
                    'params': {
                        'hidden_size': 256,
                        'num_layers': 2,
                        'dropout': 0.1
                    },
                    'reason': 'Combines CNN feature extraction with LSTM'
                }
            }
        
        elif task_type == 'volatility':
            return {
                'primary': {
                    'model': 'TFTModel',
                    'params': {
                        'hidden_size': 128,
                        'attention_heads': 4
                    },
                    'reason': 'Temporal Fusion Transformer for volatility'
                }
            }
        
        else:
            return self._get_gpu_recommended_models('stock_prediction')
    
    def create_model(self, model_config: dict):
        """
        Create a model instance from configuration
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Model instance
        """
        model_class = model_config['model']
        
        if isinstance(model_class, str):
            # Handle GPU models (imported as strings)
            if self.gpu_available:
                from .gpu_models import TransformerModel, LSTMModel, CNNLSTMModel
                model_map = {
                    'TransformerModel': TransformerModel,
                    'LSTMModel': LSTMModel,
                    'CNNLSTMModel': CNNLSTMModel
                }
                model_class = model_map.get(model_class, XGBoostModel)
            else:
                # Fallback to CPU models
                model_class = XGBoostModel
        
        return model_class(**model_config['params'])
    
    def get_hardware_info(self):
        """Get hardware information and recommendations"""
        info = {
            'gpu_available': self.gpu_available,
            'gpu_info': self.gpu_info if self.gpu_available else None,
            'recommended_approach': 'GPU' if self.gpu_available else 'CPU',
            'performance_tier': self._get_performance_tier()
        }
        
        if not self.gpu_available:
            info['cpu_recommendations'] = [
                "Use XGBoost with tree_method='hist' for best CPU performance",
                "Enable n_jobs=-1 to use all CPU cores",
                "Consider ensemble methods for better accuracy"
            ]
        else:
            info['gpu_recommendations'] = [
                "Use mixed precision training for faster GPU utilization",
                "Batch size can be larger with GPU memory",
                "Consider Transformer models for best performance"
            ]
        
        return info
    
    def _get_performance_tier(self):
        """Determine performance tier based on hardware"""
        if not self.gpu_available:
            return 'CPU-Optimized'
        
        # Simple GPU tier classification based on memory
        if 'RTX 4090' in self.gpu_info or 'A100' in self.gpu_info:
            return 'High-End GPU'
        elif 'RTX 3090' in self.gpu_info or 'V100' in self.gpu_info:
            return 'Mid-Range GPU'
        else:
            return 'Entry-Level GPU'

# Global selector instance
model_selector = ModelSelector()

def get_best_models(task_type: str = 'stock_prediction'):
    """
    Convenience function to get best models for current hardware
    
    Args:
        task_type: Type of prediction task
        
    Returns:
        Dictionary with recommended models
    """
    return model_selector.get_recommended_models(task_type)

def create_optimal_model(task_type: str = 'stock_prediction', model_type: str = 'primary'):
    """
    Create the optimal model for current hardware
    
    Args:
        task_type: Type of prediction task
        model_type: 'primary', 'secondary', or 'ensemble'
        
    Returns:
        Model instance
    """
    models = get_best_models(task_type)
    
    if model_type not in models:
        model_type = 'primary'
    
    return model_selector.create_model(models[model_type])

if __name__ == "__main__":
    # Test the model selector
    selector = ModelSelector()
    
    print("🤖 Intelligent Model Selector")
    print("=" * 40)
    
    hardware_info = selector.get_hardware_info()
    print(f"🖥️ Hardware: {hardware_info['recommended_approach']}")
    print(f"📊 Performance Tier: {hardware_info['performance_tier']}")
    
    if hardware_info['gpu_available']:
        print(f"🎮 GPU: {hardware_info['gpu_info']}")
        print("\n💡 GPU Recommendations:")
        for rec in hardware_info['gpu_recommendations']:
            print(f"   • {rec}")
    else:
        print("\n💡 CPU Recommendations:")
        for rec in hardware_info['cpu_recommendations']:
            print(f"   • {rec}")
    
    # Get recommended models
    models = get_best_models('stock_prediction')
    print(f"\n🎯 Recommended Models for Stock Prediction:")
    print(f"   Primary: {models['primary']['model'].__name__}")
    print(f"   Secondary: {models['secondary']['model'].__name__}")
    
    # Create optimal model
    optimal_model = create_optimal_model()
    print(f"\n✅ Created optimal model: {type(optimal_model).__name__}")
