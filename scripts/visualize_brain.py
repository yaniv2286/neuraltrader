#!/usr/bin/env python3
"""
NeuralTrader Brain Visualization - Feature Importance Analysis
===========================================================

Analyzes and visualizes feature importance from the Modern Era ensemble models.

Usage:
    python scripts/visualize_brain.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BrainVisualizer:
    """Visualizes NeuralTrader brain feature importance"""
    
    def __init__(self):
        """Initialize brain visualizer"""
        self.project_root = Path(__file__).parent.parent
        self.production_dir = self.project_root / 'models' / 'production'
        self.reports_dir = self.project_root / 'reports'
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🧠 Brain Visualizer initialized")
        logger.info(f"   Production models: {self.production_dir}")
        logger.info(f"   Reports output: {self.reports_dir}")
    
    def load_production_models(self) -> Dict[str, object]:
        """Load the three production models"""
        logger.info("📥 Loading production models...")
        
        models = {}
        model_files = {
            'xgboost': 'xgboost_model.pkl',
            'lightgbm': 'lightgbm_model.pkl',
            'rf': 'rf_model.pkl'
        }
        
        # Add project root to path to handle core module imports
        import sys
        sys.path.insert(0, str(self.project_root))
        
        for model_name, filename in model_files.items():
            model_path = self.production_dir / filename
            
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        models[model_name] = pickle.load(f)
                    logger.info(f"   ✅ Loaded: {filename}")
                except Exception as e:
                    logger.error(f"   ❌ Failed to load {filename}: {e}")
                    # Try alternative loading method
                    try:
                        # Try loading with custom unpickler that ignores missing modules
                        class SafeUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                if module.startswith('core.'):
                                    # Return a dummy class for missing core modules
                                    return type(name, (), {})
                                return super().find_class(module, name)
                        
                        with open(model_path, 'rb') as f:
                            models[model_name] = SafeUnpickler(f).load()
                        logger.info(f"   ✅ Loaded with safe unpickler: {filename}")
                    except Exception as e2:
                        logger.error(f"   ❌ Safe unpickler also failed for {filename}: {e2}")
            else:
                logger.error(f"   ❌ Model not found: {filename}")
        
        return models
    
    def load_feature_names(self) -> List[str]:
        """Load feature names from production metadata"""
        logger.info("📋 Loading feature names...")
        
        # Try to load from feature_names.pkl first
        feature_names_path = self.production_dir / 'feature_names.pkl'
        
        if feature_names_path.exists():
            try:
                with open(feature_names_path, 'rb') as f:
                    feature_names = pickle.load(f)
                logger.info(f"   ✅ Loaded {len(feature_names)} feature names from feature_names.pkl")
                return feature_names
            except Exception as e:
                logger.error(f"   ❌ Failed to load feature_names.pkl: {e}")
        
        # Fallback to ensemble_metadata.pkl
        metadata_path = self.production_dir / 'ensemble_metadata.pkl'
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                if 'feature_names' in metadata:
                    feature_names = metadata['feature_names']
                    logger.info(f"   ✅ Loaded {len(feature_names)} feature names from ensemble_metadata.pkl")
                    return feature_names
                else:
                    logger.error("   ❌ No feature_names found in ensemble_metadata.pkl")
            except Exception as e:
                logger.error(f"   ❌ Failed to load ensemble_metadata.pkl: {e}")
        
        logger.error("   ❌ No feature names found!")
        return []
    
    def extract_feature_importance(self, models: Dict[str, object], feature_names: List[str]) -> pd.DataFrame:
        """Extract and normalize feature importance from all models"""
        logger.info("🔍 Extracting feature importance...")
        
        importance_data = []
        
        for model_name, model in models.items():
            try:
                importances = None
                
                # Try different methods to get feature importances
                actual_model = None
                
                # Check if model has a .model attribute (wrapper class)
                if hasattr(model, 'model'):
                    actual_model = model.model
                    logger.info(f"   🔍 Found wrapped model for {model_name}")
                else:
                    actual_model = model
                
                if hasattr(actual_model, 'feature_importances_'):
                    importances = actual_model.feature_importances_
                    logger.info(f"   ✅ Found feature_importances_ for {model_name}")
                elif hasattr(actual_model, 'get_booster') and model_name == 'xgboost':
                    # XGBoost special case
                    booster = actual_model.get_booster()
                    importance_dict = booster.get_score(importance_type='gain')
                    importances = np.zeros(len(feature_names))
                    
                    # Map importance dict to array
                    for feature, importance in importance_dict.items():
                        if feature.startswith('f'):  # XGBoost feature names like f0, f1, etc.
                            idx = int(feature[1:])  # Extract number from f0, f1, etc.
                            if idx < len(importances):
                                importances[idx] = importance
                    logger.info(f"   ✅ Extracted XGBoost booster importance for {model_name}")
                elif hasattr(actual_model, 'coef_'):
                    # Linear models use coefficients
                    importances = np.abs(actual_model.coef_)
                    if len(importances.shape) > 1:
                        importances = importances[0]  # Take first row for multi-class
                    logger.info(f"   ✅ Found coef_ for {model_name}")
                elif hasattr(actual_model, 'feature_importances'):
                    # Some models might use singular form
                    importances = actual_model.feature_importances
                    logger.info(f"   ✅ Found feature_importances for {model_name}")
                else:
                    # Debug: Print available attributes
                    logger.warning(f"   ⚠️  No standard importance attribute found for {model_name}")
                    logger.info(f"   📋 Available attributes for {model_name}: {[attr for attr in dir(actual_model) if not attr.startswith('_')]}")
                    continue
                
                if importances is None or len(importances) == 0:
                    logger.warning(f"   ⚠️  No importances found for {model_name}")
                    continue
                
                # Ensure we have the right number of features
                if len(importances) != len(feature_names):
                    logger.warning(f"   ⚠️  Feature count mismatch for {model_name}: {len(importances)} vs {len(feature_names)}")
                    # Try to pad or truncate
                    if len(importances) > len(feature_names):
                        importances = importances[:len(feature_names)]
                    else:
                        importances = np.pad(importances, (0, len(feature_names) - len(importances)), 'constant')
                
                # Normalize to 0-1 range
                if importances.max() > 0:
                    normalized_importances = importances / importances.max()
                else:
                    normalized_importances = importances
                
                # Create DataFrame for this model
                model_importance = pd.DataFrame({
                    'feature': feature_names,
                    f'{model_name}_importance': normalized_importances
                })
                
                importance_data.append(model_importance)
                logger.info(f"   ✅ Extracted {model_name} importance (max: {normalized_importances.max():.3f})")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to extract {model_name} importance: {e}")
                import traceback
                logger.error(f"   📋 Traceback: {traceback.format_exc()}")
        
        if not importance_data:
            logger.error("   ❌ No feature importance data extracted!")
            return pd.DataFrame()
        
        # Merge all model importances
        combined_importance = importance_data[0]
        for df in importance_data[1:]:
            combined_importance = combined_importance.merge(df, on='feature', how='outer')
        
        # Calculate Council Consensus (average importance)
        importance_columns = [col for col in combined_importance.columns if col.endswith('_importance')]
        combined_importance['council_consensus'] = combined_importance[importance_columns].mean(axis=1)
        
        # Sort by consensus importance
        combined_importance = combined_importance.sort_values('council_consensus', ascending=False)
        
        logger.info(f"   ✅ Combined importance for {len(combined_importance)} features")
        logger.info(f"   📊 Top feature: {combined_importance.iloc[0]['feature']} ({combined_importance.iloc[0]['council_consensus']:.3f})")
        
        return combined_importance
    
    def create_visualization(self, importance_df: pd.DataFrame) -> plt.Figure:
        """Create horizontal bar chart of top features"""
        logger.info("🎨 Creating feature importance visualization...")
        
        # Get top 20 features
        top_features = importance_df.head(20).copy()
        
        # Reverse order for better visualization (most important at top)
        top_features = top_features.iloc[::-1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create color map
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(top_features)), top_features['council_consensus'], color=colors)
        
        # Customize the plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.set_xlabel('Council Consensus Importance', fontsize=12, fontweight='bold')
        ax.set_title('NeuralTrader v4.0 - Council Feature Importance (Modern Era)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_features['council_consensus'])):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontsize=9)
        
        # Style the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_top_features(self, importance_df: pd.DataFrame):
        """Print top features to console"""
        logger.info("📊 TOP 10 FEATURES:")
        
        top_10 = importance_df.head(10)
        
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            feature_name = row['feature']
            score = row['council_consensus']
            print(f"   {i:2d}. {feature_name:<30} (Score: {score:.3f})")
    
    def run_visualization(self):
        """Run the complete brain visualization"""
        try:
            logger.info("🧠 Starting NeuralTrader Brain Visualization...")
            
            # Step 1: Load models
            models = self.load_production_models()
            
            if len(models) < 2:
                logger.error("❌ Insufficient models loaded for analysis")
                return False
            
            # Step 2: Load feature names
            feature_names = self.load_feature_names()
            
            if not feature_names:
                logger.error("❌ No feature names available")
                return False
            
            # Step 3: Extract feature importance
            importance_df = self.extract_feature_importance(models, feature_names)
            
            if importance_df.empty:
                logger.error("❌ No feature importance data available")
                return False
            
            # Step 4: Print top features
            self.print_top_features(importance_df)
            
            # Step 5: Create visualization
            fig = self.create_visualization(importance_df)
            
            # Step 6: Save visualization
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.reports_dir / f'feature_importance_modern_{timestamp}.png'
            
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Visualization saved: {output_file}")
            
            # Step 7: Display summary
            logger.info("=" * 60)
            logger.info("🧠 NEURALTRADER BRAIN ANALYSIS COMPLETE")
            logger.info("=" * 60)
            logger.info(f"   Models Analyzed: {len(models)}")
            logger.info(f"   Features Analyzed: {len(importance_df)}")
            logger.info(f"   Top Feature: {importance_df.iloc[0]['feature']}")
            logger.info(f"   Top Score: {importance_df.iloc[0]['council_consensus']:.3f}")
            logger.info(f"   Visualization: {output_file}")
            logger.info("=" * 60)
            
            plt.close(fig)
            return True
            
        except Exception as e:
            logger.error(f"❌ Brain visualization failed: {e}")
            return False


def main():
    """Main execution function"""
    visualizer = BrainVisualizer()
    success = visualizer.run_visualization()
    
    if success:
        logger.info("🚀 Brain visualization completed successfully!")
    else:
        logger.error("❌ Brain visualization failed!")
        exit(1)


if __name__ == "__main__":
    main()
