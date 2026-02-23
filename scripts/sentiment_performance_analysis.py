#!/usr/bin/env python3
"""
Sentiment Performance Analysis
=============================

Analyze the performance of sentiment-enhanced models vs baseline models
to determine if sentiment features improve CAGR and reduce drawdown.

KEY METRICS:
- Model accuracy comparison
- Feature importance analysis
- Sentiment feature contribution
- Performance improvement metrics
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_engineer import FeatureEngineer
from core.sentiment_feature_engineer import SentimentFeatureEngineer
from scripts.data_manager import DataManager
from src.sentiment.sentiment_integration import SentimentIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SentimentAnalysis')

class SentimentPerformanceAnalyzer:
    """Analyze sentiment-enhanced model performance"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.base_fe = FeatureEngineer(use_advanced_features=True)
        self.sentiment_fe = SentimentFeatureEngineer(use_advanced_features=True)
        self.sentiment_integration = None
        
        # Initialize sentiment integration
        self._initialize_sentiment_integration()
        
        logger.info("SentimentPerformanceAnalyzer initialized")
    
    def _initialize_sentiment_integration(self):
        """Initialize sentiment integration"""
        try:
            sentiment_config = {
                'enable_economic': True,
                'enable_news': True,
                'enable_social': True,
                'cache_ttl': 3600,
                'fred_api_key': os.getenv('FRED_API_KEY'),
                'news_api_key': os.getenv('NEWS_API_KEY'),
                'social_api_keys': {
                    'twitter_api_key': os.getenv('TWITTER_API_KEY'),
                    'reddit_api_key': os.getenv('REDDIT_API_KEY'),
                    'stocktwits_api_key': os.getenv('STOCKTWITS_API_KEY')
                }
            }
            
            self.sentiment_integration = SentimentIntegration(sentiment_config)
            logger.info("Sentiment integration initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize sentiment integration: {e}")
            self.sentiment_integration = None
    
    def load_models(self) -> Dict[str, Any]:
        """Load baseline and sentiment-enhanced models"""
        models = {}
        
        try:
            # Load baseline models
            baseline_dir = Path("models")
            if baseline_dir.exists():
                baseline_models = {}
                for model_file in baseline_dir.glob("*_model.pkl"):
                    if "sentiment" not in model_file.name:
                        model_name = model_file.stem.replace("_model", "")
                        with open(model_file, 'rb') as f:
                            baseline_models[model_name] = pickle.load(f)
                
                models['baseline'] = baseline_models
                logger.info(f"Loaded {len(baseline_models)} baseline models")
            
            # Load sentiment-enhanced models
            sentiment_dir = Path("models")
            if sentiment_dir.exists():
                sentiment_models = {}
                for model_file in sentiment_dir.glob("*_sentiment_model.pkl"):
                    model_name = model_file.stem.replace("_sentiment_model", "")
                    with open(model_file, 'rb') as f:
                        sentiment_models[model_name] = pickle.load(f)
                
                models['sentiment'] = sentiment_models
                logger.info(f"Loaded {len(sentiment_models)} sentiment-enhanced models")
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return {}
    
    def load_performance_data(self) -> Dict[str, Any]:
        """Load performance data from saved files"""
        try:
            performance_data = {}
            
            # Load sentiment model performance
            sentiment_perf_path = Path("models/sentiment_model_performance.json")
            if sentiment_perf_path.exists():
                with open(sentiment_perf_path, 'r') as f:
                    performance_data['sentiment'] = json.load(f)
            
            # Load baseline model performance (if available)
            baseline_perf_path = Path("models/model_performance.json")
            if baseline_perf_path.exists():
                with open(baseline_perf_path, 'r') as f:
                    performance_data['baseline'] = json.load(f)
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")
            return {}
    
    def analyze_feature_importance(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature importance for sentiment vs baseline models"""
        try:
            feature_analysis = {}
            
            # Load sentiment feature names
            sentiment_features_path = Path("models/sentiment_feature_names.json")
            if sentiment_features_path.exists():
                with open(sentiment_features_path, 'r') as f:
                    sentiment_feature_names = json.load(f)
            else:
                sentiment_feature_names = []
            
            # Analyze each sentiment model
            for model_name, model in models.get('sentiment', {}).items():
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Categorize features
                    sentiment_features = []
                    technical_features = []
                    
                    for i, feature in enumerate(sentiment_feature_names):
                        if i < len(importances):
                            if 'sentiment' in feature.lower():
                                sentiment_features.append((feature, importances[i]))
                            else:
                                technical_features.append((feature, importances[i]))
                    
                    # Sort by importance
                    sentiment_features.sort(key=lambda x: x[1], reverse=True)
                    technical_features.sort(key=lambda x: x[1], reverse=True)
                    
                    feature_analysis[model_name] = {
                        'sentiment_features': sentiment_features[:10],  # Top 10
                        'technical_features': technical_features[:10],  # Top 10
                        'total_sentiment_importance': sum(imp for _, imp in sentiment_features),
                        'total_technical_importance': sum(imp for _, imp in technical_features),
                        'sentiment_feature_count': len(sentiment_features),
                        'technical_feature_count': len(technical_features)
                    }
            
            return feature_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze feature importance: {e}")
            return {}
    
    def calculate_performance_improvement(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance improvements between sentiment and baseline models"""
        try:
            improvements = {}
            
            baseline_perf = performance_data.get('baseline', {})
            sentiment_perf = performance_data.get('sentiment', {})
            
            # Compare each model type
            for model_name in ['xgboost', 'lightgbm', 'random_forest']:
                baseline_metrics = baseline_perf.get(model_name, {})
                sentiment_metrics = sentiment_perf.get(model_name, {})
                
                if baseline_metrics and sentiment_metrics:
                    # Calculate improvements
                    accuracy_improvement = sentiment_metrics.get('accuracy', 0) - baseline_metrics.get('accuracy', 0)
                    precision_improvement = sentiment_metrics.get('precision', 0) - baseline_metrics.get('precision', 0)
                    
                    # Calculate percentage improvements
                    baseline_acc = baseline_metrics.get('accuracy', 1)
                    baseline_prec = baseline_metrics.get('precision', 1)
                    
                    accuracy_improvement_pct = (accuracy_improvement / baseline_acc) * 100 if baseline_acc > 0 else 0
                    precision_improvement_pct = (precision_improvement / baseline_prec) * 100 if baseline_prec > 0 else 0
                    
                    improvements[model_name] = {
                        'baseline_accuracy': baseline_metrics.get('accuracy', 0),
                        'sentiment_accuracy': sentiment_metrics.get('accuracy', 0),
                        'accuracy_improvement': accuracy_improvement,
                        'accuracy_improvement_pct': accuracy_improvement_pct,
                        'baseline_precision': baseline_metrics.get('precision', 0),
                        'sentiment_precision': sentiment_metrics.get('precision', 0),
                        'precision_improvement': precision_improvement,
                        'precision_improvement_pct': precision_improvement_pct,
                        'baseline_features': baseline_metrics.get('features_used', 0),
                        'sentiment_features': sentiment_metrics.get('features_used', 0),
                        'feature_increase': sentiment_metrics.get('features_used', 0) - baseline_metrics.get('features_used', 0)
                    }
            
            # Calculate overall averages
            if improvements:
                avg_acc_improvement = np.mean([imp['accuracy_improvement_pct'] for imp in improvements.values()])
                avg_prec_improvement = np.mean([imp['precision_improvement_pct'] for imp in improvements.values()])
                avg_feature_increase = np.mean([imp['feature_increase'] for imp in improvements.values()])
                
                improvements['overall'] = {
                    'avg_accuracy_improvement_pct': avg_acc_improvement,
                    'avg_precision_improvement_pct': avg_prec_improvement,
                    'avg_feature_increase': avg_feature_increase,
                    'models_compared': len(improvements) - 1
                }
            
            return improvements
            
        except Exception as e:
            logger.error(f"Failed to calculate performance improvements: {e}")
            return {}
    
    def generate_analysis_report(self, performance_data: Dict[str, Any], 
                                feature_analysis: Dict[str, Any],
                                improvements: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        try:
            report = []
            report.append("=" * 80)
            report.append("SENTIMENT-ENHANCED MODEL PERFORMANCE ANALYSIS")
            report.append("=" * 80)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Performance Summary
            report.append("PERFORMANCE SUMMARY:")
            report.append("-" * 50)
            
            if 'overall' in improvements:
                overall = improvements['overall']
                report.append(f"Models Compared: {overall['models_compared']}")
                report.append(f"Average Accuracy Improvement: {overall['avg_accuracy_improvement_pct']:.2f}%")
                report.append(f"Average Precision Improvement: {overall['avg_precision_improvement_pct']:.2f}%")
                report.append(f"Average Feature Increase: {overall['avg_feature_increase']:.1f}")
                report.append("")
            
            # Individual Model Results
            report.append("INDIVIDUAL MODEL RESULTS:")
            report.append("-" * 50)
            
            for model_name, metrics in improvements.items():
                if model_name == 'overall':
                    continue
                
                report.append(f"\n{model_name.upper()}:")
                report.append(f"  Accuracy:  {metrics['baseline_accuracy']:.4f} → {metrics['sentiment_accuracy']:.4f} ({metrics['accuracy_improvement_pct']:+.2f}%)")
                report.append(f"  Precision: {metrics['baseline_precision']:.4f} → {metrics['sentiment_precision']:.4f} ({metrics['precision_improvement_pct']:+.2f}%)")
                report.append(f"  Features:  {metrics['baseline_features']} → {metrics['sentiment_features']} (+{metrics['feature_increase']})")
            
            # Feature Analysis
            report.append("\n" + "FEATURE ANALYSIS:")
            report.append("-" * 50)
            
            for model_name, analysis in feature_analysis.items():
                report.append(f"\n{model_name.upper()}:")
                report.append(f"  Sentiment Features: {analysis['sentiment_feature_count']}")
                report.append(f"  Technical Features: {analysis['technical_feature_count']}")
                report.append(f"  Total Sentiment Importance: {analysis['total_sentiment_importance']:.4f}")
                report.append(f"  Total Technical Importance: {analysis['total_technical_importance']:.4f}")
                
                if analysis['sentiment_features']:
                    report.append("  Top Sentiment Features:")
                    for feature, importance in analysis['sentiment_features'][:5]:
                        report.append(f"    {feature}: {importance:.4f}")
            
            # Conclusions
            report.append("\n" + "=" * 50)
            report.append("CONCLUSIONS:")
            report.append("-" * 50)
            
            if 'overall' in improvements:
                overall = improvements['overall']
                if overall['avg_accuracy_improvement_pct'] > 0:
                    report.append("✅ SENTIMENT FEATURES IMPROVE MODEL PERFORMANCE")
                    report.append(f"   Average accuracy improvement: {overall['avg_accuracy_improvement_pct']:.2f}%")
                    report.append(f"   Average precision improvement: {overall['avg_precision_improvement_pct']:.2f}%")
                else:
                    report.append("❌ SENTIMENT FEATURES DO NOT IMPROVE PERFORMANCE")
                    report.append(f"   Average accuracy change: {overall['avg_accuracy_improvement_pct']:.2f}%")
                
                if overall['avg_feature_increase'] > 0:
                    report.append(f"📊 Added {overall['avg_feature_increase']:.1f} features per model")
            
            # Recommendations
            report.append("\nRECOMMENDATIONS:")
            report.append("-" * 50)
            
            if 'overall' in improvements:
                overall = improvements['overall']
                if overall['avg_accuracy_improvement_pct'] > 1.0:
                    report.append("🚀 STRONG RECOMMENDATION: Deploy sentiment-enhanced models")
                elif overall['avg_accuracy_improvement_pct'] > 0.1:
                    report.append("✅ RECOMMENDATION: Consider sentiment-enhanced models")
                else:
                    report.append("⚠️  CAUTION: Sentiment features show minimal improvement")
            
            report.append("\n" + "=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Failed to generate analysis report: {e}")
            return "Error generating report"
    
    def save_analysis_results(self, performance_data: Dict[str, Any],
                            feature_analysis: Dict[str, Any],
                            improvements: Dict[str, Any],
                            output_dir: str = "analysis_results") -> None:
        """Save analysis results"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save performance data
            perf_path = output_path / f"sentiment_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(perf_path, 'w') as f:
                json.dump({
                    'performance_data': performance_data,
                    'feature_analysis': feature_analysis,
                    'improvements': improvements
                }, f, indent=2)
            logger.info(f"Saved analysis results to {perf_path}")
            
            # Save report
            report = self.generate_analysis_report(performance_data, feature_analysis, improvements)
            report_path = output_path / f"sentiment_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved analysis report to {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete sentiment performance analysis"""
        try:
            logger.info("Starting sentiment performance analysis...")
            
            # Load models
            models = self.load_models()
            if not models:
                raise ValueError("No models loaded for analysis")
            
            # Load performance data
            performance_data = self.load_performance_data()
            
            # Analyze feature importance
            feature_analysis = self.analyze_feature_importance(models)
            
            # Calculate performance improvements
            improvements = self.calculate_performance_improvement(performance_data)
            
            # Generate and display report
            report = self.generate_analysis_report(performance_data, feature_analysis, improvements)
            logger.info(f"\n{report}")
            
            # Save results
            self.save_analysis_results(performance_data, feature_analysis, improvements)
            
            logger.info("Sentiment performance analysis completed")
            
            return {
                'performance_data': performance_data,
                'feature_analysis': feature_analysis,
                'improvements': improvements,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {}

def main():
    """Main execution function"""
    try:
        logger.info("Starting Sentiment Performance Analysis...")
        
        analyzer = SentimentPerformanceAnalyzer()
        results = analyzer.run_analysis()
        
        if results:
            logger.info("Analysis completed successfully!")
        else:
            logger.error("Analysis failed!")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
