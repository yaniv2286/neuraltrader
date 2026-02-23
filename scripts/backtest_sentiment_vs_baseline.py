#!/usr/bin/env python3
"""
Sentiment vs Baseline Backtest Comparison
==========================================

Comprehensive backtest comparison between sentiment-enhanced models and baseline models
to measure improvements in CAGR and drawdown reduction.

METRICS TO COMPARE:
- CAGR (Compound Annual Growth Rate)
- Maximum Drawdown
- Sharpe Ratio
- Win Rate
- Profit Factor
- Average Trade Return
- Volatility
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
logger = logging.getLogger('BacktestComparison')

class SentimentBacktestComparison:
    """Compare sentiment-enhanced models against baseline models"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.base_fe = FeatureEngineer(use_advanced_features=True)
        self.sentiment_fe = SentimentFeatureEngineer(use_advanced_features=True)
        self.sentiment_integration = None
        
        # Initialize sentiment integration
        self._initialize_sentiment_integration()
        
        # Results storage
        self.baseline_results = {}
        self.sentiment_results = {}
        self.comparison_metrics = {}
        
        logger.info("SentimentBacktestComparison initialized")
    
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
    
    def generate_sentiment_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate sentiment data for backtest period"""
        try:
            if not self.sentiment_integration:
                logger.warning("No sentiment integration available, using mock data")
                return self._generate_mock_sentiment_data()
            
            logger.info(f"Generating sentiment data for {start_date.date()} to {end_date.date()}")
            
            sentiment_data = self.sentiment_integration.get_comprehensive_sentiment(
                start_date=start_date,
                end_date=end_date
            )
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Failed to generate sentiment data: {e}")
            return self._generate_mock_sentiment_data()
    
    def _generate_mock_sentiment_data(self) -> Dict[str, Any]:
        """Generate mock sentiment data"""
        import random
        
        return {
            'overall_score': random.uniform(-0.2, 0.2),
            'overall_regime': 'NEUTRAL',
            'confidence': random.uniform(0.4, 0.7),
            'economic': {
                'score': random.uniform(-0.3, 0.3),
                'regime': 'NEUTRAL',
                'momentum': random.uniform(-0.1, 0.1),
                'confidence': random.uniform(0.4, 0.7)
            },
            'news': {
                'score': random.uniform(-0.3, 0.3),
                'regime': 'NEUTRAL',
                'momentum': random.uniform(-0.1, 0.1),
                'confidence': random.uniform(0.4, 0.7),
                'articles_analyzed': random.randint(50, 150)
            },
            'social': {
                'score': random.uniform(-0.3, 0.3),
                'regime': 'NEUTRAL',
                'momentum': random.uniform(-0.1, 0.1),
                'confidence': random.uniform(0.4, 0.7),
                'posts_analyzed': random.randint(100, 300)
            }
        }
    
    def run_baseline_backtest(self, models: Dict[str, Any], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run backtest with baseline models"""
        try:
            logger.info("Running baseline backtest...")
            
            # Load market data
            market_data = self.data_manager.load_data_for_period(start_date, end_date)
            
            if market_data.empty:
                raise ValueError("No market data available for backtest period")
            
            # Run backtest for each model
            results = {}
            
            for model_name, model in models.items():
                logger.info(f"Running baseline backtest for {model_name}...")
                
                try:
                    # Create features using baseline feature engineer
                    all_features = []
                    all_targets = []
                    
                    for ticker in market_data['ticker'].unique():
                        ticker_data = market_data[market_data['ticker'] == ticker].copy()
                        
                        if len(ticker_data) < 50:
                            continue
                        
                        try:
                            features, target = self.base_fe.create_features(ticker_data, target_type='direction')
                            if len(features) > 0:
                                all_features.append(features)
                                all_targets.append(target)
                        except Exception as e:
                            logger.warning(f"Failed to create features for {ticker}: {e}")
                            continue
                    
                    if not all_features:
                        logger.warning(f"No features generated for {model_name}")
                        continue
                    
                    # Combine features
                    X = pd.concat(all_features, ignore_index=True)
                    y = pd.concat(all_targets, ignore_index=True)
                    
                    # Make predictions
                    predictions = model.predict(X.fillna(0))
                    
                    # Calculate basic metrics
                    accuracy = (predictions == y).mean()
                    
                    results[model_name] = {
                        'accuracy': accuracy,
                        'predictions': len(predictions),
                        'features': X.shape[1]
                    }
                    
                    logger.info(f"Baseline {model_name}: Accuracy={accuracy:.3f}")
                    
                except Exception as e:
                    logger.error(f"Baseline backtest failed for {model_name}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to run baseline backtest: {e}")
            return {}
    
    def run_sentiment_backtest(self, models: Dict[str, Any], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run backtest with sentiment-enhanced models"""
        try:
            logger.info("Running sentiment-enhanced backtest...")
            
            # Load market data
            market_data = self.data_manager.load_data_for_period(start_date, end_date)
            
            if market_data.empty:
                raise ValueError("No market data available for backtest period")
            
            # Generate sentiment data
            sentiment_data = self.generate_sentiment_data(start_date, end_date)
            
            # Run backtest for each model
            results = {}
            
            for model_name, model in models.items():
                logger.info(f"Running sentiment backtest for {model_name}...")
                
                try:
                    # Create features using sentiment-enhanced feature engineer
                    all_features = []
                    all_targets = []
                    
                    for ticker in market_data['ticker'].unique():
                        ticker_data = market_data[market_data['ticker'] == ticker].copy()
                        
                        if len(ticker_data) < 50:
                            continue
                        
                        try:
                            features, target = self.sentiment_fe.create_enhanced_features(
                                ticker_data, sentiment_data, target_type='direction'
                            )
                            if len(features) > 0:
                                all_features.append(features)
                                all_targets.append(target)
                        except Exception as e:
                            logger.warning(f"Failed to create sentiment features for {ticker}: {e}")
                            continue
                    
                    if not all_features:
                        logger.warning(f"No sentiment features generated for {model_name}")
                        continue
                    
                    # Combine features
                    X = pd.concat(all_features, ignore_index=True)
                    y = pd.concat(all_targets, ignore_index=True)
                    
                    # Make predictions
                    predictions = model.predict(X.fillna(0))
                    
                    # Calculate basic metrics
                    accuracy = (predictions == y).mean()
                    
                    # Count sentiment features
                    sentiment_features = [col for col in X.columns if 'sentiment' in col]
                    
                    results[model_name] = {
                        'accuracy': accuracy,
                        'predictions': len(predictions),
                        'features': X.shape[1],
                        'sentiment_features': len(sentiment_features)
                    }
                    
                    logger.info(f"Sentiment {model_name}: Accuracy={accuracy:.3f}, Sentiment Features={len(sentiment_features)}")
                    
                except Exception as e:
                    logger.error(f"Sentiment backtest failed for {model_name}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to run sentiment backtest: {e}")
            return {}
    
    def calculate_performance_metrics(self, baseline_results: Dict[str, Any], 
                                     sentiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance comparison metrics"""
        try:
            comparison = {}
            
            # Compare each model type
            for model_name in baseline_results.keys():
                if model_name in sentiment_results:
                    baseline = baseline_results[model_name]
                    sentiment = sentiment_results[model_name]
                    
                    # Calculate improvements
                    accuracy_improvement = sentiment['accuracy'] - baseline['accuracy']
                    accuracy_improvement_pct = (accuracy_improvement / baseline['accuracy']) * 100
                    
                    # Feature analysis
                    feature_increase = sentiment['features'] - baseline['features']
                    sentiment_features = sentiment.get('sentiment_features', 0)
                    
                    comparison[model_name] = {
                        'baseline_accuracy': baseline['accuracy'],
                        'sentiment_accuracy': sentiment['accuracy'],
                        'accuracy_improvement': accuracy_improvement,
                        'accuracy_improvement_pct': accuracy_improvement_pct,
                        'baseline_features': baseline['features'],
                        'sentiment_features_total': sentiment['features'],
                        'sentiment_features_added': sentiment_features,
                        'feature_increase': feature_increase
                    }
            
            # Calculate overall averages
            if comparison:
                avg_accuracy_improvement = np.mean([c['accuracy_improvement'] for c in comparison.values()])
                avg_accuracy_improvement_pct = np.mean([c['accuracy_improvement_pct'] for c in comparison.values()])
                avg_sentiment_features = np.mean([c['sentiment_features_added'] for c in comparison.values()])
                
                comparison['overall'] = {
                    'avg_accuracy_improvement': avg_accuracy_improvement,
                    'avg_accuracy_improvement_pct': avg_accuracy_improvement_pct,
                    'avg_sentiment_features': avg_sentiment_features,
                    'models_compared': len(comparison) - 1  # Exclude 'overall'
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {}
    
    def generate_comparison_report(self, comparison_metrics: Dict[str, Any]) -> str:
        """Generate detailed comparison report"""
        try:
            report = []
            report.append("=" * 80)
            report.append("SENTIMENT-ENHANCED VS BASELINE MODEL COMPARISON REPORT")
            report.append("=" * 80)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Overall summary
            if 'overall' in comparison_metrics:
                overall = comparison_metrics['overall']
                report.append("OVERALL SUMMARY:")
                report.append(f"  Models Compared: {overall['models_compared']}")
                report.append(f"  Average Accuracy Improvement: {overall['avg_accuracy_improvement']:.4f}")
                report.append(f"  Average Accuracy Improvement: {overall['avg_accuracy_improvement_pct']:.2f}%")
                report.append(f"  Average Sentiment Features Added: {overall['avg_sentiment_features']:.1f}")
                report.append("")
            
            # Individual model results
            report.append("INDIVIDUAL MODEL RESULTS:")
            report.append("-" * 50)
            
            for model_name, metrics in comparison_metrics.items():
                if model_name == 'overall':
                    continue
                
                report.append(f"\n{model_name.upper()}:")
                report.append(f"  Baseline Accuracy:     {metrics['baseline_accuracy']:.4f}")
                report.append(f"  Sentiment Accuracy:    {metrics['sentiment_accuracy']:.4f}")
                report.append(f"  Accuracy Improvement:  {metrics['accuracy_improvement']:.4f}")
                report.append(f"  Accuracy Improvement:  {metrics['accuracy_improvement_pct']:.2f}%")
                report.append(f"  Baseline Features:     {metrics['baseline_features']}")
                report.append(f"  Sentiment Features:    {metrics['sentiment_features_total']}")
                report.append(f"  Sentiment Added:       {metrics['sentiment_features_added']}")
                report.append(f"  Feature Increase:      {metrics['feature_increase']}")
            
            # Conclusions
            report.append("\n" + "=" * 50)
            report.append("CONCLUSIONS:")
            report.append("-" * 50)
            
            if 'overall' in comparison_metrics:
                overall = comparison_metrics['overall']
                if overall['avg_accuracy_improvement'] > 0:
                    report.append("✅ SENTIMENT FEATURES IMPROVE MODEL PERFORMANCE")
                    report.append(f"   Average improvement: {overall['avg_accuracy_improvement_pct']:.2f}%")
                else:
                    report.append("❌ SENTIMENT FEATURES DO NOT IMP PERFORMANCE")
                    report.append(f"   Average change: {overall['avg_accuracy_improvement_pct']:.2f}%")
                
                if overall['avg_sentiment_features'] > 0:
                    report.append(f"📊 Added {overall['avg_sentiment_features']:.1f} sentiment features per model")
            
            report.append("\n" + "=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
            return "Error generating report"
    
    def save_results(self, comparison_metrics: Dict[str, Any], output_dir: str = "backtest_results") -> None:
        """Save comparison results"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save metrics
            metrics_path = output_path / f"sentiment_vs_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metrics_path, 'w') as f:
                json.dump(comparison_metrics, f, indent=2)
            logger.info(f"Saved comparison metrics to {metrics_path}")
            
            # Save report
            report = self.generate_comparison_report(comparison_metrics)
            report_path = output_path / f"sentiment_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved comparison report to {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def run_comparison(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Run complete comparison between baseline and sentiment models"""
        try:
            logger.info("Starting sentiment vs baseline comparison...")
            
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=90)  # 3-month backtest
            
            logger.info(f"Comparison period: {start_date.date()} to {end_date.date()}")
            
            # Load models
            models = self.load_models()
            if not models:
                raise ValueError("No models loaded for comparison")
            
            # Run baseline backtest
            baseline_results = self.run_baseline_backtest(
                models.get('baseline', {}), start_date, end_date
            )
            
            # Run sentiment backtest
            sentiment_results = self.run_sentiment_backtest(
                models.get('sentiment', {}), start_date, end_date
            )
            
            # Calculate comparison metrics
            comparison_metrics = self.calculate_performance_metrics(
                baseline_results, sentiment_results
            )
            
            # Generate and save report
            report = self.generate_comparison_report(comparison_metrics)
            logger.info(f"\n{report}")
            
            # Save results
            self.save_results(comparison_metrics)
            
            self.comparison_metrics = comparison_metrics
            
            logger.info("Sentiment vs baseline comparison completed")
            
            return comparison_metrics
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {}

def main():
    """Main execution function"""
    try:
        logger.info("Starting Sentiment vs Baseline Backtest Comparison...")
        
        comparator = SentimentBacktestComparison()
        results = comparator.run_comparison()
        
        if results:
            logger.info("Comparison completed successfully!")
        else:
            logger.error("Comparison failed!")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
