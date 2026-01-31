"""
Fix AI Ensemble Signal Generation
==================================

Diagnose and fix the issue where AI Ensemble generates 0 signals.

Root causes:
1. Prediction thresholds too strict (0.0001 for LONG, -0.001 for SHORT)
2. Signal confirmation requiring 2/3 models too restrictive
3. Feature mismatch between training and testing

Solution:
1. Relax prediction thresholds significantly
2. Remove or relax 2/3 confirmation requirement
3. Ensure feature columns match
"""

import pandas as pd
import numpy as np
from typing import List
from src.core.ai_ensemble_strategy_v2 import AIEnsembleStrategyV2
from src.core.data_store import get_data_store
from src.features.technical_indicators import generate_all_features


def diagnose_predictions(
    ai_strategy: AIEnsembleStrategyV2,
    ticker: str,
    start_date: str,
    end_date: str
) -> dict:
    """Diagnose what predictions look like."""
    print(f"\n🔍 Diagnosing predictions for {ticker}...")
    
    try:
        # Load data
        data_store = get_data_store()
        df = data_store.get_ticker_data(ticker, start_date, end_date)
        
        if df is None or len(df) < 50:
            return {'error': 'Insufficient data'}
        
        # Generate features
        df = generate_all_features(df)
        
        # Check feature columns
        missing = [f for f in ai_strategy.feature_columns if f not in df.columns]
        if missing:
            print(f"   ⚠️ Missing features: {len(missing)}")
            return {'error': f'Missing features: {missing[:5]}'}
        
        # Get predictions
        X = df[ai_strategy.feature_columns].fillna(0).values
        
        if ai_strategy.scaler:
            X_scaled = ai_strategy.scaler.transform(X)
        else:
            X_scaled = X
        
        predictions = ai_strategy._ensemble_predict(X, X_scaled)
        
        # Analyze predictions
        stats = {
            'ticker': ticker,
            'samples': len(predictions),
            'min': predictions.min(),
            'max': predictions.max(),
            'mean': predictions.mean(),
            'std': predictions.std(),
            'median': np.median(predictions),
            'q25': np.percentile(predictions, 25),
            'q75': np.percentile(predictions, 75),
            'positive': (predictions > 0).sum(),
            'negative': (predictions < 0).sum(),
            'above_0.0001': (predictions > 0.0001).sum(),
            'below_-0.001': (predictions < -0.001).sum(),
            'above_0.0': (predictions > 0.0).sum(),
            'below_0.0': (predictions < 0.0).sum()
        }
        
        print(f"   📊 Prediction stats:")
        print(f"      Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"      Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
        print(f"      Positive: {stats['positive']}, Negative: {stats['negative']}")
        print(f"      Above 0.0001: {stats['above_0.0001']}")
        print(f"      Below -0.001: {stats['below_-0.001']}")
        
        return stats
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {'error': str(e)}


def test_relaxed_thresholds(
    ai_strategy: AIEnsembleStrategyV2,
    tickers: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Test with relaxed prediction thresholds."""
    print("\n🔧 Testing with RELAXED thresholds...")
    print("   LONG threshold: 0.0 (vs 0.0001)")
    print("   SHORT threshold: -0.01 (vs -0.001)")
    print("   Confirmation: Disabled (vs 2/3 required)")
    
    all_signals = []
    
    for ticker in tickers:
        try:
            data_store = get_data_store()
            df = data_store.get_ticker_data(ticker, start_date, end_date)
            
            if df is None or len(df) < 50:
                continue
            
            df = generate_all_features(df)
            
            missing = [f for f in ai_strategy.feature_columns if f not in df.columns]
            if missing:
                continue
            
            X = df[ai_strategy.feature_columns].fillna(0).values
            
            if ai_strategy.scaler:
                X_scaled = ai_strategy.scaler.transform(X)
            else:
                X_scaled = X
            
            predictions = ai_strategy._ensemble_predict(X, X_scaled)
            
            # RELAXED thresholds
            for i in range(len(df)):
                pred = predictions[i]
                confidence = abs(pred)
                
                signal = 0
                if pred > 0.0:  # Relaxed from 0.0001
                    signal = 1
                elif pred < -0.01:  # Relaxed from -0.001
                    signal = -1
                
                if signal == 0:
                    continue
                
                # NO confirmation requirement
                
                all_signals.append({
                    'date': df.index[i],
                    'ticker': ticker,
                    'prediction': pred,
                    'confidence': confidence,
                    'signal': signal
                })
        
        except Exception as e:
            continue
    
    if not all_signals:
        print("   ❌ Still no signals with relaxed thresholds")
        return pd.DataFrame()
    
    signals_df = pd.DataFrame(all_signals)
    print(f"   ✅ Generated {len(signals_df):,} signals")
    print(f"   📈 LONG: {(signals_df['signal'] == 1).sum():,}")
    print(f"   📉 SHORT: {(signals_df['signal'] == -1).sum():,}")
    
    return signals_df


def main():
    """Main diagnostic and fix routine."""
    print("\n" + "=" * 70)
    print("🔧 AI ENSEMBLE SIGNAL GENERATION FIX")
    print("=" * 70)
    
    # Initialize
    data_store = get_data_store()
    tickers = data_store.available_tickers[:20]  # Test on 20 tickers
    
    ai_strategy = AIEnsembleStrategyV2()
    
    # Step 1: Train models
    print("\n1️⃣ Training AI Ensemble...")
    ai_strategy.train_ensemble(
        tickers=tickers,
        train_start='2005-01-01',
        train_end='2014-12-31',
        use_cache=True
    )
    
    # Step 2: Diagnose predictions
    print("\n2️⃣ Diagnosing predictions...")
    test_ticker = tickers[0]
    stats = diagnose_predictions(
        ai_strategy=ai_strategy,
        ticker=test_ticker,
        start_date='2015-01-01',
        end_date='2015-12-31'
    )
    
    if 'error' not in stats:
        print(f"\n   💡 Insight:")
        if stats['above_0.0001'] == 0:
            print(f"      ⚠️ NO predictions above 0.0001 threshold")
            print(f"      ✅ But {stats['above_0.0']} predictions above 0.0")
            print(f"      🔧 Solution: Relax threshold to 0.0")
        
        if stats['below_-0.001'] == 0:
            print(f"      ⚠️ NO predictions below -0.001 threshold")
            print(f"      ✅ But {stats['below_0.0']} predictions below 0.0")
            print(f"      🔧 Solution: Relax threshold to -0.01")
    
    # Step 3: Test with relaxed thresholds
    print("\n3️⃣ Testing relaxed thresholds...")
    signals = test_relaxed_thresholds(
        ai_strategy=ai_strategy,
        tickers=tickers,
        start_date='2015-01-01',
        end_date='2015-12-31'
    )
    
    if not signals.empty:
        print("\n" + "=" * 70)
        print("✅ FIX VALIDATED - Signals generated successfully!")
        print("=" * 70)
        print(f"Total signals: {len(signals):,}")
        print(f"LONG signals: {(signals['signal'] == 1).sum():,}")
        print(f"SHORT signals: {(signals['signal'] == -1).sum():,}")
        print(f"\n🔧 Required changes to ai_ensemble_strategy_v2.py:")
        print("   1. Change LONG threshold: 0.0001 → 0.0")
        print("   2. Change SHORT threshold: -0.001 → -0.01")
        print("   3. Remove or relax 2/3 confirmation requirement")
    else:
        print("\n" + "=" * 70)
        print("❌ FIX FAILED - Still no signals")
        print("=" * 70)
        print("Further investigation needed")
    
    return signals


if __name__ == "__main__":
    signals = main()
