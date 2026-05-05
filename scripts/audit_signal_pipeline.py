#!/usr/bin/env python3
"""
FULL AUDIT: Signal Pipeline Diagnostic
=======================================
Traces every step from raw data → features → model prediction → signal output.
Identifies exactly where and why signals are being lost.
"""
import os
import sys
import pickle
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('AUDIT')
logger.setLevel(logging.INFO)

DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
MODELS_DIR = PROJECT_ROOT / 'models'

def audit_step1_model_files():
    """Check what model files exist and their metadata"""
    print("=" * 70)
    print("STEP 1: MODEL FILE AUDIT")
    print("=" * 70)
    
    expected_files = [
        'xgboost_model.pkl', 'lightgbm_model.pkl', 'rf_model.pkl',
        'feature_scaler.pkl', 'feature_names.pkl', 'ensemble_metadata.pkl',
        'regime_classifier.pkl', 'regime_scaler.pkl'
    ]
    
    for f in expected_files:
        path = MODELS_DIR / f
        if path.exists():
            size = path.stat().st_size
            mtime = pd.Timestamp.fromtimestamp(path.stat().st_mtime)
            print(f"  OK  {f} ({size:,} bytes, modified {mtime})")
        else:
            print(f"  MISSING  {f}")
    
    # Load feature names
    fn_path = MODELS_DIR / 'feature_names.pkl'
    if fn_path.exists():
        with open(fn_path, 'rb') as f:
            feature_names = pickle.load(f)
        print(f"\n  Training feature count: {len(feature_names)}")
        print(f"  Training features: {feature_names[:10]}...")
        return feature_names
    return None

def audit_step2_feature_generation(ticker='AAPL'):
    """Check what features are generated at inference time"""
    print("\n" + "=" * 70)
    print(f"STEP 2: FEATURE GENERATION AUDIT (ticker={ticker})")
    print("=" * 70)
    
    from core.feature_engineer import FeatureEngineer
    
    # Load ticker data
    df = pd.read_parquet(DATA_DIR / f'{ticker}.parquet')
    if 'date' in df.columns:
        df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    print(f"  Data rows: {len(df)}, columns: {list(df.columns)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # Generate features using FeatureEngineer
    fe = FeatureEngineer(use_advanced_features=True, verbose=False)
    feats, _ = fe.create_features(df, target_type='direction')
    
    print(f"  FeatureEngineer output: {feats.shape[0]} rows x {feats.shape[1]} columns")
    print(f"  Feature names: {list(feats.columns[:10])}...")
    print(f"  Total features from FeatureEngineer: {len(feats.columns)}")
    
    return feats, df

def audit_step3_feature_mismatch(training_features, inference_features):
    """Compare training features vs inference features"""
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE MISMATCH AUDIT")
    print("=" * 70)
    
    if training_features is None:
        print("  CANNOT AUDIT - training features not loaded")
        return
    
    train_set = set(training_features)
    infer_set = set(inference_features.columns)
    
    missing_in_inference = train_set - infer_set
    extra_in_inference = infer_set - train_set
    common = train_set & infer_set
    
    print(f"  Training features: {len(train_set)}")
    print(f"  Inference features: {len(infer_set)}")
    print(f"  Common features: {len(common)}")
    print(f"  MISSING at inference (in training but not inference): {len(missing_in_inference)}")
    if missing_in_inference:
        for f in sorted(missing_in_inference):
            print(f"    MISSING: {f}")
    print(f"  EXTRA at inference (in inference but not training): {len(extra_in_inference)}")
    if extra_in_inference:
        for f in sorted(extra_in_inference):
            print(f"    EXTRA: {f}")

def audit_step4_model_predictions(ticker='AAPL'):
    """Check raw model predictions"""
    print("\n" + "=" * 70)
    print(f"STEP 4: RAW MODEL PREDICTIONS AUDIT (ticker={ticker})")
    print("=" * 70)
    
    from core.ai_models import EnsemblePredictor
    
    ensemble = EnsemblePredictor()
    
    # Load data
    df = pd.read_parquet(DATA_DIR / f'{ticker}.parquet')
    if 'date' in df.columns:
        df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Generate features the SAME WAY the orchestrator does
    features = ensemble.generate_features(df)
    print(f"  Features generated: {features.shape}")
    
    # Get prediction with NO threshold
    signal_raw, conf_raw, details_raw = ensemble.predict(features, threshold=None)
    print(f"\n  Raw prediction (no threshold):")
    print(f"    Signal: {signal_raw}")
    print(f"    Confidence: {conf_raw:.4f}")
    print(f"    P(NEUTRAL): {details_raw['ensemble_prob_neutral']:.4f}")
    print(f"    P(LONG):    {details_raw['ensemble_prob_long']:.4f}")
    print(f"    P(SHORT):   {details_raw['ensemble_prob_short']:.4f}")
    
    # Get prediction with 0.55 threshold (BULL regime)
    signal_55, conf_55, details_55 = ensemble.predict(features, threshold=0.55)
    print(f"\n  With threshold=0.55 (BULL):")
    print(f"    Signal: {signal_55}")
    print(f"    Confidence: {conf_55:.4f}")
    
    # Get prediction with 0.65 threshold
    signal_65, conf_65, details_65 = ensemble.predict(features, threshold=0.65)
    print(f"\n  With threshold=0.65:")
    print(f"    Signal: {signal_65}")
    print(f"    Confidence: {conf_65:.4f}")
    
    return signal_raw, conf_raw, details_raw

def audit_step5_orchestrator_pipeline():
    """Check what the orchestrator does with signals"""
    print("\n" + "=" * 70)
    print("STEP 5: ORCHESTRATOR SIGNAL HANDLING AUDIT")
    print("=" * 70)
    
    # Read orchestrator source
    orch_path = PROJECT_ROOT / 'main_orchestrator_ist.py'
    with open(orch_path, 'r', encoding='utf-8') as f:
        orch_code = f.read()
    
    # Check for signal handlers
    has_buy = "signal == 'BUY'" in orch_code
    has_sell = "signal == 'SELL'" in orch_code
    has_sell_short = "signal == 'SELL_SHORT'" in orch_code or "SELL_SHORT" in orch_code
    has_hold = "signal == 'HOLD'" in orch_code
    
    print(f"  Handler for BUY:        {'YES' if has_buy else 'MISSING!'}")
    print(f"  Handler for SELL:       {'YES' if has_sell else 'MISSING!'}")
    print(f"  Handler for SELL_SHORT: {'YES' if has_sell_short else 'MISSING! <-- CRITICAL BUG'}")
    print(f"  Handler for HOLD:       {'YES' if has_hold else 'N/A (expected)'}")
    
    # Check what get_ensemble_signal returns
    print(f"\n  get_ensemble_signal() calls predict_from_ohlcv()")
    print(f"  predict_from_ohlcv() calls predict(features, threshold=None)")
    print(f"  --> Model returns highest-probability class (no threshold filtering)")
    print(f"  --> Possible returns: BUY, SELL_SHORT, HOLD")
    
    if not has_sell_short:
        print(f"\n  *** CRITICAL: SELL_SHORT signals are SILENTLY DROPPED ***")
        print(f"  *** Orchestrator only handles BUY and SELL, not SELL_SHORT ***")

def audit_step6_scan_all_tickers():
    """Scan multiple tickers to see signal distribution"""
    print("\n" + "=" * 70)
    print("STEP 6: SIGNAL DISTRIBUTION ACROSS TICKERS")
    print("=" * 70)
    
    from core.ai_models import get_ensemble_signal
    
    test_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                    'JPM', 'BAC', 'XOM', 'NVDA', 'META', 'VXX', 'GLD', 'TLT']
    
    results = {'BUY': 0, 'SELL': 0, 'SELL_SHORT': 0, 'HOLD': 0}
    details_list = []
    
    for ticker in test_tickers:
        try:
            path = DATA_DIR / f'{ticker}.parquet'
            if not path.exists():
                continue
            
            df = pd.read_parquet(path)
            if 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            if len(df) < 50:
                continue
            
            signal, confidence, details = get_ensemble_signal(df)
            results[signal] = results.get(signal, 0) + 1
            
            pn = details.get('ensemble_prob_neutral', 0)
            pl = details.get('ensemble_prob_long', 0)
            ps = details.get('ensemble_prob_short', 0)
            
            print(f"  {ticker:6s}: {signal:12s} (conf={confidence:.3f}) N={pn:.3f} L={pl:.3f} S={ps:.3f}")
            details_list.append({
                'ticker': ticker, 'signal': signal, 'confidence': confidence,
                'prob_neutral': pn, 'prob_long': pl, 'prob_short': ps
            })
            
        except Exception as e:
            print(f"  {ticker:6s}: ERROR - {e}")
    
    print(f"\n  Signal Distribution:")
    for sig, count in results.items():
        print(f"    {sig}: {count}")
    
    if details_list:
        df_details = pd.DataFrame(details_list)
        print(f"\n  Average probabilities across all tickers:")
        print(f"    P(NEUTRAL): {df_details['prob_neutral'].mean():.4f}")
        print(f"    P(LONG):    {df_details['prob_long'].mean():.4f}")
        print(f"    P(SHORT):   {df_details['prob_short'].mean():.4f}")

def audit_step7_regime_detection():
    """Check what regime is detected"""
    print("\n" + "=" * 70)
    print("STEP 7: REGIME DETECTION AUDIT")
    print("=" * 70)
    
    from core.regime_detector import RegimeDetector
    
    detector = RegimeDetector()
    
    # Load SPY and VXX
    spy_df = pd.read_parquet(DATA_DIR / 'SPY.parquet')
    if 'date' in spy_df.columns:
        spy_df = spy_df.set_index('date')
    spy_df.index = pd.to_datetime(spy_df.index)
    
    vxx_path = DATA_DIR / 'VXX.parquet'
    vxx_df = None
    if vxx_path.exists():
        vxx_df = pd.read_parquet(vxx_path)
        if 'date' in vxx_df.columns:
            vxx_df = vxx_df.set_index('date')
        vxx_df.index = pd.to_datetime(vxx_df.index)
    
    regime_code, regime_name, threshold = detector.detect_regime(spy_df, vxx_df)
    
    print(f"  Current Regime: {regime_name} ({regime_code})")
    print(f"  Confidence Threshold: {threshold}")
    print(f"  Regime Thresholds: {detector.regime_thresholds}")

def main():
    print("=" * 70)
    print("  NEURALTRADER FULL SIGNAL PIPELINE AUDIT")
    print("  Date: " + str(pd.Timestamp.now()))
    print("=" * 70)
    
    # Step 1: Model files
    training_features = audit_step1_model_files()
    
    # Step 2: Feature generation
    inference_features, _ = audit_step2_feature_generation('AAPL')
    
    # Step 3: Feature mismatch
    audit_step3_feature_mismatch(training_features, inference_features)
    
    # Step 4: Raw model predictions
    audit_step4_model_predictions('AAPL')
    
    # Step 5: Orchestrator pipeline
    audit_step5_orchestrator_pipeline()
    
    # Step 6: Scan all tickers
    audit_step6_scan_all_tickers()
    
    # Step 7: Regime detection
    audit_step7_regime_detection()
    
    # SUMMARY
    print("\n" + "=" * 70)
    print("  AUDIT SUMMARY - ROOT CAUSES")
    print("=" * 70)

if __name__ == '__main__':
    main()
