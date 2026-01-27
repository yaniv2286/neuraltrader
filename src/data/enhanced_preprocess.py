"""
Enhanced Data Preprocessing Pipeline
Integrates all feature engineering modules with robust data validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .tiingo_loader import load_tiingo_data

# Import feature modules (with error handling)
try:
    from features.indicators import apply_indicators
    from features.advanced_features import detect_support_resistance, detect_patterns
    from features.momentum import calculate_momentum_features
    from features.volatility import calculate_volatility_features
    from features.regime_detector import detect_market_regime
    FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Feature modules not available: {e}")
    FEATURES_AVAILABLE = False

# Basic indicators fallback
def apply_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Basic indicators when feature modules are not available"""
    df = df.copy()
    
    if 'close' in df.columns:
        # Basic moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_20d'] = df['close'].pct_change(20)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
    
    # Volume features
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df

class DataValidator:
    """Validates data quality and consistency"""
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, any]:
        """Comprehensive data quality checks"""
        issues = []
        
        # Check for missing data
        missing_pct = df.isnull().sum() / len(df) * 100
        if missing_pct.max() > 5:
            issues.append(f"High missing data: {missing_pct.max():.1f}%")
        
        # Check for gaps in time series
        if df.index.duplicated().any():
            issues.append("Duplicate timestamps found")
        
        # Check for outliers (using IQR method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'price' in col.lower() or 'close' in col.lower():
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > len(df) * 0.01:  # More than 1% outliers
                    issues.append(f"Many outliers in {col}: {outliers}")
        
        # Check for zero/negative prices
        price_cols = [col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()]
        for col in price_cols:
            if (df[col] <= 0).any():
                issues.append(f"Non-positive prices in {col}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'missing_pct': missing_pct.max(),
            'total_records': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}"
        }
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean data based on validation results"""
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Forward fill small gaps (max 3 consecutive days)
        df_clean = df_clean.groupby(df_clean.index.to_period('D')).apply(
            lambda x: x if len(x) > 0 else None
        ).dropna()
        
        # Handle outliers using winsorization
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'close' in col.lower() or 'price' in col.lower():
                Q1 = df_clean[col].quantile(0.01)
                Q3 = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(Q1, Q3)
        
        return df_clean

class FeatureEngineer:
    """Comprehensive feature engineering pipeline"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all technical indicators and features"""
        df_features = df.copy()
        
        if not FEATURES_AVAILABLE:
            print("⚠️ Feature modules not available, using basic features only")
            return apply_basic_indicators(df_features)
        
        try:
            # Basic indicators
            df_features = apply_indicators(df_features)
            
            # Advanced features
            df_features = detect_support_resistance(df_features)
            df_features = detect_patterns(df_features)
            
            # Momentum features
            df_features = calculate_momentum_features(df_features)
            
            # Volatility features
            df_features = calculate_volatility_features(df_features)
            
            # Market regime
            df_features = detect_market_regime(df_features)
            
        except Exception as e:
            print(f"⚠️ Error in feature modules: {e}")
            print("🔄 Using basic features instead")
            return apply_basic_indicators(df_features)
        
        # Additional custom features (always available)
        df_features = self._add_custom_features(df_features)
        
        # Store feature names for reference
        self.feature_names = [col for col in df_features.columns if col not in df.columns]
        
        return df_features
    
    def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom trading features"""
        df = df.copy()
        
        # Price-based features
        if 'close' in df.columns:
            # Returns at different horizons
            for period in [1, 3, 5, 10, 20]:
                df[f'return_{period}d'] = df['close'].pct_change(period)
            
            # Moving averages and ratios
            for period in [5, 10, 20, 50, 200]:
                df[f'ma_{period}'] = df['close'].rolling(period).mean()
                df[f'price_to_ma_{period}'] = df['close'] / df[f'ma_{period}']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Price momentum
            df['price_momentum_5'] = df['close'].pct_change(5)
            df['price_momentum_20'] = df['close'].pct_change(20)
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['volume_price_trend'] = (df['close'].pct_change() * df['volume']).rolling(5).sum()
        
        # Volatility features
        if 'close' in df.columns:
            df['volatility_5'] = df['close'].pct_change().rolling(5).std()
            df['volatility_20'] = df['close'].pct_change().rolling(20).std()
            df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators when feature modules aren't available"""
        df = df.copy()
        
        # Basic moving averages
        if 'close' in df.columns:
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Returns
            df['returns'] = df['close'].pct_change()
            df['returns_5d'] = df['close'].pct_change(5)
            df['returns_20d'] = df['close'].pct_change(20)
            
            # Volatility
            df['volatility'] = df['returns'].rolling(20).std()
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df

def build_enhanced_model_input(
    ticker: str, 
    timeframes: List[str] = ["1d"], 
    start: str = "2020-01-01", 
    end: str = "2024-01-01",
    validate_data: bool = True,
    create_features: bool = True
) -> Optional[pd.DataFrame]:
    """
    Enhanced model input building with validation and comprehensive features
    
    Args:
        ticker: Stock symbol
        timeframes: List of timeframes (e.g., ["1d", "1wk"])
        start: Start date
        end: End date
        validate_data: Whether to validate data quality
        create_features: Whether to create technical indicators
    
    Returns:
        Enhanced DataFrame with all features or None if data is invalid
    """
    print(f"🔧 Building enhanced model input for {ticker}...")
    
    # Step 1: Fetch raw data from Tiingo cache
    df_dict = {}
    for tf in timeframes:
        print(f"   📊 Loading {ticker} data for {tf} from cache...")
        df = load_tiingo_data(ticker, start_date=start, end_date=end)
        
        if df is None or df.empty:
            print(f"   ❌ No data for {ticker} at {tf}")
            continue
        
        # Add timeframe prefix (for multi-timeframe support)
        if len(timeframes) > 1:
            df = df.add_prefix(f"{tf}_")
        df_dict[tf] = df
    
    if not df_dict:
        print(f"   ❌ No valid data retrieved for {ticker}")
        return None
    
    # Step 2: Combine timeframes
    combined = pd.concat(df_dict.values(), axis=1, join="outer")
    combined = combined.dropna()
    
    print(f"   📈 Combined data shape: {combined.shape}")
    
    # Step 3: Data validation
    if validate_data:
        validator = DataValidator()
        validation_result = validator.check_data_quality(combined)
        
        if not validation_result['valid']:
            print(f"   ⚠️ Data quality issues found:")
            for issue in validation_result['issues']:
                print(f"      - {issue}")
            
            # Clean the data
            combined = validator.clean_data(combined)
            print(f"   ✅ Data cleaned. New shape: {combined.shape}")
        else:
            print(f"   ✅ Data validation passed")
    
    # Step 4: Feature engineering
    if create_features:
        engineer = FeatureEngineer()
        combined = engineer.create_all_features(combined)
        print(f"   🎯 Features created. Total features: {len(combined.columns)}")
        print(f"   📋 New features: {len(engineer.feature_names)}")
    
    # Step 5: Final validation
    if combined.empty or combined.isnull().all().all():
        print(f"   ❌ No valid data after processing")
        return None
    
    print(f"   ✅ Enhanced model input ready: {combined.shape}")
    return combined

def create_sequences(
    df: pd.DataFrame, 
    target_col: str = "1d_close", 
    sequence_length: int = 30,
    prediction_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series models
    
    Args:
        df: Feature DataFrame
        target_col: Target column name
        sequence_length: Length of input sequences
        prediction_horizon: Days ahead to predict
    
    Returns:
        Tuple of (X, y) arrays
    """
    # Remove any remaining NaN values
    df_clean = df.dropna()
    
    if target_col not in df_clean.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Create target (shifted by prediction horizon)
    target = df_clean[target_col].shift(-prediction_horizon)
    
    # Remove NaN from target
    valid_idx = target.dropna().index
    df_clean = df_clean.loc[valid_idx]
    target = target.loc[valid_idx]
    
    # Create sequences
    X, y = [], []
    feature_cols = [col for col in df_clean.columns if col != target_col]
    
    for i in range(sequence_length, len(df_clean)):
        X.append(df_clean[feature_cols].iloc[i-sequence_length:i].values)
        y.append(target.iloc[i])
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Test the enhanced pipeline
    print("🧪 Testing enhanced data pipeline...")
    
    ticker = "AAPL"
    df = build_enhanced_model_input(
        ticker=ticker,
        timeframes=["1d"],
        start="2023-01-01",
        end="2024-01-01"
    )
    
    if df is not None:
        print(f"✅ Success! Final DataFrame shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns[:10])}...")
        
        # Test sequence creation
        X, y = create_sequences(df, sequence_length=30)
        print(f"🎯 Sequences created: X={X.shape}, y={y.shape}")
    else:
        print("❌ Pipeline test failed")
