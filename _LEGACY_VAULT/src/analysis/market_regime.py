"""
Market Regime Classifier - HMM-based Market State Detection
===========================================================

Uses Hidden Markov Models with market breadth to classify market states:
- State 0: Bull Market (Low Volatility, High Breadth)
- State 1: Chop/Transition (Medium Volatility, Medium Breadth)  
- State 2: Bear Market (High Volatility, Low Breadth)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler

# Import HMM
from hmmlearn.hmm import GaussianHMM

# Configure logging
logger = logging.getLogger('MarketRegime')

class MarketBreadth:
    """Calculates market breadth indicators for regime detection"""
    
    def __init__(self):
        self.logger = logging.getLogger('MarketBreadth')
    
    def calculate_mmtw(self, universe_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculate Market Momentum Trend Weight (MMTW)
        
        Args:
            universe_data: Dict of ticker DataFrames with OHLCV data
            
        Returns:
            pd.Series: Daily breadth values (0.0 to 1.0)
        """
        self.logger.info("Calculating market breadth (MMTW)...")
        
        breadth_data = {}
        
        for ticker, df in universe_data.items():
            try:
                # Calculate 20-day SMA
                df = df.copy()
                df['SMA_20'] = df['Close'].rolling(window=20, min_periods=10).mean()
                
                # Determine if stock is above SMA (healthy)
                df['above_sma'] = (df['Close'] > df['SMA_20']).astype(int)
                
                # Store date and above_sma
                breadth_data[ticker] = df[['Date', 'above_sma']].set_index('Date')
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate breadth for {ticker}: {e}")
        
        # Combine all tickers
        if not breadth_data:
            return pd.Series(dtype=float)
        
        # Concatenate all breadth data
        combined_breadth = pd.concat(breadth_data.values(), axis=1)
        
        # Calculate percentage of stocks above SMA
        mmtw = combined_breadth.mean(axis=1)
        
        # Forward fill any missing values
        mmtw = mmtw.ffill().fillna(0.5)
        
        self.logger.info(f"Calculated MMTW for {len(mmtw)} days")
        self.logger.info(f"MMTW stats - Min: {mmtw.min():.3f}, Max: {mmtw.max():.3f}, Mean: {mmtw.mean():.3f}")
        
        return mmtw
    
    def calculate_advance_decline(self, universe_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculate Advance/Decline ratio
        
        Args:
            universe_data: Dict of ticker DataFrames with OHLCV data
            
        Returns:
            pd.Series: Daily A/D ratio
        """
        adv_dec_data = {}
        
        for ticker, df in universe_data.items():
            try:
                df = df.copy()
                df['advance'] = (df['Close'] > df['Open']).astype(int)
                df['decline'] = (df['Close'] < df['Open']).astype(int)
                
                adv_dec_data[ticker] = df[['Date', 'advance', 'decline']].set_index('Date')
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate A/D for {ticker}: {e}")
        
        if not adv_dec_data:
            return pd.Series(dtype=float)
        
        # Sum advances and declines across all stocks
        combined = pd.concat(adv_dec_data.values(), axis=1)
        total_advances = combined['advance'].sum(axis=1)
        total_declines = combined['decline'].sum(axis=1)
        
        # Calculate A/D ratio
        ad_ratio = total_advances / (total_advances + total_declines)
        ad_ratio = ad_ratio.fillna(0.5).ffill()
        
        return ad_ratio


class RegimeClassifier:
    """HMM-based market regime classifier"""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the regime classifier
        
        Args:
            models_dir: Path to models directory
        """
        self.logger = logging.getLogger('RegimeClassifier')
        
        # Set models directory
        if models_dir is None:
            project_root = Path(__file__).parent.parent.parent
            models_dir = project_root / "models"
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model components (loaded once into memory)
        self.hmm_model = None
        self.scaler = None
        self.state_mapping = None  # Maps HMM states to Bull/Chop/Bear
        
        # Initialize market breadth calculator
        self.breadth_calculator = MarketBreadth()
        
        # Load model immediately into memory
        self.load_model()
    
    def prepare_features(self, spy_data: pd.DataFrame, vix_data: pd.DataFrame, 
                        mmtw_data: pd.Series) -> np.ndarray:
        """
        Prepare features for HMM training
        
        Args:
            spy_data: SPY OHLCV data
            vix_data: VIX data
            mmtw_data: Market breadth series
            
        Returns:
            np.ndarray: Feature matrix (N_samples, 3)
        """
        # Calculate SPY returns
        spy_returns = spy_data['Close'].pct_change().fillna(0)
        
        # Ensure all data is aligned by date
        self.logger.info(f"SPY data shape: {spy_data.shape}")
        self.logger.info(f"VIX data shape: {vix_data.shape}")
        self.logger.info(f"MMTW data shape: {mmtw_data.shape}")
        
        # Convert mmtw to DataFrame if it's a Series
        if isinstance(mmtw_data, pd.Series):
            mmtw_df = mmtw_data.reset_index()
            mmtw_df.columns = ['Date', 'MMTW']
            # Ensure datetime format without timezone
            mmtw_df['Date'] = pd.to_datetime(mmtw_df['Date']).dt.tz_localize(None)
        else:
            mmtw_df = mmtw_data
        
        # Align all data by date
        features_df = pd.DataFrame({
            'Date': pd.to_datetime(spy_data['Date']).dt.tz_localize(None),
            'Returns': spy_returns.values,
            'VIX': vix_data['Close'].values if 'Close' in vix_data.columns else vix_data.iloc[:, 0].values
        })
        
        # Merge with MMTW
        features_df = features_df.merge(mmtw_df, on='Date', how='inner')
        
        self.logger.info(f"After merge, features_df shape: {features_df.shape}")
        
        # Drop any rows with NaN values
        features_df = features_df.dropna()
        
        self.logger.info(f"After dropping NaN, features_df shape: {features_df.shape}")
        
        # Return only the feature columns
        return features_df[['Returns', 'VIX', 'MMTW']].values
    
    def fit(self, spy_data: pd.DataFrame, vix_data: pd.DataFrame, 
            mmtw_data: pd.Series, n_components: int = 3) -> Dict:
        """
        Fit HMM model to market data
        
        Args:
            spy_data: SPY OHLCV data
            vix_data: VIX data
            mmtw_data: Market breadth series
            n_components: Number of HMM states (default 3)
            
        Returns:
            Dict: Training results and state characteristics
        """
        self.logger.info("Training HMM regime classifier...")
        
        # Prepare features
        features = self.prepare_features(spy_data, vix_data, mmtw_data)
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Initialize and fit HMM
        self.hmm_model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        self.hmm_model.fit(features_scaled)
        
        # Get state characteristics
        state_means = self.hmm_model.means_
        state_covars = self.hmm_model.covars_
        
        # Analyze states to determine mapping
        # State 0 (Bull): Lowest VIX, Highest MMTW
        # State 2 (Bear): Highest VIX, Lowest MMTW  
        # State 1 (Chop): Middle ground
        
        # Sort states by VIX (index 1) and MMTW (index 2)
        state_scores = []
        for i in range(n_components):
            vix_mean = state_means[i][1]  # VIX is feature 1
            mmtw_mean = state_means[i][2]  # MMTW is feature 2
            # Lower VIX and higher MMTW = better (bullish)
            score = -vix_mean + mmtw_mean  # Negative VIX + positive MMTW
            state_scores.append((i, score, vix_mean, mmtw_mean))
        
        # Sort by score (highest = most bullish)
        state_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create state mapping
        self.state_mapping = {}
        for rank, (state_idx, score, vix_mean, mmtw_mean) in enumerate(state_scores):
            if rank == 0:
                self.state_mapping[state_idx] = 0  # Bull
                regime_name = "BULL"
            elif rank == n_components - 1:
                self.state_mapping[state_idx] = 2  # Bear
                regime_name = "BEAR"
            else:
                self.state_mapping[state_idx] = 1  # Chop
                regime_name = "CHOP"
            
            self.logger.info(f"State {state_idx} -> {regime_name} (VIX: {vix_mean:.3f}, MMTW: {mmtw_mean:.3f})")
        
        # Save model
        model_path = self.models_dir / "hmm_regime.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'hmm_model': self.hmm_model,
                'scaler': self.scaler,
                'state_mapping': self.state_mapping,
                'state_means': state_means,
                'state_covars': state_covars
            }, f)
        
        self.logger.info(f"HMM model saved to {model_path}")
        
        # Get predicted states for training data
        predicted_states = self.hmm_model.predict(features_scaled)
        mapped_states = [self.state_mapping[s] for s in predicted_states]
        
        # Calculate state statistics
        state_stats = {}
        for regime in range(n_components):
            regime_mask = np.array(mapped_states) == regime
            if regime_mask.any():
                regime_features = features[regime_mask]
                state_stats[regime] = {
                    'count': int(regime_mask.sum()),
                    'percentage': regime_mask.mean() * 100,
                    'avg_return': np.mean(regime_features[:, 0]),
                    'avg_vix': np.mean(regime_features[:, 1]),
                    'avg_mmtw': np.mean(regime_features[:, 2])
                }
        
        results = {
            'model_path': str(model_path),
            'n_components': n_components,
            'state_mapping': self.state_mapping,
            'state_stats': state_stats,
            'training_samples': len(features)
        }
        
        self.logger.info("HMM training completed successfully")
        return results
    
    def load_model(self) -> bool:
        """
        Load HMM model into memory (called once during initialization)
        
        Returns:
            bool: True if model loaded successfully
        """
        model_path = self.models_dir / "hmm_regime.pkl"
        
        if not model_path.exists():
            self.logger.error(f"No model found at {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.hmm_model = model_data['hmm_model']
                self.scaler = model_data['scaler']
                self.state_mapping = model_data['state_mapping']
            
            self.logger.info("HMM model loaded into memory")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_state(self, current_data: Tuple[np.ndarray, np.ndarray, float]) -> Optional[int]:
        """
        Predict current market regime using in-memory model
        
        Args:
            current_data: Tuple of (spy_return, vix_value, mmtw_value)
            
        Returns:
            int: Predicted regime (0=Bull, 1=Chop, 2=Bear) or None if failed
        """
        # Check if model is loaded in memory
        if self.hmm_model is None or self.scaler is None or self.state_mapping is None:
            # Default to Bull/Green state if no model
            return 0
        
        try:
            # Prepare current features
            spy_return, vix_value, mmtw_value = current_data
            features = np.array([[spy_return, vix_value, mmtw_value]])
            
            # Check for NaN or infinite values
            if not np.all(np.isfinite(features)):
                self.logger.warning(f"Non-finite values in features: {features}")
                return 0  # Default to Bull
            
            # Scale features using in-memory scaler
            features_scaled = self.scaler.transform(features)
            
            # Predict state using in-memory model
            hmm_state = self.hmm_model.predict(features_scaled)[0]
            regime_state = self.state_mapping[hmm_state]
            
            return regime_state
            
        except Exception as e:
            self.logger.error(f"Failed to predict regime: {e}")
            return 0  # Default to Bull on error
    
    def get_regime_name(self, state: int) -> str:
        """Get human-readable regime name"""
        names = {0: "BULL", 1: "CHOP", 2: "BEAR"}
        return names.get(state, "UNKNOWN")
    
    def analyze_regime_history(self, spy_data: pd.DataFrame, vix_data: pd.DataFrame,
                              mmtw_data: pd.Series) -> pd.DataFrame:
        """
        Analyze historical regime predictions
        
        Returns:
            pd.DataFrame: Date, Regime, and features
        """
        if not self.load_model():
            return pd.DataFrame()
        
        # Prepare features
        features = self.prepare_features(spy_data, vix_data, mmtw_data)
        features_scaled = self.scaler.transform(features)
        
        # Predict states
        hmm_states = self.hmm_model.predict(features_scaled)
        mapped_states = [self.state_mapping[s] for s in hmm_states]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': spy_data['Date'].iloc[len(spy_data) - len(mapped_states):],
            'Regime': mapped_states,
            'RegimeName': [self.get_regime_name(s) for s in mapped_states],
            'Returns': features[:, 0],
            'VIX': features[:, 1],
            'MMTW': features[:, 2]
        })
        
        return results
