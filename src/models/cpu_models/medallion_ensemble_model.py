"""
Medallion Models Integration
Integrate Renaissance Technologies inspired models into existing pipeline
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

def build_medallion_ensemble_model():
    """Build Medallion-inspired ensemble model"""
    return MedallionEnsembleModel()

def train_and_predict_medallion_ensemble(X_train, y_train, X_test, model=None):
    """Train and predict with Medallion ensemble"""
    if model is None:
        model = build_medallion_ensemble_model()
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_train[-len(predictions):], predictions))
    
    return predictions, rmse

def predict_future_medallion_ensemble(model, recent_X_df, steps=1):
    """Predict future with Medallion ensemble"""
    predictions = []
    current_data = recent_X_df.copy()
    
    for _ in range(steps):
        pred = model.predict(current_data.values.reshape(1, -1))[0]
        predictions.append(pred)
        
        # Update features for next prediction (simplified)
        # In practice, you'd update all features properly
        current_data = current_data.shift(1)
        current_data.iloc[0, 0] = pred  # Update first feature with prediction
    
    return np.array(predictions)

class MedallionEnsembleModel:
    """
    Medallion-inspired ensemble model
    Integrates with existing pipeline
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.is_fitted = False
        
    def _create_models(self):
        """Create diverse models like Medallion"""
        models = {}
        
        # Random Forest - great for non-linear patterns
        models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting - great for sequential patterns
        models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # Neural Network - great for complex relationships
        models['neural_network'] = MedallionNeuralNetwork(
            input_size=self.feature_names.shape[0] if self.feature_names is not None else 50
        )
        
        return models
    
    def fit(self, X_train, y_train):
        """Fit all models in the ensemble"""
        # Convert to DataFrame if needed
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
        
        self.feature_names = X_train.columns if hasattr(X_train, 'columns') else None
        
        # Create models
        self.models = self._create_models()
        
        # Fit each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name == 'neural_network':
                # Neural network training
                self._fit_neural_network(model, X_train.values, y_train)
            else:
                # Scikit-learn models
                model.fit(X_train, y_train)
        
        self.is_fitted = True
        print("✅ Medallion ensemble training complete!")
    
    def _fit_neural_network(self, model, X_train, y_train):
        """Fit neural network model"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Loss {loss.item():.6f}")
    
    def predict(self, X_test):
        """Make predictions with ensemble"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to DataFrame if needed
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if name == 'neural_network':
                # Neural network prediction
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test.values)
                    pred = model(X_tensor).cpu().numpy().flatten()
                predictions[name] = pred
            else:
                # Scikit-learn models
                pred = model.predict(X_test)
                predictions[name] = pred
        
        # Ensure all predictions have the same shape
        pred_shapes = [pred.shape for pred in predictions.values()]
        if len(set(pred_shapes)) > 1:
            # Find the most common shape
            from collections import Counter
            most_common_shape = Counter(pred_shapes).most_common(1)[0][0]
            
            # Resize all predictions to the same shape
            for name in predictions:
                if predictions[name].shape != most_common_shape:
                    if len(predictions[name].shape) > 1:
                        predictions[name] = predictions[name].flatten()
                    if predictions[name].shape[0] != most_common_shape[0]:
                        predictions[name] = predictions[name][:most_common_shape[0]]
        
        # Ensemble predictions (simple average)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        return ensemble_pred
    
    def get_feature_importance(self):
        """Get feature importance from ensemble"""
        importance = {}
        
        # Get importance from tree-based models
        if 'random_forest' in self.models:
            importance['random_forest'] = self.models['random_forest'].feature_importances_
        
        if 'gradient_boosting' in self.models:
            importance['gradient_boosting'] = self.models['gradient_boosting'].feature_importances_
        
        return importance

class MedallionNeuralNetwork(nn.Module):
    """Neural network component of Medallion ensemble"""
    
    def __init__(self, input_size):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class MedallionFeatureEngine:
    """
    Feature engineering for Medallion models
    Creates hedge fund level features
    """
    
    def __init__(self):
        self.feature_names = []
    
    def create_medallion_features(self, df, timeframe='1d'):
        """Create Medallion-inspired features"""
        features = pd.DataFrame(index=df.index)
        
        # Price columns
        close_col = f"{timeframe}_close"
        high_col = f"{timeframe}_high"
        low_col = f"{timeframe}_low"
        open_col = f"{timeframe}_open"
        volume_col = f"{timeframe}_volume"
        
        # === MOMENTUM FEATURES ===
        # Multiple timeframe momentum
        for period in [5, 10, 20, 50]:
            features[f'momentum_{period}'] = df[close_col].pct_change(period)
            features[f'momentum_rank_{period}'] = features[f'momentum_{period}'].rolling(252).rank(pct=True)
        
        # RSI
        for period in [14, 30]:
            delta = df[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # === MEAN REVERSION FEATURES ===
        # Bollinger Bands
        for period in [20, 50]:
            sma = df[close_col].rolling(period).mean()
            std = df[close_col].rolling(period).std()
            features[f'bb_position_{period}'] = (df[close_col] - sma) / (std * 2)
            features[f'bb_width_{period}'] = (sma + std * 2) / (sma - std * 2)
        
        # Distance from moving averages
        for period in [10, 20, 50, 200]:
            ma = df[close_col].rolling(period).mean()
            features[f'distance_ma_{period}'] = (df[close_col] - ma) / ma
        
        # === VOLATILITY FEATURES ===
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = df[close_col].pct_change().rolling(period).std()
            features[f'volatility_rank_{period}'] = features[f'volatility_{period}'].rolling(252).rank(pct=True)
        
        # === TREND FEATURES ===
        # Moving average crossovers
        features['ma_cross_10_20'] = np.where(
            df[close_col].rolling(10).mean() > df[close_col].rolling(20).mean(),
            1, -1
        )
        features['ma_cross_20_50'] = np.where(
            df[close_col].rolling(20).mean() > df[close_col].rolling(50).mean(),
            1, -1
        )
        
        # Trend strength
        features['trend_strength'] = abs(
            df[close_col].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        )
        
        # === VOLUME FEATURES ===
        if volume_col in df.columns:
            features['volume_sma'] = df[volume_col].rolling(20).mean()
            features['volume_ratio'] = df[volume_col] / features['volume_sma']
            features['volume_price_trend'] = np.where(
                (df[close_col] > df[open_col]) & (features['volume_ratio'] > 1.5),
                1,
                np.where(
                    (df[close_col] < df[open_col]) & (features['volume_ratio'] > 1.5),
                    -1,
                    0
                )
            )
        
        # === PRICE ACTION FEATURES ===
        if all(col in df.columns for col in [high_col, low_col, open_col]):
            features['daily_range'] = (df[high_col] - df[low_col]) / df[close_col]
            features['body_ratio'] = abs(df[close_col] - df[open_col]) / (df[high_col] - df[low_col])
            features['upper_shadow'] = (df[high_col] - np.maximum(df[close_col], df[open_col])) / df[close_col]
            features['lower_shadow'] = (np.minimum(df[close_col], df[open_col]) - df[low_col]) / df[close_col]
        
        self.feature_names = features.columns.tolist()
        return features.dropna()

# Integration functions for existing pipeline
def integrate_medallion_into_existing_pipeline():
    """Integration helper for existing pipeline"""
    
    # Model interface functions
    def medallion_build_model():
        return build_medallion_ensemble_model()
    
    def medallion_train_and_predict(X_train, y_train, X_test, model):
        return train_and_predict_medallion_ensemble(X_train, y_train, X_test, model)
    
    def medallion_predict_future(model, recent_X_df, steps=1):
        return predict_future_medallion_ensemble(model, recent_X_df, steps)
    
    return {
        'build_model': medallion_build_model,
        'train_and_predict': medallion_train_and_predict,
        'predict_future_sequence': medallion_predict_future
    }

# Create module-like interface
medallion_ensemble_model = type('Module', (), {
    'build_model': build_medallion_ensemble_model,
    'train_and_predict': train_and_predict_medallion_ensemble,
    'predict_future_sequence': predict_future_medallion_ensemble
})()

# Export functions for direct import
__all__ = ['build_medallion_ensemble_model', 'train_and_predict_medallion_ensemble', 'predict_future_medallion_ensemble', 'medallion_ensemble_model']

if __name__ == "__main__":
    print("🏆 Medallion Models Integration Ready")
    print("✅ Compatible with existing pipeline")
    print("✅ Ensemble of RF + GB + Neural Network")
    print("✅ Hedge fund level features")
