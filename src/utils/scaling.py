from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def rolling_scale_features(
    X: pd.DataFrame,
    window: int = 252,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply rolling z-score scaling to each feature using a trailing window.
    Returns scaled DataFrame and the last fitted scaler (for inference).
    """
    scaled = X.copy()
    scalers = {}
    for col in X.columns:
        scaler = StandardScaler()
        # Fit on trailing window and transform; for early rows, use expanding window
        rolling_vals = X[col].rolling(window=window, min_periods=1)
        scaled[col] = rolling_vals.apply(lambda s: scaler.fit_transform(s.values.reshape(-1, 1)).flatten()[-1])
        scalers[col] = scaler
    # Return the last scaler for inference use
    last_scaler = StandardScaler()
    last_scaler.fit(X.iloc[-window:])
    return scaled, last_scaler
