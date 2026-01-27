from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from models import xgboost_model


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    param_grid: Dict[str, List[Any]] | None = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Simple grid search for XGBoost hyperparameters.
    Returns best_params and validation RMSE.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [200, 300],
            "max_depth": [3, 4],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

    best_params = {}
    best_rmse = float("inf")

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total = int(np.prod([len(v) for v in values]))

    for i, combo in enumerate(np.meshgrid(*values)):
        params = dict(zip(keys, combo))
        model = xgboost_model.build_model(params)
        preds = xgboost_model.train_and_predict(X_train, y_train, X_val, model)
        preds = np.array(preds).flatten()[:len(y_val)]
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    return best_params, best_rmse
