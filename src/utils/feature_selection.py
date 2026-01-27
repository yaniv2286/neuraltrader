from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.base import clone

from models import xgboost_model


def select_features_rfe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features_to_select: int = 20,
    step: float = 0.1,
) -> Tuple[List[str], RFE]:
    """
    Recursive Feature Elimination using XGBoost.
    Returns list of selected feature names and the fitted RFE object.
    """
    estimator = xgboost_model.build_model()
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
    rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_].tolist()
    return selected_features, rfe
