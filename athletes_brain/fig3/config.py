"""Configuration module for figure 2 modeling settings."""

from pathlib import Path
from typing import Dict, Any
from sklearn.model_selection import GroupKFold
from athletes_brain.config import FIGURES_DIR, DATA_DIR
from athletes_brain.fig1.config import Fig1Config
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import (
    ElasticNet,
    RidgeCV,
    ElasticNetCV,
    LassoCV,
    Ridge,
    Lasso,
    SGDRegressor,
)

# xgb
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

REGION_COL = "index"


# --- Modular Config Class for Fig2 ---
class Fig2Config:
    """Configuration class for Figure 2 modeling settings."""

    # Atlas configuration
    ATLAS = "schaefer2018tian2020_400_7"
    REGION_COL = "index"

    # Metrics to analyze
    METRICS = ["gm_vol", "wm_vol", "csf_vol", "adc", "fa", "ad", "rd"]
    # METRICS = ["gm_vol"]
    DISTRIBUTION_METRIC = "qfmean"
    BAD_SUBJECTS = ["IN120120"]

    # Model Definitions
    AVAILABLE_MODELS = {
        "ElasticNet": ElasticNet(max_iter=int(1e6)),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=int(1e6)),
        "HistGB": HistGradientBoostingRegressor(),
        "xgboost": xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    }
    ALPHAS = np.logspace(-3, 4, 30)  # ridge λ grid

    # Hyperparameter Grids
    AVAILABLE_PARAMS = {
        "ElasticNet": {
            "estimator__l1_ratio": [0.2, 0.5, 0.9, 1.0],  # l1_ratio for ElasticNet
            "estimator__alpha": ALPHAS,  # alpha for ElasticNet
        },
        "Ridge": {
            "estimator__alpha": ALPHAS,  # alpha for Ridge
        },
        "Lasso": {
            "estimator__alpha": ALPHAS,  # alpha for Lasso
        },
        "HistGB": {
            "estimator__max_iter": [100, 200, 300],
            "estimator__max_depth": [3, 5, 7],
            "estimator__learning_rate": [0.01, 0.1, 0.2],
        },
        "xgboost": {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__max_depth": [3, 5, 7],
            "estimator__learning_rate": [0.01, 0.1, 0.2],
            "estimator__subsample": [0.8, 1.0],
            "estimator__colsample_bytree": [0.8, 1.0],
        },
    }

    # Cross-validation
    CV_SPLITS = 5
    cv = GroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)

    # Data and Output Directories
    DATA_DIR = DATA_DIR
    OUTPUT_DIR = FIGURES_DIR / "fig2"

    # Experiment Parameters
    N_PERMUTATIONS = 1000
    MODEL_NAME = "sgd"
    GROUPS = [None, "climbing", "bjj"]
    GROUP_NAMES = {
        None: "all",
        "climbing": "climbing",
        "bjj": "bjj",
    }

    # Demographic and Target Columns for long_to_wide
    DEMOGRAPHIC_COLS = [
        "age_at_scan",
        "sex",
        "group",
        "target",
        "tiv",
    ]

    @classmethod
    def get_metric_columns(cls) -> Dict[str, str]:
        """Get the column names for each metric."""
        return {
            metric: "volume" if "vol" in metric else cls.DISTRIBUTION_METRIC
            for metric in cls.METRICS
        }
