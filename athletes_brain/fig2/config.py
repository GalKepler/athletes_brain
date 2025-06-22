from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GroupKFold

# Model Definitions
AVAILABLE_MODELS = {
    "sgd": SGDClassifier(random_state=42, class_weight="balanced", max_iter=int(1e6)),
    "rf": RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
    "svm": SVC(random_state=42, class_weight="balanced", kernel="rbf", probability=True),
    "logreg": LogisticRegression(random_state=42, class_weight="balanced", max_iter=int(1e6)),
}

# Hyperparameter Grids
AVAILABLE_PARAMS = {
    "sgd": {
        "classifier__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__loss": ["modified_huber", "log_loss"],
        "classifier__penalty": ["l1", "l2", "elasticnet"],
        "classifier__l1_ratio": [0.15, 0.5, 0.85],
    },
    "rf": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [5, 50, 100, None],
        "classifier__max_features": ["auto", "sqrt"],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__bootstrap": [True, False],
    },
    "svm": {
        "classifier__C": [0.1, 1, 10, 100, 1000],
        "classifier__kernel": ["linear", "rbf", "poly"],
        "classifier__gamma": ["scale", "auto"],
        "classifier__degree": [2, 3, 4],  # Only relevant for 'poly' kernel
    },
    "logreg": {
        "classifier__C": [0.1, 1, 10, 100, 1000],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["liblinear"],  # 'liblinear' supports 'l1' and 'l2' penalties
    },
}

# Cross-validation
CV_SPLITS = 5
cv = GroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)

# Atlas and Region Column
ATLAS = "schaefer2018tian2020_400_7"
REGION_COL = "index"

# Data and Output Directories
DATA_DIR = Path("/home/galkepler/Projects/athletes_brain/data")
OUTPUT_DIR = Path("/home/galkepler/Projects/athletes_brain/figures/fig2")

# Experiment Parameters
METRICS = ["gm_vol", "wm_vol", "csf_vol", "adc", "fa", "ad", "rd"]
DISTRIBUTION_METRIC = "qfmean"
BAD_SUBJECTS = ["IN120120"]
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
