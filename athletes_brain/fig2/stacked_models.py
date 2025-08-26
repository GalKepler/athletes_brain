"""
Stacked modeling for Figure 2, matching the exact logic of the Jupyter notebook.
This script will:
1. Train a model for each ROI (region) using all metrics as features, with permutation testing and OOF predictions.
2. Train a meta-model on the OOF predictions from all ROI models (plus demographics).
3. Save all results, predictions, and models to disk, matching the notebook's outputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    cross_val_predict,
    permutation_test_score,
    GroupKFold,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

from athletes_brain.fig2.config import Fig2Config
from athletes_brain.fig2.data_loader import Fig2DataLoader
from athletes_brain.fig2.preprocessing import long_to_wide
from athletes_brain.fig2.utils import save_results, save_predictions, save_model


def init_preprocessor(X):
    return ColumnTransformer(
        [
            ("num", StandardScaler(), X.select_dtypes(include="number").columns),
            ("cat", OneHotEncoder(), X.select_dtypes(include="object").columns),
        ]
    )


def run_stacked_modeling(
    group_name="all", base_models_name: str = "xgb", meta_model_name="sgd", force_retrain=True
):
    config = Fig2Config()
    data_loader = Fig2DataLoader(config)
    parcels = data_loader.load_parcels()
    raw_metric_data = data_loader.load_and_preprocess_metric_data()

    # Convert to wide format and filter for common sessions
    data_wide = {}
    for metric, df in raw_metric_data.items():
        data_wide[metric] = long_to_wide(
            df, columns_to_pivot=config.REGION_COL, demographic_cols=config.DEMOGRAPHIC_COLS
        )
    common_sessions = data_loader.find_common_sessions(data_wide)
    data_wide = data_loader.filter_by_common_sessions(data_wide, common_sessions)

    # Prepare template for stacking
    tmp = pd.DataFrame(
        index=list(common_sessions),
        columns=["subject_code", "age_at_scan", "sex", "group", "target"] + list(data_wide.keys()),
    )
    for metric, df in data_wide.items():

        m_v = df.set_index("session_id")
        # m_v = m_v[~m_v.index.duplicated(keep="first")]

        tmp[metric] = m_v[
            parcels[config.REGION_COL][0]
        ]  # Use first ROI as placeholder, will be replaced per ROI
        tmp[["subject_code", "age_at_scan", "sex", "group", "target"]] = m_v[
            ["subject_code", "age_at_scan", "sex", "group", "target"]
        ]
    if group_name != "all":
        tmp = tmp.drop(
            tmp[tmp["target"] & (tmp["group"].astype(str).str.lower() != group_name)].index
        )
    subjects = tmp["subject_code"].astype(str)
    cv = GroupKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=42)

    # --- Train per-ROI models ---
    stacked_models = parcels.copy()
    stacked_estimators = {roi: {} for roi in parcels[config.REGION_COL]}
    oof_preds = pd.DataFrame(index=tmp.index, columns=parcels[config.REGION_COL])
    results_dir = Path(config.OUTPUT_DIR) / group_name / "stacked"
    results_dir.mkdir(parents=True, exist_ok=True)

    for i, row in parcels.iterrows():
        roi = row[config.REGION_COL]
        vals = pd.DataFrame(
            index=list(common_sessions),
            columns=["subject_code", "age_at_scan", "sex", "group", "target"]
            + list(data_wide.keys()),
        )
        for metric, df in data_wide.items():
            m_v = df.set_index("session_id")
            m_v = m_v[~m_v.index.duplicated(keep="first")]
            vals[metric] = m_v[roi]
            vals[["subject_code", "age_at_scan", "sex", "group", "target"]] = m_v[
                ["subject_code", "age_at_scan", "sex", "group", "target"]
            ]
        if group_name != "all":
            vals = vals.drop(
                vals[vals["target"] & (vals["group"].astype(str).str.lower() != group_name)].index
            )
        y = vals["target"]
        X_roi = vals.drop(columns=["subject_code", "group", "target"])
        X_roi.columns = X_roi.columns.astype(str)
        preprocessor = init_preprocessor(X_roi)
        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("imputer", SimpleImputer(strategy="mean")),
                ("classifier", config.AVAILABLE_MODELS[base_models_name]),
            ]
        )
        grid = GridSearchCV(
            pipe,
            config.AVAILABLE_PARAMS[base_models_name],
            cv=cv,
            n_jobs=-1,
            scoring="average_precision",
            verbose=1,
        )
        grid.fit(X_roi, y, groups=subjects)
        stacked_estimators[roi]["estimator"] = grid.best_estimator_
        for scorer in ["average_precision", "roc_auc", "f1"]:
            scores = cross_val_score(
                grid.best_estimator_, X_roi, y, cv=cv, scoring=scorer, groups=subjects, n_jobs=-1
            )
            print(f"{scorer} mean: {np.mean(scores)}, std: {np.std(scores)}")
            splits = cv.split(X_roi, y, groups=subjects)
            perm_score, permutation_scores, pvalue = permutation_test_score(
                grid.best_estimator_,
                X_roi,
                y,
                cv=splits,
                scoring=scorer,
                n_permutations=config.N_PERMUTATIONS,
                n_jobs=-1,
            )
            for fold, score in enumerate(scores):
                stacked_models.loc[i, f"{scorer}-{fold}"] = score
        # Store OOF predictions
        predictions_df = pd.DataFrame(
            {
                "true": y,
                "proba": cross_val_predict(
                    grid.best_estimator_,
                    X_roi,
                    y,
                    cv=cv,
                    groups=subjects,
                    n_jobs=-1,
                    method="predict_proba",
                )[:, 1],
            }
        )
        oof_preds[roi] = predictions_df["proba"].values
        # Save per-ROI model
        roi_dir = results_dir / f"roi_{roi}"
        roi_dir.mkdir(parents=True, exist_ok=True)
        save_model(grid.best_estimator_, roi_dir, "best_model.pkl")
        save_predictions(predictions_df, roi_dir, "predictions.csv")
        # Optionally save X and y
        X_roi.to_csv(roi_dir / "X.csv", index=False)
        y.to_csv(roi_dir / "y.csv", index=False)

    # Save all per-ROI results
    stacked_models.to_csv(results_dir / "results.csv")
    oof_preds.to_csv(results_dir / "oof_preds.csv")

    # --- Train meta-model on OOF predictions ---
    meta_X = oof_preds.copy()
    meta_X["age_at_scan"] = tmp["age_at_scan"]
    meta_X["sex"] = tmp["sex"].astype(str)
    y_meta = tmp["target"]
    meta_X.columns = meta_X.columns.astype(str)
    meta_dir = results_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = init_preprocessor(meta_X)
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("imputer", SimpleImputer(strategy="mean")),
            ("classifier", config.AVAILABLE_MODELS[meta_model_name]),
        ]
    )
    grid = GridSearchCV(
        pipe,
        config.AVAILABLE_PARAMS[meta_model_name],
        cv=cv,
        n_jobs=-1,
        scoring="average_precision",
        verbose=1,
    )
    grid.fit(meta_X, y_meta, groups=subjects)
    meta_model = grid.best_estimator_
    # Save meta-model
    save_model(meta_model, meta_dir, "best_model.pkl")
    # Save meta-model predictions
    meta_predictions = pd.DataFrame(
        {
            "true": y_meta,
            "proba": cross_val_predict(
                meta_model,
                meta_X,
                y_meta,
                cv=cv,
                groups=subjects,
                n_jobs=-1,
                method="predict_proba",
            )[:, 1],
        }
    )
    save_predictions(meta_predictions, meta_dir, "predictions.csv")
    # Save meta-model X and y
    meta_X.to_csv(meta_dir / "X.csv", index=False)
    y_meta.to_csv(meta_dir / "y.csv", index=False)

    # Save meta-model CV and permutation results
    meta_results = []
    for scorer in ["average_precision", "roc_auc", "f1"]:
        scores = cross_val_score(
            meta_model, meta_X, y_meta, cv=cv, scoring=scorer, groups=subjects, n_jobs=-1
        )
        print(f"Meta {scorer} mean: {np.mean(scores)}, std: {np.std(scores)}")
        splits = cv.split(meta_X, y_meta, groups=subjects)
        perm_score, permutation_scores, pvalue = permutation_test_score(
            meta_model,
            meta_X,
            y_meta,
            cv=splits,
            scoring=scorer,
            n_permutations=config.N_PERMUTATIONS,
            n_jobs=-1,
        )
        for fold, score in enumerate(scores):
            meta_results.append(
                {
                    "scorer": scorer,
                    "fold": fold,
                    "score": score,
                    "permutation_score": perm_score,
                    "pvalue": pvalue,
                    "permutation_scores": permutation_scores,
                }
            )
    meta_results_df = pd.DataFrame(meta_results)
    meta_results_df.to_csv(meta_dir / "results.csv", index=False)


if __name__ == "__main__":
    run_stacked_modeling(
        group_name="all", base_models_name="sgd", meta_model_name="sgd", force_retrain=True
    )
