import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pathlib import Path
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    permutation_test_score,
    cross_val_predict,
)

from athletes_brain.fig2.config import (
    AVAILABLE_MODELS,
    AVAILABLE_PARAMS,
    cv,
    N_PERMUTATIONS,
    MODEL_NAME,
)
from athletes_brain.fig2.preprocessing import init_preprocessor
from athletes_brain.fig2.utils import (
    save_results,
    save_predictions,
    save_model,
    check_and_load_existing_artifact,
)
from athletes_brain.fig2.config import OUTPUT_DIR, REGION_COL


def train_and_evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_name: str,
    output_dir: Path,
    scorer_list: list = None,
    force_retrain: bool = False,
):
    """
    Trains, evaluates, and saves a machine learning model.
    If force_retrain is False and existing artifacts are found, they are loaded instead.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame.
    y : pd.Series
        Target Series.
    groups : pd.Series
        Group labels for cross-validation.
    model_name : str
        Name of the model to use (key in AVAILABLE_MODELS).
    output_dir : Path
        Directory to save results and model.
    scorer_list : list, optional
        List of scoring metrics to use. Defaults to ["average_precision", "roc_auc", "f1"].
    force_retrain : bool, optional
        If True, forces retraining even if results exist. Defaults to False.

    Returns
    -------
    tuple
        (measure_df, predictions_df, best_estimator)
    """
    if scorer_list is None:
        scorer_list = ["average_precision", "roc_auc", "f1"]

    results_path = output_dir / "results.csv"
    predictions_path = output_dir / "predictions.csv"
    model_path = output_dir / "best_model.pkl"

    if (
        not force_retrain
        and results_path.exists()
        and predictions_path.exists()
        and model_path.exists()
    ):
        print(f"Loading existing results, predictions, and model from {output_dir}")
        measure_df = pd.read_csv(results_path)
        predictions_df = pd.read_csv(predictions_path)
        best_estimator = check_and_load_existing_artifact(model_path, "model")
        return measure_df, predictions_df, best_estimator

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Target distribution (absolute):\n", y.value_counts(normalize=False))
    print("Target distribution (normalized):\n", y.value_counts(normalize=True))

    preprocessor = init_preprocessor(X)
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("imputer", SimpleImputer(strategy="mean")),
            ("classifier", AVAILABLE_MODELS[model_name]),
        ]
    )

    grid = GridSearchCV(
        pipe,
        AVAILABLE_PARAMS[model_name],
        cv=cv,
        n_jobs=-1,
        scoring="average_precision",
        verbose=1,
    )
    grid.fit(X, y, groups=groups)

    best_estimator = grid.best_estimator_

    measure_df_data = []
    for scorer in scorer_list:
        scores = cross_val_score(
            best_estimator, X, y, cv=cv, scoring=scorer, groups=groups, n_jobs=-1
        )
        print(f"{scorer} mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}")

        splits = cv.split(X, y, groups=groups)
        perm_score, permutation_scores, pvalue = permutation_test_score(
            best_estimator,
            X,
            y,
            cv=splits,
            scoring=scorer,
            n_permutations=N_PERMUTATIONS,
            n_jobs=-1,
        )

        for fold, score in enumerate(scores):
            measure_df_data.append(
                {
                    "scorer": scorer,
                    "fold": fold,
                    "score": score,
                    "permutation_score": perm_score,
                    "pvalue": pvalue,
                    "permutation_scores": permutation_scores,
                }
            )
    measure_df = pd.DataFrame(measure_df_data)

    predictions_df = pd.DataFrame(
        {
            "true": y,
            "proba": cross_val_predict(
                best_estimator, X, y, cv=cv, groups=groups, n_jobs=-1, method="predict_proba"
            )[:, 1],
        }
    )

    save_results(measure_df, output_dir, "results.csv")
    save_predictions(predictions_df, output_dir, "predictions.csv")
    save_model(best_estimator, output_dir, "best_model.pkl")
    # save X and y
    X.to_csv(output_dir / "X.csv", index=False)
    y.to_csv(output_dir / "y.csv", index=False)

    return measure_df, predictions_df, best_estimator


def train_base_models(data_wide, parcels, group_name, force_retrain: bool = False):
    """
    Trains and evaluates base models for each metric.

    Parameters
    ----------
    data_wide : dict
        Dictionary of wide-format DataFrames for each metric.
    parcels : pd.DataFrame
        DataFrame containing parcel information.
    group_name : str
        Name of the current group (e.g., 'all', 'climbing').
    force_retrain : bool, optional
        If True, forces retraining even if results exist. Defaults to False.

    Returns
    -------
    dict, dict
        results: Dictionary to store evaluation results for each metric.
        predictions: Dictionary to store predictions for each metric.
    """
    results = {}
    predictions = {}

    for metric in data_wide.keys():
        CUR_DEST = OUTPUT_DIR / group_name / metric
        CUR_DEST.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Evaluating ({group_name}) - {metric.upper()} ---")
        m_data = data_wide[metric].copy()

        # Handle group filtering if applicable
        if group_name != "all":
            m_data = m_data.drop(
                m_data[
                    m_data["target"] & (m_data["group"].astype(str).str.lower() != group_name)
                ].index
            )

        X = m_data.drop(columns=["subject_code", "session_id", "group", "target"])
        X["sex"] = X["sex"].astype("str")
        y = m_data["target"]
        groups = m_data["subject_code"].astype("str")

        if "vol" in metric:
            tiv = m_data["tiv"]  # Use tiv from the filtered m_data
            X[parcels[REGION_COL]] = X[parcels[REGION_COL]].div(tiv, axis=0)
        X.columns = X.columns.astype(str)

        metric_results, metric_predictions, _ = train_and_evaluate_model(
            X, y, groups, MODEL_NAME, CUR_DEST, force_retrain=force_retrain
        )
        results[metric] = metric_results
        predictions[metric] = metric_predictions
    return results, predictions


def train_stacked_base_models(
    common_sessions, data_wide, parcels, group_name, force_retrain: bool = False
):
    """
    Trains base models for each ROI, used as inputs for the final stacked model.

    Parameters
    ----------
    common_sessions : set
        Set of common session IDs.
    data_wide : dict
        Dictionary of wide-format DataFrames for each metric.
    parcels : pd.DataFrame
        DataFrame containing parcel information.
    group_name : str
        Name of the current group (e.g., 'all', 'climbing').
    force_retrain : bool, optional
        If True, forces retraining even if results exist. Defaults to False.

    Returns
    -------
    dict, pd.DataFrame, pd.DataFrame
        predictions_base_stacked: Dictionary storing OOF predictions for each ROI.
        stacked_models_results: DataFrame storing evaluation results for each ROI.
        common_data_template: DataFrame containing common demographic/target data.
    """
    CUR_DEST = OUTPUT_DIR / group_name / "stacked"
    CUR_DEST.mkdir(parents=True, exist_ok=True)

    roi_results_path = CUR_DEST / "roi_results.csv"
    # Predictions for stacked base models are stored per ROI, so check the first one as an indicator
    # or implement a more robust check for all ROI predictions

    # Create a template DataFrame for merging metric values and demographic data
    tmp_df_template = pd.DataFrame(
        index=list(common_sessions),
        columns=["subject_code", "age_at_scan", "sex", "group", "target"] + list(data_wide.keys()),
    )

    # Populate the template with demographic/target info from the first metric
    first_metric_df = data_wide[list(data_wide.keys())[0]].set_index("session_id")
    # drop duplicate indexes to avoid issues with multiple sessions
    first_metric_df = first_metric_df[~first_metric_df.index.duplicated(keep="first")]
    tmp_df_template[["subject_code", "age_at_scan", "sex", "group", "target"]] = (
        first_metric_df.loc[
            list(common_sessions), ["subject_code", "age_at_scan", "sex", "group", "target"]
        ]
    )

    # Filter subjects based on group for the overall stacked process
    if group_name != "all":
        tmp_df_template = tmp_df_template.drop(
            tmp_df_template[
                tmp_df_template["target"]
                & (tmp_df_template["group"].astype(str).str.lower() != group_name)
            ].index
        )
    subjects = tmp_df_template["subject_code"].astype(str)

    stacked_models_results = parcels.copy()
    predictions_base_stacked = {}

    # Check if all ROI models were previously trained and predictions saved
    all_roi_predictions_exist = True
    if not force_retrain:
        for i, row in parcels.iterrows():
            roi_dest = CUR_DEST / f"roi_{row[REGION_COL]}"
            roi_dest.mkdir(parents=True, exist_ok=True)
            roi = row[REGION_COL]
            roi_pred_path = roi_dest / "predictions.csv"
            if not roi_pred_path.exists():
                all_roi_predictions_exist = False
                break

    if not force_retrain and roi_results_path.exists() and all_roi_predictions_exist:
        print(f"Loading existing stacked base model results and predictions from {CUR_DEST}")
        stacked_models_results = pd.read_csv(roi_results_path)
        for i, row in parcels.iterrows():
            roi_dest = CUR_DEST / f"roi_{row[REGION_COL]}"
            roi = row[REGION_COL]
            predictions_base_stacked[roi] = pd.read_csv(roi_dest / "predictions.csv", index_col=0)
        return predictions_base_stacked, stacked_models_results, tmp_df_template

    # Initiate CV for subjects to ensure groups are preserved
    gkf_splits = list(cv.split(tmp_df_template, tmp_df_template["target"], groups=subjects))

    for i, row in parcels.iterrows():
        roi_dest = CUR_DEST / f"roi_{row[REGION_COL]}"
        roi_dest.mkdir(parents=True, exist_ok=True)
        roi = row[REGION_COL]
        print(f"\n--- Training stacked base model for Parcel {roi} ({group_name}) ---")

        vals_data = {}
        for metric, df in data_wide.items():
            m_v = df.set_index("session_id")
            m_v = m_v[~m_v.index.duplicated(keep="first")]
            vals_data[metric] = m_v[roi].loc[list(common_sessions)]

        X_roi = pd.DataFrame(vals_data)
        X_roi["age_at_scan"] = tmp_df_template["age_at_scan"]
        X_roi["sex"] = tmp_df_template["sex"]
        X_roi = X_roi.loc[tmp_df_template.index]

        y = tmp_df_template["target"]

        X_roi.columns = X_roi.columns.astype(str)

        preprocessor = init_preprocessor(X_roi)
        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("imputer", SimpleImputer(strategy="mean")),
                ("classifier", AVAILABLE_MODELS[MODEL_NAME]),
            ]
        )

        grid = GridSearchCV(
            pipe,
            AVAILABLE_PARAMS[MODEL_NAME],
            cv=cv,
            n_jobs=-1,
            scoring="average_precision",
            verbose=0,
        )
        grid.fit(X_roi, y, groups=subjects)

        best_estimator = grid.best_estimator_

        for scorer in ["average_precision", "roc_auc", "f1"]:
            scores = cross_val_score(
                best_estimator,
                X_roi,
                y,
                cv=cv,
                scoring=scorer,
                groups=subjects,
                n_jobs=-1,
            )
            print(
                f"  Parcel {roi} {scorer} mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}"
            )

            perm_splits = cv.split(X_roi, y, groups=subjects)
            perm_score, _, pvalue = permutation_test_score(
                best_estimator,
                X_roi,
                y,
                cv=perm_splits,
                scoring=scorer,
                n_permutations=N_PERMUTATIONS,
                n_jobs=-1,
            )
            for fold, score in enumerate(scores):
                stacked_models_results.loc[i, f"{scorer}-{fold}"] = score

        roi_predictions_df = pd.DataFrame(
            {
                "true": y,
                "proba": cross_val_predict(
                    best_estimator,
                    X_roi,
                    y,
                    cv=cv,
                    groups=subjects,
                    n_jobs=-1,
                    method="predict_proba",
                )[:, 1],
            },
            index=y.index,
        )
        predictions_base_stacked[roi] = roi_predictions_df
        # Save individual ROI predictions for easier loading later
        save_predictions(roi_predictions_df, roi_dest, "predictions.csv")
        # Save X and y for the ROI
        X_roi.to_csv(roi_dest / "X.csv", index=False)
        y.to_csv(roi_dest / "y.csv", index=False)

    return predictions_base_stacked, stacked_models_results, tmp_df_template


def train_final_stacked_model(
    predictions_base_stacked, common_data_template, group_name, force_retrain: bool = False
):
    """
    Trains the final stacked model using predictions from base ROI models.
    If force_retrain is False and existing artifacts are found, they are loaded instead.

    Parameters
    ----------
    predictions_base_stacked : dict
        Dictionary of OOF predictions from base models for each ROI.
    common_data_template : pd.DataFrame
        DataFrame containing common demographic/target data for all sessions/subjects.
    group_name : str
        Name of the current group (e.g., 'all', 'climbing').
    force_retrain : bool, optional
        If True, forces retraining even if results exist. Defaults to False.

    Returns
    -------
    pd.DataFrame, pd.DataFrame, object
        final_stacked_results: DataFrame containing evaluation results for the final stacked model.
        final_stacked_predictions: DataFrame containing predictions for the final stacked model.
        final_stacked_best_model: The trained best estimator for the stacked model.
    """
    print(f"\n--- Fitting final stacked model for {group_name} ---")

    CUR_DEST = OUTPUT_DIR / group_name / "stacked"
    # Ensure aligned_predictions is created regardless of force_retrain to construct new_X
    aligned_predictions = {}
    for roi, df in predictions_base_stacked.items():
        aligned_predictions[roi] = df.reindex(common_data_template.index)

    new_X = pd.DataFrame(
        {roi: aligned_predictions[roi]["proba"] for roi in aligned_predictions.keys()},
        index=common_data_template.index,
    )

    new_X = new_X.join(common_data_template[["age_at_scan", "sex"]])
    new_X["sex"] = new_X["sex"].astype(str)

    new_X.columns = new_X.columns.astype(str)
    y = common_data_template["target"]
    groups = common_data_template["subject_code"].astype(str)

    final_stacked_results, final_stacked_predictions, final_stacked_best_model = (
        train_and_evaluate_model(
            new_X, y, groups, MODEL_NAME, CUR_DEST, force_retrain=force_retrain
        )
    )
    return final_stacked_results, final_stacked_predictions, final_stacked_best_model
