import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier  # or any other classifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier

# fit the model (split)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import permutation_test_score

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_predict

AVAILABLE_MODELS = {
    "sgd": SGDClassifier(random_state=42, class_weight="balanced", max_iter=int(1e6)),
    "rf": RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
    "svm": SVC(random_state=42, class_weight="balanced", kernel="rbf", probability=True),
    "logreg": LogisticRegression(random_state=42, class_weight="balanced", max_iter=int(1e6)),
}

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

cv = GroupKFold(n_splits=5, shuffle=True, random_state=42)

ATLAS = "schaefer2018tian2020_400_7"
region_col = "index"
# Load important files
DATA_DIR = Path("/home/galkepler/Projects/athletes_brain/data")

# Output directory for figures
OUTPUT_DIR = Path("/home/galkepler/Projects/athletes_brain/figures/fig2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the data
parcels = pd.read_csv(DATA_DIR / "external" / "atlases" / ATLAS / "parcels.csv", index_col=0)
nifti = DATA_DIR / "external" / "atlases" / ATLAS / "atlas.nii.gz"
nifti_matlab = DATA_DIR / "external" / "atlases" / ATLAS / "atlas_matlab.nii"


# Assuming 'region_col' is defined globally (e.g., 'Region_Name' or 'Region_ID')
# Make sure to define it if not already:
# region_col = 'Region_ID' # Example, adapt to your actual region column name


# --- Adapted `long_to_wide` Function ---
def long_to_wide(
    long_df,
    *,
    index_cols=["subject_code", "session_id"],  # Use both for unique sessions
    columns_to_pivot=region_col,  # Your region column name
    values_col="value",
    demographic_cols=[
        "age_at_scan",
        "sex",
        "group",
        "target",
        "tiv",
    ],  # Include all relevant demographics/targets
):
    """
    Pivots a long metric table to a wide DataFrame (sessions x features),
    while preserving demographic and target columns.

    Parameters
    ----------
    long_df : pd.DataFrame
        Input DataFrame in long format with columns like index_cols, columns_to_pivot,
        values_col, and demographic_cols.
    index_cols : list of str
        Columns to use as the index for pivoting (e.g., ['Participant_ID', 'Session_ID']).
    columns_to_pivot : str
        Column containing the region names/IDs to become new columns.
    values_col : str
        Column containing the metric values.
    demographic_cols : list of str
        Other columns to preserve (demographics, target variables).

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with (Participant_ID, Session_ID) as a multi-index,
        region values as columns (prefixed with metric type), and
        demographic/target columns.
    """
    # Ensure all index_cols and demographic_cols are in the DataFrame
    if not all(col in long_df.columns for col in index_cols + [columns_to_pivot, values_col]):
        raise ValueError(
            f"Missing essential columns for pivoting. Required: {index_cols + [columns_to_pivot, values_col]}"
        )

    # Filter demographic_cols to only those actually present in the long_df
    present_demographic_cols = [col for col in demographic_cols if col in long_df.columns]

    # Handle multiple sessions per participant if Session_ID is not explicit
    if "session_id" not in long_df.columns:
        long_df["session_id"] = long_df.groupby("Participant_ID").cumcount() + 1
        print(
            "  Warning: 'Session_ID' not found in long_df. Generated dummy Session_ID for pivoting."
        )

    # Select columns to pivot and drop duplicates for the index/demographics
    # We drop duplicates to ensure only one set of demographic/target data per session
    meta_df = long_df[index_cols + present_demographic_cols].drop_duplicates(subset=index_cols)
    meta_df = meta_df.set_index(index_cols)

    # Pivot the metric data
    wide_metric = long_df.pivot_table(
        index=index_cols, columns=columns_to_pivot, values=values_col
    )

    # Merge demographics back. Ensure column names from wide_metric are unique.
    # The columns from wide_metric are already the region names, no prefix needed here yet.
    full_wide_df = meta_df.join(
        wide_metric, how="inner"
    )  # Inner join to ensure only sessions with metric data

    # Optional: Missing value thresholding
    # These thresholds might need careful consideration in a multimodal context
    # as dropping columns from one metric might affect others later.
    # For initial combined ML, it's often better to drop NaNs at the end on the full feature set.
    # You can re-enable if you want to filter per-metric-wise first.
    # thresh_cols = int(0.8 * len(full_wide_df))
    # full_wide_df = full_wide_df.dropna(axis=1, thresh=thresh_cols)
    # thresh_rows = int(0.8 * len(wide_metric.columns)) # Assuming parcels is wide_metric.columns
    # full_wide_df = full_wide_df.dropna(axis=0, thresh=thresh_rows)

    return full_wide_df.reset_index()


def init_preprocessor(X):
    return ColumnTransformer(
        [
            ("num", StandardScaler(), X.select_dtypes(include="number").columns),
            ("cat", OneHotEncoder(), X.select_dtypes(include="object").columns),
        ]
    )


if __name__ == "__main__":
    metrics = ["gm_vol", "wm_vol", "csf_vol", "adc", "fa", "ad", "rd"]
    distribution_metric = "qfmean"

    bad_subjects = ["IN120120"]

    # Load the data
    data = {}
    for metric in metrics:
        data[metric] = pd.read_csv(
            DATA_DIR / "processed" / f"{metric}.csv", index_col=0
        ).reset_index(drop=True)
        # data[metric] = data[metric].drop_duplicates(subset=["subject_code", region_col], keep="last")
        # drop problematic subjects
        data[metric] = data[metric][~data[metric]["subject_code"].isin(bad_subjects)]
        data[metric]["sex"] = data[metric]["sex"].map({"M": 0, "F": 1})

    # data["age_squared"] = data["age_at_scan"] ** 2

    metric_cols = {
        metric: "volume" if "vol" in metric else distribution_metric for metric in metrics
    }

    for m, df in data.items():
        df = df.rename(columns={metric_cols[m]: "value"})
        data[m] = df

    common_sessions = []
    data_wide = {}
    for metric, df in data.items():
        data_wide[metric] = long_to_wide(df)
        common_sessions.append(set(data_wide[metric]["session_id"].unique()))
    # Find common sessions across all metrics
    common_sessions = reduce(lambda x, y: x.intersection(y), common_sessions)
    # Filter each metric DataFrame to only include common sessions
    for metric in data_wide:
        data_wide[metric] = data_wide[metric][
            data_wide[metric]["session_id"].isin(common_sessions)
        ]

    N_PERMUTATIONS = 1000  # Number of permutations for significance testing
    MODEL = "sgd"
    GROUPS = [None, "climbing", "bjj"]
    GROUP_NAMES = {
        None: "all",
        "climbing": "climbing",
        "bjj": "bjj",
    }
    df_template = pd.DataFrame(
        columns=["scorer", "fold", "score", "permutation_score", "pvalue", "permutations"]
    )
    predictions_template = pd.DataFrame(columns=["true", "proba"])

    results = {g: {m: df_template.copy() for m in data_wide.keys()} for g in GROUP_NAMES.values()}
    predictions = {
        g: {m: predictions_template.copy() for m in data_wide.keys()} for g in GROUP_NAMES.values()
    }

    for GROUP, GROUP_NAME in GROUP_NAMES.items():
        for i, metric in enumerate(data_wide.keys()):
            CUR_DEST = OUTPUT_DIR / GROUP_NAME / metric
            CUR_DEST.mkdir(parents=True, exist_ok=True)
            print(f"\n--- Evaluating ({GROUP_NAME}) - {metric.upper()} ---")
            measure_df = df_template.copy()
            predictions_df = predictions_template.copy()
            m_data = data_wide[metric].copy()
            if GROUP is not None:
                # drop athletes not in the group
                m_data = m_data.drop(
                    m_data[
                        m_data["target"] & (m_data["group"].astype(str).str.lower() != GROUP)
                    ].index
                )
            X = m_data.drop(columns=["subject_code", "session_id", "group", "target"])
            X["sex"] = X["sex"].astype("str")
            y = m_data["target"]
            groups = m_data["subject_code"].astype("str")
            if "vol" in metric:
                # divide by TIV to normalize
                tiv = data_wide[metric]["tiv"]
                X[parcels[region_col]] = X[parcels[region_col]].div(tiv, axis=0)
            X.columns = X.columns.astype(
                str
            )  # Ensure all column names are strings for consistency

            print(f"X shape: {X.shape}, y shape: {y.shape}")
            print(y.value_counts(normalize=False))
            print(y.value_counts(normalize=True))

            preprocessor = init_preprocessor(X)
            # Define the pipeline
            pipe = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("imputer", SimpleImputer(strategy="mean")),
                    # ("oversampler", RandomOverSampler(random_state=42)),
                    ("classifier", AVAILABLE_MODELS[MODEL]),
                ]
            )

            # Define the grid search
            grid = GridSearchCV(
                pipe,
                AVAILABLE_PARAMS[MODEL],
                cv=cv,
                n_jobs=-1,
                scoring="average_precision",
                verbose=1,
            )

            # Fit the grid search
            grid.fit(X, y, groups=groups)

            for scorer in ["average_precision", "roc_auc", "f1"]:
                scores = cross_val_score(
                    grid.best_estimator_, X, y, cv=cv, scoring=scorer, groups=groups, n_jobs=-1
                )
                print(f"{scorer} mean: {np.mean(scores)}, std: {np.std(scores)}")
                splits = cv.split(X, y, groups=groups)
                perm_score, permutation_scores, pvalue = permutation_test_score(
                    grid.best_estimator_,
                    X,
                    y,
                    cv=splits,
                    scoring=scorer,
                    n_permutations=N_PERMUTATIONS,
                    n_jobs=-1,
                )
                # break
                for fold, score in enumerate(scores):
                    measure_df.loc[len(measure_df)] = [
                        scorer,
                        fold,
                        score,
                        perm_score,
                        pvalue,
                        permutation_scores,
                    ]
            results[GROUP_NAME][metric] = measure_df
            predictions_df["true"] = y
            predictions_df["proba"] = cross_val_predict(
                grid.best_estimator_, X, y, cv=cv, groups=groups, n_jobs=-1, method="predict_proba"
            )[:, 1]
            predictions[GROUP_NAME][metric] = predictions_df
            print(f"Saving results for {GROUP_NAME} - {metric}")
            results[GROUP_NAME][metric].to_csv(CUR_DEST / "results.csv")
            predictions[GROUP_NAME][metric].to_csv(CUR_DEST / "predictions.csv")
            # Save the best model
            best_model = grid.best_estimator_
            with open(CUR_DEST / "best_model.pkl", "wb") as f:
                import pickle

                pickle.dump(best_model, f)
            print(f"Best model saved for {GROUP_NAME} - {metric}")

    for GROUP, GROUP_NAME in GROUP_NAMES.items():
        print(f"\n--- Applying stacking for {GROUP_NAME} ---")
        stacked_models = parcels.copy()
        stacked_estimators = {roi: {} for roi in stacked_models[region_col]}
        predictions[GROUP_NAME]["base_stacked"] = {}

        alphas = np.logspace(-3, 4, 30)  # ridge grid

        # initiate CV
        cv = GroupKFold(n_splits=5, shuffle=True, random_state=42)

        tmp = pd.DataFrame(
            index=list(common_sessions),
            columns=["subject_code", "group", "target"] + list(data_wide.keys()),
        )
        for metric, df in data_wide.items():
            # get the values for the current parcel
            m_v = df.set_index("session_id")
            # drop duplicated index
            m_v = m_v[~m_v.index.duplicated(keep="first")]
            tmp[metric] = m_v[1]
            #     df.set_index("session_id")[roi].astype(float).drop_duplicates(keep="first")
            # )
            # add subject_code, session_id, group, target
            tmp[["subject_code", "age_at_scan", "sex", "group", "target"]] = m_v[
                ["subject_code", "age_at_scan", "sex", "group", "target"]
            ]
        if GROUP is not None:
            # drop athletes not in the group
            tmp = tmp.drop(
                tmp[tmp["target"] & (tmp["group"].astype(str).str.lower() != GROUP)].index
            )
        # get subjects from the common sessions
        subjects = tmp["subject_code"].astype(str)

        # fit cv to subjects to ensure that the groups are preserved across folds
        splits = cv.split(subjects)

        for i, row in parcels.iterrows():  # i == parcel index (0..453)
            learner_df = df_template.copy()
            # ------------- build design matrix for parcel i -----------------
            # X_roi : (n_subjects , 5 metrics)
            # X_cov = cov[cov_names["gm_vol"]].to_numpy()
            roi = row[region_col]  # e.g. 'V1'
            vals = pd.DataFrame(
                index=list(common_sessions),
                columns=["subject_code", "group", "target"] + list(data_wide.keys()),
            )
            for metric, df in data_wide.items():
                # get the values for the current parcel
                m_v = df.set_index("session_id")
                # drop duplicated index
                m_v = m_v[~m_v.index.duplicated(keep="first")]
                vals[metric] = m_v[roi]
                #     df.set_index("session_id")[roi].astype(float).drop_duplicates(keep="first")
                # )
                # add subject_code, session_id, group, target
                vals[["subject_code", "age_at_scan", "sex", "group", "target"]] = m_v[
                    ["subject_code", "age_at_scan", "sex", "group", "target"]
                ]

            # drop athletes not in the group
            if GROUP is not None:
                vals = vals.drop(
                    vals[vals["target"] & (vals["group"].astype(str).str.lower() != GROUP)].index
                )

            y = vals["target"]
            X_roi = vals.drop(columns=["subject_code", "group", "target"])

            print(f"Parcel {roi} - X shape: {X_roi.shape}, y shape: {y.shape}")
            print(y.value_counts(normalize=True))
            # ------------- fit the model -----------------
            preprocessor = init_preprocessor(X_roi)
            # Define the pipeline
            pipe = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("imputer", SimpleImputer(strategy="mean")),
                    # ("oversampler", RandomOverSampler(random_state=42)),
                    ("classifier", AVAILABLE_MODELS[MODEL]),
                ]
            )
            # Define the grid search
            grid = GridSearchCV(
                pipe,
                AVAILABLE_PARAMS[MODEL],
                cv=cv,
                n_jobs=-1,
                scoring="average_precision",
                verbose=1,
            )
            # Fit the grid search
            grid.fit(X_roi, y, groups=subjects)

            # Store the best estimator for this parcel
            stacked_estimators[roi]["estimator"] = grid.best_estimator_
            for scorer in ["average_precision", "roc_auc", "f1"]:
                scores = cross_val_score(
                    grid.best_estimator_,
                    X_roi,
                    y,
                    cv=cv,
                    scoring=scorer,
                    groups=subjects,
                    n_jobs=-1,
                )
                print(f"{scorer} mean: {np.mean(scores)}, std: {np.std(scores)}")
                splits = cv.split(X_roi, y, groups=subjects)
                perm_score, permutation_scores, pvalue = permutation_test_score(
                    grid.best_estimator_,
                    X_roi,
                    y,
                    cv=splits,
                    scoring=scorer,
                    n_permutations=N_PERMUTATIONS,
                    n_jobs=-1,
                )
                # break
                for fold, score in enumerate(scores):
                    learner_df.loc[len(learner_df)] = [
                        scorer,
                        fold,
                        score,
                        perm_score,
                        pvalue,
                        permutation_scores,
                    ]
                    stacked_models.loc[i, f"{scorer}-{fold}"] = score
            # Store the predictions for this parcel
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
            predictions[GROUP_NAME]["base_stacked"][roi] = predictions_df

        # Save the results for the parcel
        CUR_DEST = OUTPUT_DIR / GROUP_NAME / "stacked"
        CUR_DEST.mkdir(parents=True, exist_ok=True)
        stacked_models.to_csv(CUR_DEST / "results.csv")
        # save OOF predictions
        # predictions_df = pd.DataFrame(predictions["base_stacked"]).T
    for GROUP, GROUP_NAME in GROUP_NAMES.items():
        print(f"\n--- Fitting stacked model for {GROUP_NAME} ---")
        learner_df = df_template.copy()
        # use predictions of base stacked
        new_X = pd.DataFrame(
            {roi: predictions["base_stacked"][roi]["proba"] for roi in stacked_estimators.keys()},
            index=predictions["base_stacked"][list(stacked_estimators.keys())[0]].index,
        )
        # add sex, age, group, target
        new_X = new_X.join(tmp[["age_at_scan", "sex"]])

        new_X.columns = new_X.columns.astype(
            str
        )  # Ensure all column names are strings for consistency

        y = tmp["target"]

        # Fit a new model on the stacked predictions
        preprocessor = init_preprocessor(new_X)
        # Define the pipeline
        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("imputer", SimpleImputer(strategy="mean")),
                # ("oversampler", RandomOverSampler(random_state=42)),
                ("classifier", AVAILABLE_MODELS[MODEL]),
            ]
        )
        # Define the grid search
        grid = GridSearchCV(
            pipe,
            AVAILABLE_PARAMS[MODEL],
            cv=cv,
            n_jobs=-1,
            scoring="average_precision",
            verbose=1,
        )
        # Fit the grid search
        grid.fit(new_X, y, groups=subjects)

        # Store the best estimator for the stacked model
        stacked_estimators["stacked_model"] = grid.best_estimator_

        for scorer in ["average_precision", "roc_auc", "f1"]:
            scores = cross_val_score(
                grid.best_estimator_, new_X, y, cv=cv, scoring=scorer, groups=subjects, n_jobs=-1
            )
            print(f"{scorer} mean: {np.mean(scores)}, std: {np.std(scores)}")
            splits = cv.split(new_X, y, groups=subjects)
            perm_score, permutation_scores, pvalue = permutation_test_score(
                grid.best_estimator_,
                new_X,
                y,
                cv=splits,
                scoring=scorer,
                n_permutations=N_PERMUTATIONS,
                n_jobs=-1,
            )
            # break
            for fold, score in enumerate(scores):
                learner_df.loc[len(learner_df)] = [
                    scorer,
                    fold,
                    score,
                    perm_score,
                    pvalue,
                    permutation_scores,
                ]
        # Store the predictions for the stacked model
        predictions_df = pd.DataFrame(
            {
                "true": y,
                "proba": cross_val_predict(
                    grid.best_estimator_,
                    new_X,
                    y,
                    cv=cv,
                    groups=subjects,
                    n_jobs=-1,
                    method="predict_proba",
                )[:, 1],
            }
        )
        predictions[GROUP_NAME]["stacked"] = predictions_df
        # Save the results for the stacked model
        CUR_DEST = OUTPUT_DIR / GROUP_NAME / "stacked"
        CUR_DEST.mkdir(parents=True, exist_ok=True)
        learner_df.to_csv(CUR_DEST / "results.csv")
        # Save the best model
        with open(CUR_DEST / "best_model.pkl", "wb") as f:
            import pickle

            pickle.dump(grid.best_estimator_, f)
