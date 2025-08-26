"""Modeling class for Figure 2: encapsulates model training, evaluation, and saving."""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_predict

from athletes_brain.fig2.config import Fig2Config
from athletes_brain.fig2.preprocessing import init_preprocessor, long_to_wide
from athletes_brain.fig2.utils import save_results, save_predictions, save_model


class StackedModeling:
    """Class for model training and evaluation for Figure 2."""

    def __init__(
        self,
        data: pd.DataFrame,
        parcels: pd.DataFrame,
        group_name: str,
        config: Optional[Fig2Config] = None,
        force_retrain: bool = False,
    ):
        self.config = config or Fig2Config()
        self.data = data
        self.parcels = parcels
        self.group_name = group_name
        self.force_retrain = force_retrain

    def extract_constants(self) -> pd.Series:
        """
        Get the constant columns for modeling.

        Returns
        -------
        pd.Series
            The target variable and grouping variable.
        """
        base_metric = self.config.METRICS[0]
        base_data = self.data[base_metric]
        covariates = base_data[self.config.DEMOGRAPHIC_COLS].copy()
        target = covariates["age_at_scan"]
        groups = base_data["subject_code"]
        return target, groups, covariates.drop(columns=["age_at_scan", "group", "target"])

    def train_roi_models(self, scorer_list=None):
        """
        Train a model for each ROI (region), storing OOF predictions and models.
        Returns:
            roi_models: dict of {roi: best_estimator}
            roi_oof_preds: DataFrame (index: sample, columns: roi, values: OOF proba)
            roi_results: dict of {roi: measure_df}
        """
        import numpy as np
        from sklearn.model_selection import cross_val_predict

        roi_models = {}
        roi_results = {}
        roi_oof_preds = pd.DataFrame(
            index=self.data[self.config.METRICS[0]].index,
            columns=self.parcels[self.config.REGION_COL].astype(str),
        )
        y, groups, covariates = self.extract_constants()

        for i, row in self.parcels.iterrows():
            roi = row[self.config.REGION_COL]
            roi_X = pd.DataFrame()
            for metric, X in self.data.items():
                X.columns = X.columns.astype(str)
                roi_X[metric] = X[str(roi)]
            roi_X = pd.concat([roi_X, covariates], axis=1)
            output_dir = Path(self.config.OUTPUT_DIR) / self.group_name / f"roi_{roi}"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Training model for ROI: {roi} in group: {self.group_name}")
            print(f"Output directory: {output_dir}")
            measure_df, predictions_df, best_estimator = self.train_base_models(
                roi_X,
                y,
                groups,
                "rf",
                output_dir,
                scorer_list,
                self.force_retrain,
            )
            roi_models[roi] = best_estimator
            roi_results[roi] = measure_df
            # OOF predictions for stacking
            roi_oof_preds[roi] = predictions_df["proba"].values
        return roi_models, roi_oof_preds, roi_results

    def train_meta_model(
        self,
        roi_oof_preds: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        covariates: pd.DataFrame,
        scorer_list=None,
    ):
        """
        Train a meta-model using OOF predictions from all ROI models as features.
        Returns:
            meta_measure_df, meta_predictions_df, meta_model
        """
        # Optionally add demographics
        meta_X = roi_oof_preds.copy()
        if "age_at_scan" in covariates.columns:
            meta_X["age_at_scan"] = covariates["age_at_scan"]
        if "sex" in covariates.columns:
            meta_X["sex"] = covariates["sex"].astype(str)
        if "tiv" in covariates.columns:
            meta_X["tiv"] = covariates["tiv"]
        output_dir = Path(self.config.OUTPUT_DIR) / self.group_name / "meta_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        return self.train_base_models(
            meta_X, y, groups, self.config.MODEL_NAME, output_dir, scorer_list, self.force_retrain
        )

    def fit(self, scorer_list=None):
        """
        Run the full stacking pipeline: train ROI models, then train meta-model on OOF predictions.
        Returns:
            {
                'roi_models': ...,
                'roi_oof_preds': ...,
                'roi_results': ...,
                'meta_results': ...,
                'meta_predictions': ...,
                'meta_model': ...
            }
        """
        y, groups, covariates = self.extract_constants()
        roi_models, roi_oof_preds, roi_results = self.train_roi_models(scorer_list)
        meta_results, meta_predictions, meta_model = self.train_meta_model(
            roi_oof_preds, y, groups, covariates, scorer_list
        )
        return {
            "roi_models": roi_models,
            "roi_oof_preds": roi_oof_preds,
            "roi_results": roi_results,
            "meta_results": meta_results,
            "meta_predictions": meta_predictions,
            "meta_model": meta_model,
        }

    def train_base_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        model_name: str,
        output_dir: Path,
        scorer_list: Optional[list] = None,
        force_retrain: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
        """
        Trains, evaluates, and saves a machine learning model.
        If force_retrain is False and existing artifacts are found, they are loaded instead.
        Returns (measure_df, predictions_df, best_estimator)
        Includes permutation test results for each scorer, as in the original model_training.py.
        """
        import numpy as np
        from sklearn.model_selection import cross_val_score, permutation_test_score

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
            measure_df = pd.read_csv(results_path)
            predictions_df = pd.read_csv(predictions_path)
            with open(model_path, "rb") as f:
                import pickle

                best_estimator = pickle.load(f)
            return measure_df, predictions_df, best_estimator

        X.columns = X.columns.astype(str)  # Ensure columns are str for consistency
        preprocessor = init_preprocessor(X)
        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("imputer", SimpleImputer(strategy="mean")),
                ("classifier", self.config.AVAILABLE_MODELS[model_name]),
            ]
        )
        print("Pipeline created successfully.")

        grid = GridSearchCV(
            pipe,
            self.config.AVAILABLE_PARAMS[model_name],
            cv=self.config.cv,
            n_jobs=-1,
            scoring="average_precision",
            verbose=1,
        )
        grid.fit(X, y, groups=groups)
        best_estimator = grid.best_estimator_

        measure_df_data = []
        for scorer in scorer_list:
            scores = cross_val_score(
                best_estimator, X, y, cv=self.config.cv, groups=groups, scoring=scorer, n_jobs=-1
            )
            print(f"{scorer} mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}")

            splits = self.config.cv.split(X, y, groups=groups)
            perm_score, permutation_scores, pvalue = permutation_test_score(
                best_estimator,
                X,
                y,
                cv=splits,
                scoring=scorer,
                n_permutations=self.config.N_PERMUTATIONS,
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
                    best_estimator,
                    X,
                    y,
                    cv=self.config.cv,
                    groups=groups,
                    n_jobs=-1,
                    method="predict_proba",
                )[:, 1],
            }
        )

        save_results(measure_df, output_dir, "results.csv")
        save_predictions(predictions_df, output_dir, "predictions.csv")
        save_model(best_estimator, output_dir, "best_model.pkl")
        X.to_csv(output_dir / "X.csv", index=False)
        y.to_csv(output_dir / "y.csv", index=False)

        return measure_df, predictions_df, best_estimator


if __name__ == "__main__":
    # Example usage for running the full stacking pipeline
    from athletes_brain.fig2.data_loader import Fig2DataLoader
    from athletes_brain.fig2.config import Fig2Config

    # Initialize config and data loader
    config = Fig2Config()
    data_loader = Fig2DataLoader(config)

    # Load data (replace with your actual data loading logic as needed)
    parcels = data_loader.load_parcels()
    raw_metric_data = data_loader.load_and_preprocess_metric_data()

    data_wide = {}
    for metric, df in raw_metric_data.items():
        data_wide[metric] = long_to_wide(
            df, columns_to_pivot=config.REGION_COL, demographic_cols=config.DEMOGRAPHIC_COLS
        )
    common_sessions = data_loader.find_common_sessions(data_wide)
    data_wide = data_loader.filter_by_common_sessions(data_wide, common_sessions)

    group_name = "all"  # or "climbing", "bjj", etc.
    stacking = StackedModeling(
        data=data_wide,
        parcels=parcels,
        group_name=group_name,
        config=config,
        force_retrain=True,
    )

    # Run the full stacking pipeline
    stacking_results = stacking.fit()

    # Access results:
    # stacking_results['roi_models']        # dict of per-ROI models
    # stacking_results['roi_oof_preds']     # DataFrame of OOF predictions per ROI
    # stacking_results['roi_results']       # dict of per-ROI results DataFrames
    # stacking_results['meta_results']      # meta-model results DataFrame
    # stacking_results['meta_predictions']  # meta-model predictions DataFrame
    # stacking_results['meta_model']        # trained meta-model object
