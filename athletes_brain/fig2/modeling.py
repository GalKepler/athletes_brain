"""Modeling class for Figure 2: encapsulates model training, evaluation, and saving."""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_predict

from athletes_brain.fig2.config import Fig2Config
from athletes_brain.fig2.preprocessing import init_preprocessor
from athletes_brain.fig2.utils import save_results, save_predictions, save_model


class Fig2Modeling:
    """Class for model training and evaluation for Figure 2."""

    def __init__(self, config: Optional[Fig2Config] = None):
        self.config = config or Fig2Config()

    def train_and_evaluate_model(
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

    def train_base_models(
        self,
        data_wide: Dict[str, pd.DataFrame],
        parcels: pd.DataFrame,
        group_name: str,
        force_retrain: bool = False,
    ) -> Tuple[Dict, Dict]:
        """Trains and evaluates base models for each metric."""
        results = {}
        predictions = {}
        for metric, df in data_wide.items():
            X = df.drop(columns=["target", "group", "session_id", "subject_code"])
            y = df["target"]
            print(y.value_counts())
            print(X.columns)
            groups = df["subject_code"] if "subject_code" in df.columns else None
            output_dir = self.config.OUTPUT_DIR / group_name / metric
            output_dir.mkdir(parents=True, exist_ok=True)
            measure_df, predictions_df, _ = self.train_and_evaluate_model(
                X, y, groups, self.config.MODEL_NAME, output_dir, force_retrain=force_retrain
            )
            results[metric] = measure_df
            predictions[metric] = predictions_df
        return results, predictions
