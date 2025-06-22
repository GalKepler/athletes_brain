import pandas as pd
import numpy as np
from pathlib import Path
from functools import reduce

from athletes_brain.fig2.config import (
    OUTPUT_DIR,
    GROUP_NAMES,
    REGION_COL,
    METRICS,
    MODEL_NAME,
    DEMOGRAPHIC_COLS,
)
from athletes_brain.fig2.data_loader import (
    load_parcels,
    load_and_preprocess_metric_data,
    find_common_sessions,
    filter_by_common_sessions,
)
from athletes_brain.fig2.preprocessing import long_to_wide
from athletes_brain.fig2.model_training import (
    train_base_models,
    train_stacked_base_models,
    train_final_stacked_model,
)
from athletes_brain.fig2.utils import save_results, save_predictions, save_model


FORCE = False


def main(force_retrain: bool = False):
    """
    Main function to run the athlete's brain analysis pipeline.

    Parameters
    ----------
    force_retrain : bool, optional
        If True, forces retraining of models even if previous results exist.
        If False, attempts to load existing results/models to save time.
        Defaults to False.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    parcels = load_parcels()
    raw_metric_data = load_and_preprocess_metric_data()

    # 2. Convert to Wide Format and Find Common Sessions
    data_wide = {}
    for metric, df in raw_metric_data.items():
        data_wide[metric] = long_to_wide(
            df, columns_to_pivot=REGION_COL, demographic_cols=DEMOGRAPHIC_COLS
        )
    common_sessions = find_common_sessions(data_wide)
    data_wide = filter_by_common_sessions(data_wide, common_sessions)

    # Initialize dictionaries to store all results and predictions
    all_results = {g: {} for g in GROUP_NAMES.values()}
    all_predictions = {g: {} for g in GROUP_NAMES.values()}

    # 3. Train Base Models for Each Metric (Individual Analysis)
    for group, group_name in GROUP_NAMES.items():
        print(f"\n--- Running individual metric analysis for group: {group_name} ---")

        group_data_wide = {}
        for metric, df in data_wide.items():
            temp_df = df.copy()
            if group is not None:
                temp_df = temp_df.drop(
                    temp_df[
                        temp_df["target"] & (temp_df["group"].astype(str).str.lower() != group)
                    ].index
                )
            group_data_wide[metric] = temp_df

        metric_results, metric_predictions = train_base_models(
            group_data_wide, parcels, group_name, force_retrain
        )
        all_results[group_name]["individual_metrics"] = metric_results
        all_predictions[group_name]["individual_metrics"] = metric_predictions

    # 4. Train Stacked Base Models (ROI-wise)
    for group, group_name in GROUP_NAMES.items():
        print(f"\n--- Running stacked base model training for group: {group_name} ---")
        predictions_base_stacked, stacked_models_results, common_data_template = (
            train_stacked_base_models(
                common_sessions, data_wide, parcels, group_name, force_retrain
            )
        )

        all_results[group_name]["stacked_base_models"] = stacked_models_results
        all_predictions[group_name]["base_stacked_predictions"] = predictions_base_stacked

        CUR_DEST = OUTPUT_DIR / group_name / "stacked"
        CUR_DEST.mkdir(parents=True, exist_ok=True)
        # Always save ROI-wise results if they were generated or loaded
        save_results(stacked_models_results, CUR_DEST, "roi_results.csv")

        # 5. Train Final Stacked Model
        final_stacked_results, final_stacked_predictions, final_stacked_best_model = (
            train_final_stacked_model(
                predictions_base_stacked, common_data_template, group_name, force_retrain
            )
        )
        all_results[group_name]["final_stacked_model"] = final_stacked_results
        all_predictions[group_name]["final_stacked_predictions"] = final_stacked_predictions

    print("\n--- All analyses complete ---")


if __name__ == "__main__":
    # To use the force option, run: python main.py --force
    import argparse

    parser = argparse.ArgumentParser(description="Run athlete's brain analysis pipeline.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining of models even if previous results exist.",
    )
    args = parser.parse_args()
    main(force_retrain=args.force)
