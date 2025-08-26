from athletes_brain.fig2.config import Fig2Config
from athletes_brain.fig2.data_loader import Fig2DataLoader
from athletes_brain.fig2.preprocessing import long_to_wide
from athletes_brain.fig2.modeling import Fig2Modeling


def main(force_retrain: bool = True):
    """
    Main function to run the athlete's brain analysis pipeline (modular version).
    """
    config = Fig2Config()
    data_loader = Fig2DataLoader(config)
    modeling = Fig2Modeling(config)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    parcels = data_loader.load_parcels()
    raw_metric_data = data_loader.load_and_preprocess_metric_data()

    # 2. Convert to Wide Format and Find Common Sessions
    data_wide = {}
    for metric, df in raw_metric_data.items():
        data_wide[metric] = long_to_wide(
            df, columns_to_pivot=config.REGION_COL, demographic_cols=config.DEMOGRAPHIC_COLS
        )
    common_sessions = data_loader.find_common_sessions(data_wide)
    data_wide = data_loader.filter_by_common_sessions(data_wide, common_sessions)

    # Initialize dictionaries to store all results and predictions
    all_results = {g: {} for g in config.GROUP_NAMES.values()}
    all_predictions = {g: {} for g in config.GROUP_NAMES.values()}

    # 3. Train Base Models for Each Metric (Individual Analysis)
    for group, group_name in config.GROUP_NAMES.items():
        print(f"\n--- Running individual metric analysis for group: {group_name} ---")

        group_data_wide = {}
        for metric, df in data_wide.items():
            temp_df = df.copy()
            if group is not None:
                temp_df = temp_df.drop(
                    temp_df[
                        temp_df["target"].astype(bool)
                        & (temp_df["group"].astype(str).str.lower() != group)
                    ].index
                )
            # convert columns to str
            temp_df.columns = temp_df.columns.astype(str)
            group_data_wide[metric] = temp_df
        print(f"  Number of sessions in group '{group_name}': {group_data_wide[metric].shape[0]}")
        print(
            f"  Number of unique subjects in group '{group_name}': {group_data_wide[metric]['subject_code'].nunique()}"
        )
        print(f"force is set to {force_retrain}")

        metric_results, metric_predictions = modeling.train_base_models(
            group_data_wide, parcels, group_name, force_retrain=force_retrain
        )
        all_results[group_name]["individual_metrics"] = metric_results
        all_predictions[group_name]["individual_metrics"] = metric_predictions

    print("\n--- All analyses complete ---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run athlete's brain analysis pipeline.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining of models even if previous results exist.",
    )
    args = parser.parse_args()
    main(force_retrain=args.force)
