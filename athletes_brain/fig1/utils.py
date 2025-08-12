"""Utility functions for Figure 1 analysis."""

from typing import Dict, Optional
import pandas as pd
from loguru import logger

from .config import Fig1Config
from .data_loader import AthletesBrainDataLoader
from .statistical_analysis import GroupComparison


def load_and_analyze_all(
    config: Optional[Fig1Config] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load data and perform all statistical comparisons.

    Args:
        config: Configuration object

    Returns:
        Dictionary containing all comparison results
    """
    config = config or Fig1Config()

    # Initialize components
    data_loader = AthletesBrainDataLoader(config)
    group_comparison = GroupComparison(config)

    # Load data
    metric_data, parcels, _, _ = data_loader.load_all_data()

    # Perform all comparisons
    results = {
        "athletes_vs_controls": group_comparison.athletes_vs_controls(metric_data, parcels),
        "climbers_vs_controls": group_comparison.sport_vs_controls(
            metric_data, parcels, config.CLIMBER_GROUP_LABEL
        ),
        "bjj_vs_controls": group_comparison.sport_vs_controls(
            metric_data, parcels, config.BJJ_GROUP_LABEL
        ),
        "climbers_vs_bjj": group_comparison.climbers_vs_bjj(metric_data, parcels),
    }

    logger.success("Completed all statistical comparisons")
    return results


def find_significant_regions(
    results: Dict[str, pd.DataFrame], p_threshold: float = 0.05, value_threshold: float = 0
) -> Dict[str, pd.DataFrame]:
    """Find significant regions across all metrics.

    Args:
        results: Dictionary of results DataFrames
        p_threshold: P-value threshold for significance
        value_threshold: Minimum absolute value threshold

    Returns:
        Dictionary of significant regions for each metric
    """
    significant_regions = {}

    for metric, df in results.items():
        significant = df[
            (df["adjusted_p_value"] < p_threshold) & (df["t_statistic"].abs() > value_threshold)
        ].sort_values("adjusted_p_value")

        significant_regions[metric] = significant
        logger.info(f"Found {len(significant)} significant regions for {metric}")

    return significant_regions


def compare_metrics_across_comparisons(
    all_results: Dict[str, Dict[str, pd.DataFrame]], metric: str = "gm_vol"
) -> pd.DataFrame:
    """Compare a specific metric across all comparison types.

    Args:
        all_results: Dictionary containing all comparison results
        metric: Metric to compare

    Returns:
        DataFrame comparing the metric across comparisons
    """
    comparison_data = []

    for comparison_name, results in all_results.items():
        if metric in results:
            df = results[metric].copy()
            df["comparison"] = comparison_name
            comparison_data.append(df)

    if comparison_data:
        combined_df = pd.concat(comparison_data, ignore_index=True)
        logger.info(f"Combined {metric} data across {len(comparison_data)} comparisons")
        return combined_df
    else:
        logger.warning(f"No data found for metric {metric}")
        return pd.DataFrame()


def summarize_results(results: Dict[str, pd.DataFrame], top_n: int = 5) -> pd.DataFrame:
    """Create a summary of the most significant results across metrics.

    Args:
        results: Dictionary of results DataFrames
        top_n: Number of top regions to include per metric

    Returns:
        Summary DataFrame
    """
    summary_data = []

    for metric, df in results.items():
        top_regions = df.nsmallest(top_n, "adjusted_p_value")
        for _, row in top_regions.iterrows():
            summary_data.append(
                {
                    "metric": metric,
                    "region": row.get("region_name", row[Fig1Config.REGION_COL]),
                    "hemisphere": row.get("hemisphere", ""),
                    "p_value": row["p_value"],
                    "adjusted_p_value": row["adjusted_p_value"],
                    "t_statistic": row["t_statistic"],
                    "effect_size": row.get("coefficient", ""),
                }
            )

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("adjusted_p_value")

    logger.info(f"Created summary with {len(summary_df)} entries")
    return summary_df
