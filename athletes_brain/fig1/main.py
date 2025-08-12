"""Main module for generating Figure 1 of the athletes brain study."""

from pathlib import Path
from typing import Optional
from loguru import logger

from .config import Fig1Config
from .data_loader import AthletesBrainDataLoader
from .statistical_analysis import GroupComparison
from .brain_visualization import BrainPlotter


def generate_figure1(
    output_dir: Optional[Path] = None, config: Optional[Fig1Config] = None
) -> None:
    """Generate complete Figure 1 for the athletes brain study.

    This function orchestrates the entire Figure 1 generation process:
    1. Sets up visualization configuration
    2. Loads and preprocesses data
    3. Performs statistical comparisons
    4. Creates brain visualizations

    Args:
        output_dir: Directory to save figures. If None, uses default from config.
        config: Configuration object. If None, uses default Fig1Config.
    """
    # Initialize configuration and components
    config = config or Fig1Config()
    config.setup_matplotlib_config()

    logger.info("Starting Figure 1 generation")

    # Initialize components
    data_loader = AthletesBrainDataLoader(config)
    group_comparison = GroupComparison(config)
    brain_plotter = BrainPlotter(config)

    # Set up output directory
    if output_dir is None:
        output_dir = data_loader.get_output_directory("fig1")

    logger.info(f"Output directory: {output_dir}")

    # Load data
    logger.info("Loading data...")
    metric_data, parcels, nifti, nifti_matlab = data_loader.load_all_data()

    # Perform statistical comparisons
    logger.info("Performing statistical comparisons...")

    # Athletes vs Controls
    athletes_vs_controls_results = group_comparison.athletes_vs_controls(metric_data, parcels)

    # Individual sports vs Controls
    climbers_vs_controls_results = group_comparison.sport_vs_controls(
        metric_data, parcels, config.CLIMBER_GROUP_LABEL
    )
    bjj_vs_controls_results = group_comparison.sport_vs_controls(
        metric_data, parcels, config.BJJ_GROUP_LABEL
    )

    # Climbers vs BJJ
    climbers_vs_bjj_results = group_comparison.climbers_vs_bjj(metric_data, parcels)

    # Create brain visualizations
    logger.info("Creating brain visualizations...")

    # Athletes vs Controls
    brain_plotter.create_brain_plots(
        athletes_vs_controls_results, "athletes_vs_controls", output_dir
    )

    # Climbers vs Controls
    brain_plotter.create_brain_plots(
        climbers_vs_controls_results, "climbers_vs_controls", output_dir
    )

    # BJJ vs Controls
    brain_plotter.create_brain_plots(bjj_vs_controls_results, "bjj_vs_controls", output_dir)

    # Climbers vs BJJ
    brain_plotter.create_brain_plots(climbers_vs_bjj_results, "climbers_vs_bjj", output_dir)

    logger.success(f"Figure 1 generation completed! Results saved to {output_dir}")


def analyze_specific_comparison(
    comparison_type: str,
    metric: str = "adc",
    output_dir: Optional[Path] = None,
    config: Optional[Fig1Config] = None,
) -> None:
    """Analyze a specific comparison and metric combination.

    Args:
        comparison_type: Type of comparison ('athletes_vs_controls', 'climbers_vs_controls',
                        'bjj_vs_controls', 'climbers_vs_bjj')
        metric: Specific metric to analyze (default: 'adc')
        output_dir: Directory to save figures
        config: Configuration object
    """
    config = config or Fig1Config()
    config.setup_matplotlib_config()

    logger.info(f"Analyzing {comparison_type} for {metric}")

    # Initialize components
    data_loader = AthletesBrainDataLoader(config)
    group_comparison = GroupComparison(config)
    brain_plotter = BrainPlotter(config)

    # Set up output directory
    if output_dir is None:
        output_dir = data_loader.get_output_directory("fig1")

    # Load data
    metric_data, parcels, _, _ = data_loader.load_all_data()

    # Perform specific comparison
    if comparison_type == "athletes_vs_controls":
        results = group_comparison.athletes_vs_controls(metric_data, parcels)
    elif comparison_type == "climbers_vs_controls":
        results = group_comparison.sport_vs_controls(
            metric_data, parcels, config.CLIMBER_GROUP_LABEL
        )
    elif comparison_type == "bjj_vs_controls":
        results = group_comparison.sport_vs_controls(metric_data, parcels, config.BJJ_GROUP_LABEL)
    elif comparison_type == "climbers_vs_bjj":
        results = group_comparison.climbers_vs_bjj(metric_data, parcels)
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type}")

    # Create visualization for specific metric
    if metric in results:
        fig_dest = output_dir / comparison_type / metric
        fig_dest.mkdir(parents=True, exist_ok=True)

        brain_plotter.plot_single_metric(results[metric], metric, fig_dest)

        logger.success(f"Analysis completed for {comparison_type} - {metric}")
    else:
        logger.error(f"Metric {metric} not found in results")


def get_most_significant_regions(
    comparison_type: str,
    metric: str = "gm_vol",
    n_regions: int = 10,
    config: Optional[Fig1Config] = None,
) -> None:
    """Get and display the most significant regions for a comparison.

    Args:
        comparison_type: Type of comparison
        metric: Specific metric to analyze
        n_regions: Number of top regions to display
        config: Configuration object
    """
    config = config or Fig1Config()

    # Initialize components
    data_loader = AthletesBrainDataLoader(config)
    group_comparison = GroupComparison(config)

    # Load data
    metric_data, parcels, _, _ = data_loader.load_all_data()

    # Perform comparison
    if comparison_type == "athletes_vs_controls":
        results = group_comparison.athletes_vs_controls(metric_data, parcels)
    elif comparison_type == "climbers_vs_controls":
        results = group_comparison.sport_vs_controls(
            metric_data, parcels, config.CLIMBER_GROUP_LABEL
        )
    elif comparison_type == "bjj_vs_controls":
        results = group_comparison.sport_vs_controls(metric_data, parcels, config.BJJ_GROUP_LABEL)
    elif comparison_type == "climbers_vs_bjj":
        results = group_comparison.climbers_vs_bjj(metric_data, parcels)
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type}")

    if metric in results:
        most_significant = results[metric].sort_values(by="adjusted_p_value").head(n_regions)

        # logger.info(f"Most significant regions for {comparison_type} - {metric}:")
        # print(most_significant[["region_name", "adjusted_p_value", "t_statistic", "p_value"]])

        return most_significant
    else:
        logger.error(f"Metric {metric} not found in results")
