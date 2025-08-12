"""Brain visualization module for athletes brain study Figure 1."""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from surfplot import Plot
from neuromaps.datasets import fetch_fslr
from brainspace.datasets import load_parcellation
from subcortex_visualization.plotting import plot_subcortical_data
from loguru import logger

from .config import Fig1Config, savefig_nice


class BrainPlotter:
    """Class for creating brain visualizations."""

    def __init__(self, config: Optional[Fig1Config] = None):
        """Initialize the brain plotter.

        Args:
            config: Configuration object. If None, uses default Fig1Config.
        """
        self.config = config or Fig1Config()
        self.surfaces = None
        self.parcellations = None

    def setup_surfaces(self) -> None:
        """Set up brain surfaces and parcellations."""
        logger.info("Setting up brain surfaces and parcellations")

        # Fetch standard surfaces
        self.surfaces = fetch_fslr()

        # Load parcellation
        lh_parc, rh_parc = load_parcellation("schaefer")
        self.parcellations = {"lh": lh_parc, "rh": rh_parc}

        logger.success("Brain surfaces and parcellations loaded successfully")

    def prepare_value_maps(
        self,
        results_df: pd.DataFrame,
        p_column: str = "adjusted_p_value",
        value_column: str = "t_statistic",
    ) -> Tuple[Dict, Dict, pd.DataFrame]:
        """Prepare value maps for brain visualization.

        Args:
            results_df: DataFrame containing statistical results
            p_column: Column name for p-values
            value_column: Column name for values to visualize

        Returns:
            Tuple of (left hemisphere map, right hemisphere map, subcortical DataFrame)
        """
        logger.info(f"Preparing value maps for {value_column}")

        value_map_lh = {}
        value_map_rh = {}
        value_map_subcortex = {"region": [], "value": [], "Hemisphere": []}

        for i, row in results_df.iterrows():
            label = row[self.config.REGION_COL]
            hemi_row = row["hemisphere"]

            # Apply thresholds
            if (
                row[p_column] < self.config.P_THRESHOLD
                and np.abs(row[value_column]) > self.config.VALUE_THRESHOLD
            ):
                value = row[value_column]
            else:
                value = np.nan

            # Assign to appropriate hemisphere or subcortex
            if "schaefer" in self.config.ATLAS:
                atlas_size = int(self.config.ATLAS.split("_")[1])
                if label > atlas_size:
                    # Subcortical region
                    value_map_subcortex["region"].append(label)
                    value_map_subcortex["value"].append(value)
                    value_map_subcortex["Hemisphere"].append(row["hemisphere"])
                else:
                    # Cortical region
                    if hemi_row == "L":
                        value_map_lh[label] = value
                    elif hemi_row == "R":
                        value_map_rh[label] = value

        # Create subcortical DataFrame
        subcort_df = pd.DataFrame(value_map_subcortex)
        if "schaefer" in self.config.ATLAS:
            atlas_size = int(self.config.ATLAS.split("_")[1])
            subcort_df["region"] = subcort_df["region"] - atlas_size

        logger.info(
            f"Prepared maps with {len(value_map_lh)} LH, {len(value_map_rh)} RH, "
            f"and {len(subcort_df)} subcortical regions"
        )

        return value_map_lh, value_map_rh, subcort_df

    def plot_cortical_surface(
        self,
        value_map_lh: Dict,
        value_map_rh: Dict,
        output_dir: Path,
        metric_key: str,
        value_column: str = "t_statistic",
    ) -> None:
        """Plot cortical surface maps.

        Args:
            value_map_lh: Left hemisphere value mapping
            value_map_rh: Right hemisphere value mapping
            output_dir: Directory to save figures
            metric_key: Key identifying the metric being visualized
            value_column: Name of the value column for colorbar label
        """
        logger.info("Creating cortical surface plots")

        if self.parcellations is None:
            raise ValueError("Surfaces not set up. Call setup_surfaces() first.")

        lh_parc, rh_parc = self.parcellations["lh"], self.parcellations["rh"]
        lh_surf, rh_surf = self.surfaces["veryinflated"]

        # Vectorize mapping for hemispheres
        vec_lh = np.vectorize(lambda x: value_map_lh.get(x, np.nan))
        data_lh_mapped = vec_lh(lh_parc)

        vec_rh = np.vectorize(lambda x: value_map_rh.get(x, np.nan))
        data_rh_mapped = vec_rh(rh_parc)

        # Plot each hemisphere
        hemisphere_data = [
            ("left", lh_surf, data_lh_mapped, lh_parc, ["medial", "lateral"]),
            ("right", rh_surf, data_rh_mapped, rh_parc, ["lateral", "medial"]),
        ]

        for hemi_key, hemi_surf_data, mapped_data, parcellation_data, views in hemisphere_data:
            logger.info(f"Plotting {hemi_key} hemisphere")

            # Set up surface plot
            surf_dict = {
                "surf_lh": hemi_surf_data if hemi_key == "left" else None,
                "surf_rh": hemi_surf_data if hemi_key == "right" else None,
            }

            p = Plot(
                **surf_dict,
                views=views,
                size=(800, 300),
                zoom=1.6,
                layout="row",
                mirror_views=False,
            )

            # Add data layer
            p.add_layer(
                {hemi_key: mapped_data},
                cmap="coolwarm",
                color_range=(self.config.VMIN, self.config.VMAX),
                cbar_label=value_column,
                cbar=True,
            )

            # Add outline layer
            p.add_layer({hemi_key: parcellation_data}, cmap="gray", as_outline=True, cbar=False)

            # Build and save figure
            fig = p.build()
            out_fname = output_dir / f"{hemi_key}_{metric_key}.png"
            savefig_nice(fig, out_fname, tight=True, dpi=300)

            logger.info(f"Saved {hemi_key} hemisphere plot to {out_fname}")

    def plot_subcortical(
        self, subcort_df: pd.DataFrame, output_dir: Path, metric_key: str
    ) -> None:
        """Plot subcortical regions.

        Args:
            subcort_df: DataFrame containing subcortical data
            output_dir: Directory to save figures
            metric_key: Key identifying the metric being visualized
        """
        if subcort_df.empty:
            logger.info("No subcortical data to plot")
            return

        logger.info("Creating subcortical plots")

        for hemi in ["L", "R"]:
            logger.info(f"Plotting {hemi} subcortical regions")

            fig = plot_subcortical_data(
                subcort_df,
                atlas="Melbourne_S3",
                show_legend=True,
                hemisphere=hemi,
                cmap="coolwarm",
                line_color="black",
                line_thickness=2,
                vmin=self.config.VMIN,
                vmax=self.config.VMAX,
                show_figure=False,
            )

            out_fname = output_dir / f"subcort_{metric_key}_{hemi}.png"
            savefig_nice(fig, out_fname, dpi=400)
            logger.info(f"Saved {hemi} subcortical plot to {out_fname}")

    def create_brain_plots(
        self,
        results_dict: Dict[str, pd.DataFrame],
        comparison_name: str,
        output_base_dir: Path,
        p_column: str = "adjusted_p_value",
        value_column: str = "t_statistic",
    ) -> None:
        """Create complete brain plots for a comparison.

        Args:
            results_dict: Dictionary mapping metric names to results DataFrames
            comparison_name: Name of the comparison (e.g., 'athletes_vs_naive')
            output_base_dir: Base directory for saving figures
            p_column: Column name for p-values
            value_column: Column name for values to visualize
        """
        logger.info(f"Creating brain plots for {comparison_name}")

        if self.surfaces is None:
            self.setup_surfaces()

        for metric_key, results_df in results_dict.items():
            logger.info(f"Processing {metric_key} for {comparison_name}")

            # Create output directory
            fig_dest = output_base_dir / comparison_name / metric_key
            fig_dest.mkdir(parents=True, exist_ok=True)

            # Prepare value maps
            value_map_lh, value_map_rh, subcort_df = self.prepare_value_maps(
                results_df, p_column, value_column
            )

            # Plot cortical surfaces
            self.plot_cortical_surface(
                value_map_lh, value_map_rh, fig_dest, metric_key, value_column
            )

            # Plot subcortical regions
            self.plot_subcortical(subcort_df, fig_dest, metric_key)

        logger.success(f"Completed brain plots for {comparison_name}")

    def plot_single_metric(
        self,
        results_df: pd.DataFrame,
        metric_key: str,
        output_dir: Path,
        p_column: str = "adjusted_p_value",
        value_column: str = "t_statistic",
    ) -> None:
        """Plot a single metric's results.

        Args:
            results_df: DataFrame containing statistical results
            metric_key: Key identifying the metric
            output_dir: Directory to save figures
            p_column: Column name for p-values
            value_column: Column name for values to visualize
        """
        logger.info(f"Creating brain plots for {metric_key}")

        if self.surfaces is None:
            self.setup_surfaces()

        # Prepare value maps
        value_map_lh, value_map_rh, subcort_df = self.prepare_value_maps(
            results_df, p_column, value_column
        )

        # Plot cortical surfaces
        self.plot_cortical_surface(
            value_map_lh, value_map_rh, output_dir, metric_key, value_column
        )

        # Plot subcortical regions
        self.plot_subcortical(subcort_df, output_dir, metric_key)

        logger.success(f"Completed brain plots for {metric_key}")
