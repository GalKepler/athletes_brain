"""Data loading module for athletes brain study Figure 1."""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from loguru import logger

from athletes_brain.config import DATA_DIR, EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR
from .config import Fig1Config


class AthletesBrainDataLoader:
    """Data loader for athletes brain study."""

    def __init__(self, config: Optional[Fig1Config] = None):
        """Initialize the data loader.

        Args:
            config: Configuration object. If None, uses default Fig1Config.
        """
        self.config = config or Fig1Config()
        self.data_dir = DATA_DIR
        self.external_dir = EXTERNAL_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR

    def load_atlas_data(self) -> tuple[pd.DataFrame, Path, Path]:
        """Load atlas-related data files.

        Returns:
            Tuple of (parcels DataFrame, nifti path, nifti_matlab path)
        """
        logger.info(f"Loading atlas data for {self.config.ATLAS}")

        atlas_dir = self.external_dir / "atlases" / self.config.ATLAS
        parcels = pd.read_csv(atlas_dir / "parcels.csv", index_col=0)
        nifti = atlas_dir / "atlas.nii.gz"
        nifti_matlab = atlas_dir / "atlas_matlab.nii"

        logger.info(f"Loaded {len(parcels)} parcels from atlas")
        return parcels, nifti, nifti_matlab

    def load_metric_data(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess all metric data.

        Returns:
            Dictionary mapping metric names to processed DataFrames
        """
        logger.info("Loading metric data...")

        data = {}
        metric_cols = self.config.get_metric_columns()

        for metric in self.config.METRICS:
            logger.info(f"Loading {metric} data")

            # Load raw data
            df = pd.read_csv(self.processed_dir / f"{metric}.csv", index_col=0).reset_index(
                drop=True
            )

            # Remove duplicates
            df = df.drop_duplicates(subset=["subject_code", self.config.REGION_COL], keep="last")

            # Remove problematic subjects
            df = df[~df["subject_code"].isin(self.config.BAD_SUBJECTS)]

            # Encode sex as numeric
            df["sex"] = df["sex"].map({"M": 0, "F": 1})

            # Rename value column for consistency
            df = df.rename(columns={metric_cols[metric]: "value"})

            data[metric] = df
            logger.info(f"Loaded {len(df)} records for {metric}")

        logger.success(f"Successfully loaded {len(data)} metrics")
        return data

    def get_output_directory(self, figure_name: str = "fig1") -> Path:
        """Get and create output directory for figures.

        Args:
            figure_name: Name of the figure directory

        Returns:
            Path to the output directory
        """
        output_dir = Path.home() / "Projects" / "athletes_brain" / "figures" / figure_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def load_all_data(self) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame, Path, Path]:
        """Load all required data for Figure 1.

        Returns:
            Tuple of (metric_data_dict, parcels_df, nifti_path, nifti_matlab_path)
        """
        metric_data = self.load_metric_data()
        parcels, nifti, nifti_matlab = self.load_atlas_data()

        return metric_data, parcels, nifti, nifti_matlab
