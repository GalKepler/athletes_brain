"""Data loading module for athletes brain study Figure 2."""

import pandas as pd
from pathlib import Path
from functools import reduce
from typing import Dict, Optional

from athletes_brain.fig2.config import Fig2Config


# --- Modular DataLoader Class for Fig2 ---
class Fig2DataLoader:
    """Data loader for athletes brain study Figure 2."""

    def __init__(self, config: Optional[Fig2Config] = None):
        self.config = config or Fig2Config()
        self.data_dir = self.config.DATA_DIR

    def load_parcels(self) -> pd.DataFrame:
        """Loads the parcels CSV file."""
        return pd.read_csv(
            self.data_dir / "external" / "atlases" / self.config.ATLAS / "parcels.csv", index_col=0
        )

    def load_nifti_paths(self) -> Dict[str, Path]:
        """Returns paths to NIfTI files."""
        base = self.data_dir / "external" / "atlases" / self.config.ATLAS
        return {
            "nifti": base / "atlas.nii.gz",
            "nifti_matlab": base / "atlas_matlab.nii",
        }

    def load_and_preprocess_metric_data(self) -> Dict[str, pd.DataFrame]:
        """Loads raw metric data, performs initial cleaning, and renames value columns."""
        data = {}
        metric_cols = self.config.get_metric_columns()
        for metric in self.config.METRICS:
            df = pd.read_csv(
                self.data_dir / "processed" / f"{metric}.csv", index_col=0
            ).reset_index(drop=True)
            df = df[~df["subject_code"].isin(self.config.BAD_SUBJECTS)]
            df["subject_code"] = (
                df["subject_code"]
                .astype(str)
                .str.replace("-", "")
                .str.replace(" ", "")
                .str.replace("_", "")
                .str.zfill(4)
            )
            df["sex"] = df["sex"].map({"M": 0, "F": 1})
            df = df.rename(columns={metric_cols[metric]: "value"})
            data[metric] = df
        return data

    @staticmethod
    def find_common_sessions(data_wide: Dict[str, pd.DataFrame]) -> set:
        """Finds sessions common across all wide-format metric DataFrames."""
        common_sessions = []
        for metric, df in data_wide.items():
            common_sessions.append(set(df["session_id"].unique()))
        return reduce(lambda x, y: x.intersection(y), common_sessions)

    @staticmethod
    def filter_by_common_sessions(
        data_wide: Dict[str, pd.DataFrame], common_sessions: set
    ) -> Dict[str, pd.DataFrame]:
        """Filters each DataFrame in data_wide to include only common sessions."""
        filtered_data_wide = {}
        for metric, df in data_wide.items():
            filtered_data_wide[metric] = df[df["session_id"].isin(common_sessions)]
        return filtered_data_wide
