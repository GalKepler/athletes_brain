import pandas as pd
from pathlib import Path
import nibabel as nib
from functools import reduce

from athletes_brain.fig2.config import (
    DATA_DIR,
    ATLAS,
    REGION_COL,
    BAD_SUBJECTS,
    METRICS,
    DISTRIBUTION_METRIC,
)


def load_parcels():
    """Loads the parcels CSV file."""
    return pd.read_csv(DATA_DIR / "external" / "atlases" / ATLAS / "parcels.csv", index_col=0)


def load_nifti_paths():
    """Returns paths to NIfTI files."""
    return {
        "nifti": DATA_DIR / "external" / "atlases" / ATLAS / "atlas.nii.gz",
        "nifti_matlab": DATA_DIR / "external" / "atlases" / ATLAS / "atlas_matlab.nii",
    }


def load_and_preprocess_metric_data():
    """
    Loads raw metric data, performs initial cleaning, and renames value columns.

    Returns
    -------
    dict
        A dictionary where keys are metric names and values are preprocessed DataFrames.
    """
    data = {}
    metric_cols = {
        metric: "volume" if "vol" in metric else DISTRIBUTION_METRIC for metric in METRICS
    }

    for metric in METRICS:
        df = pd.read_csv(DATA_DIR / "processed" / f"{metric}.csv", index_col=0).reset_index(
            drop=True
        )
        df = df[~df["subject_code"].isin(BAD_SUBJECTS)]
        df["sex"] = df["sex"].map({"M": 0, "F": 1})
        df = df.rename(columns={metric_cols[metric]: "value"})
        data[metric] = df
    return data


def find_common_sessions(data_wide):
    """
    Finds sessions common across all wide-format metric DataFrames.

    Parameters
    ----------
    data_wide : dict
        A dictionary of wide-format DataFrames for each metric.

    Returns
    -------
    set
        A set of session IDs common to all metrics.
    """
    common_sessions = []
    for metric, df in data_wide.items():
        common_sessions.append(set(df["session_id"].unique()))
    return reduce(lambda x, y: x.intersection(y), common_sessions)


def filter_by_common_sessions(data_wide, common_sessions):
    """
    Filters each DataFrame in data_wide to include only common sessions.

    Parameters
    ----------
    data_wide : dict
        A dictionary of wide-format DataFrames for each metric.
    common_sessions : set
        A set of session IDs to filter by.

    Returns
    -------
    dict
        Filtered data_wide dictionary.
    """
    filtered_data_wide = {}
    for metric, df in data_wide.items():
        filtered_data_wide[metric] = df[df["session_id"].isin(common_sessions)]
    return filtered_data_wide
