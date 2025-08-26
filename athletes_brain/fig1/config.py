"""Configuration module for figure 1 visualization settings."""

from pathlib import Path
from typing import Dict, Any
import matplotlib as mpl
import seaborn as sns
from matplotlib import font_manager


class Fig1Config:
    """Configuration class for Figure 1 visualization settings."""

    # Atlas configuration
    ATLAS = "schaefer2018tian2020_400_7"
    REGION_COL = "index"

    # Metrics to analyze
    METRICS = ["gm_vol", "wm_vol", "csf_vol", "adc", "fa", "ad", "rd"]
    # METRICS = ["gm_vol", "adc"]
    DISTRIBUTION_METRIC = "qfmean"

    # Bad subjects to exclude
    BAD_SUBJECTS = ["IN120120"]

    # Group labels
    CONTROL_GROUP_LABEL = "Control"
    CLIMBER_GROUP_LABEL = "Climbing"
    BJJ_GROUP_LABEL = "Bjj"

    # Visualization parameters
    VMIN = -5
    VMAX = 5
    P_THRESHOLD = 0.05
    VALUE_THRESHOLD = 0
    DPI = 400  # Default DPI for saving figures
    # Colormap configuration
    COLORMAP = "coolwarm"  # Default colormap for visualization

    # Colors
    COL_RAW = "#ffb300"
    COL_WEIGHTED = "#7F0099CE"
    COL_REF = "0.25"
    COL_CENSUS = "#A5A5A5"
    COL_WHITE = "white"

    @classmethod
    def setup_matplotlib_config(cls) -> None:
        """Set up matplotlib configuration for consistent visualization."""
        # Add Calibri font if available
        font_path = Path.home() / ".fonts/calibri-regular.ttf"
        if font_path.exists():
            font_manager.fontManager.addfont(font_path)

        mpl.rcParams.update(
            {
                # Canvas size & resolution
                "figure.figsize": (12, 8),
                "figure.dpi": 200,
                "savefig.dpi": 400,
                # Fonts
                "font.family": "Calibri",
                "font.sans-serif": ["Calibri", "DejaVu Sans", "Arial"],
                "axes.titlesize": 24,
                "axes.labelsize": 24,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 20,
                # Axis & spine aesthetics
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.linewidth": 1,
                "axes.grid": True,
                "grid.color": "#E6E6E6",
                "grid.linewidth": 0.4,
                "grid.alpha": 0.8,
                # Color cycle
                "axes.prop_cycle": mpl.cycler(color=sns.color_palette("Set2")),
                # Figure background
                "figure.facecolor": "white",
            }
        )

        # Set seaborn theme
        sns.set_theme(
            context="talk",
            style="whitegrid",
            palette="Set2",
        )

    @classmethod
    def get_colormaps(cls) -> Dict[str, Any]:
        """Generate colormaps for visualization."""
        col_raw_rgb = mpl.colors.to_rgb(cls.COL_RAW)
        col_weighted_rgb = mpl.colors.to_rgb(cls.COL_WEIGHTED)

        cmap_raw = mpl.colors.LinearSegmentedColormap.from_list(
            "raw", [cls.COL_WHITE, col_raw_rgb], N=256
        )
        cmap_weighted = mpl.colors.LinearSegmentedColormap.from_list(
            "weighted", [cls.COL_WHITE, col_weighted_rgb], N=256
        )

        return {"raw": cmap_raw, "weighted": cmap_weighted}

    @classmethod
    def get_metric_columns(cls) -> Dict[str, str]:
        """Get the column names for each metric."""
        return {
            metric: "volume" if "vol" in metric else cls.DISTRIBUTION_METRIC
            for metric in cls.METRICS
        }


def savefig_nice(fig, filename, *, tight=True, dpi=300, **savefig_kwargs):
    """Save figure with tight layout and correct DPI."""
    if tight:
        fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight", transparent=True, **savefig_kwargs)
