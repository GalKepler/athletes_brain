"""Figure 1 generation module for athletes brain study."""

from .config import Fig1Config, savefig_nice
from .data_loader import AthletesBrainDataLoader
from .statistical_analysis import GroupComparison
from .brain_visualization import BrainPlotter
from .main import generate_figure1, analyze_specific_comparison, get_most_significant_regions
from .utils import (
    load_and_analyze_all,
    find_significant_regions,
    compare_metrics_across_comparisons,
    summarize_results,
)

__all__ = [
    "Fig1Config",
    "savefig_nice",
    "AthletesBrainDataLoader",
    "GroupComparison",
    "BrainPlotter",
    "generate_figure1",
    "analyze_specific_comparison",
    "get_most_significant_regions",
    "load_and_analyze_all",
    "find_significant_regions",
    "compare_metrics_across_comparisons",
    "summarize_results",
]
