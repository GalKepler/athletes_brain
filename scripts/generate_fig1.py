#!/usr/bin/env python3
"""
Figure 1 Generation Script

This script demonstrates how to use the refactored athletes_brain.fig1 module
to generate Figure 1 for the athletes brain study.

Usage:
    python generate_fig1.py [--output-dir OUTPUT_DIR] [--quick]
"""

import argparse
from pathlib import Path
from loguru import logger

from athletes_brain.fig1 import (
    generate_figure1,
    analyze_specific_comparison,
    get_most_significant_regions,
    Fig1Config,
)


def main():
    """Main function to generate Figure 1."""
    parser = argparse.ArgumentParser(description="Generate Figure 1 for athletes brain study")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: ~/Projects/athletes_brain/figures/fig1)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick analysis (athletes vs controls for ADC only)",
    )
    parser.add_argument(
        "--comparison",
        choices=[
            "athletes_vs_controls",
            "climbers_vs_controls",
            "bjj_vs_controls",
            "climbers_vs_bjj",
        ],
        default="athletes_vs_controls",
        help="Specific comparison to analyze (used with --quick)",
    )
    parser.add_argument(
        "--metric",
        choices=["gm_vol", "wm_vol", "csf_vol", "adc", "fa", "ad", "rd"],
        default="adc",
        help="Specific metric to analyze (used with --quick)",
    )

    args = parser.parse_args()

    # Initialize configuration
    config = Fig1Config()

    if args.quick:
        logger.info(f"Running quick analysis: {args.comparison} for {args.metric}")

        # Run specific analysis
        analyze_specific_comparison(
            comparison_type=args.comparison,
            metric=args.metric,
            output_dir=args.output_dir,
            config=config,
        )

        # Show most significant regions
        logger.info("Most significant regions:")
        get_most_significant_regions(
            comparison_type=args.comparison, metric=args.metric, n_regions=10, config=config
        )

    else:
        logger.info("Running complete Figure 1 generation")

        # Generate complete figure
        generate_figure1(output_dir=args.output_dir, config=config)

    logger.success("Analysis completed!")


if __name__ == "__main__":
    main()
