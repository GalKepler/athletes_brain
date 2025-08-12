# Figure 1 Module

This module provides a refactored, modular approach to generating Figure 1 for the athletes brain study. The original notebook has been reorganized into reusable, well-documented modules.

## Module Structure

```
athletes_brain/fig1/
├── __init__.py              # Main module imports
├── config.py                # Configuration and visualization settings
├── data_loader.py           # Data loading and preprocessing
├── statistical_analysis.py  # Statistical comparisons (ANCOVA)
├── brain_visualization.py   # Brain surface and subcortical plotting
├── main.py                  # High-level functions for complete analysis
└── utils.py                 # Utility functions for data analysis
```

## Key Features

- **Modular Design**: Each component has a single responsibility
- **Reusable Functions**: Use components independently or together
- **Consistent Configuration**: Centralized settings for reproducibility
- **Comprehensive Logging**: Track progress and identify issues
- **Flexible Analysis**: Run complete analysis or focus on specific comparisons

## Quick Start

### Complete Figure Generation

```python
from athletes_brain.fig1 import generate_figure1

# Generate all comparisons and visualizations
generate_figure1()
```

### Step-by-Step Analysis

```python
from athletes_brain.fig1 import (
    Fig1Config, AthletesBrainDataLoader, 
    GroupComparison, BrainPlotter
)

# Initialize components
config = Fig1Config()
data_loader = AthletesBrainDataLoader(config)
group_comparison = GroupComparison(config)
brain_plotter = BrainPlotter(config)

# Load data
metric_data, parcels, nifti, nifti_matlab = data_loader.load_all_data()

# Perform comparisons
results = group_comparison.athletes_vs_controls(metric_data, parcels)

# Create visualizations
brain_plotter.create_brain_plots(results, "athletes_vs_controls", output_dir)
```

### Focused Analysis

```python
from athletes_brain.fig1.main import analyze_specific_comparison

# Analyze specific comparison and metric
analyze_specific_comparison(
    comparison_type="athletes_vs_controls",
    metric="adc"
)
```

## Available Comparisons

1. **Athletes vs Controls**: All athletes compared to control participants
2. **Climbers vs Controls**: Rock climbers compared to controls
3. **BJJ vs Controls**: Brazilian Jiu-Jitsu athletes compared to controls  
4. **Climbers vs BJJ**: Direct comparison between sports

## Available Metrics

- **Structural**: `gm_vol`, `wm_vol`, `csf_vol` (gray matter, white matter, CSF volumes)
- **Diffusion**: `adc`, `fa`, `ad`, `rd` (apparent diffusion coefficient, fractional anisotropy, axial/radial diffusivity)

## Configuration

The `Fig1Config` class centralizes all configuration options:

```python
from athletes_brain.fig1 import Fig1Config

config = Fig1Config()
print(f"Atlas: {config.ATLAS}")
print(f"P-value threshold: {config.P_THRESHOLD}")
print(f"Visualization range: [{config.VMIN}, {config.VMAX}]")
```

## Command Line Usage

A command-line script is provided for easy execution:

```bash
# Generate complete figure
python scripts/generate_fig1.py

# Quick analysis for specific comparison
python scripts/generate_fig1.py --quick --comparison athletes_vs_controls --metric adc

# Custom output directory
python scripts/generate_fig1.py --output-dir /path/to/output
```

## Notebook Usage

See `notebooks/fig1/fig1_refactored.ipynb` for an interactive demonstration of the refactored modules.

## Differences from Original

### Improvements
- **Organized Code**: Functions grouped by purpose
- **Error Handling**: Robust error handling and logging
- **Documentation**: Comprehensive docstrings and type hints
- **Flexibility**: Easy to modify parameters or add new analyses
- **Reusability**: Components can be used in other analyses

### Preserved Functionality
- All statistical methods (weighted least squares, multiple comparisons correction)
- All visualization capabilities (cortical surfaces, subcortical regions)
- All comparison types and metrics
- Same output format and quality

## Dependencies

The module requires the same dependencies as the original notebook:
- pandas, numpy, scipy
- statsmodels (statistical analysis)
- matplotlib, seaborn (basic plotting)
- nibabel, nilearn (neuroimaging)
- surfplot, neuromaps, brainspace (brain visualization)
- subcortex_visualization (subcortical plotting)

## Future Extensions

The modular design makes it easy to:
- Add new brain atlases
- Include additional statistical methods
- Support new visualization styles
- Extend to other figures or analyses
