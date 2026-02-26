"""
Centralized output path configuration for topologies_for_llm scripts.

This module provides a single source of truth for all output paths,
making it easy to reorganize outputs or add new variants in the future.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"

# Variant directories
STANDARD_DIR = OUTPUTS_DIR / "standard"
MOE_DIR = OUTPUTS_DIR / "moe"
NO_MOE_DIR = OUTPUTS_DIR / "no_moe"
COMPARISONS_DIR = OUTPUTS_DIR / "comparisons"
METADATA_DIR = OUTPUTS_DIR / "metadata"


def get_variant_paths(variant="standard"):
    """
    Get output paths for a specific variant.

    Args:
        variant: "standard", "moe", or "no_moe"

    Returns:
        dict with keys:
            - dataframes: Path to dataframes directory
            - plots_overhead: Path to overhead communication plots
            - plots_simulation: Path to simulation plots
            - heatmaps: Path to heatmaps directory

    Raises:
        ValueError: If variant is not recognized
    """
    variant_map = {
        "standard": STANDARD_DIR,
        "moe": MOE_DIR,
        "no_moe": NO_MOE_DIR
    }

    if variant not in variant_map:
        raise ValueError(f"Unknown variant: {variant}. Must be one of {list(variant_map.keys())}")

    base = variant_map[variant]

    return {
        "dataframes": base / "dataframes",
        "plots_overhead": base / "plots" / "overhead_communication",
        "plots_simulation": base / "plots" / "simulation",
        "heatmaps": base / "heatmaps"
    }


def get_comparison_paths():
    """
    Get paths for cross-variant comparison outputs.

    Returns:
        Path to comparisons directory
    """
    return COMPARISONS_DIR


def get_metadata_paths():
    """
    Get paths for metadata and processing artifacts.

    Returns:
        Path to metadata directory
    """
    return METADATA_DIR
