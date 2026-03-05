# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 17:26:05 2025

@author: elcha
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import re
import numpy as np
import os
import argparse
from output_config import get_variant_paths

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate plots from SimAI simulation results')
parser.add_argument('--moe-active', action='store_true',
                    help='Use MOE-enabled workload data instead of regular workloads')
parser.add_argument('--all-archs', action='store_true',
                    help='Combine data from both MOE and standard workloads (~300 total workloads)')
args = parser.parse_args()

# Validate mutually exclusive flags
if args.moe_active and args.all_archs:
    print("❌ Error: --moe-active and --all-archs are mutually exclusive. Choose one.")
    exit(1)

#%% Configuration

# Path to pkl files (depends on flags)
if args.all_archs:
    variant = "all_archs"
    paths = get_variant_paths(variant)
    # Get paths for both standard and moe data
    standard_paths = get_variant_paths("standard")
    moe_paths = get_variant_paths("moe")
    PKL_DIR_STANDARD = str(standard_paths["dataframes"])
    PKL_DIR_MOE = str(moe_paths["dataframes"])
    output_suffix = "_all_archs"
    print("🌐 All Architectures Mode: Combining MOE and standard workloads")
else:
    variant = "moe" if args.moe_active else "standard"
    paths = get_variant_paths(variant)
    PKL_DIR = str(paths["dataframes"])
    output_suffix = "_moe" if args.moe_active else ""
    
    if args.moe_active:
        print("🔬 MOE Mode: Using MOE-enabled simulation data")
    else:
        print("📊 Standard Mode: Using regular simulation data")

# Check if directories exist
if args.all_archs:
    missing_dirs = []
    if not os.path.exists(PKL_DIR_STANDARD):
        missing_dirs.append(f"Standard: {PKL_DIR_STANDARD}")
    if not os.path.exists(PKL_DIR_MOE):
        missing_dirs.append(f"MOE: {PKL_DIR_MOE}")
    
    if missing_dirs:
        print("❌ Missing data directories:")
        for d in missing_dirs:
            print(f"   - {d}")
        print("\n   Run 'Topologies Runtime.py' (standard) and 'Topologies Runtime.py --moe-active' (MOE) first")
        exit(1)
    
    print(f"📁 Reading standard data from: {PKL_DIR_STANDARD}")
    print(f"📁 Reading MOE data from: {PKL_DIR_MOE}")
else:
    if not os.path.exists(PKL_DIR):
        print(f"❌ Directory not found: {PKL_DIR}")
        if args.moe_active:
            print("   Make sure to run 'Topologies Runtime.py --moe-active' first to generate MOE DataFrames")
        else:
            print("   Make sure to run 'Topologies Runtime.py' first to generate DataFrames")
        exit(1)
    
    print(f"📁 Reading data from: {PKL_DIR}")

# List of pkl files to process
PKL_FILES = ['only_tp.pkl', 'only_dp.pkl', 'only_pp.pkl', 'Total.pkl']

# Topologies to compare
TOPOLOGY_COLS = ["fat tree", "DragonFly+", "HyperX"]

# Number of bins for histograms
NUM_BINS = 10

# Create output directory structure
OUTPUT_DIR = str(paths["plots_simulation"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📊 Saving plots to: {OUTPUT_DIR}")

#%% Process each pkl file

for filename in PKL_FILES:
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    name = filename[:-4]  # without ".pkl"
    
    if args.all_archs:
        # Load and concatenate from both standard and MOE directories
        filepath_standard = os.path.join(PKL_DIR_STANDARD, filename)
        filepath_moe = os.path.join(PKL_DIR_MOE, filename)
        
        dfs_to_concat = []
        
        if os.path.exists(filepath_standard):
            df_standard = pd.read_pickle(filepath_standard)
            # Add source column to track origin
            df_standard['source'] = 'standard'
            dfs_to_concat.append(df_standard)
            print(f"  Loaded {len(df_standard)} standard workloads")
        else:
            print(f"  ⚠️  Standard file not found: {filepath_standard}")
        
        if os.path.exists(filepath_moe):
            df_moe = pd.read_pickle(filepath_moe)
            # Add source column to track origin
            df_moe['source'] = 'moe'
            dfs_to_concat.append(df_moe)
            print(f"  Loaded {len(df_moe)} MOE workloads")
        else:
            print(f"  ⚠️  MOE file not found: {filepath_moe}")
        
        if not dfs_to_concat:
            print(f"  ⚠️  No data found for {filename}, skipping")
            continue
        
        total_runtime_df = pd.concat(dfs_to_concat, ignore_index=True)
        total_runtime_df = total_runtime_df.sort_values(by='HyperX', ascending=True)
        print(f"  Combined total: {len(total_runtime_df)} workloads")
    else:
        filepath = os.path.join(PKL_DIR, filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            continue

        total_runtime_df = pd.read_pickle(filepath)
        total_runtime_df = total_runtime_df.sort_values(by='HyperX', ascending=True)
        print(f"Loaded {len(total_runtime_df)} workloads")

    # Create subdirectory for this parallelism strategy
    strategy_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(strategy_dir, exist_ok=True)


    #%% Simple Graph:

    plt.figure(dpi = 500)
    plt.plot(total_runtime_df["file"], total_runtime_df["fat tree"], "o", label="Fat Tree", color="#1f77b4", ms=6) # כחול עמוק
    plt.plot(total_runtime_df["file"], total_runtime_df["DragonFly+"], "o", label="DragonFly+", color="#ff7f0e", ms=2.5) # כתום בוהק
    plt.plot(total_runtime_df["file"], total_runtime_df["HyperX"], "o", label="HyperX", color="#2ca02c", ms=3) # ירוק חי (דומה לקיים)

    plt.xlabel("Workload index")
    plt.ylabel("Overhead Communication")
    title = f"Overhead Communication of Topologies - {name}"
    if args.all_archs:
        title += " (All Architectures)"
    elif args.moe_active:
        title += " (MOE)"
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.xticks([])
    plt.yscale('log')

    safe_title = re.sub(r'[\\/*?:"<>|]', "_", plt.gca().get_title())
    scatter_filename = os.path.join(strategy_dir, f"scatter_{name}.png")
    plt.savefig(scatter_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved scatter plot: {scatter_filename}")
    
    # Save scatter plot data to CSV
    scatter_csv_filename = os.path.join(strategy_dir, f"scatter_{name}.csv")
    scatter_df = total_runtime_df[["file"] + TOPOLOGY_COLS].copy()
    scatter_df.to_csv(scatter_csv_filename, index=False)
    print(f"📄 Saved scatter CSV: {scatter_csv_filename}")

    #%% split by num of GPUs:

    def extract_world_size(file_name):
        match = re.search(r'world_size(\d+)', file_name)
        if match:
            return int(match.group(1))
        return None

    dfs_by_world_size = {
        key: group
        for key, group in total_runtime_df.groupby(total_runtime_df['file'].apply(extract_world_size))
    }

    large_group_dfs = {
        key: sub_df
        for key, sub_df in dfs_by_world_size.items()
        if len(sub_df) > 6
    }

    print(f"Found {len(large_group_dfs)} GPU count groups with >6 workloads: {list(large_group_dfs.keys())}")

    if len(large_group_dfs) == 0:
        print(f"⚠️  No groups with >6 workloads found, skipping histograms for {name}")
        continue

    #%% Calculate global bins for unified histograms

    # Find global min/max across all GPU counts
    global_min_val = float('inf')
    global_max_val = float('-inf')

    for size, sub_df in large_group_dfs.items():
        all_values = pd.concat([sub_df[col] for col in TOPOLOGY_COLS])
        positive_values = all_values[all_values > 0]

        if not positive_values.empty:
            global_min_val = min(global_min_val, positive_values.min())
            global_max_val = max(global_max_val, positive_values.max())

    if global_min_val == float('inf'):
        print(f"⚠️  No positive data found for {name}, skipping histograms")
        continue

    # Create unified bins
    unified_bins = np.logspace(np.log10(global_min_val), np.log10(global_max_val), NUM_BINS + 1)
    unified_bins[-1] = unified_bins[-1] * 1.0000001  # Include max value

    # Calculate global scaling factor
    scaling_power = int(np.floor(np.log10(global_max_val)))
    power_to_show = scaling_power - 3
    SCALING_FACTOR = 10**power_to_show

    #%% Generate histogram for each GPU count

    for size, sub_df in large_group_dfs.items():
        # Use unified bins
        bins = unified_bins

        # Prepare DataFrame for counting matrices
        plot_data = pd.DataFrame()
        original_bins_intervals = None

        # Count matrices in each bin for each topology
        for topology in TOPOLOGY_COLS:
            counts = pd.cut(sub_df[topology], bins=bins, right=True, include_lowest=True).value_counts().sort_index()
            plot_data[topology] = counts

            if original_bins_intervals is None:
                original_bins_intervals = counts.index

        # Format bin names with scaling
        plot_data.index = [
            f"[{b.left / SCALING_FACTOR:.0f} - {b.right / SCALING_FACTOR:.0f})"
            for b in original_bins_intervals
        ]

        # Create histogram plot
        plt.figure(figsize=(10, 6), dpi=500)
        ax = plot_data.plot(kind='bar', ax=plt.gca(), colormap='viridis')

        title = f"Topology Performance Comparison ({name}):\nTraffic Matrices Count per Overhead Communication Range for a Network of {size} GPUs"
        if args.all_archs:
            title += " (All Architectures)"
        elif args.moe_active:
            title += " (MOE)"
        plt.title(title, pad=20)
        plt.xlabel(f"Overhead Communication Time Bins ($\\times 10^{{{power_to_show}}}$)")
        plt.ylabel("Number of Matrices (Count)")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Topology")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Force y-axis to show only whole numbers
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()

        # Save histogram
        hist_filename = os.path.join(strategy_dir, f"histogram_world_size_{size}.png")
        plt.savefig(hist_filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  ✅ Saved histogram for world_size={size}: {hist_filename}")
        
        # Save histogram data to CSV
        hist_csv_filename = os.path.join(strategy_dir, f"histogram_world_size_{size}.csv")
        plot_data.to_csv(hist_csv_filename, index_label="bin_range")
        print(f"  📄 Saved histogram CSV: {hist_csv_filename}")

print("\n" + "="*60)
print("✅ All plots generated successfully!")
print("="*60)