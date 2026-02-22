# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 17:26:05 2025

@author: elcha
"""
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate plots from SimAI simulation results')
parser.add_argument('--moe-active', action='store_true',
                    help='Use MOE-enabled workload data instead of regular workloads')
args = parser.parse_args()

#%% Configuration

# Path to pkl files (depends on MOE flag)
if args.moe_active:
    PKL_DIR = "saved DataFrames_moe"
    output_suffix = "_moe"
    print("🔬 MOE Mode: Using MOE-enabled simulation data")
else:
    PKL_DIR = "saved DataFrames"
    output_suffix = ""
    print("📊 Standard Mode: Using regular simulation data")

# Check if directory exists
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
OUTPUT_DIR = f"simulation_plots{output_suffix}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📊 Saving plots to: {OUTPUT_DIR}")

#%% Process each pkl file

for filename in PKL_FILES:
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    name = filename[:-4]  # without ".pkl"
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
    if args.moe_active:
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
        if args.moe_active:
            title += " (MOE)"
        plt.title(title, pad=20)
        plt.xlabel(f"Overhead Communication Time Bins ($\\times 10^{{{power_to_show}}}$)")
        plt.ylabel("Number of Matrices (Count)")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Topology")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save histogram
        hist_filename = os.path.join(strategy_dir, f"histogram_world_size_{size}.png")
        plt.savefig(hist_filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  ✅ Saved histogram for world_size={size}: {hist_filename}")

print("\n" + "="*60)
print("✅ All plots generated successfully!")
print("="*60)