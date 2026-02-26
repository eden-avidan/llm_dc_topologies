# -*- coding: utf-8 -*-
"""
Process only the 123 non-MOE configurations that match the MOE workloads.
This creates a fair comparison by using the same configurations.

Based on: Topologies Runtime.py
Modified to process specific subset with _no_moe suffix
"""
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from output_config import get_variant_paths, METADATA_DIR

# Load the list of files to process
script_dir = Path(__file__).resolve().parent
files_to_process_file = METADATA_DIR / "non_moe_files_to_process.txt"

if not files_to_process_file.exists():
    print("❌ File list not found. Run 'process_matching_no_moe.py' first!")
    exit(1)

with open(files_to_process_file, 'r') as f:
    files_to_process = set(line.strip() for line in f if line.strip())

print(f"📋 Loaded {len(files_to_process)} files to process")

# Configuration
output_suffix = "_no_moe"
paths = get_variant_paths("no_moe")

# Gather data from the SimAI simulations:
invalid_dirs = ["workload", "heatmaps", "workload_moe"]

folder = (script_dir / "../simai/final_output").resolve()

# Use regular (non-MOE) directories
print("📊 Processing Non-MOE data (subset matching MOE configurations)")
dirs_list = [p for p in folder.iterdir() if p.is_dir() and p.name not in invalid_dirs and not p.name.endswith('_moe')]

if not dirs_list:
    print(f"❌ No directories found in {folder}")
    exit(1)

print(f"Found {len(dirs_list)} directories: {[d.name for d in dirs_list]}")

files_list = []

def get_matrix(file_index=0):
    """Get the transport matrix from the file."""
    file_path = files_list[file_index]
    dataframe = pd.read_csv(file_path)
    transport_matrix = dataframe.iloc[:, 1:].to_numpy()
    if len(transport_matrix) != len(transport_matrix[0]):
        print("Error - the transpose matrix is not square!")
    return transport_matrix

def get_N_hbi(file_index: int) -> int | None:
    """Get the number of GPUs in each HBI from the file name."""
    match = re.search(r"-tp(\d+)-", os.path.basename(files_list[file_index]))
    return int(match.group(1)) if match else None

# Constants
GPUs_num = 0
N_hbi = 0
N_nodes_1dim = 0

T_link = 1
T_link_out = T_link
T_link_in = T_link

# Location classes
@dataclass
class Location:
    HBI_index: int
    GPU_index: int

@dataclass
class HyperX_Location:
    HBI_index_1: int
    HBI_index_2: int
    GPU_index: int

def index_to_location(i):
    return Location(int(i / N_hbi), i % N_hbi)

def location_to_HyperX_location(loc):
    return HyperX_Location(int(loc.HBI_index / N_nodes_1dim), loc.HBI_index % N_nodes_1dim, loc.GPU_index)

# Latency functions
def Fat_Tree_latency(i_location, j_location):
    if i_location == j_location:
        return 0
    elif i_location.HBI_index == j_location.HBI_index:
        return 2 * T_link_in
    elif i_location.GPU_index == j_location.GPU_index:
        return 2 * T_link_out
    else:
        return 4 * T_link_out

def Rail_Only_latency(i_location, j_location):
    if i_location.HBI_index != j_location.HBI_index and i_location.GPU_index != j_location.GPU_index:
        return None
    else:
        return Fat_Tree_latency(i_location, j_location)

def dragonFlyP_latency(i_location, j_location):
    if i_location == j_location:
        return 0
    elif i_location.HBI_index == j_location.HBI_index:
        return 2 * T_link_in
    else:
        return 3 * T_link

def HyperX_latency(i_location, j_location):
    if (i_location == j_location):
        return 0
    i_HyperX_loc = location_to_HyperX_location(i_location)
    j_HyperX_loc = location_to_HyperX_location(j_location)
    result = 0
    if i_HyperX_loc.GPU_index != j_HyperX_loc.GPU_index:
        result = result + 1
    if i_HyperX_loc.HBI_index_1 != j_HyperX_loc.HBI_index_1:
        result = result + 1
    if i_HyperX_loc.HBI_index_2 != j_HyperX_loc.HBI_index_2:
        result = result + 1
    return result

# Topology data
@dataclass
class Topology_Data:
    latency_function: callable
    last_runtime: int

topologies_dict = {
    "fat tree": Topology_Data(Fat_Tree_latency, None),
    "rail only": Topology_Data(Rail_Only_latency, None),
    "HyperX": Topology_Data(HyperX_latency, None),
    "DragonFly+": Topology_Data(dragonFlyP_latency, None)
}

TP_Runtimes_Table = []
DP_Runtimes_Table = []
PP_Runtimes_Table = []
Matrices_Runtimes_Table = []

tables_dict = {
    "only_pp": PP_Runtimes_Table,
    "only_tp": TP_Runtimes_Table,
    "only_dp": DP_Runtimes_Table,
    "matrices": Matrices_Runtimes_Table
}

def update_Runtimes_Table(table, file_name):
    # Remove .csv and any suffix like _tp_only, _dp_only, _pp_only
    clean_name = file_name.replace('_tp_only.csv', '.csv').replace('_dp_only.csv', '.csv').replace('_pp_only.csv', '.csv')
    clean_name = clean_name[:-4]  # Remove .csv
    table.append({
        "file": clean_name,
        "fat tree": topologies_dict["fat tree"].last_runtime,
        "rail only": topologies_dict["rail only"].last_runtime,
        "HyperX": topologies_dict["HyperX"].last_runtime,
        "DragonFly+": topologies_dict["DragonFly+"].last_runtime
    })

def Topology_Runtime(topology, matrix):
    result = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            i_location = index_to_location(i)
            j_location = index_to_location(j)
            if matrix[i][j] != 0:
                i_j_latency = topologies_dict[topology].latency_function(i_location, j_location)
                if i_j_latency == None:
                    return None
                result = result + i_j_latency * matrix[i][j]
    return result

# Process files
total_processed = 0
total_skipped = 0

for directory in dirs_list:
    all_files = sorted([f for f in directory.iterdir() if f.is_file() and '.csv' in os.path.basename(f)])

    # Filter to only process matching files
    # Handle different naming conventions in each directory
    if directory.name == "only_tp":
        expected_files = {f.replace('.csv', '_tp_only.csv') for f in files_to_process}
    elif directory.name == "only_dp":
        expected_files = {f.replace('.csv', '_dp_only.csv') for f in files_to_process}
    elif directory.name == "only_pp":
        expected_files = {f.replace('.csv', '_pp_only.csv') for f in files_to_process}
    else:  # matrices
        expected_files = files_to_process

    files_list = [f for f in all_files if os.path.basename(f) in expected_files]

    print(f"\n{'='*70}")
    print(f"Directory: {directory.name}")
    print(f"  Total files: {len(all_files)}")
    print(f"  Processing: {len(files_list)} (matching MOE subset)")
    print(f"{'='*70}")

    for file_index in range(len(files_list)):
        print(f"Matrix {file_index+1}/{len(files_list)}: {os.path.basename(files_list[file_index])}")

        GPUs_num = len(get_matrix(file_index))
        N_hbi = get_N_hbi(file_index)
        N_nodes_1dim = int((GPUs_num/N_hbi)**0.5) + 1

        if N_hbi == None:
            print("  ⚠️  Error: Could not extract tp size")
            total_skipped += 1
            continue

        for topology in topologies_dict.keys():
            topologies_dict[topology].last_runtime = Topology_Runtime(topology, get_matrix(file_index))

        update_Runtimes_Table(tables_dict[directory.name], os.path.basename(files_list[file_index]))
        total_processed += 1

print(f"\n{'='*70}")
print(f"Processing Complete:")
print(f"  Processed: {total_processed}")
print(f"  Skipped: {total_skipped}")
print(f"{'='*70}")

# Arrange the data
Runtime_dfs = {name: pd.DataFrame(table) for name, table in tables_dict.items()}
dfs_list = list(Runtime_dfs.values())
total_runtime_df = dfs_list[0].copy()

for df in dfs_list[1:]:
    for col in total_runtime_df.columns:
        if total_runtime_df[col].dtype != "object":
            total_runtime_df[col] += df[col]

Runtime_dfs["Total"] = total_runtime_df

# Save DataFrames
save_dir = str(paths["dataframes"])
os.makedirs(save_dir, exist_ok=True)

print(f"\n💾 Saving DataFrames to: {save_dir}/")
for name, df in Runtime_dfs.items():
    df.to_pickle(f"{save_dir}/{name}.pkl")
    print(f"   Saved: {name}.pkl ({len(df)} workloads)")

# Generate plots
plots_dir = str(paths["plots_overhead"])
os.makedirs(plots_dir, exist_ok=True)

print(f"\n📊 Generating plots in: {plots_dir}/")
for name, df in Runtime_dfs.items():
    plt.figure(dpi=500)
    plt.plot(df["file"], df["HyperX"], "o", label="HyperX", color="#1f77b4", ms=1)

    plt.xlabel("Transport Matrix")
    plt.ylabel("Overhead Communication")
    title = f"Overhead Communication of Topologie - {name} (Densed)"
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.xticks([])

    safe_title = re.sub(r'[\\/*?:"<>|]', "_", plt.gca().get_title())
    filename = f"{plots_dir}/{safe_title}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {safe_title}.png")

print("\n✅ Processing complete!")
print(f"\n📁 Output directories:")
print(f"  DataFrames: {save_dir}/")
print(f"  Plots: {plots_dir}/")
