# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 17:26:05 2025

@author: elcha
@contributor: Eden Avidan (Feb 2026)
"""
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import argparse
from output_config import get_variant_paths

# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze topology runtime for SimAI workloads')
parser.add_argument('--moe-active', action='store_true',
                    help='Use MOE-enabled workload data instead of regular workloads')
parser.add_argument('--heatmaps-only', action='store_true',
                    help='Generate only topology latency heatmaps and skip runtime/dataframe/overhead plots')
args = parser.parse_args()

# Configure variant-specific output paths once, for all run modes
variant = "moe" if args.moe_active else "standard"
paths = get_variant_paths(variant)

if args.heatmaps_only:
    print("🗺️  Heatmaps-only mode: skipping workload runtime processing")
    dirs_list = []
else:
    # Gather data from the SimAI simulations:
    invalid_dirs = ["workload", "heatmaps", "workload_moe"]

    script_dir = Path(__file__).resolve().parent
    folder = (script_dir / "../simai/final_output").resolve()

    # Filter directories based on MOE flag
    if args.moe_active:
        print("🔬 MOE Mode: Using MOE-enabled workload data")
        dirs_list = [p for p in folder.iterdir() if p.is_dir() and p.name.endswith('_moe')]
    else:
        print("📊 Standard Mode: Using regular workload data")
        dirs_list = [p for p in folder.iterdir() if p.is_dir() and p.name not in invalid_dirs and not p.name.endswith('_moe')]

    if not dirs_list:
        print(f"❌ No directories found in {folder}")
        if args.moe_active:
            print("   Make sure MOE workloads have been processed (run process_moe_workloads.py)")
        exit(1)

    print(f"Found {len(dirs_list)} directories to process: {[d.name for d in dirs_list]}")

files_list = [] #sorted([f for f in dirs_list[0].iterdir() if f.is_file() and '.csv' in os.path.basename(f)])
    
def get_matrix(file_index = 0):
    """
    Get the transport matrix from the file.
    Transport matrix is a square matrix of size GPUs_num x GPUs_num, where GPUs_num is the number of GPUs in the simulation.
    The matrix is read from the file and returned as a numpy array, and represents how much data needs to be sent from GPU i to GPU j.
    """

    file_path = files_list[file_index]
    dataframe = pd.read_csv(file_path)
    transport_matrix = dataframe.iloc[:, 1:].to_numpy()
    if len(transport_matrix) != len(transport_matrix[0]):
        print ("Error - the transpose matrix is not square!")
    return  transport_matrix

def save_heatmap(file_index = 0, out_put_dir="heatmaps"):
    """
    Show the transport matrix as a heatmap.
    Note - heatmaps are already saved in simai.
    """
    os.makedirs(out_put_dir, exist_ok=True)
    matrix = get_matrix(file_index)
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color="white")
    plt.figure(dpi = 300)
    plt.imshow(np.ma.masked_where(matrix == 0, matrix), cmap=cmap, aspect="auto")
    plt.colorbar(label="amount of data")
    plt.title(f"Heatmap of GPU Communication Matrix:\n{os.path.basename(files_list[file_index])}")
    plt.xlabel("source GPU")
    plt.ylabel("destination GPU")
    plt.savefig(f"{out_put_dir}/{os.path.basename(files_list[file_index])}.png", dpi=300, bbox_inches="tight")
    plt.close()


def get_N_hbi(file_index: int) -> int | None:
    """
    Get the number of GPUs in each HBI from the file name.
    The file name is expected to be in the format "transport_matrix_<topology>_<GPUs_num>g_<GPUs_per_server>gps_<GPU_type>.csv"
    """
    match = re.search(r"-tp(\d+)-", os.path.basename(files_list[file_index]))
    return int(match.group(1)) if match else None

#%% Constants:

GPUs_num = 0 #len(get_matrix()) # below - update for each matrix
N_hbi = 8 # amount of GPUs in each HBI
N_nodes_1dim = 0#int((GPUs_num/N_hbi)**0.5)

# def update_constants(file_index):
    # GPUs_num = len(get_matrix(file_index))
HBI_link = 1
T_link = 1
T_link_out = T_link
T_link_in = T_link


#%% Location Defination:

@dataclass
class Location:
    HBI_index: int # index of HBI
    GPU_index: int # index of GPU in HBI

@dataclass
class HyperX_Location:
    HBI_index_1: int
    HBI_index_2: int
    GPU_index: int

def index_to_location(i):
    return Location(int(i / N_hbi), i % N_hbi)

def location_to_HyperX_location(loc):
    return HyperX_Location(int(loc.HBI_index / N_nodes_1dim), loc.HBI_index % N_nodes_1dim, loc.GPU_index)
    
#%% Latency Between 2 GPUs:

HBI_SIZE = 8
SWITCH_UP_LINKS = 64
SWITCH_DOWN_LINKS = 64
DRAGONFLY_PLUS_GROUP_LEAF_COUNT = 4
DRAGONFLY_PLUS_GROUP_SPINE_COUNT = 4

def Fat_Tree_latency(i_location, j_location):
    latency = 0
    if i_location == j_location:
        # Same GPU
        latency = 0
    elif i_location.HBI_index == j_location.HBI_index:
        # same HBI
        latency = 1 * HBI_link
    elif i_location.GPU_index == j_location.GPU_index and i_location.HBI_index // SWITCH_DOWN_LINKS == j_location.HBI_index // SWITCH_DOWN_LINKS:
        # different HBI but under the same Rail
        latency = 2 * T_link
    elif i_location.GPU_index != j_location.GPU_index and i_location.HBI_index // SWITCH_DOWN_LINKS == j_location.HBI_index // SWITCH_DOWN_LINKS:
        # non-relevant rail under the same spine: gpu1 - rail1 - spine1 - rail2 - gpu2
        latency = 4 * T_link
    else:
        # gpu1 - rail1 - spine1 - super spine1 - spine2 - rail2 - gpu2
        latency = 6 * T_link
        # print(f"GPU {i_location.HBI_index}, {i_location.GPU_index} to GPU {j_location.HBI_index}, {j_location.GPU_index}: {latency}")
    return latency


def Rail_Only_latency(i_location, j_location):
    if i_location.HBI_index != j_location.HBI_index and i_location.GPU_index != j_location.GPU_index:
        return None
    else:
        return Fat_Tree_latency(i_location, j_location)


def dragonFlyP_latency(i_location, j_location):
    latency = 0
    if i_location == j_location:
        # Same GPU
        latency = 0
    elif i_location.HBI_index == j_location.HBI_index:
        # Same HBI
        latency = 1 * HBI_link
    elif i_location.HBI_index // DRAGONFLY_PLUS_GROUP_LEAF_COUNT == j_location.HBI_index // DRAGONFLY_PLUS_GROUP_LEAF_COUNT:
        # Same group different HBI
        latency = 4 * T_link
    else:
        latency = 5 * T_link
    return latency


from typing import Tuple
from math import ceil, floor

def minimal_balanced_3d_dims(num_gpus: int, gpus_per_hbi: int = 8):
    """
    Find (Sx, Sy, Sz) such that:
      - Sx*Sy*Sz >= routers = ceil(num_gpus/gpus_per_hbi)
      - Volume is minimal
      - Among minimal volumes, dims are as balanced as possible (closest to cube)

    Returns dims sorted descending (Sx >= Sy >= Sz) for determinism.
    """
    R = ceil(num_gpus / gpus_per_hbi)
    if R <= 1:
        return (1, 1, 1)

    best = None  # (volume, imbalance, spread, dims)

    # Bound search:
    # Sz <= Sy <= Sx and Sz up to cube root, Sy up to sqrt(R/Sz)
    max_sz = ceil(R ** (1/3)) + 2  # small cushion
    for Sz in range(1, max_sz + 1):
        # minimal possible Sy for this Sz is 1
        # maximal Sy we need to consider is around sqrt(R/Sz)
        max_sy = ceil((R / Sz) ** 0.5) + 2
        for Sy in range(Sz, max_sy + 1):
            # minimal Sx needed to reach >= R
            Sx = ceil(R / (Sy * Sz))
            if Sx < Sy:
                Sx = Sy  # enforce Sx >= Sy

            V = Sx * Sy * Sz
            if V < R:
                continue

            dims = (Sx, Sy, Sz)
            imbalance = Sx - Sz
            spread = (Sx - Sy) ** 2 + (Sy - Sz) ** 2 + (Sx - Sz) ** 2

            cand = (V, imbalance, spread, dims)
            if best is None or cand < best:
                best = cand

        # Optional early break:
        # once Sz gets too large, volume will generally grow, but keeping it simple is fine.

    return best[3]

@dataclass(frozen=True)
class HyperXGPUPlacement:
    """
    Maps (HBI/router index, GPU index within that HBI) -> (x, y, z) router coords.

    Assumes:
      - Routers are indexed 0..(Sx*Sy*Sz - 1) in row-major order where z is fastest:
            idx = (x * Sy + y) * Sz + z
      - gpu_in_hbi is validated but not used for coords (coords are router-only).
    """
    dims: Tuple[int, int, int]  # (Sx, Sy, Sz)

    def router_coords(self, *, hbi_idx: int, gpu_in_hbi: int) -> Tuple[int, int, int]:
        sx, sy, sz = self.dims
        if sx <= 0 or sy <= 0 or sz <= 0:
            raise ValueError(f"dims must be positive, got {self.dims!r}")

        r = sx * sy * sz
        if not (0 <= hbi_idx < r):
            raise ValueError(f"hbi_idx out of range: {hbi_idx} not in [0, {r-1}]")

        # gpu_in_hbi is not used for coordinate calculation - coords are router-only
        # (validation removed since N_hbi varies per workload)

        # Row-major with z fastest: idx = (x*Sy + y)*Sz + z
        x = hbi_idx // (sy * sz)
        rem = hbi_idx % (sy * sz)
        y = rem // sz
        z = rem % sz
        return int(x), int(y), int(z)


def HyperX_latency_8_nodes_under_switch(i_location, j_location, num_gpus=128):
    # print(f"i_location: {i_location.HBI_index}, {i_location.GPU_index}")
    # print(f"j_location: {j_location.HBI_index}, {j_location.GPU_index}")
    hyperx_gpu_placement = HyperXGPUPlacement(dims=minimal_balanced_3d_dims(num_gpus=num_gpus))
    # print(f"HyperX GPU placement: {hyperx_gpu_placement.dims}")
    latency = 0
    if i_location == j_location:
        # Same GPU
        latency = 0
    elif i_location.HBI_index == j_location.HBI_index:
        # Same HBI
        latency = 1 * HBI_link
    else:
        i_x, i_y, i_z = hyperx_gpu_placement.router_coords(hbi_idx=i_location.HBI_index, gpu_in_hbi=i_location.GPU_index)
        j_x, j_y, j_z = hyperx_gpu_placement.router_coords(hbi_idx=j_location.HBI_index, gpu_in_hbi=j_location.GPU_index)
        latency = int(i_x != j_x) + int(i_y != j_y) + int(i_z != j_z) + 2
        # print(f"GPU i: ({i_x}, {i_y}, {i_z}) to GPU j: ({j_x}, {j_y}, {j_z}): {latency}")
    return latency

# def HyperX_latency_1_nodes_under_switch(i_location, j_location):
    
#     if (i_location == j_location):
#         return 0
#     i_HyperX_loc = location_to_HyperX_location(i_location)
#     j_HyperX_loc = location_to_HyperX_location(j_location)
#     result = 2
#     if i_HyperX_loc.GPU_index != j_HyperX_loc.GPU_index:
#         result = result+1
#     if i_HyperX_loc.HBI_index_1 != j_HyperX_loc.HBI_index_1:
#         result = result+1
#     if i_HyperX_loc.HBI_index_2 != j_HyperX_loc.HBI_index_2:
#         result = result+1
#     return result



#%% Topologies:

@dataclass
class Topology_Data:
    # topology_name: str
    latency_function: callable
    last_runtime: int

topologies_dict = {"fat tree": Topology_Data(Fat_Tree_latency, None),
                      "rail only": Topology_Data(Rail_Only_latency, None),
                    #   "HyperX_1": Topology_Data(HyperX_latency_1_nodes_under_switch, None),
                      "DragonFly+": Topology_Data(dragonFlyP_latency, None),
                      "HyperX_8": Topology_Data(HyperX_latency_8_nodes_under_switch, None)
                      }

TP_Runtimes_Table = []
DP_Runtimes_Table = []
PP_Runtimes_Table = []
Matrices_Runtimes_Table = []

# Create tables_dict with appropriate keys based on MOE mode
if args.moe_active:
    tables_dict = {"only_pp_moe": PP_Runtimes_Table,
                   "only_tp_moe": TP_Runtimes_Table,
                   "only_dp_moe": DP_Runtimes_Table,
                   "matrices_moe": Matrices_Runtimes_Table}
else:
    tables_dict = {"only_pp": PP_Runtimes_Table,
                   "only_tp": TP_Runtimes_Table,
                   "only_dp": DP_Runtimes_Table,
                   "matrices": Matrices_Runtimes_Table}

def update_Runtimes_Table(table, file_name):
    table.append({
        "file": file_name[:-4], # without ".csv"
        "fat tree": topologies_dict["fat tree"].last_runtime,
        "rail only": topologies_dict["rail only"].last_runtime,
        # "HyperX_1": topologies_dict["HyperX_1"].last_runtime,
        "DragonFly+": topologies_dict["DragonFly+"].last_runtime,
        "HyperX_8": topologies_dict["HyperX_8"].last_runtime
    })

# latency_functions = {"fat tree": Fat_Tree_latency,
                     # "rail only": Rail_Only_latency,
                     # "HyperX": HyperX_latency}

#%% Runtime of topologies:

def Topology_Runtime(topology, matrix, GPUs_num):
    result = 0
    for i in range (len(matrix)):
        for j in range (len(matrix)):
            i_location = index_to_location(i)
            j_location = index_to_location(j)
            if matrix[i][j] != 0:
                # print(f"i_location: {i_location.HBI_index}, {i_location.GPU_index}")
                if topology == "HyperX_8":
                    i_j_latency = topologies_dict[topology].latency_function(i_location, j_location, GPUs_num)
                else:
                    i_j_latency = topologies_dict[topology].latency_function(i_location, j_location)
                if i_j_latency == None:
                    return None
                result = result + i_j_latency * matrix[i][j]
    return result


if not args.heatmaps_only:
    for directory in dirs_list:
        files_list = sorted([f for f in directory.iterdir() if f.is_file() and '.csv' in os.path.basename(f)])
        for file_index in range(len(files_list)):
            print (f"martix {file_index} from {len(files_list)} in dir {directory.name}:")
            print (f"Transport matrix from the file: {os.path.basename(files_list[file_index])}")
            # save_heatmap(file_index) # heatmaps are already saved in simai.
            GPUs_num = len(get_matrix(file_index))
            # N_hbi = get_N_hbi(file_index)
            N_nodes_1dim = int((GPUs_num/N_hbi)**0.5)+1
            if N_hbi == None:
                print ("Error in get_N_hbi: The name of the file is not contaion \"_tp<i>_\"!")
                continue
            
            for topology in topologies_dict.keys():
                topologies_dict[topology].last_runtime = Topology_Runtime(topology, get_matrix(file_index), GPUs_num)
                print (topology, "runtime is: ", topologies_dict[topology].last_runtime)
                
            update_Runtimes_Table(tables_dict[directory.name], os.path.basename(files_list[file_index]))
            print()
    #%% arrange the data

    Runtime_dfs = {name: pd.DataFrame(table) for name, table in tables_dict.items()}
    # total_runtime_df = sum(Runtime_dfs.values())
    dfs_list = list(Runtime_dfs.values())
    total_runtime_df = dfs_list[0].copy()

    for df in dfs_list[1:]:
        for col in total_runtime_df.columns:
            if total_runtime_df[col].dtype != "object":  # רק עמודות מספריות
                total_runtime_df[col] += df[col]

    # Add Total with appropriate key
    if args.moe_active:
        Runtime_dfs["Total_moe"] = total_runtime_df
    else:
        Runtime_dfs["Total"] = total_runtime_df

    # for df in Runtime_dfs.values():
        # df = df.sort_values(by="fat tree")
        
    # Runtime_df = pd.DataFrame(Runtimes_Table)
    # Runtime_df = Runtime_df.sort_values(by="fat tree")
    #print (Runtime_df)

    #%% save the DataFrames

    save_dir = str(paths["dataframes"])
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n💾 Saving DataFrames to: {save_dir}/")
    for name, df in Runtime_dfs.items():
        # Remove _moe suffix from pickle filename for consistency
        clean_name = name.replace('_moe', '')
        df.to_pickle(f"{save_dir}/{clean_name}.pkl")
        print(f"   Saved: {clean_name}.pkl ({len(df)} workloads)")

    #%% read the DataFrames

    import pandas as pd

    loaded_dfs = {}
    for filename in os.listdir(save_dir):
        if filename.endswith(".pkl"):
            name = filename[:-4]  # without ".pkl"
            loaded_dfs[name] = pd.read_pickle(f"{save_dir}/{filename}")


    #%% Plot the results:
    # labels = Runtime_df["file"]
    # fat = Runtime_df["fat tree"]
    # rail = Runtime_df["rail only"]
    # hyperx = Runtime_df["HyperX"]

    # x_axis_loc = np.arange(len(labels))
    # width = 0.2

    # fig, ax = plt.subplots(figsize=(10, 6), dpi = 300)

    # rects1 = ax.bar(x_axis_loc - width, fat, width, label="Fat Tree")
    # rects2 = ax.bar(x_axis_loc, rail, width, label="Rail Only")
    # rects3 = ax.bar(x_axis_loc + width, hyperx, width, label="HyperX")

    # ax.set_ylabel("Runtime")
    # ax.set_xlabel("File Name")
    # ax.set_title("Runtime Comparison Across Topologies")
    # ax.set_xticks(x_axis_loc)
    # ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    plots_dir = str(paths["plots_overhead"])
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n📊 Generating plots in: {plots_dir}/")
    for name, df in Runtime_dfs.items():
        plt.figure(dpi = 500)
        # plt.plot(df["file"], df["fat tree"], "o", label="Fat Tree", color="#2ca02c", ms=7)
        # plt.plot(df["file"], df["rail only"], "o", label="Rail Only", color="gold", ms=5)
        # plt.plot(df["file"], df["DragonFly+"], "o", label="DragonFly+", color="#d62728", ms=3)
        plt.plot(df["file"], df["HyperX"], "o", label="HyperX", color="#1f77b4", ms=1)

        plt.xlabel("Transport Matrix")
        plt.ylabel("Overhead Communication")
        title = f"Overhead Communication of Topologie - {name}"
        if args.moe_active:
            title += " (MOE)"
        plt.title(title)
        plt.legend()
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.xticks([])

        safe_title = re.sub(r'[\\/*?:"<>|]', "_", plt.gca().get_title())
        filename = f"{plots_dir}/{safe_title}.png"
        # title = plt.gca().get_title()
        # filename = f"plots/Overhead_Communication_of_Topologie_{name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"   Saved: {safe_title}.png")

#%%

# different_rows1 = Runtime_df[Runtime_df["rail only"] != Runtime_df["fat tree"]]
# different_rows2 = Runtime_df[Runtime_df["DragonFly+"] != Runtime_df["HyperX"]]

#%% delete the files

# import shutil
# if os.path.exists(save_dir):
    # shutil.rmtree(save_dir)
    
#%%


GPUs_num = 128
N_hbi = 8
N_nodes_1dim = 4

dist_matrices = {}
statistic_dict = {}
for topo_name, topo_data in topologies_dict.items():
    latency_func = topo_data.latency_function
    matrix = np.zeros((GPUs_num, GPUs_num), dtype=float)
    statistic_dict[topo_name] = {}

    for i in range(GPUs_num):
        for j in range(GPUs_num):
            # HyperX_8 needs num_gpus parameter for 3D coordinate calculation
            if topo_name == "HyperX_8":
                val = latency_func(index_to_location(i), index_to_location(j), GPUs_num)
            else:
                val = latency_func(index_to_location(i), index_to_location(j))
            matrix[i, j] = val
            
            statistic_dict[topo_name][val] = statistic_dict[topo_name].get(val, 0) + 1

    dist_matrices[topo_name] = matrix

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import ListedColormap, BoundaryNorm

# נוודא שתיקיית הפלט קיימת
output_dir = str(paths["heatmaps"])
os.makedirs(output_dir, exist_ok=True)
print(f"\n🗺️  Generating heatmaps in: {output_dir}/")

for topo_name, matrix in dist_matrices.items():
    plt.figure(figsize=(10, 8))

    # Determine discrete values from matrix (matching assign_and_plot.py logic)
    valid_values = matrix[~np.isnan(matrix)]
    if valid_values.size > 0:
        min_dist = int(np.min(valid_values))
        max_dist = int(np.max(valid_values))
    else:
        min_dist, max_dist = 0, 5
    
    # Ensure at least 7 discrete color values
    min_colors = 7
    if max_dist - min_dist + 1 < min_colors:
        max_dist = min_dist + min_colors - 1
    discrete_values = list(range(min_dist, max_dist + 1))

    # Cold-to-warm color scale (same as assign_and_plot.py)
    cold_to_warm = [
        '#313695',  # deep navy (coldest)
        '#74add1',  # distinct mid-blue
        '#e0f3f8',  # pale icy blue
        '#fee090',  # pale warm yellow
        '#f46d43',  # vibrant orange-red
        '#a50026',  # deep crimson (warmest)
    ]
    
    # Select colors based on number of discrete values needed
    num_colors = len(discrete_values)
    if num_colors <= len(cold_to_warm):
        # Pick evenly spaced colors from our palette
        indices = np.linspace(0, len(cold_to_warm) - 1, num_colors).astype(int)
        colors = [cold_to_warm[i] for i in indices]
    else:
        # Fall back to coolwarm colormap for more colors
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_colors))
    cmap = ListedColormap(colors)

    # Boundaries for discrete colors
    bounds = np.arange(len(discrete_values) + 1) - 0.5 + min_dist
    norm = BoundaryNorm(bounds, cmap.N)

    sns.heatmap(
        matrix,
        cmap=cmap,
        norm=norm,                 # 👈 זה מה שעושה בדיד
        square=True,
        cbar=True,
        mask=np.isnan(matrix),
        vmin=min(discrete_values),
        vmax=max(discrete_values),
        linewidths=0,
        linecolor=None
        )
    cbar = plt.gca().collections[0].colorbar
    cbar.set_ticks(discrete_values)           # [0,1,2,3,4,5]
    cbar.set_ticklabels(discrete_values) 

    plt.title(f"{topo_name} – Latency Heatmap", fontsize=14)
    plt.xlabel("GPU j", fontsize=12)
    plt.ylabel("GPU i", fontsize=12)

    # שמירה לקובץ
    plt.tight_layout()
    filename = f"{output_dir}/{topo_name.replace(' ', '_')}_heatmap.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"✅ Saved heatmap for {topo_name} → {filename}")
    
    # Save as CSV with GPU indices
    csv_filename = f"{output_dir}/{topo_name.replace(' ', '_')}_heatmap.csv"
    gpu_labels = [f"GPU{i}" for i in range(GPUs_num)]
    df_heatmap = pd.DataFrame(matrix, index=gpu_labels, columns=gpu_labels)
    df_heatmap.to_csv(csv_filename)
    print(f"📄 Saved CSV for {topo_name} → {csv_filename}")    
    
for topo_name, statistic in statistic_dict.items():
    print(topo_name)
    print(statistic)