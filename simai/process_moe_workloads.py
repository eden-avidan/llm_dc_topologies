#!/usr/bin/env python3
"""
Process all MOE workloads to generate matrices and analysis outputs.
Generates: full matrices, TP-only, DP-only, PP-only matrices, and heatmaps.
"""

import os
import sys
import subprocess
from pathlib import Path
import csv
import shutil

# Configuration
SCRIPT_DIR = Path(__file__).parent
WORKLOAD_MOE_DIR = SCRIPT_DIR / "final_output" / "workload_moe"
OUTPUT_BASE_DIR = SCRIPT_DIR / "final_output"
ANALYZER_SCRIPT = SCRIPT_DIR / "workload_analyzer_my_version.py"

# Output directories
OUTPUT_DIRS = {
    "matrices": OUTPUT_BASE_DIR / "matrices_moe",
    "only_tp": OUTPUT_BASE_DIR / "only_tp_moe",
    "only_dp": OUTPUT_BASE_DIR / "only_dp_moe",
    "only_pp": OUTPUT_BASE_DIR / "only_pp_moe",
    "heatmaps": OUTPUT_BASE_DIR / "heatmaps_moe",
}

def create_output_directories():
    """Create all necessary output directories."""
    for dir_path in OUTPUT_DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def parse_config_from_filename(filename):
    """Extract configuration from filename."""
    config = {}
    parts = filename.replace(".txt", "").split("-")

    for part in parts:
        if "world_size" in part:
            config["world_size"] = int(part.replace("world_size", ""))
        elif part.startswith("tp") and "gbs" not in part:
            config["tp"] = int(part.replace("tp", ""))
        elif part.startswith("pp") and "gbs" not in part:
            config["pp"] = int(part.replace("pp", ""))
        elif part.startswith("ep") and "gbs" not in part:
            config["ep"] = int(part.replace("ep", ""))

    return config

def get_base_name(workload_filename):
    """Get base name without A100- prefix and .txt extension."""
    base = workload_filename.replace("A100-", "").replace(".txt", "")
    return base

def generate_full_matrix(workload_file, output_file):
    """Generate full communication matrix using the analyzer."""
    cmd = [
        "python3",
        str(ANALYZER_SCRIPT),
        str(workload_file),
        "--matrix",
        "--output", str(output_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0 and output_file.exists():
            return True
        else:
            print(f"    Error: {result.stderr[:200]}")
            return False

    except Exception as e:
        print(f"    Exception: {e}")
        return False

def load_matrix_from_csv(csv_file):
    """Load matrix from CSV file."""
    matrix = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            matrix.append([int(val) for val in row[1:]])  # Skip GPU column
    return matrix

def save_matrix_to_csv(matrix, csv_file, n_gpus):
    """Save matrix to CSV file."""
    with open(csv_file, 'w') as f:
        # Write header
        f.write("GPU")
        for j in range(n_gpus):
            f.write(f",GPU{j}")
        f.write("\n")

        # Write data rows
        for i in range(n_gpus):
            f.write(f"GPU{i}")
            for j in range(n_gpus):
                f.write(f",{matrix[i][j]}")
            f.write("\n")

def extract_tp_only_matrix(full_matrix, config):
    """Extract TP-only communication from full matrix."""
    n_gpus = config["world_size"]
    tp_size = config["tp"]
    pp_size = config["pp"]
    dp_size = n_gpus // (tp_size * pp_size)

    tp_matrix = [[0] * n_gpus for _ in range(n_gpus)]

    # TP groups are consecutive GPUs within each DP rank of each PP stage
    for pp_stage in range(pp_size):
        for dp_rank in range(dp_size):
            # Start of this TP group
            group_start = pp_stage * (tp_size * dp_size) + dp_rank * tp_size

            # Copy TP communication within this group
            for i in range(tp_size):
                for j in range(tp_size):
                    src = group_start + i
                    dst = group_start + j
                    if src < n_gpus and dst < n_gpus:
                        tp_matrix[src][dst] = full_matrix[src][dst]

    return tp_matrix

def extract_dp_only_matrix(full_matrix, config):
    """Extract DP-only communication from full matrix."""
    n_gpus = config["world_size"]
    tp_size = config["tp"]
    pp_size = config["pp"]
    dp_size = n_gpus // (tp_size * pp_size)

    dp_matrix = [[0] * n_gpus for _ in range(n_gpus)]

    # DP groups span across TP groups within same PP stage
    for pp_stage in range(pp_size):
        for tp_idx in range(tp_size):
            # GPUs at same TP position across different DP ranks
            for dp_i in range(dp_size):
                for dp_j in range(dp_size):
                    if dp_i != dp_j:
                        src = pp_stage * (tp_size * dp_size) + dp_i * tp_size + tp_idx
                        dst = pp_stage * (tp_size * dp_size) + dp_j * tp_size + tp_idx
                        if src < n_gpus and dst < n_gpus:
                            dp_matrix[src][dst] = full_matrix[src][dst]

    return dp_matrix

def extract_pp_only_matrix(full_matrix, config):
    """Extract PP-only communication from full matrix."""
    n_gpus = config["world_size"]
    tp_size = config["tp"]
    pp_size = config["pp"]
    dp_size = n_gpus // (tp_size * pp_size)

    pp_matrix = [[0] * n_gpus for _ in range(n_gpus)]

    gpus_per_stage = tp_size * dp_size

    # PP communication is between adjacent stages at same TP/DP position
    for pp_i in range(pp_size):
        for pp_j in range(pp_size):
            if abs(pp_i - pp_j) == 1:  # Adjacent stages
                stage_i_start = pp_i * gpus_per_stage
                stage_j_start = pp_j * gpus_per_stage

                # Copy communication between all GPUs at same position
                for offset in range(gpus_per_stage):
                    src = stage_i_start + offset
                    dst = stage_j_start + offset
                    if src < n_gpus and dst < n_gpus:
                        pp_matrix[src][dst] = full_matrix[src][dst]

    return pp_matrix

def generate_heatmap(csv_file, output_png):
    """Generate heatmap from CSV matrix."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np

        # Load matrix
        matrix = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                matrix.append([int(val) for val in row[1:]])

        matrix = np.array(matrix)

        # Create heatmap
        plt.figure(figsize=(12, 10), dpi=150)

        # Mask zeros for better visualization
        masked_matrix = np.ma.masked_where(matrix == 0, matrix)

        plt.imshow(masked_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Data Transfer (bytes)')
        plt.title(f'GPU Communication Matrix\n{csv_file.stem}')
        plt.xlabel('Destination GPU')
        plt.ylabel('Source GPU')

        plt.tight_layout()
        plt.savefig(output_png, dpi=150, bbox_inches='tight')
        plt.close()

        return True

    except ImportError:
        print("    Warning: matplotlib not available, skipping heatmap")
        return False
    except Exception as e:
        print(f"    Heatmap error: {e}")
        return False

def process_workload(workload_file):
    """Process a single workload file."""
    filename = workload_file.name
    base_name = get_base_name(filename)

    print(f"\n  Processing: {filename}")

    # Parse configuration
    config = parse_config_from_filename(filename)
    if not config.get("world_size"):
        print("    ⚠️  Could not parse configuration, skipping")
        return False

    # 1. Generate full matrix
    full_matrix_file = OUTPUT_DIRS["matrices"] / f"{base_name}.csv"
    print(f"    [1/5] Generating full matrix...")
    if not generate_full_matrix(workload_file, full_matrix_file):
        print("    ❌ Failed to generate full matrix")
        return False
    print(f"    ✅ Full matrix: {full_matrix_file.name}")

    # 2. Load full matrix
    try:
        full_matrix = load_matrix_from_csv(full_matrix_file)
    except Exception as e:
        print(f"    ❌ Failed to load matrix: {e}")
        return False

    # 3. Extract TP-only matrix
    print(f"    [2/5] Extracting TP-only matrix...")
    tp_matrix = extract_tp_only_matrix(full_matrix, config)
    tp_matrix_file = OUTPUT_DIRS["only_tp"] / f"{base_name}_tp_only.csv"
    save_matrix_to_csv(tp_matrix, tp_matrix_file, config["world_size"])
    print(f"    ✅ TP-only: {tp_matrix_file.name}")

    # 4. Extract DP-only matrix
    print(f"    [3/5] Extracting DP-only matrix...")
    dp_matrix = extract_dp_only_matrix(full_matrix, config)
    dp_matrix_file = OUTPUT_DIRS["only_dp"] / f"{base_name}_dp_only.csv"
    save_matrix_to_csv(dp_matrix, dp_matrix_file, config["world_size"])
    print(f"    ✅ DP-only: {dp_matrix_file.name}")

    # 5. Extract PP-only matrix
    print(f"    [4/5] Extracting PP-only matrix...")
    pp_matrix = extract_pp_only_matrix(full_matrix, config)
    pp_matrix_file = OUTPUT_DIRS["only_pp"] / f"{base_name}_pp_only.csv"
    save_matrix_to_csv(pp_matrix, pp_matrix_file, config["world_size"])
    print(f"    ✅ PP-only: {pp_matrix_file.name}")

    # 6. Generate heatmap
    print(f"    [5/5] Generating heatmap...")
    heatmap_file = OUTPUT_DIRS["heatmaps"] / f"{base_name}_heatmap.png"
    if generate_heatmap(full_matrix_file, heatmap_file):
        print(f"    ✅ Heatmap: {heatmap_file.name}")

    return True

def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Process MOE workloads to generate matrices")
    parser.add_argument("--limit", type=int, help="Limit number of workloads to process")
    parser.add_argument("--skip-heatmaps", action="store_true", help="Skip heatmap generation")
    args = parser.parse_args()

    print("=" * 70)
    print("MOE Workload Matrix Generator")
    print("=" * 70)

    # Check analyzer script exists
    if not ANALYZER_SCRIPT.exists():
        print(f"❌ Analyzer script not found: {ANALYZER_SCRIPT}")
        sys.exit(1)

    # Create output directories
    print("\nCreating output directories...")
    create_output_directories()

    # Get all MOE workload files
    workload_files = sorted(WORKLOAD_MOE_DIR.glob("*.txt"))
    if not workload_files:
        print(f"\n❌ No workload files found in: {WORKLOAD_MOE_DIR}")
        sys.exit(1)

    print(f"\nFound {len(workload_files)} MOE workload files")

    # Apply limit if specified
    if args.limit:
        workload_files = workload_files[:args.limit]
        print(f"Limited to first {len(workload_files)} workloads")

    # Process each workload
    print("\n" + "=" * 70)
    print("Processing Workloads")
    print("=" * 70)

    successful = 0
    failed = 0

    for i, workload_file in enumerate(workload_files, 1):
        print(f"\n[{i}/{len(workload_files)}]", end="")

        if process_workload(workload_file):
            successful += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(workload_files)}")

    print(f"\n📁 Output directories:")
    for name, path in OUTPUT_DIRS.items():
        count = len(list(path.glob("*")))
        print(f"  {name:12} → {count:3} files in {path}")

    print("\n✅ Processing complete!")

if __name__ == "__main__":
    main()
