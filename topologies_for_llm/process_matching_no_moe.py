#!/usr/bin/env python3
"""
Process non-MOE workloads that match the 123 MOE configurations.
This creates a fair comparison by processing the same configurations.
"""

import os
import re
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import numpy as np

# Import from Topologies Runtime.py
script_dir = Path(__file__).resolve().parent
simai_output = (script_dir / "../simai/final_output").resolve()

# Directories
moe_matrices_dir = simai_output / "matrices_moe"
non_moe_dirs = {
    "matrices": simai_output / "matrices",
    "only_tp": simai_output / "only_tp",
    "only_dp": simai_output / "only_dp",
    "only_pp": simai_output / "only_pp"
}

output_suffix = "_no_moe"

def parse_workload_config(filename):
    """Extract configuration from workload filename."""
    config = {}

    # Remove A100- prefix if present
    name = filename.replace("A100-", "").replace(".csv", "").replace("_tp_only", "").replace("_dp_only", "").replace("_pp_only", "")

    parts = name.split("-")

    for part in parts:
        if part.startswith("gpt_") or part.startswith("llama_"):
            config["model"] = part
        elif "world_size" in part:
            config["world_size"] = part.replace("world_size", "")
        elif part.startswith("tp") and "gbs" not in part:
            config["tp"] = part.replace("tp", "")
        elif part.startswith("pp") and "gbs" not in part:
            config["pp"] = part.replace("pp", "")
        elif part.startswith("gbs"):
            config["gbs"] = part.replace("gbs", "")
        elif part.startswith("mbs"):
            config["mbs"] = part.replace("mbs", "")
        elif part.startswith("seq"):
            config["seq"] = part.replace("seq", "")

    return config

def find_matching_non_moe_file(moe_config, non_moe_dir):
    """Find the non-MOE file that matches the MOE configuration."""
    # Build search pattern (ignoring ep and MOE/GEMM flags)
    pattern_parts = [
        moe_config.get("model", ""),
        f"world_size{moe_config.get('world_size', '')}",
        f"tp{moe_config.get('tp', '')}",
        f"pp{moe_config.get('pp', '')}",
        f"ep1",  # Non-MOE typically has ep1
        f"gbs{moe_config.get('gbs', '')}",
        f"mbs{moe_config.get('mbs', '')}",
        f"seq{moe_config.get('seq', '')}",
        "MOE-False"
    ]

    # Search for matching file
    for file in non_moe_dir.iterdir():
        if not file.is_file() or not file.name.endswith('.csv'):
            continue

        # Check if all key parts match
        filename = file.name
        if all(part in filename for part in pattern_parts[:8]):  # Check main config parts
            return file

    return None

print("="*70)
print("Processing Matching Non-MOE Configurations")
print("="*70)

# Get list of processed MOE workloads
moe_files = sorted(moe_matrices_dir.glob("*.csv"))
print(f"\nFound {len(moe_files)} MOE workloads")

# Find matching non-MOE files
matching_configs = []
for moe_file in moe_files:
    moe_config = parse_workload_config(moe_file.name)

    # Try to find matching non-MOE file
    non_moe_file = find_matching_non_moe_file(moe_config, non_moe_dirs["matrices"])

    if non_moe_file:
        matching_configs.append({
            "moe_file": moe_file.name,
            "non_moe_file": non_moe_file.name,
            "config": moe_config
        })
    else:
        print(f"⚠️  No match found for: {moe_file.name}")

print(f"\n✅ Found {len(matching_configs)} matching non-MOE configurations")
print(f"   (out of {len(moe_files)} MOE workloads)")

# Save the mapping
mapping_file = script_dir / "moe_to_non_moe_mapping.txt"
with open(mapping_file, 'w') as f:
    f.write("MOE File -> Non-MOE File\n")
    f.write("="*100 + "\n")
    for match in matching_configs:
        f.write(f"{match['moe_file']}\n")
        f.write(f"  -> {match['non_moe_file']}\n\n")

print(f"\n📄 Mapping saved to: {mapping_file}")

# Extract just the non-MOE filenames to process
non_moe_files_to_process = [m["non_moe_file"] for m in matching_configs]

print(f"\n💾 Creating list of files to process...")
list_file = script_dir / "non_moe_files_to_process.txt"
with open(list_file, 'w') as f:
    for filename in non_moe_files_to_process:
        f.write(filename + "\n")

print(f"   Saved to: {list_file}")

print("\n" + "="*70)
print("Next Steps:")
print("="*70)
print("Run the modified Topologies Runtime script to process only these files:")
print(f"  python3 'Topologies Runtime No MOE Subset.py'")
print("\nThis will generate:")
print(f"  - saved DataFrames{output_suffix}/")
print(f"  - plots - overhead communication{output_suffix}/")
print(f"  - heatmaps{output_suffix}/")
