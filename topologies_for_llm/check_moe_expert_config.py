#!/usr/bin/env python3
"""
Quick utility to check expert configuration for MOE workloads.
Usage: python3 check_moe_expert_config.py [workload_filename]
"""

import sys
import re
from pathlib import Path

def analyze_moe_config(filename):
    """Parse MOE workload filename and show expert distribution."""

    # Extract parameters
    match_ws = re.search(r'world_size(\d+)', filename)
    match_tp = re.search(r'-tp(\d+)-', filename)
    match_pp = re.search(r'-pp(\d+)-', filename)
    match_ep = re.search(r'-ep(\d+)-', filename)
    match_model = re.search(r'(gpt_\d+B|llama_\d+B)', filename)

    if not all([match_ws, match_tp, match_pp, match_ep, match_model]):
        print(f"Error: Could not parse configuration from filename: {filename}")
        return

    world_size = int(match_ws.group(1))
    tp = int(match_tp.group(1))
    pp = int(match_pp.group(1))
    ep = int(match_ep.group(1))
    model = match_model.group(1)

    # MOE parameters (from generate_moe_workloads.py)
    num_experts = 16
    top_k = 2

    # Calculate derived values
    dp = world_size // (tp * pp * ep)
    experts_per_ep_rank = num_experts // ep

    # Print analysis
    print("="*80)
    print(f"MOE WORKLOAD CONFIGURATION: {filename}")
    print("="*80)
    print(f"\n📊 MODEL & SCALE")
    print(f"  Model:              {model}")
    print(f"  Total GPUs:         {world_size}")

    print(f"\n🔀 PARALLELISM STRATEGY")
    print(f"  Tensor Parallel:    {tp:>4} (TP)")
    print(f"  Pipeline Parallel:  {pp:>4} (PP)")
    print(f"  Expert Parallel:    {ep:>4} (EP)")
    print(f"  Data Parallel:      {dp:>4} (DP = {world_size}/{tp}×{pp}×{ep})")

    print(f"\n🧠 EXPERT CONFIGURATION")
    print(f"  Total Experts:      {num_experts}")
    print(f"  Router Top-K:       {top_k} (each token → {top_k} experts)")
    print(f"  Experts per EP:     {experts_per_ep_rank} expert(s) per EP rank")

    print(f"\n📦 DISTRIBUTION")
    print(f"  {num_experts} experts distributed across {ep} EP ranks")
    print(f"  → Each EP rank handles {experts_per_ep_rank} expert(s)")

    if dp > 1:
        print(f"  → {dp} complete model replicas (data parallel)")
        print(f"  → Each replica has its own set of {num_experts} experts")
    else:
        print(f"  → Single model replica (no data parallelism)")

    print(f"\n💡 INTERPRETATION")
    if ep == 16:
        print(f"  • Maximum expert distribution (1 expert per EP rank)")
        print(f"  • More ALLTOALL communication between experts")
    elif ep == 2:
        print(f"  • Minimum expert distribution (8 experts per EP rank)")
        print(f"  • Less communication, more computation per rank")
    else:
        print(f"  • Medium expert distribution ({experts_per_ep_rank} experts per EP rank)")

    if tp > 1:
        print(f"  • Each expert is sharded across {tp} GPUs (tensor parallel)")

    if pp > 1:
        print(f"  • Model split into {pp} pipeline stages")

    print("\n" + "="*80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Analyze specific file
        filename = sys.argv[1]
        analyze_moe_config(filename)
    else:
        # Show example
        print("Usage: python3 check_moe_expert_config.py <workload_filename>")
        print("\nExample:")
        example = "A100-gpt_13B-world_size1024-tp16-pp2-ep16-gbs2048-mbs1-seq4096-MOE-True-GEMM-True-flash_attn-False.txt"
        analyze_moe_config(example)
