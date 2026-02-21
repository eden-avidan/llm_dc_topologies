#!/usr/bin/env python3
"""
Generate MOE-enabled workloads matching all existing MOE-False configurations.
This script parses existing workload filenames and generates corresponding MOE-True workloads.
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
AICB_DIR = Path(__file__).parent / "aicb"
WORKLOAD_DIR = Path(__file__).parent / "final_output" / "workload"
AICB_OUTPUT_DIR = AICB_DIR / "results" / "workload"
FINAL_OUTPUT_DIR = Path(__file__).parent / "final_output" / "workload_moe"
GPU_TYPE = "A100"

# Model size mapping for -m parameter
MODEL_SIZE_MAP = {
    "gpt_7B": "7",
    "gpt_13B": "13",
    "gpt_22B": "22",
    "gpt_175B": "175",
    "llama_65B": "65",  # May need custom parameters
    "llama_405B": "405",  # May need custom parameters
}

# MOE parameters - adjust these as needed
MOE_PARAMS = {
    "num_experts": 16,
    "moe_router_topk": 2,
    "moe_grouped_gemm": True,
}

def parse_workload_filename(filename):
    """Parse a workload filename to extract configuration parameters."""
    # Remove .txt extension
    name = filename.replace(".txt", "")

    # Split by hyphens
    parts = name.split("-")

    config = {}

    for part in parts:
        if part.startswith("gpt_") or part.startswith("llama_"):
            config["model"] = part
        elif part.startswith("world_size"):
            config["world_size"] = int(part.replace("world_size", ""))
        elif part.startswith("tp") and "gbs" not in part and "seq" not in part:
            config["tp"] = int(part.replace("tp", ""))
        elif part.startswith("pp") and "gbs" not in part:
            config["pp"] = int(part.replace("pp", ""))
        elif part.startswith("ep") and "gbs" not in part:
            config["ep"] = int(part.replace("ep", ""))
        elif part.startswith("gbs"):
            config["gbs"] = int(part.replace("gbs", ""))
        elif part.startswith("mbs"):
            config["mbs"] = int(part.replace("mbs", ""))
        elif part.startswith("seq"):
            config["seq"] = int(part.replace("seq", ""))
        elif part.startswith("flash_attn"):
            config["flash_attn"] = part.replace("flash_attn-", "")

    return config

def calculate_ep_for_moe(world_size, tp, pp):
    """
    Calculate expert parallelism size.
    EP should divide evenly into: world_size / (tp * pp)
    """
    dp_size = world_size // (tp * pp)

    # Try EP values: 16, 8, 4, 2, 1
    for ep in [16, 8, 4, 2, 1]:
        if dp_size % ep == 0 and MOE_PARAMS["num_experts"] % ep == 0:
            return ep

    # Fallback to 1
    return 1

def generate_moe_workload(config, dry_run=False):
    """Generate a MOE-enabled workload using AICB."""

    # Get model size parameter
    model_size = MODEL_SIZE_MAP.get(config["model"])
    if not model_size:
        print(f"⚠️  Unknown model: {config['model']}, skipping...")
        return False

    # Calculate EP size
    ep = calculate_ep_for_moe(config["world_size"], config["tp"], config["pp"])

    # Build command
    cmd = [
        "sh",
        "scripts/megatron_workload_with_aiob.sh",
        "-m", model_size,
        "--world_size", str(config["world_size"]),
        "--tensor_model_parallel_size", str(config["tp"]),
        "--pipeline_model_parallel", str(config["pp"]),
        "--sp",  # Required for MOE + TP
        "--ep", str(ep),
        "--num_experts", str(MOE_PARAMS["num_experts"]),
        "--moe_router_topk", str(MOE_PARAMS["moe_router_topk"]),
        "--moe_enable",
        "--frame", "Megatron",
        "--global_batch", str(config["gbs"]),
        "--micro_batch", str(config["mbs"]),
        "--seq_length", str(config["seq"]),
        "--gpu_type", GPU_TYPE,
    ]

    # Add optional flags
    if MOE_PARAMS["moe_grouped_gemm"]:
        cmd.append("--moe_grouped_gemm")

    if config.get("flash_attn") == "True":
        cmd.append("--use_flash_attn")

    # Print command
    config_str = f"{config['model']}-ws{config['world_size']}-tp{config['tp']}-pp{config['pp']}-ep{ep}"
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Generating: {config_str}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        return True

    # Execute command
    try:
        result = subprocess.run(
            cmd,
            cwd=AICB_DIR,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per workload
        )

        if result.returncode == 0:
            print(f"✅ Successfully generated: {config_str}")

            # Move the generated file to final_output/workload_moe
            try:
                # Find the generated file (should be the most recent one)
                import glob
                import shutil

                pattern = f"{config['model']}-world_size{config['world_size']}-tp{config['tp']}-pp{config['pp']}-ep{ep}-*-MOE-True-*.txt"
                generated_files = glob.glob(str(AICB_OUTPUT_DIR / pattern))

                if generated_files:
                    src_file = generated_files[-1]  # Get most recent
                    filename = os.path.basename(src_file)
                    dst_file = FINAL_OUTPUT_DIR / filename

                    shutil.move(src_file, dst_file)
                    print(f"   Moved to: {dst_file}")

            except Exception as move_error:
                print(f"   ⚠️  Warning: Could not move file: {move_error}")

            return True
        else:
            print(f"❌ Failed to generate: {config_str}")
            print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"❌ Timeout generating: {config_str}")
        return False
    except Exception as e:
        print(f"❌ Error generating {config_str}: {e}")
        return False

def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate MOE workloads")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--limit", type=int, help="Limit number of workloads to generate (for testing)")
    parser.add_argument("--model", help="Only generate for specific model (e.g., gpt_7B)")
    parser.add_argument("--world-size", type=int, help="Only generate for specific world size")
    args = parser.parse_args()

    # Check AICB directory exists
    if not AICB_DIR.exists():
        print(f"❌ AICB directory not found: {AICB_DIR}")
        print("Please run: git clone https://github.com/aliyun/aicb.git")
        sys.exit(1)

    # Create output directory if it doesn't exist
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FINAL_OUTPUT_DIR}")

    # Get all existing workload files
    workload_files = sorted(WORKLOAD_DIR.glob("*-MOE-False-*.txt"))
    print(f"Found {len(workload_files)} MOE-False workloads to convert")

    # Parse configurations
    configs = []
    for wf in workload_files:
        config = parse_workload_filename(wf.name)

        # Apply filters
        if args.model and config.get("model") != args.model:
            continue
        if args.world_size and config.get("world_size") != args.world_size:
            continue

        configs.append(config)

    print(f"Will generate {len(configs)} MOE-enabled workloads")

    # Apply limit if specified
    if args.limit:
        configs = configs[:args.limit]
        print(f"Limited to first {len(configs)} workloads")

    if not args.dry_run:
        response = input("\nProceed with generation? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Generate workloads
    successful = 0
    failed = 0

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] ", end="")

        if generate_moe_workload(config, dry_run=args.dry_run):
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(configs)}")

    if not args.dry_run:
        print(f"\n✅ Workloads saved to: {FINAL_OUTPUT_DIR}")
        print(f"\nGenerated files have format:")
        print(f"  {GPU_TYPE}-gpt_7B-world_size64-tp4-pp2-ep8-gbs2048-mbs1-seq2048-MOE-True-GEMM-True-flash_attn-False.txt")

if __name__ == "__main__":
    main()
