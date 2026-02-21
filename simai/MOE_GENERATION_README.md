# MOE Workload Generation Guide

This guide explains how to generate MOE-enabled workloads matching your existing 184 MOE-False configurations.

## Prerequisites

1. **Clone AICB repository** (if not already done):
   ```bash
   cd /Users/eavidan/Documents/topology_repo/simai
   git clone https://github.com/aliyun/aicb.git
   ```

2. **Install dependencies** (if needed):
   ```bash
   cd aicb
   # Check if requirements.txt exists
   pip install -r requirements.txt  # if available
   ```

## Available Scripts

Two scripts are provided - choose based on your preference:

### Option 1: Python Script (Recommended)

**Script:** `generate_moe_workloads.py`

**Advantages:**
- Better error handling
- More detailed output
- Easier to customize MOE parameters

**Usage:**

```bash
cd /Users/eavidan/Documents/topology_repo/simai

# Dry run to see what will be generated
./generate_moe_workloads.py --dry-run

# Generate first 5 workloads (for testing)
./generate_moe_workloads.py --limit 5

# Generate only GPT-7B workloads
./generate_moe_workloads.py --model gpt_7B

# Generate only world_size=64 workloads
./generate_moe_workloads.py --world-size 64

# Generate ALL 184 MOE workloads
./generate_moe_workloads.py
```

### Option 2: Shell Script

**Script:** `generate_moe_workloads.sh`

**Advantages:**
- No Python dependencies
- Simpler to understand
- Easier to modify for shell users

**Usage:**

```bash
cd /Users/eavidan/Documents/topology_repo/simai

# Dry run to see what will be generated
./generate_moe_workloads.sh --dry-run

# Generate first 5 workloads (for testing)
./generate_moe_workloads.sh --limit 5

# Generate only GPT-7B workloads
./generate_moe_workloads.sh --model gpt_7B

# Generate ALL 184 MOE workloads
./generate_moe_workloads.sh
```

## MOE Configuration

Both scripts use these default MOE parameters:

```python
num_experts = 16          # Total number of experts
moe_router_topk = 2       # Experts activated per token
moe_grouped_gemm = True   # Optimize grouped operations
```

**Expert Parallelism (EP)** is automatically calculated based on:
- `EP = world_size / (TP * PP)`
- Must divide evenly into both DP size and num_experts
- Scripts try: 16, 8, 4, 2, 1 (in that order)

### Customizing MOE Parameters

**For Python script**, edit these lines:
```python
# Line 16-20 in generate_moe_workloads.py
MOE_PARAMS = {
    "num_experts": 16,        # Change this
    "moe_router_topk": 2,     # Change this
    "moe_grouped_gemm": True, # Change this
}
```

**For Shell script**, edit these lines:
```bash
# Line 12-13 in generate_moe_workloads.sh
NUM_EXPERTS=16        # Change this
MOE_ROUTER_TOPK=2     # Change this
```

## Output Location

Generated workloads will be saved to:
```
/Users/eavidan/Documents/topology_repo/simai/aicb/results/workload/
```

### Filename Format

Generated files will have this format:
```
None-gpt_7B-world_size64-tp4-pp2-ep8-gbs2048-mbs1-seq2048-MOE-True-GEMM-True-flash_attn-False.txt
```

**Note:** The `None-` prefix is from the missing GPU type parameter.

## Removing the "None-" Prefix

After generation, remove the prefix:

```bash
cd /Users/eavidan/Documents/topology_repo/simai/aicb/results/workload/

# Remove "None-" prefix from all files
for f in None-*.txt; do
    mv "$f" "${f#None-}"
done
```

Or specify GPU type during generation by modifying the scripts to add `--gpu_type A100`.

## Moving to final_output Directory

After generation and renaming:

```bash
# Copy to final_output/workload
cp /Users/eavidan/Documents/topology_repo/simai/aicb/results/workload/*-MOE-True-*.txt \
   /Users/eavidan/Documents/topology_repo/simai/final_output/workload/
```

## Expected Runtime

- **Per workload:** ~5-30 seconds (depending on size)
- **All 184 workloads:** ~15-90 minutes

The scripts include:
- Progress indicators
- Success/failure tracking
- Summary statistics
- Timeout handling (5 minutes per workload)

## Troubleshooting

### Issue: "Failed to import 'torch'"

This is normal - the workload generator works without PyTorch for non-AIOB mode.

### Issue: "AssertionError: moe must be enabled with sequence parallel"

The scripts already include `--sp` flag. If you see this, ensure you're using the provided scripts.

### Issue: Script can't find AICB

Make sure AICB is cloned in the correct location:
```bash
ls /Users/eavidan/Documents/topology_repo/simai/aicb/
```

### Issue: Permission denied

Make scripts executable:
```bash
chmod +x generate_moe_workloads.py
chmod +x generate_moe_workloads.sh
```

## Testing Before Full Generation

**Always test with a small batch first:**

```bash
# Test with just 3 workloads
./generate_moe_workloads.py --limit 3

# Verify output
ls -lh aicb/results/workload/
```

## Example: Complete Workflow

```bash
# 1. Navigate to SimAI directory
cd /Users/eavidan/Documents/topology_repo/simai

# 2. Ensure AICB is cloned
git clone https://github.com/aliyun/aicb.git

# 3. Test generation with dry run
./generate_moe_workloads.py --dry-run --limit 5

# 4. Generate test batch
./generate_moe_workloads.py --limit 3

# 5. Verify workload is valid
head -20 aicb/results/workload/None-gpt_*.txt

# 6. If good, generate all workloads
./generate_moe_workloads.py

# 7. Remove "None-" prefix
cd aicb/results/workload/
for f in None-*.txt; do mv "$f" "${f#None-}"; done

# 8. Copy to final_output
cd /Users/eavidan/Documents/topology_repo/simai
cp aicb/results/workload/*-MOE-True-*.txt final_output/workload/

# 9. Verify
ls -lh final_output/workload/ | grep "MOE-True" | wc -l
```

## Summary

- 📊 **184 configurations** will be converted
- ⚙️ **MOE parameters:** 16 experts, top-2 routing
- 🎯 **Expert Parallelism:** Auto-calculated per configuration
- ⏱️ **Estimated time:** 15-90 minutes for all workloads
- 📁 **Output:** `aicb/results/workload/`

**Recommendation:** Start with `--limit 5` to test before generating all 184 workloads!
