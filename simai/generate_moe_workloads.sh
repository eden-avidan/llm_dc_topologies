#!/bin/bash
# Generate MOE-enabled workloads for all existing MOE-False configurations
# This script reads existing workload filenames and generates corresponding MOE workloads

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AICB_DIR="${SCRIPT_DIR}/aicb"
WORKLOAD_DIR="${SCRIPT_DIR}/final_output/workload"
AICB_OUTPUT_DIR="${AICB_DIR}/results/workload"
FINAL_OUTPUT_DIR="${SCRIPT_DIR}/final_output/workload_moe"

# MOE parameters
NUM_EXPERTS=16
MOE_ROUTER_TOPK=2
GPU_TYPE="A100"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "MOE Workload Batch Generator"
echo "================================================"

# Check if AICB exists
if [ ! -d "$AICB_DIR" ]; then
    echo -e "${RED}❌ AICB directory not found: $AICB_DIR${NC}"
    echo "Please run: cd $SCRIPT_DIR && git clone https://github.com/aliyun/aicb.git"
    exit 1
fi

# Create output directory
mkdir -p "$FINAL_OUTPUT_DIR"
echo "Output directory: $FINAL_OUTPUT_DIR"
echo ""

cd "$AICB_DIR"

# Count total workloads
TOTAL=$(ls "$WORKLOAD_DIR"/*-MOE-False-*.txt 2>/dev/null | wc -l | tr -d ' ')
echo "Found $TOTAL MOE-False workloads to convert"
echo ""

# Function to extract value from filename
extract_param() {
    local filename=$1
    local param=$2
    echo "$filename" | grep -oP "(?<=${param})\d+" || echo "1"
}

# Function to extract model name
extract_model() {
    local filename=$1
    echo "$filename" | grep -oP "(gpt|llama)_\d+[A-Z]?"
}

# Function to get model size code
get_model_size() {
    local model=$1
    case "$model" in
        gpt_7B) echo "7" ;;
        gpt_13B) echo "13" ;;
        gpt_22B) echo "22" ;;
        gpt_175B) echo "175" ;;
        llama_65B) echo "65" ;;
        llama_405B) echo "405" ;;
        *) echo "" ;;
    esac
}

# Function to calculate EP size
calculate_ep() {
    local world_size=$1
    local tp=$2
    local pp=$3

    local dp_size=$((world_size / (tp * pp)))

    # Try EP values: 16, 8, 4, 2, 1
    for ep in 16 8 4 2 1; do
        if [ $((dp_size % ep)) -eq 0 ] && [ $((NUM_EXPERTS % ep)) -eq 0 ]; then
            echo "$ep"
            return
        fi
    done

    # Fallback
    echo "1"
}

# Parse command line arguments
DRY_RUN=false
LIMIT=""
FILTER_MODEL=""
FILTER_WS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --model)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --world-size)
            FILTER_WS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run         Print commands without executing"
            echo "  --limit N         Only generate first N workloads"
            echo "  --model MODEL     Only generate for specific model (e.g., gpt_7B)"
            echo "  --world-size WS   Only generate for specific world size"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Process workloads
COUNTER=0
SUCCESS=0
FAILED=0

for workload_file in "$WORKLOAD_DIR"/*-MOE-False-*.txt; do
    [ -f "$workload_file" ] || continue

    filename=$(basename "$workload_file")

    # Extract parameters
    model=$(extract_model "$filename")
    world_size=$(extract_param "$filename" "world_size")
    tp=$(extract_param "$filename" "tp")
    pp=$(extract_param "$filename" "pp")
    gbs=$(extract_param "$filename" "gbs")
    mbs=$(extract_param "$filename" "mbs")
    seq=$(extract_param "$filename" "seq")

    # Apply filters
    if [ -n "$FILTER_MODEL" ] && [ "$model" != "$FILTER_MODEL" ]; then
        continue
    fi

    if [ -n "$FILTER_WS" ] && [ "$world_size" != "$FILTER_WS" ]; then
        continue
    fi

    # Get model size
    model_size=$(get_model_size "$model")
    if [ -z "$model_size" ]; then
        echo -e "${YELLOW}⚠️  Unknown model: $model, skipping...${NC}"
        continue
    fi

    # Calculate EP
    ep=$(calculate_ep "$world_size" "$tp" "$pp")

    COUNTER=$((COUNTER + 1))

    # Check limit
    if [ -n "$LIMIT" ] && [ "$COUNTER" -gt "$LIMIT" ]; then
        break
    fi

    # Build command
    CMD="sh scripts/megatron_workload_with_aiob.sh -m ${model_size} \
        --world_size ${world_size} \
        --tensor_model_parallel_size ${tp} \
        --pipeline_model_parallel ${pp} \
        --sp \
        --ep ${ep} \
        --num_experts ${NUM_EXPERTS} \
        --moe_router_topk ${MOE_ROUTER_TOPK} \
        --moe_grouped_gemm \
        --moe_enable \
        --frame Megatron \
        --global_batch ${gbs} \
        --micro_batch ${mbs} \
        --seq_length ${seq} \
        --gpu_type ${GPU_TYPE}"

    echo ""
    echo "[$COUNTER/$TOTAL] Generating: ${model}-ws${world_size}-tp${tp}-pp${pp}-ep${ep}"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC} $CMD"
        SUCCESS=$((SUCCESS + 1))
    else
        if eval "$CMD" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Success${NC}"

            # Move generated file to final_output/workload_moe
            PATTERN="${model}-world_size${world_size}-tp${tp}-pp${pp}-ep${ep}-*-MOE-True-*.txt"
            GENERATED_FILE=$(ls -t "${AICB_OUTPUT_DIR}/"${PATTERN} 2>/dev/null | head -1)

            if [ -f "$GENERATED_FILE" ]; then
                mv "$GENERATED_FILE" "$FINAL_OUTPUT_DIR/"
                echo "   Moved to: $FINAL_OUTPUT_DIR/$(basename "$GENERATED_FILE")"
            fi

            SUCCESS=$((SUCCESS + 1))
        else
            echo -e "${RED}❌ Failed${NC}"
            FAILED=$((FAILED + 1))
        fi
    fi
done

# Summary
echo ""
echo "================================================"
echo "Summary:"
echo "  Successful: $SUCCESS"
echo "  Failed: $FAILED"
echo "  Total: $COUNTER"
echo "================================================"

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "✅ Workloads saved to: ${FINAL_OUTPUT_DIR}"
    echo ""
    echo "Generated files have format:"
    echo "  ${GPU_TYPE}-gpt_7B-world_size64-tp4-pp2-ep8-gbs2048-mbs1-seq2048-MOE-True-GEMM-True-flash_attn-False.txt"
fi
