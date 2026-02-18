#!/bin/bash

# SimAI Simulation Runner Script
# Usage: ./run_simai_simulation.sh <topo> <num_gpus> <gpu_per_server>
# Example: ./run_simai_simulation.sh Spectrum-X 128 8

# Check if correct number of arguments provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <topo> <num_gpus> <gpu_per_server>"
    echo "Example: $0 Spectrum-X 128 8"
    echo "Supported topologies: Spectrum-X, HPN"
    exit 1
fi

# Parse arguments
TOPO=$1
NUM_GPUS=$2
GPU_PER_SERVER=$3

# Validate topology
if [ "$TOPO" != "Spectrum-X" ] && [ "$TOPO" != "HPN" ]; then
    echo "Error: Unsupported topology '$TOPO'. Supported topologies: Spectrum-X, HPN"
    exit 1
fi

# Set default values based on topology
if [ "$TOPO" = "Spectrum-X" ]; then
    GPU_TYPE="A100"
    BANDWIDTH="100Gbps"
    NV_BANDWIDTH="2400Gbps"
elif [ "$TOPO" = "HPN" ]; then
    GPU_TYPE="A100"
    BANDWIDTH="200Gbps"
    NV_BANDWIDTH="2400Gbps"
fi

echo "=== SimAI Simulation Configuration ==="
echo "Topology: $TOPO"
echo "Number of GPUs: $NUM_GPUS"
echo "GPU per Server: $GPU_PER_SERVER"
echo "GPU Type: $GPU_TYPE"
echo "Bandwidth: $BANDWIDTH"
echo "NVLink Bandwidth: $NV_BANDWIDTH"
echo "======================================"

# Step 1: Generate topology
echo "Step 1: Generating topology..."
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    -topo $TOPO \
    -g $NUM_GPUS \
    -gps $GPU_PER_SERVER \
    -gt $GPU_TYPE \
    -bw $BANDWIDTH \
    -nvbw $NV_BANDWIDTH

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate topology"
    exit 1
fi

# Get the generated topology file name
if [ "$TOPO" = "Spectrum-X" ]; then
    TOPO_FILE="${TOPO}_${NUM_GPUS}g_${GPU_PER_SERVER}gps_${BANDWIDTH}_${GPU_TYPE}"
else
    TOPO_FILE="Rail_Opti_SingleToR_${NUM_GPUS}g_${GPU_PER_SERVER}gps_${BANDWIDTH}_${GPU_TYPE}"
fi

echo "Generated topology file: $TOPO_FILE"

# Step 2: Create modified workload file
echo "Step 2: Creating modified workload file..."
WORKLOAD_SOURCE="./example/L7B-M1-C04_Llama7B_megatron_tp2_pp1_mbs1_A100_new_format.txt"
WORKLOAD_MODIFIED="./example/workload_modified.txt"

if [ ! -f "$WORKLOAD_SOURCE" ]; then
    echo "Error: Source workload file not found: $WORKLOAD_SOURCE"
    exit 1
fi

# Calculate new parameters based on input
TOTAL_GPUS=$NUM_GPUS
MODEL_PARALLEL_SIZE=$GPU_PER_SERVER
DATA_PARALLEL_SIZE=$((TOTAL_GPUS / MODEL_PARALLEL_SIZE))

# Create modified workload with updated header
{
    # Update the header line with new parameters
    echo "HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: $MODEL_PARALLEL_SIZE ep: 1 pp: 1 vpp: $MODEL_PARALLEL_SIZE ga: 1 all_gpus: $TOTAL_GPUS checkpoints: 0 checkpoint_initiates: 0"
    
    # Copy the rest of the file (skip the first line)
    tail -n +2 "$WORKLOAD_SOURCE"
} > "$WORKLOAD_MODIFIED"

echo "Created modified workload file: $WORKLOAD_MODIFIED"

# Step 3: Run SimAI simulation
echo "Step 3: Running SimAI simulation..."
AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator \
    -t 16 \
    -w "$WORKLOAD_MODIFIED" \
    -n "./$TOPO_FILE" \
    -c astra-sim-alibabacloud/inputs/config/SimAI.conf

if [ $? -ne 0 ]; then
    echo "Error: SimAI simulation failed"
    exit 1
fi

# Step 4: Download and rename transport matrix file
echo "Step 4: Downloading and renaming transport matrix file..."

# Create filename for the transport matrix
TRANSPORT_MATRIX_NAME="transport_matrix_${TOPO}_${NUM_GPUS}g_${GPU_PER_SERVER}gps_${GPU_TYPE}.csv"

# Download the file (for Google Colab environment)
if command -v wget >/dev/null 2>&1; then
    wget -O "$TRANSPORT_MATRIX_NAME" "https://raw.githubusercontent.com/your-repo/transport_matrix_Nitay.csv" 2>/dev/null
elif command -v curl >/dev/null 2>&1; then
    curl -o "$TRANSPORT_MATRIX_NAME" "https://raw.githubusercontent.com/your-repo/transport_matrix_Nitay.csv" 2>/dev/null
else
    echo "Warning: Neither wget nor curl found. Please manually download transport_matrix_Nitay.csv and rename it to: $TRANSPORT_MATRIX_NAME"
fi

# Check if download was successful
if [ -f "$TRANSPORT_MATRIX_NAME" ]; then
    echo "Successfully downloaded and renamed transport matrix to: $TRANSPORT_MATRIX_NAME"
else
    echo "Note: Transport matrix file not found. Please manually download transport_matrix_Nitay.csv and rename it to: $TRANSPORT_MATRIX_NAME"
fi

echo "=== Simulation completed successfully ==="
echo "Generated files:"
echo "- Topology: $TOPO_FILE"
echo "- Modified workload: $WORKLOAD_MODIFIED"
echo "- Transport matrix: $TRANSPORT_MATRIX_NAME"