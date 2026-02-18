#!/bin/bash

# Script to demonstrate transport matrix extraction from SimAI simulation
# This script shows how to run a simulation with transport matrix extraction enabled

echo "SimAI Transport Matrix Extraction Demo"
echo "====================================="

# Check if the simulator binary exists
if [ ! -f "./bin/SimAI_simulator" ]; then
    echo "Error: SimAI_simulator binary not found. Please build the project first."
    echo "Run: ./scripts/build.sh ns3"
    exit 1
fi

# Example topology and config files (adjust paths as needed)
TOPOLOGY_FILE="inputs/topo/example_topo.txt"
CONFIG_FILE="inputs/config/SimAI.conf"

# Check if example files exist
if [ ! -f "$TOPOLOGY_FILE" ]; then
    echo "Warning: Topology file $TOPOLOGY_FILE not found. Using default."
    TOPOLOGY_FILE=""
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file $CONFIG_FILE not found. Using default."
    CONFIG_FILE=""
fi

echo "Running SimAI simulation with transport matrix extraction enabled..."
echo "Command: ./bin/SimAI_simulator -n $TOPOLOGY_FILE -c $CONFIG_FILE -e"

# Run the simulation with transport matrix extraction
if [ -n "$TOPOLOGY_FILE" ] && [ -n "$CONFIG_FILE" ]; then
    ./bin/SimAI_simulator -n "$TOPOLOGY_FILE" -c "$CONFIG_FILE" -e
else
    ./bin/SimAI_simulator -e
fi

echo ""
echo "Transport matrix extraction completed!"
echo "Generated files:"
echo "  - transport_matrix_paths.txt: Routing paths between nodes"
echo "  - transport_matrix_bandwidth.txt: Link bandwidths"
echo "  - transport_matrix_delay.txt: Link delays"
echo "  - transport_matrix_rtt.txt: End-to-end RTTs"
echo "  - transport_matrix_topology.txt: Physical topology"
echo "  - transport_matrix_summary.txt: Summary information"
echo ""
echo "You can now analyze these files to understand the network transport matrix." 