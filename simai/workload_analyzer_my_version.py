#!/usr/bin/env python3
"""
Workload Analyzer - Compute GPU-to-GPU Data Transfer
Analyzes SimAI workload files and calculates total data transfer between GPUs
based on collective communication patterns and parallelism configuration.
"""

import sys
import argparse
from typing import Dict, List, Tuple
import math

class WorkloadAnalyzer:
    def __init__(self, workload_file: str):
        self.workload_file = workload_file
        self.layers = []
        self.config = {}
        self.parse_workload()
        self.matrix = [[0 for _ in range(self.config['total_gpus'])] for _ in range(self.config['total_gpus'])]
        
    def parse_workload(self):
        """Parse the workload file and extract configuration and layers."""
        with open(self.workload_file, 'r') as f:
            lines = f.readlines()
        
        # Parse header line
        header = lines[0].strip()
        self.parse_header(header)
        
        # Parse number of layers
        num_layers = int(lines[1].strip())
        
        # Parse each layer
        for i in range(2, 2 + num_layers):
            layer = self.parse_layer(lines[i].strip())
            self.layers.append(layer)
    
    def parse_header(self, header: str):
        """Parse the workload header to extract parallelism configuration."""
        parts = header.split()
        
        # Initialize defaults
        self.config['pp_comm_size'] = 0
        self.config['vpp'] = 1
        self.config['ga'] = 1
        self.config['tp_size'] = 1
        self.config['ep_size'] = 1
        self.config['pp_size'] = 1
        self.config['total_gpus'] = 1
        
        # Extract key parameters
        for i, part in enumerate(parts):
            if "model_parallel_NPU_group:" in part and i + 1 < len(parts):
                self.config['tp_size'] = int(parts[i + 1])
            elif part == "ep:" and i + 1 < len(parts):
                self.config['ep_size'] = int(parts[i + 1])
            elif part == "pp:" and i + 1 < len(parts):
                self.config['pp_size'] = int(parts[i + 1])
            elif part == "vpp:" and i + 1 < len(parts):
                self.config['vpp'] = int(parts[i + 1])
            elif part == "ga:" and i + 1 < len(parts):
                self.config['ga'] = int(parts[i + 1])
            elif part == "all_gpus:" and i + 1 < len(parts):
                self.config['total_gpus'] = int(parts[i + 1])
            elif part == "pp_comm:" and i + 1 < len(parts):
                self.config['pp_comm_size'] = int(float(parts[i + 1]))
        
        
        # Calculate derived parameters
        # DP size = total_gpus / (tp_size * pp_size)
        # Each DP group spans across all pipeline stages
        self.config['dp_size'] = self.config['total_gpus'] // (self.config['tp_size'] * self.config['pp_size'])
        
        print(f"Configuration:")
        print(f"  Total GPUs: {self.config['total_gpus']}")
        print(f"  TP Size: {self.config['tp_size']}")
        print(f"  DP Size: {self.config['dp_size']}")
        print(f"  EP Size: {self.config['ep_size']}")
        print(f"  PP Size: {self.config['pp_size']}")
    
    def parse_layer(self, line: str) -> Dict:
        """Parse a single layer line into a structured format."""
        parts = line.split()
        
        layer = {
            'name': parts[0],
            'dependency': int(parts[1]),
            'fwd_compute': int(parts[2]),
            'fwd_comm_type': parts[3],
            'fwd_comm_size': int(parts[4]),
            'fwd_enabled': int(parts[5]),
            'bwd_input_comm_type': parts[6],
            'bwd_input_size': int(parts[7]),
            'bwd_input_enabled': int(parts[8]),
            'bwd_weight_comm_type': parts[9],
            'bwd_weight_size': int(parts[10]),
            'layer_id': int(parts[11])
        }
        
        return layer
    
    def tp_pair(self, gpu_id: int) -> int:
        candidate = gpu_id + 1
        if self.get_tp_group(candidate) !=  self.get_tp_group(gpu_id):
            candidate = candidate - self.config['tp_size']
        return candidate
    
    def dp_pair(self, gpu_id: int) -> int:
        candidate = gpu_id + self.config['tp_size']
        gpus_in_dp = self.config['tp_size'] * self.config['dp_size']
        if self.get_dp_group(candidate) != self.get_dp_group(gpu_id):
            candidate = candidate - gpus_in_dp
        return candidate

    def pp_pair(self, gpu_id: int, is_forward: bool) -> int:
        pp_stage_size = self.config['tp_size'] * self.config['dp_size']
        if is_forward:
            candidate = gpu_id + pp_stage_size
            if candidate >= self.config['total_gpus']:
                candidate = candidate - self.config['total_gpus']
        else:
            candidate = gpu_id - pp_stage_size
            if candidate < 0:
                candidate = candidate + self.config['total_gpus']
        return candidate
    
    def get_tp_group(self, gpu_id: int) -> int:
        """Get the tensor parallel group ID for a given GPU."""
        return gpu_id // self.config['tp_size']
    
    def get_dp_group(self, gpu_id: int) -> int:
        """Get the data parallel group ID for a given GPU."""
        return gpu_id // (self.config['dp_size']*self.config['tp_size'])
    
    def get_tp_group_members(self, gpu_id: int) -> List[int]:
        """Get all GPUs in the same tensor parallel group."""
        tp_size = self.config['tp_size']
        pp_size = self.config['pp_size']
        dp_size = self.config['dp_size']
        
        # Get which pipeline stage this GPU is in
        gpus_per_pp_stage = tp_size * dp_size
        pp_stage = gpu_id // gpus_per_pp_stage
        
        # Get position within this pipeline stage
        local_gpu_id = gpu_id % gpus_per_pp_stage
        
        # Which TP group within this stage (0 to tp_size-1)
        tp_group_in_stage = local_gpu_id // dp_size
        
        # TP group members are consecutive GPUs in this TP group
        tp_group_start = pp_stage * gpus_per_pp_stage + tp_group_in_stage * dp_size
        return list(range(tp_group_start, tp_group_start + dp_size))
    
    def get_dp_group_members(self, gpu_id: int) -> List[int]:
        """Get all GPUs in the same data parallel group (same DP position within same PP stage)."""
        tp_size = self.config['tp_size']
        pp_size = self.config['pp_size']
        dp_size = self.config['dp_size']
        
        # Get which pipeline stage this GPU is in
        gpus_per_pp_stage = tp_size * dp_size
        pp_stage = gpu_id // gpus_per_pp_stage
        
        # Get DP position within this pipeline stage
        local_gpu_id = gpu_id % gpus_per_pp_stage
        dp_position = local_gpu_id % dp_size
        
        # DP group includes all GPUs with same DP position within SAME pipeline stage
        members = []
        for tp_group in range(tp_size):
            member_gpu = pp_stage * gpus_per_pp_stage + tp_group * dp_size + dp_position
            members.append(member_gpu)
        return members
    
    def get_pp_group(self, gpu_id: int) -> int:
        """Get the pipeline parallel stage ID for a given GPU."""
        tp_size = self.config['tp_size']
        pp_size = self.config['pp_size']
        dp_size = self.config['dp_size']
        
        # Each pipeline stage contains tp_size * dp_size GPUs
        gpus_per_pp_stage = tp_size * dp_size
        return gpu_id // gpus_per_pp_stage
    
    def get_pp_stage_members(self, gpu_id: int) -> List[int]:
        """Get all GPUs in the same pipeline stage."""
        pp_stage = self.get_pp_group(gpu_id)
        tp_size = self.config['tp_size']
        dp_size = self.config['dp_size']
        gpus_per_pp_stage = tp_size * dp_size
        start = pp_stage * gpus_per_pp_stage
        return list(range(start, start + gpus_per_pp_stage))
    
    def get_pp_neighbors(self, gpu_id: int) -> Tuple[List[int], List[int]]:
        """Get previous and next pipeline stage GPUs for a given GPU."""
        pp_stage = self.get_pp_group(gpu_id)
        pp_size = self.config['pp_size']
        tp_size = self.config['tp_size']
        dp_size = self.config['dp_size']
        gpus_per_pp_stage = tp_size * dp_size
        
        prev_stage_gpus = []
        next_stage_gpus = []
        
        # Previous stage (for backward pass)
        if pp_stage > 0:
            prev_start = (pp_stage - 1) * gpus_per_pp_stage
            prev_stage_gpus = list(range(prev_start, prev_start + gpus_per_pp_stage))
        
        # Next stage (for forward pass)
        if pp_stage < pp_size - 1:
            next_start = (pp_stage + 1) * gpus_per_pp_stage
            next_stage_gpus = list(range(next_start, next_start + gpus_per_pp_stage))
        
        return prev_stage_gpus, next_stage_gpus
    
    def calculate_ring_allreduce_transfer(self, src: int, dst: int, message_size: int, group_members: List[int]) -> int:
        """Calculate data transfer between two GPUs in Ring AllReduce."""
        if src not in group_members or dst not in group_members:
            return 0  # No communication outside the group
        
        if src == dst:
            return 0  # No self-transfer
        
        n = len(group_members)
        
        # Ring AllReduce transfer factor: 2 * (n-1)/n
        transfer_factor = 2.0 * (n - 1) / n
        
        # Check if src and dst are adjacent in the ring
        src_idx = group_members.index(src)
        dst_idx = group_members.index(dst)
        
        # In ring topology, each GPU only sends to its next neighbor
        if (dst_idx - src_idx) % n == 1:  # dst is next neighbor of src
            return int(message_size * transfer_factor)
        else:
            return 0  # No direct transfer in ring
    
    def calculate_reduce_scatter_transfer(self, src: int, dst: int, message_size: int, group_members: List[int]) -> int:
        """Calculate data transfer between two GPUs in ReduceScatter."""
        if src not in group_members or dst not in group_members:
            return 0
        
        if src == dst:
            return 0
        
        n = len(group_members)
        transfer_factor = (n - 1) / n
        
        src_idx = group_members.index(src)
        dst_idx = group_members.index(dst)
        
        if (dst_idx - src_idx) % n == 1:
            return int(message_size * transfer_factor)
        else:
            return 0
    
    def calculate_all_gather_transfer(self, src: int, dst: int, message_size: int, group_members: List[int]) -> int:
        """Calculate data transfer between two GPUs in AllGather."""
        if src not in group_members or dst not in group_members:
            return 0
        
        if src == dst:
            return 0
        
        n = len(group_members)
        transfer_factor = (n - 1) / n
        
        src_idx = group_members.index(src)
        dst_idx = group_members.index(dst)
        
        if (dst_idx - src_idx) % n == 1:
            return int(message_size * transfer_factor)
        else:
            return 0
    
    def calculate_broadcast_transfer(self, src: int, dst: int, message_size: int, group_members: List[int], root: int = 0) -> int:
        """Calculate data transfer between two GPUs in Broadcast."""
        if src not in group_members or dst not in group_members:
            return 0
        
        if src == dst:
            return 0
        
        # Assuming ring-based broadcast
        n = len(group_members)
        transfer_factor = (n - 1) / n
        
        src_idx = group_members.index(src)
        dst_idx = group_members.index(dst)
        
        if (dst_idx - src_idx) % n == 1:
            return int(message_size * transfer_factor)
        else:
            return 0
    
    def calculate_pipeline_transfer(self, src: int, dst: int) -> int:
        """Calculate pipeline parallelism data transfer between two GPUs."""
        if self.config['pp_size'] <= 1 or self.config['pp_comm_size'] <= 0:
            return 0  # No pipeline communication
        
        src_stage = self.get_pp_group(src)
        dst_stage = self.get_pp_group(dst)
        
        # Pipeline communication only happens between adjacent stages
        if abs(src_stage - dst_stage) != 1:
            return 0
        
        # GPUs must be in corresponding positions within their stages
        tp_size = self.config['tp_size']
        dp_size = self.config['dp_size']
        gpus_per_pp_stage = tp_size * dp_size
        
        src_local_id = src % gpus_per_pp_stage
        dst_local_id = dst % gpus_per_pp_stage
        
        # Only communicate between GPUs in the same local position
        if src_local_id != dst_local_id:
            return 0
        
        # Calculate transfers based on AICB formula and pipeline scheduling
        vpp = self.config['vpp']
        ga = self.config['ga']
        pp_comm_size = self.config['pp_comm_size']
        
        # Forward pass: src_stage < dst_stage (activations flow forward)
        # Backward pass: src_stage > dst_stage (gradients flow backward)
        
        # Each pipeline stage sends to the next in forward pass and receives from next in backward
        # Total transfers = forward activations + backward gradients
        # Multiply by VPP (virtual pipeline stages) and GA (gradient accumulation)
        
        if src_stage == dst_stage - 1:  # Forward direction (src to next stage)
            return pp_comm_size * vpp * ga
        elif src_stage == dst_stage + 1:  # Backward direction (gradients back)
            return pp_comm_size * vpp * ga
        
        return 0
    
    def calculate_layer_transfer(self, layer: Dict, src_gpu: int, dst_gpu: int) -> int:
        """Calculate total data transfer for a single layer between two GPUs."""
        total_transfer = 0
        
        # Forward pass communication
        if layer['fwd_enabled'] and layer['fwd_comm_size'] > 0:
            if layer['fwd_comm_type'] == 'ALLREDUCE':
                tp_members = self.get_tp_group_members(src_gpu)
                total_transfer += self.calculate_ring_allreduce_transfer(
                    src_gpu, dst_gpu, layer['fwd_comm_size'], tp_members)
            
            elif layer['fwd_comm_type'] == 'ALLGATHER':
                tp_members = self.get_tp_group_members(src_gpu)
                total_transfer += self.calculate_all_gather_transfer(
                    src_gpu, dst_gpu, layer['fwd_comm_size'], tp_members)
            
            elif layer['fwd_comm_type'] == 'BROADCAST':
                tp_members = self.get_tp_group_members(src_gpu)
                total_transfer += self.calculate_broadcast_transfer(
                    src_gpu, dst_gpu, layer['fwd_comm_size'], tp_members)
        
        # Backward input gradient communication
        if layer['bwd_input_enabled'] and layer['bwd_input_size'] > 0:
            if layer['bwd_input_comm_type'] == 'ALLREDUCE':
                tp_members = self.get_tp_group_members(src_gpu)
                total_transfer += self.calculate_ring_allreduce_transfer(
                    src_gpu, dst_gpu, layer['bwd_input_size'], tp_members)
        
        # Backward weight gradient communication
        if layer['bwd_weight_size'] > 0:
            if layer['bwd_weight_comm_type'] == 'ALLREDUCE':
                dp_members = self.get_dp_group_members(src_gpu)
                total_transfer += self.calculate_ring_allreduce_transfer(
                    src_gpu, dst_gpu, layer['bwd_weight_size'], dp_members)
            
            elif layer['bwd_weight_comm_type'] == 'REDUCESCATTER':
                dp_members = self.get_dp_group_members(src_gpu)
                total_transfer += self.calculate_reduce_scatter_transfer(
                    src_gpu, dst_gpu, layer['bwd_weight_size'], dp_members)
        
        return total_transfer
    
    def compute_transfer_between_2_gpus(self, comm_size: int, comm_type: str, is_tp: bool) -> int:
        """Compute TP transfer between two GPUs."""
        group_size = self.config['tp_size'] if is_tp else (self.config['dp_size']*self.config['tp_size'])
        total_transfer = ((group_size - 1) / group_size) * comm_size
        if comm_type == 'ALLREDUCE':
            total_transfer = 2*total_transfer
        actual_total_transfer = int(int(total_transfer)/self.config['pp_size'])
        return actual_total_transfer
    
    def calculate_layer_total_transfer(self, layer: Dict):
        """Calculate total data transfer for a single layer."""
        #tp communication
        tp_transfer_between_2_gpus = 0
        dp_transfer_between_2_gpus = 0
        # Forward pass communication
        if layer['fwd_comm_size'] > 0:
            tp_transfer_between_2_gpus += self.compute_transfer_between_2_gpus(layer['fwd_comm_size'], layer['fwd_comm_type'], is_tp=True)
        # Backward input gradient communication
        if layer['bwd_input_size'] > 0:
            tp_transfer_between_2_gpus += self.compute_transfer_between_2_gpus(layer['bwd_input_size'], layer['bwd_input_comm_type'], is_tp=True)
        # Backward weight gradient communication
        if layer['bwd_weight_size'] > 0:
            dp_transfer_between_2_gpus += self.compute_transfer_between_2_gpus(layer['bwd_weight_size'], layer['bwd_weight_comm_type'], is_tp=False)
        for src_gpu in range(self.config['total_gpus']):
            self.matrix[src_gpu][self.tp_pair(src_gpu)] += tp_transfer_between_2_gpus
            self.matrix[src_gpu][self.dp_pair(src_gpu)] += dp_transfer_between_2_gpus


    def calculate_total_transfer(self, src_gpu: int, dst_gpu: int, iterations: int = 1) -> int:
        """Calculate total data transfer between two GPUs across all layers."""
        total_transfer = 0
        
        print(f"\nAnalyzing transfer from GPU {src_gpu} to GPU {dst_gpu}:")
        print(f"  GPU {src_gpu} TP group: {self.get_tp_group_members(src_gpu)}")
        print(f"  GPU {src_gpu} DP group: {self.get_dp_group_members(src_gpu)}")
        
        for layer in self.layers:
            layer_transfer = self.calculate_layer_transfer(layer, src_gpu, dst_gpu)
            if layer_transfer > 0:
                print(f"  Layer '{layer['name']}': {layer_transfer:,} bytes ({layer_transfer/1024/1024:.1f} MB)")
            total_transfer += layer_transfer
        
        # Add pipeline parallelism communication
        if self.config['pp_size'] > 1:
            pp_transfer = self.calculate_pipeline_transfer(src_gpu, dst_gpu)
            if pp_transfer > 0:
                print(f"  Pipeline communication: {pp_transfer:,} bytes ({pp_transfer/1024/1024:.1f} MB)")
            total_transfer += pp_transfer
        
        total_transfer *= iterations
        
        print(f"\nTotal transfer (GPU {src_gpu} → GPU {dst_gpu}):")
        print(f"  {total_transfer:,} bytes")
        print(f"  {total_transfer/1024/1024:.1f} MB")
        print(f"  {total_transfer/1024/1024/1024:.3f} GB")
        print(f"  Iterations: {iterations}")
        
        return total_transfer
    
    def calculate_layer_transfer_silent(self, src_gpu: int, dst_gpu: int) -> int:
        """Calculate total data transfer for all layers between two GPUs (silent version)."""
        total_transfer = 0
        
        for layer in self.layers:
            total_transfer += self.calculate_layer_transfer(layer, src_gpu, dst_gpu)
        
        # Add pipeline parallelism communication (silent)
        if self.config['pp_size'] > 1:
            pp_transfer = self.calculate_pipeline_transfer(src_gpu, dst_gpu)
            total_transfer += pp_transfer
        
        return total_transfer
    
    def generate_full_matrix(self, output_file: str = None, verbose: bool = False):
        """Generate the complete GPU-to-GPU transfer matrix."""
        n_gpus = self.config['total_gpus']
        
        print(f"\nGenerating full {n_gpus}x{n_gpus} transfer matrix...")
        for layer in self.layers:
            self.calculate_layer_total_transfer(layer)
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                # Write header row (GPU column numbers)
                f.write("GPU")
                for j in range(n_gpus):
                    f.write(f",GPU{j}")
                f.write("\n")
                
                # Write each row (GPU i to all other GPUs)
                for i in range(n_gpus):
                    f.write(f"GPU{i}")
                    for j in range(n_gpus):
                        f.write(f",{self.matrix[i][j]}")
                    f.write("\n")
            print(f"  Matrix saved to: {output_file}")
        
        return self.matrix
    
    def add_pp_comm_to_matrix(self, iterations: int):
        """Add pp_comm to the matrix."""

        total_pp_comm = self.config['pp_comm_size'] * iterations

        for i in range(self.config['total_gpus']):
            # Forward pass
            self.matrix[i][self.pp_pair(i, is_forward=True)] += total_pp_comm
            # Backward pass
            self.matrix[i][self.pp_pair(i, is_forward=False)] += total_pp_comm
    
    def calculate_pp_comm_size(self, hidden_size: int = 4096, seq_length: int = 2048, micro_batch: int = 1, bytes_per_element: int = 2) -> int:
        """Calculate pp_comm_size using AICB formula: 2 * hidden_size * seq_length * micro_batch."""
        if self.config['pp_size'] <= 1:
            print("\nPipeline Communication Analysis:")
            print("  PP Size = 1, no pipeline communication needed")
            print("  Recommendation: Do not include pp_comm in workload header")
            return 0
        
        # AICB formula: pp_comm_size = bytes_per_element * hidden_size * seq_length * micro_batch
        calculated_pp_comm = bytes_per_element * hidden_size * seq_length * micro_batch
        
        print(f"\nPipeline Communication Analysis:")
        print(f"  Using AICB formula: pp_comm_size = {bytes_per_element} * {hidden_size} * {seq_length} * {micro_batch}")
        print(f"  Model parameters:")
        print(f"    Hidden size: {hidden_size}")
        print(f"    Sequence length: {seq_length}")
        print(f"    Micro batch: {micro_batch}")
        print(f"    Bytes per element: {bytes_per_element} ({'Float16/BFloat16' if bytes_per_element == 2 else 'Float32' if bytes_per_element == 4 else 'Custom'})")
        print(f"  Calculated pp_comm_size: {calculated_pp_comm:,} bytes ({calculated_pp_comm/1024/1024:.1f} MB)")
        
        if self.config['pp_comm_size'] == 0:
            print(f"  Recommendation: Add 'pp_comm: {calculated_pp_comm}' to your workload header")
            print(f"  Fixed header would be:")
            header_parts = []
            header_parts.append("HYBRID_TRANSFORMER_FWD_IN_BCKWD")
            header_parts.append(f"model_parallel_NPU_group: {self.config['tp_size']}")
            header_parts.append(f"ep: {self.config['ep_size']}")
            header_parts.append(f"pp: {self.config['pp_size']}")
            header_parts.append(f"vpp: {self.config['vpp']}")
            header_parts.append(f"ga: {self.config['ga']}")
            header_parts.append(f"all_gpus: {self.config['total_gpus']}")
            header_parts.append(f"pp_comm: {calculated_pp_comm}")
            header_parts.append("checkpoints: 0 checkpoint_initiates: 0")
            print(f"    {' '.join(header_parts)}")
        elif self.config['pp_comm_size'] != calculated_pp_comm:
            print(f"  Current pp_comm_size: {self.config['pp_comm_size']:,} bytes ({self.config['pp_comm_size']/1024/1024:.1f} MB)")
            print(f"  Calculated pp_comm_size: {calculated_pp_comm:,} bytes ({calculated_pp_comm/1024/1024:.1f} MB)")
            if abs(self.config['pp_comm_size'] - calculated_pp_comm) / calculated_pp_comm > 0.1:  # >10% difference
                print(f"  Warning: Significant difference detected!")
        else:
            print(f"  Current pp_comm_size matches calculation!")
        
        return calculated_pp_comm


def main():
    parser = argparse.ArgumentParser(description='Analyze SimAI workload files for GPU-to-GPU data transfer')
    parser.add_argument('workload_file', help='Path to the workload file')
    parser.add_argument('--src', type=int, help='Source GPU ID')
    parser.add_argument('--dst', type=int, help='Destination GPU ID')
    parser.add_argument('--iterations', type=int, default=1, help='Number of training iterations')
    parser.add_argument('--matrix', action='store_true', help='Generate full transfer matrix')
    parser.add_argument('--output', help='Output file for matrix (CSV format)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed transfer analysis for each GPU pair')
    parser.add_argument('--calc-pp-comm', action='store_true', help='Calculate pp_comm_size using AICB formula')
    parser.add_argument('--hidden-size', type=int, default=4096, help='Model hidden size (default: 4096)')
    parser.add_argument('--seq-length', type=int, default=2048, help='Sequence length (default: 2048)')
    parser.add_argument('--micro-batch', type=int, default=1, help='Micro batch size (default: 1)')
    parser.add_argument('--bytes-per-element', type=int, default=2, help='Bytes per element: 2=Float16/BFloat16, 4=Float32 (default: 2)')
    parser.add_argument('--heatmap', action='store_true', help='Generate heatmap visualization of the transfer matrix')
    parser.add_argument('--heatmap-output', help='Output file for heatmap (PNG format)')
    parser.add_argument('--enhance-factor', type=float, default=10.0, help='Enhancement factor for small values in heatmap (default: 10.0)')
    
    args = parser.parse_args()
    
    try:
        analyzer = WorkloadAnalyzer(args.workload_file)
        
        if args.calc_pp_comm or True:
            analyzer.calculate_pp_comm_size(args.hidden_size, args.seq_length, args.micro_batch, args.bytes_per_element)
        if analyzer.config['pp_comm_size'] > 0:
            analyzer.add_pp_comm_to_matrix(args.iterations)
        if args.matrix:
            analyzer.generate_full_matrix(args.output, args.verbose)
            
            # Generate heatmap if requested
            if args.heatmap:
                if args.output:
                    try:
                        from heatmap_generator import generate_heatmap_from_csv
                        
                        # Determine heatmap output path
                        if args.heatmap_output:
                            heatmap_path = args.heatmap_output
                        else:
                            # Auto-generate heatmap filename based on matrix filename
                            base_name = args.output.replace('.csv', '')
                            heatmap_path = f"{base_name}_heatmap.png"
                        
                        print(f"\nGenerating heatmap...")
                        heatmap_output = generate_heatmap_from_csv(
                            args.output, 
                            heatmap_path, 
                            args.enhance_factor
                        )
                        print(f"Heatmap saved to: {heatmap_output}")
                        
                    except ImportError:
                        print("Warning: matplotlib not available. Heatmap generation skipped.")
                        print("Install matplotlib to enable heatmap generation: pip install matplotlib")
                    except Exception as e:
                        print(f"Error generating heatmap: {e}")
                else:
                    print("Warning: --heatmap requires --output to specify matrix file location")
        elif args.src is not None and args.dst is not None:
            analyzer.calculate_total_transfer(args.src, args.dst, args.iterations)
        else:
            print("Please specify one of:")
            print("  --src/--dst for specific transfer analysis")
            print("  --matrix for full transfer matrix generation") 
            print("  --calc-pp-comm for pp_comm_size calculation using AICB formula")
            print("  --heatmap (with --matrix) for heatmap visualization")
            
    except FileNotFoundError:
        print(f"Error: Workload file '{args.workload_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
