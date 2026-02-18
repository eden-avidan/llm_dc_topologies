input_file = "G175B-M1-C03_GPT175B_megatron_tp8_pp1_mbs1_A100.txt"
output_file = "converted_work_load.txt"

with open(input_file) as fin, open(output_file, "w") as fout:
    # Write new header
    fout.write("HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 8 ep: 1 pp: 1 vpp: 8 ga: 1 all_gpus: 8 checkpoints: 0 checkpoint_initiates: 0\n")
    lines = fin.readlines()
    # Skip old header and count line
    op_lines = [l.strip().replace('\t', ' ') for l in lines[2:] if l.strip()]
    fout.write(f"{len(op_lines)}\n")
    for l in op_lines:
        cols = l.split()
        # Pad or trim to 12 columns
        cols = (cols + ['0']*12)[:12]
        fout.write(' '.join(cols) + '\n')
