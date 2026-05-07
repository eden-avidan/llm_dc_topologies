# LLM Data-Center Topologies

Comparative analysis of data-center network topologies for large-scale LLM training. The repo evaluates **Rail-Only**, **Fat Tree**, **Dragonfly+**, and **HyperX** networks against realistic Megatron-style transport matrices, producing runtime, cost, edge-load, and end-to-end delay results — the data behind the APNET paper *"LLM Topologies"* (PDF in repo root).

---

## What's in here

| Area | Purpose |
| --- | --- |
| `topologies_for_llm/` | Main analysis code: runtime, price, MOE variants, plotting drivers |
| `topologies_for_llm/graph_overloading/` | Edge-load and delay analysis on per-topology graphs |
| `topologies_for_llm/graph_overloading/topologies/` | Topology builders: `FatTree`, `HyperX`, `DragonflyPlus` (all extend `abstract_topology.Topology`) |
| `simai/` | [SimAI](https://github.com/aliyun/SimAI) tree (Alibaba) — generates the transport matrices used as input |
| `saved DataFrames/` | Pre-computed input matrices: `matrices`, `only_tp`, `only_dp`, `only_pp`, `Total` |
| `heatmaps/`, `plots - overhead communication/` | Headline figures (also regenerated under `topologies_for_llm/outputs/`) |
| `CSSlideshow_Eng/` | Slide deck (English) |
| `APNET___LLM_Topologies__Copy___Eden_.pdf` | Paper draft |
| `overleaf.txt` | Overleaf project link for the paper |

---

## Topologies modeled

All topologies subclass `topologies.abstract_topology.Topology` and produce a NetworkX graph plus edge attributes.

- **Rail-Only** — flat HBI mesh + per-rail ToR.
- **Fat Tree** — GPU → Rail (leaf) → Spine → SuperSpine. Hop costs fixed at 1/2/4/6 by locality (same node / same block-same-rail / same block-different-rail / cross-block).
- **Dragonfly+** — leaf-spine groups with direct intra-leaf GPU mesh; spines fully interconnect across groups.
- **HyperX** — Megatron-aware: one router per `(tp_leaf, pp, dp)` coordinate, 8 GPUs per router, "clique per dimension" inter-router edges.

---

## Pipeline at a glance

```
                ┌──────────────────────────┐
                │  SimAI (simai/)          │
                │  Megatron workload runs  │
                └─────────────┬────────────┘
                              ▼
                 simai/final_output/*.csv          ← per-(workload, parallelism) transport matrices
                              │
        ┌─────────────────────┼─────────────────────────────┐
        ▼                     ▼                             ▼
  Topologies Runtime.py   Topologies Price.py     graph_overloading/
  (per-topology runtime   (cost models per         graph_overload_analyzer.py
   from the workload)      topology, price/        (build graph → assign OD →
                           perf curves)             edge-load CDFs + delays)
        │                     │                             │
        └────────► outputs/{standard,moe,no_moe,all_archs}/ ◄┘
                       dataframes/  plots/  heatmaps/
```

---

## Setup

```bash
# Conda env (defined in environment.yml, name: topology_venv)
conda env create -f environment.yml
conda activate topology_venv
```

Core deps: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `networkx`, `scikit-learn` (+ `torch`, `transformers`, etc. for the LLM-side tooling).

> The SimAI simulator (`simai/`) is a separate ecosystem with its own toolchain — see `simai/README.md` and `simai/MOE_GENERATION_README.md`. You only need it if you want to regenerate transport matrices; otherwise use the pre-computed ones in `saved DataFrames/` and `simai/final_output/`.

---

## Running the analyses

All scripts are run from `topologies_for_llm/`.

### 1. Per-topology runtime over all workloads

```bash
cd topologies_for_llm
python "Topologies Runtime.py"                  # standard workloads
python "Topologies Runtime.py" --moe-active     # MOE-enabled workloads
python "Topologies Runtime.py" --heatmaps-only  # latency heatmaps only
```

Inputs: directories under `simai/final_output/` (each with per-config CSVs).
Outputs: variant-keyed folders under `outputs/{standard,moe}/`:
- `dataframes/` — runtime/overhead per workload
- `plots/overhead_communication/` — overhead bar/line plots
- `heatmaps/` — per-topology pair-wise latency heatmaps

The MOE-subset variant lives in `Topologies Runtime No MOE Subset.py`.

### 2. Cost / pricing curves

```bash
python "Topologies Price.py"
```

Sweeps `N_gpus` from 100 → 1000 and plots cost for each topology family using configurable HBI / link / switch unit prices.

### 3. Graph-overloading (edge load + end-to-end delay)

```bash
cd topologies_for_llm/graph_overloading
python graph_overload_analyzer.py
```

Driver that:
1. Builds each topology graph for a given `num_gpus`.
2. Reads the workload transport matrix (`OD`) from `simai/final_output/matrices/`.
3. Routes `OD` on the graph (shortest-path or even-spread variants).
4. Computes per-edge load, hop-distribution, delay-percentile, and CDF outputs.
5. Writes CSVs and PNGs under `edge_load_comparisons*/`.

For one-off diagnostics on a single workload/topology, see `run_single_workloads.py`.

### 4. Parallelism breakdown

```bash
python topologies_for_llm/calculate_parallelism_percentages.py
```

Aggregates the TP-only / DP-only / PP-only matrices in `simai/final_output/` and reports each parallelism dimension's share of total communication.

### 5. Quick simulation plots

```bash
python "topologies_for_llm/plots from our simulation.py"
python "topologies_for_llm/plots from our simulation no moe subset.py"
```

---

## Output layout

Output paths are centralized in `topologies_for_llm/output_config.py`:

```
topologies_for_llm/outputs/
├── standard/   # non-MOE workloads
│   ├── dataframes/
│   ├── plots/{overhead_communication,simulation}/
│   └── heatmaps/
├── moe/        # MOE-enabled workloads
├── no_moe/     # MOE source data, MOE-experts disabled
├── all_archs/  # cross-architecture aggregation
├── comparisons/
└── metadata/
```

Use `get_variant_paths("standard" | "moe" | "no_moe" | "all_archs")` from `output_config` instead of hard-coding paths in new scripts.

---

## Key inputs

- **Transport matrices** — square `(GPUs × GPUs)` CSVs under `simai/final_output/matrices/` (and `matrices_moe/`). Cell `(i, j)` = bytes sent from GPU *i* to GPU *j*.
- **Parallelism slices** — `only_tp/`, `only_dp/`, `only_pp/`: same shape, restricted to one parallelism dimension. Used for the parallelism-share analysis.
- **Pre-aggregated DataFrames** — `saved DataFrames/*.{csv,pkl}` mirror the above as pandas objects for faster reload.

Each CSV's filename encodes the workload config, e.g.
`gpt_13B-world_size1024-tp16-pp2-ep1-gbs2048-mbs1-seq4096-MOE-False-GEMM-False-flash_attn-False.csv`.

---

## Adding a new topology

1. Create `topologies_for_llm/graph_overloading/topologies/<name>.py`.
2. Subclass `Topology` from `abstract_topology.py` and implement `build_topology() -> TopologyBuildResult` (edges, edge attrs, meta).
3. Add it to the imports/dispatch in `graph_overload_analyzer.py` and (if relevant) `Topologies Runtime.py` / `Topologies Price.py`.
4. Re-run the drivers above.

---

## Paper / writeup

- Draft PDF: `APNET___LLM_Topologies__Copy___Eden_.pdf`
- Overleaf project: see `overleaf.txt`
- Slides: `CSSlideshow_Eng/`

---

## Authors

Original work by Elchanan ("elcha"), with extensions by Eden Avidan (Feb 2026 onwards). See commit history for change attribution.
