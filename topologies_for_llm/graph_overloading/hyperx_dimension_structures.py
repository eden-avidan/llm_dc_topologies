from __future__ import annotations

import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

def megatron_rank_to_hyperx_coords(tp: int, pp: int, dp: int, gpus_per_leaf: int = 8) -> List[Tuple[int, int, int]]:
    """
    Returns an array coords[rank] = (tp_switch, pp_idx, dp_idx) where:
      - tp_switch = (tp_idx // gpus_per_leaf)  [which 8-GPU leaf within the TP group]
      - pp_idx    = pipeline stage index
      - dp_idx    = data parallel index

    Assumes the rank layout consistent with the transport-matrix pattern:
        rank = tp_idx + tp * dp_idx + tp * dp * pp_idx

    World size = tp * pp * dp.

    Example (tp=32, pp=8, dp=4):
      rank 0   -> (0,0,0)
      rank 7   -> (0,0,0)
      rank 8   -> (1,0,0)
      rank 32  -> (0,0,1)
      rank 128 -> (0,1,0)
    """
    if tp % gpus_per_leaf != 0:
        raise ValueError(f"tp={tp} must be divisible by gpus_per_leaf={gpus_per_leaf} to form equal 8-GPU leaves.")

    world_size = tp * pp * dp
    coords: List[Tuple[int, int, int]] = []

    for rank in range(world_size):
        tp_idx = rank % tp
        dp_idx = (rank // tp) % dp
        pp_idx = rank // (tp * dp)

        tp_switch = tp_idx // gpus_per_leaf
        coords.append((tp_switch, pp_idx, dp_idx))

    return coords



def check_one_hop_from_transport_csv(
    transport_csv_path: str,
    rank_to_coords: List[Tuple[int, int, int]],
    *,
    threshold: float = 0.0,
    metric: str = "manhattan",
    allow_same_switch: bool = True,
) -> Dict[str, Any]:
    """
    Reads a transport-matrix CSV (NxN) and checks whether every communicating pair
    is at most 1 hop apart in the provided 3D coordinate system.

    Interpretation used here (matches your "3D distance to 1 hop" phrasing):
      - Each rank has a 3D switch coordinate (x,y,z) = rank_to_coords[rank]
      - Distance between two ranks is distance between their switch coordinates
      - 'one hop' means distance <= 1 (and optionally distance == 0 allowed)

    Parameters
    ----------
    transport_csv_path:
        Path to the CSV containing an NxN transport matrix.
        Common formats supported:
          - With row/col headers of ranks
          - With an index column (often unnamed)
          - Raw numeric matrix

    rank_to_coords:
        List where rank_to_coords[r] = (x, y, z) for rank r.

    threshold:
        Treat entries > threshold as "a communication edge".

    metric:
        "manhattan" (default) or "chebyshev".
        - manhattan: |dx|+|dy|+|dz|
        - chebyshev: max(|dx|,|dy|,|dz|)

    allow_same_switch:
        If True, pairs on the same switch (distance 0) are allowed.
        If False, require distance == 1 exactly for any communicating pair.

    Returns
    -------
    dict with:
      - ok: bool
      - world_size: int
      - edges_checked: int
      - violations: list of dicts with (src, dst, bytes, dist, src_coord, dst_coord)
      - max_distance_observed: int
    """

    # --- Load CSV robustly ---
    df = pd.read_csv(transport_csv_path)

    # If first column looks like an index (often unnamed), set it as index
    if df.shape[1] > 1 and (df.columns[0].startswith("Unnamed") or df.columns[0] in ("rank", "src", "")):
        # Try to set as index if it is monotonic-ish or looks like rank labels
        try:
            df2 = pd.read_csv(transport_csv_path, index_col=0)
            # Use it only if it became square-ish
            if df2.shape[0] == df2.shape[1]:
                df = df2
        except Exception:
            pass

    # If first column contains labels like GPU0, GPU1, ... use it as index:
    # strip 'GPU' prefix and convert the rest to int.
    if df.shape[1] > 1:
        first_col = df.iloc[:, 0].astype(str).str.strip()
        if (
            first_col.str.startswith("GPU").all()
            and first_col.str[3:].str.fullmatch(r"\d+").all()
        ):
            gpu_idx = first_col.str[3:].astype(int)
            df = df.iloc[:, 1:].copy()
            df.index = gpu_idx

    # Ensure numeric matrix values
    df = df.apply(pd.to_numeric, errors="coerce")
    mat = df.to_numpy()

    # Ensure numeric
    mat = np.asarray(mat, dtype=np.float64)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Transport matrix must be square NxN. Got shape {mat.shape} from {transport_csv_path}")

    n = mat.shape[0]
    if n != len(rank_to_coords):
        raise ValueError(
            f"rank_to_coords length ({len(rank_to_coords)}) must match matrix size ({n})."
        )

    def dist(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
        if metric.lower() == "manhattan":
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
        elif metric.lower() == "chebyshev":
            return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))
        elif metric.lower() == "hamming":
            # HyperX-like hop count: one hop if you differ in exactly one dimension,
            # regardless of how far apart the indices are in that dimension.
            return int(a[0] != b[0]) + int(a[1] != b[1]) + int(a[2] != b[2])
        else:
            raise ValueError("metric must be 'manhattan', 'chebyshev', or 'hamming'")

    violations: List[Dict[str, Any]] = []
    edges_checked = 0
    max_dist = 0

    # Iterate over all directed edges with traffic
    # (If your matrix is symmetric, this will count both directions.)
    for i in range(n):
        ci = rank_to_coords[i]
        row = mat[i]
        for j in range(n):
            if i == j:
                continue
            v = row[j]
            if v > threshold:
                edges_checked += 1
                cj = rank_to_coords[j]
                d = dist(ci, cj)
                max_dist = max(max_dist, d)

                # "one hop" rule
                if allow_same_switch:
                    ok_edge = (d <= 1)
                else:
                    ok_edge = (d == 1)

                if not ok_edge:
                    violations.append(
                        {
                            "src": i,
                            "dst": j,
                            "bytes": float(v),
                            "dist": int(d),
                            "src_coord": ci,
                            "dst_coord": cj,
                        }
                    )

    return {
        "ok": len(violations) == 0,
        "world_size": n,
        "edges_checked": edges_checked,
        "violations": violations,
        "max_distance_observed": max_dist,
        "metric": metric,
        "threshold": threshold,
        "allow_same_switch": allow_same_switch,
    }





FNAME_RE = re.compile(
    r"""
    ^.*?                                  # any prefix
    -world_size(?P<world_size>\d+)         # -world_size128
    -tp(?P<tp>\d+)                         # -tp4
    -pp(?P<pp>\d+)                         # -pp4
    (?:-|\.|$)                             # next separator
    """,
    re.VERBOSE,
)

@dataclass(frozen=True)
class CsvMeta:
    path: str
    filename: str
    world_size: int
    tp: int
    pp: int


def parse_megatron_csv_filename(filename: str) -> Optional[Tuple[int, int, int]]:
    """
    Extract (world_size, tp, pp) from a filename like:
      gpt_7B-world_size128-tp4-pp4-ep1-gbs4096-mbs1-seq2048-MOE-False-GEMM-False-flash_attn-False.csv
    Returns None if not matched.
    """
    m = FNAME_RE.search(filename)
    if not m:
        return None
    return int(m.group("world_size")), int(m.group("tp")), int(m.group("pp"))


def load_csv_metas(
    directory: str,
    *,
    world_size: Optional[int] = None,
    tp: Optional[int | List[int]] = None,
    pp: Optional[int] = None,
) -> List[CsvMeta]:
    """
    Scan a directory for .csv files, parse metadata from filenames, and filter.

    Filters are exact-match if provided.
    """
    metas: List[CsvMeta] = []

    # Normalize tp filter to a set (None => no filter)
    tp_set: Optional[set[int]]
    if tp is None:
        tp_set = None
    elif isinstance(tp, int):
        tp_set = {int(tp)}
    else:
        tp_values = [int(x) for x in tp]
        tp_set = set(tp_values) if len(tp_values) > 0 else None

    for fn in os.listdir(directory):
        if not fn.lower().endswith(".csv"):
            continue
        parsed = parse_megatron_csv_filename(fn)
        if parsed is None:
            continue
        ws, t, p = parsed
        if world_size is not None and ws != world_size:
            continue
        if tp_set is not None and t not in tp_set:
            continue
        if pp is not None and p != pp:
            continue
        full_path = os.path.join(directory, fn)
        metas.append(CsvMeta(full_path, fn, ws, t, p))

    # deterministic order
    metas.sort(key=lambda x: (x.world_size, x.tp, x.pp, x.filename))
    return metas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load Megatron transport-matrix CSV files from a directory and extract world_size/tp/pp from filenames."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="/Users/eavidan/Documents/topology_repo/simai/final_output/matrices",
        help="Directory containing CSV files",
    )

    parser.add_argument(
        "--tp",
        type=int,
        nargs="*",
        default=[8, 16, 32],
        help="Filter by tp. Default: 8 16 32. Pass no values (e.g. --tp) to disable tp filtering.",
    )
    parser.add_argument(
        "--pp",
        type=int,
        nargs="*",
        default=None,
        help="Filter by pp. Default: no filtering. Pass no values (e.g. --pp) to disable pp filtering.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1024,
        help="Filter by world_size. Default: no filtering. Pass no values (e.g. --world_size) to disable world_size filtering.",
    )
        # ---- add knobs for the hop-check ----
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Transport entry > threshold is treated as a communication edge. Default: 0.0",
    )
    parser.add_argument(
        "--metric",
        choices=["hamming", "manhattan", "chebyshev"],
        default="hamming",
        help="Distance metric on (x,y,z). Default: hamming (HyperX dimension hop count).",
    )
    parser.add_argument(
        "--no-same-switch",
        action="store_true",
        help="If set, require communicating pairs to be exactly 1 hop (distance==1), not 0.",
    )
    parser.add_argument(
        "--max-violations",
        type=int,
        default=0,
        help="Print at most this many violating edges per file. Default: 20",
    )

    args = parser.parse_args()

    metas = load_csv_metas(
        args.path,
        world_size=args.world_size,
        tp=args.tp,
        pp=args.pp,
    )

    if not metas:
        print("No matching CSV files found.")
        return

    print(f"Found {len(metas)} matching CSV file(s):")

    for m in metas:
        # ---- derive dp from world_size / (tp*pp) ----
        denom = m.tp * m.pp
        if denom <= 0 or (m.world_size % denom) != 0:
            print(
                f"\n[mismatch] {m.filename}: world_size={m.world_size} not divisible by tp*pp={denom}. Skipping."
            )
            continue
        dp = m.world_size // denom

        # ---- build mapping rank -> (x,y,z) for HyperX ----
        # assumes you have this function already defined:
        #   megatron_rank_to_hyperx_coords(tp:int, pp:int, dp:int, gpus_per_switch:int=8) -> List[Tuple[int,int,int]]
        coords = megatron_rank_to_hyperx_coords(m.tp, m.pp, dp, gpus_per_leaf=8)

        # ---- run hop check ----
        res = check_one_hop_from_transport_csv(
            transport_csv_path=m.path,
            rank_to_coords=coords,
            threshold=args.threshold,
            metric=args.metric,
            allow_same_switch=not args.no_same_switch,
        )

        # ---- print summary ----
        print(
            f"\n- {m.filename} | world_size={m.world_size} tp={m.tp} pp={m.pp} dp={dp}"
            f"\n  edges_checked={res['edges_checked']} max_dist={res['max_distance_observed']} ok={res['ok']}"
        )

        # ---- print a few violations (if any) ----
        if not res["ok"]:
            vios = res["violations"]
            print(f"  violations={len(vios)} (showing up to {args.max_violations})")
            for vio in vios[: args.max_violations]:
                print(
                    "   "
                    f"src={vio['src']} {vio['src_coord']} -> "
                    f"dst={vio['dst']} {vio['dst_coord']} | "
                    f"bytes={vio['bytes']:.0f} dist={vio['dist']}"
                )
        else:
            print("  ✅ all communicating pairs are within 1 hop under the provided coordinates.")

if __name__ == "__main__":
    main()