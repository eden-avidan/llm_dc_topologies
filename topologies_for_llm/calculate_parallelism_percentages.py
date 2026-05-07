import os
from pathlib import Path
import pandas as pd
import numpy as np

matrices_dir = "/Users/eavidan/Documents/topology_repo/simai/final_output/"
only_tp_dir = os.path.join(matrices_dir, "only_tp")
only_dp_dir = os.path.join(matrices_dir, "only_dp")
only_pp_dir = os.path.join(matrices_dir, "only_pp")

def sum_total_values(path):
    df = pd.read_csv(path)
    
    # sum all numerical values in the dataframe
    return df.select_dtypes(include=[np.number]).sum().sum()

def get_name_without_suffix(name):
    # Every name ends with one of the following suffixes: _tp_only, _dp_only, _pp_only
    if name.endswith("_tp_only.csv"):
        return name.replace("_tp_only.csv", "")
    elif name.endswith("_dp_only.csv"):
        return name.replace("_dp_only.csv", "")
    elif name.endswith("_pp_only.csv"):
        return name.replace("_pp_only.csv", "")
    else:
        return name

def main():
    # Create a dataframe with the columns: tp_total, dp_total, pp_total, tp_percentage, dp_percentage, pp_percentage
    df = pd.DataFrame(columns=["tp_total", "dp_total", "pp_total", "tp_percentage", "dp_percentage", "pp_percentage"])
    count = 0
    # Iterate over the only_tp_dir, only_dp_dir, only_pp_dir and add the total values to the dataframe
    for file in os.listdir(only_tp_dir):
        if "world_size1024-" not in file:
            continue
        if not any(tp_tag in file for tp_tag in ("tp8-", "tp16-", "tp32-")):
            continue
        # Get the file name
        file_name = get_name_without_suffix(os.path.basename(file))
        # This file exists also in only_dp_dir and only_pp_dir, get the total values from both
        dp_total = sum_total_values(os.path.join(only_dp_dir, f"{file_name}_dp_only.csv"))
        pp_total = sum_total_values(os.path.join(only_pp_dir, f"{file_name}_pp_only.csv"))
        tp_total = sum_total_values(os.path.join(only_tp_dir, f"{file_name}_tp_only.csv"))
        # Add one row with totals.
        row_df = pd.DataFrame([{"tp_total": tp_total, "dp_total": dp_total, "pp_total": pp_total}])
        df = pd.concat([df, row_df], ignore_index=True)

        # Calculate percentages for the last inserted row.
        denom = tp_total + dp_total + pp_total
        df.loc[df.index[-1], "tp_percentage"] = tp_total / denom if denom else 0.0
        df.loc[df.index[-1], "dp_percentage"] = dp_total / denom if denom else 0.0
        df.loc[df.index[-1], "pp_percentage"] = pp_total / denom if denom else 0.0

        print(f"File: {file_name} added to dataframe")
        count += 1
    if df.empty:
        print("No files matched the filter; no averages to report.")
        return

    print(f"Total files processed: {count}")
    numeric_cols = ["tp_total", "dp_total", "pp_total", "tp_percentage", "dp_percentage", "pp_percentage"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    avg = df[numeric_cols].mean()
    print("\nAverage values across all processed files:")
    print(
        f"tp_percentage={avg['tp_percentage']:.4f}, "
        f"dp_percentage={avg['dp_percentage']:.4f}, "
        f"pp_percentage={avg['pp_percentage']:.4f}"
    )

    print("\nMinimum values across all processed files:")
    print(
        f"tp_percentage={df['tp_percentage'].min():.4f}, "
        f"dp_percentage={df['dp_percentage'].min():.4f}, "
        f"pp_percentage={df['pp_percentage'].min():.4f}"
    )

    print("\nMaximum values across all processed files:")
    print(
        f"tp_percentage={df['tp_percentage'].max():.4f}, "
        f"dp_percentage={df['dp_percentage'].max():.4f}, "
        f"pp_percentage={df['pp_percentage'].max():.4f}"
    )

if __name__ == "__main__":
    main()