#!/usr/bin/env python3
"""
Add an `is_boundary` flag to a mobile-robot log CSV.

Usage
-----
python add_boundary.py  \
       --in   dataset_3.csv           \
       --out  dataset_3_bdry.csv      \
       --radius 0.30                  \
       --buffer 0.05                  \
       --delta  0.02

• A row is flagged boundary (=1) iff
    | min_lidar_dist – (radius+buffer) | < delta
• Safe / unsafe labels are **not modified**.
"""

import pandas as pd, argparse, numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in',   required=True, dest='csv_in')
    p.add_argument('--out',  required=True, dest='csv_out')
    p.add_argument('--radius', type=float, default=0.30)
    p.add_argument('--buffer', type=float, default=0.05)
    p.add_argument('--delta',  type=float, default=0.02)
    args = p.parse_args()

    d_safe = args.radius + args.buffer
    print(f"Boundary band: |d_min – {d_safe:.3f}| < {args.delta}")

    df = pd.read_csv(args.csv_in)
    if 'min_lidar_dist' not in df.columns:
        raise RuntimeError("CSV lacks 'min_lidar_dist' column.")

    df['is_boundary'] = (
        (df['min_lidar_dist'] - d_safe).abs() < args.delta
    ).astype(int)

    # quick report
    n_bdry = df['is_boundary'].sum()
    print(f"Flagged {n_bdry} boundary rows ({n_bdry/len(df):.2%}).")

    df.to_csv(args.csv_out, index=False)
    print(f"Exported → {args.csv_out}")

if __name__ == "__main__":
    main()
