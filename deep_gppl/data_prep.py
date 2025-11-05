
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Data preparation for GB1 -> preference GP.

import argparse
import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RNG = np.random.default_rng(0)

SEQ_COL = "seq"
Y_COL = "y"


def sample_pairs(y: np.ndarray, max_pairs_per_anchor: int = 25):
    n = y.shape[0]
    V, U = [], []
    for i in range(n):
        better = np.where(y[i] > y)[0]
        if better.size == 0:
            continue
        k = min(max_pairs_per_anchor, better.size)
        js = RNG.choice(better, size=k, replace=False)
        V.extend([i]*k)
        U.extend(js.tolist())
    return np.asarray(V, dtype=np.int64), np.asarray(U, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="processed_gb1")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--num_variable_pos", type=int, default=4)
    ap.add_argument("--prefix_filter", type=str, default="A")
    ap.add_argument("--max_pairs_per_anchor", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)  # [149361, 3]
    df.to_csv("raw.csv", index=False)

    df = df[[SEQ_COL, Y_COL]].dropna().drop_duplicates().reset_index(drop=True)
    df.rename(columns={SEQ_COL: "sequence", Y_COL: "y"}, inplace=True)
    df = df.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)

    # Optional filtering and truncation by num_variable_pos and prefix_filter
    num_fixed_pos = 4 - args.num_variable_pos
    if args.num_variable_pos is not None:
        original_size = df.shape[0]
        df = df[df["sequence"].str.startswith("A"*num_fixed_pos)].reset_index(drop=True)
        df["sequence"] = df["sequence"].str[num_fixed_pos:]
        print(f"Filtered {original_size} sequences to {df.shape[0]} sequences with {args.num_variable_pos} variable positions starting with {args.prefix_filter*num_fixed_pos}")

    trainval_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.random_state, shuffle=True)
    rel_val = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(trainval_df, test_size=rel_val, random_state=args.random_state, shuffle=True)

    y_train = train_df["y"].to_numpy()
    y_val = val_df["y"].to_numpy()
    V_train, U_train = sample_pairs(y_train, max_pairs_per_anchor=args.max_pairs_per_anchor)
    V_val, U_val = sample_pairs(y_val, max_pairs_per_anchor=args.max_pairs_per_anchor)

    meta = {
        "csv_source": args.csv,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "m_train": int(V_train.size),
        "m_val": int(V_val.size),
        "sequence_length": int(len(train_df["sequence"].iloc[0])) if len(train_df)>0 else None,
        "num_variable_pos": args.num_variable_pos,
        "prefix_filter": args.prefix_filter,
    }

    train_df.to_csv(out_dir / "train_instances.csv", index=False)
    val_df.to_csv(out_dir / "val_instances.csv", index=False)
    test_df.to_csv(out_dir / "test_instances.csv", index=False)

    import numpy as np
    np.savez(out_dir / "train_pairs.npz", V=V_train, U=U_train)
    np.savez(out_dir / "val_pairs.npz", V=V_val, U=U_val)

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"Wrote processed data to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
