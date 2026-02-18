import argparse

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV containing FishID column")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "FishID" not in df.columns:
        raise RuntimeError("Input CSV must contain FishID column")

    fish_ids = sorted({str(x) for x in df["FishID"].dropna().astype(str).tolist() if "+" not in str(x)})
    if not fish_ids:
        raise RuntimeError("No single-fish FishID values found")

    rng = np.random.default_rng(int(args.seed))
    fish_ids = list(fish_ids)
    rng.shuffle(fish_ids)

    k = int(args.k)
    if k < 2:
        raise RuntimeError("k must be >= 2")

    rows = []
    for i, fid in enumerate(fish_ids):
        rows.append({"FishID": fid, "fold": int(i % k), "seed": int(args.seed), "k": int(k)})

    out = pd.DataFrame(rows).sort_values(["fold", "FishID"], kind="mergesort").reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(args.out)
    print(out["fold"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

