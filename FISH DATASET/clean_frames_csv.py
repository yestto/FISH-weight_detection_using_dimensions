import argparse

import pandas as pd


def clean(in_path: str, out_path: str) -> None:
    df = pd.read_csv(in_path)
    df = df.copy()
    df["FishID"] = df["FishID"].astype(str)
    df = df[~df["FishID"].str.contains("+", regex=False)].copy()

    drop = [c for c in df.columns if str(c).startswith("_")]
    truth_cols = ["_Truth_Length (cm)", "Width_truth (cm)", "Area_truth (cm²)", "Perimeter_truth (cm)"]
    for c in truth_cols:
        if c in df.columns and c not in drop:
            drop.append(c)

    df = df.drop(columns=[c for c in drop if c in df.columns])
    df.to_csv(out_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()
    clean(args.in_path, args.out_path)
    print(args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

