"""
build_split.py

Reconstructs the custom train / dev / test split described in:
"Empirical Analysis of Multimodal Fusion Techniques for Offensive Meme
Classification" (Lalruatkimi & Laitonjam), using:

  1. The original Facebook Hateful Memes (FHM) dataset, downloaded
     separately by the user from Kaggle:
     https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset

  2. The ID lists in this repository's `splits/` folder
     (train_ids.csv, dev_ids.csv, test_ids.csv), each with columns
     `image_name,label`.

This script does NOT download or redistribute the original dataset. You
must download it yourself first and pass its location via --original_dir.

Usage:
    python scripts/build_split.py \\
        --original_dir /path/to/original_fhm_dataset \\
        --output_dir ./rebuilt_dataset

The script auto-detects whether the original data is provided as .jsonl
files (the original Facebook AI release format: train.jsonl, dev.jsonl,
each record with fields like "id", "img", "text", "label") or as .csv
files (some Kaggle mirrors reformat it). Adjust the column-name mapping
in load_original_as_dataframe() if your local copy uses different field
names.

Output: rebuilt_dataset/train.csv, dev.csv, test.csv, each with columns
image_name, id, text, label -- label is taken from this repo's split
files (authoritative), text/image_name spelling is taken from your local
original dataset copy.
"""

import argparse
import json
import os
import sys

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(os.path.dirname(THIS_DIR), "splits")


def load_original_as_dataframe(original_dir: str) -> pd.DataFrame:
    records = []

    jsonl_candidates = ["train.jsonl", "dev.jsonl", "dev_seen.jsonl", "dev_unseen.jsonl"]
    found_any = False

    for fname in jsonl_candidates:
        fpath = os.path.join(original_dir, fname)
        if os.path.exists(fpath):
            found_any = True
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    records.append(
                        {
                            "id": str(row.get("id", "")).zfill(5) if str(row.get("id", "")).isdigit() else row.get("id"),
                            "image_name": row.get("img", row.get("image_name", "")),
                            "text": row.get("text", row.get("sentence", "")),
                            "label": row.get("label", None),
                        }
                    )

    if not found_any:
        csv_candidates = [f for f in os.listdir(original_dir) if f.lower().endswith(".csv")]
        if not csv_candidates:
            sys.exit(
                f"Could not find train.jsonl/dev.jsonl or any .csv files in {original_dir}. "
                "Please point --original_dir at your extracted Kaggle download."
            )
        dfs = []
        for fname in csv_candidates:
            df = pd.read_csv(os.path.join(original_dir, fname))
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)

        col_map = {}
        for col in df_all.columns:
            lc = col.lower()
            if lc in ("id",):
                col_map[col] = "id"
            elif lc in ("img", "image", "image_name", "filename"):
                col_map[col] = "image_name"
            elif lc in ("text", "sentence", "caption"):
                col_map[col] = "text"
            elif lc in ("label", "class"):
                col_map[col] = "label"
        df_all = df_all.rename(columns=col_map)
        records = df_all.to_dict(orient="records")

    df = pd.DataFrame(records)

    def normalize_id(row):
        if "image_name" in row and pd.notna(row.get("image_name")):
            base = os.path.splitext(str(row["image_name"]))[0]
            return base.lstrip("0") or "0" if base.isdigit() else base
        return str(row.get("id"))

    df["id"] = df.apply(normalize_id, axis=1)
    df["id"] = df["id"].astype(str).str.lstrip("0").replace("", "0")

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df[["id", "image_name", "text", "label"]].drop_duplicates(subset=["id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", required=True)
    parser.add_argument("--output_dir", default="./rebuilt_dataset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading original dataset from: {args.original_dir}")
    original_df = load_original_as_dataframe(args.original_dir)
    print(f"Loaded {len(original_df)} labeled records from the original dataset.")

    for split_name, out_name in [("train_ids.csv", "train.csv"),
                                  ("dev_ids.csv", "dev.csv"),
                                  ("test_ids.csv", "test.csv")]:
        split_path = os.path.join(SPLITS_DIR, split_name)
        split_ids = pd.read_csv(split_path)  # columns: image_name, label

        # Derive the join key "id" from image_name (same normalization used
        # for the original dataset above): strip extension, strip leading zeros.
        def id_from_image_name(name):
            base = os.path.splitext(str(name))[0]
            return base.lstrip("0") or "0" if base.isdigit() else base

        split_ids["id"] = split_ids["image_name"].apply(id_from_image_name).astype(str)
        split_ids = split_ids.rename(columns={"label": "label_split"})

        merged = split_ids.merge(
            original_df, on="id", how="left", suffixes=("", "_orig")
        )

        missing = merged["text"].isna().sum() if "text" in merged.columns else 0
        if missing > 0:
            print(f"WARNING: {missing} of {len(merged)} IDs in {split_name} could not be matched.")

        # Use OUR label (from the split file) as the authoritative ground truth.
        merged["label"] = merged["label_split"]
        merged = merged.drop(columns=["label_split"])
        # Prefer the original dataset's image_name spelling if present, else ours.
        if "image_name_orig" in merged.columns:
            merged["image_name"] = merged["image_name_orig"].fillna(merged["image_name"])
            merged = merged.drop(columns=["image_name_orig"])

        out_path = os.path.join(args.output_dir, out_name)
        merged.to_csv(out_path, index=False)
        print(f"Wrote {len(merged)} rows to {out_path}")

    print("\nDone. Verify row counts match: train=7500, dev=500, test=1000.")


if __name__ == "__main__":
    main()
