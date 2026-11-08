"""Shrink the Food.com Kaggle download to the columns this project actually reads.

RAW_interactions.csv keeps user_id, recipe_id, date, rating (drops the long review text).
RAW_recipes.csv keeps name, id, submitted, ingredients (drops steps, description, tags,
nutrition ...). The output ZIP has the same member names, so it works unchanged with:
    python run_preprocessing.py foodcom --zip foodcom_slim.zip --output out

Usage:
    python make_foodcom_slim.py                                   # reads the project data folder
    python make_foodcom_slim.py --input "Food.com.zip"            # or a ZIP / another folder with the two CSVs
    python make_foodcom_slim.py --input "Food.com.zip" --separate  # two smaller ZIPs instead of one
"""
import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

# Project-level raw data folder (ML_TermProject_1/data)
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

KEEP = {"RAW_interactions.csv": ["user_id", "recipe_id", "date", "rating"],
        "RAW_recipes.csv": ["name", "id", "submitted", "ingredients"]}


def read_member(source, name, columns):
    source = Path(source)
    if source.is_dir():
        return pd.read_csv(source / name, usecols=columns, dtype=str, keep_default_na=False)
    with zipfile.ZipFile(source) as z:
        member = [n for n in z.namelist() if Path(n).name == name and not n.startswith("__MACOSX/")]
        if len(member) != 1:
            raise ValueError(f"Expected one {name} in {source}, found {member}")
        with z.open(member[0]) as f:
            return pd.read_csv(f, usecols=columns, dtype=str, keep_default_na=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DATA_DIR,
                   help="folder with RAW_*.csv (default: project data folder) or the Food.com ZIP from Kaggle")
    p.add_argument("--output", default="foodcom_slim.zip")
    p.add_argument("--separate", action="store_true", help="write foodcom_interactions.zip and foodcom_recipes.zip")
    a = p.parse_args()
    frames = {name: read_member(a.input, name, cols)[cols] for name, cols in KEEP.items()}
    targets = ({"RAW_interactions.csv": "foodcom_interactions.zip", "RAW_recipes.csv": "foodcom_recipes.zip"}
               if a.separate else {name: a.output for name in KEEP})
    for out in sorted(set(targets.values())):
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name, frame in frames.items():
                if targets[name] == out:
                    buf = io.StringIO(); frame.to_csv(buf, index=False)
                    z.writestr(name, buf.getvalue())
        print(f"{out}: {Path(out).stat().st_size / 1e6:.1f} MB, rows: " +
              ", ".join(f"{n}={len(f):,}" for n, f in frames.items() if targets[n] == out))


if __name__ == "__main__":
    main()
