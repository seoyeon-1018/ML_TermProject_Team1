"""Optional reproduction only. Fetch just three public CSV archives, not images.

The user's Allrecipes.zip already provides core-data-test_rating.csv.
Run only when rebuilding from the raw source; processed data is included.
"""
import argparse
import json
import urllib.request
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="raw/allrecipes")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["core-data_recipe.csv", "core-data-train_rating.csv", "core-data-valid_rating.csv"]:
        path = out / (name + ".zip")
        tmp = out / (name + ".zip.tmp")
        url = "https://www.kaggle.com/api/v1/datasets/download/elisaxxygao/foodrecsysv1/" + name + "?datasetVersionNumber=1"
        with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
            n = 0
            while block := r.read(1024 * 1024):
                n += len(block)
                if n > 150000000:
                    raise ValueError("Unexpected single-file size; stopped before downloading large image data")
                f.write(block)
        with zipfile.ZipFile(tmp) as z:
            if z.namelist() != [name] or z.testzip() is not None:
                raise ValueError(f"Unexpected source file response for {name}")
        tmp.replace(path)
        print(json.dumps({"file": str(path), "bytes": n}), flush=True)


if __name__ == "__main__":
    main()
