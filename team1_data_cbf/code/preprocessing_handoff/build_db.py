"""Pack the preprocessing outputs into the SQLite files used by cbf.py and LightFM.

Pipeline
--------
1. python run_preprocessing.py allrecipes --recipe-archive ... --train-archive ... \
       --valid-archive ... --test-archive ... --output out
2. python run_preprocessing.py foodcom --zip Food.com.zip --output out
3. python build_db.py --prepared out --dest ..          (writes allrecipes.db, foodcom.db)

The DB keeps only what the shared evaluation needs:
  recipes       normalized ingredients of every recipe in any evaluation catalog
  interactions  Allrecipes TRAIN ratings only (LightFM training / CBF vocabulary)
  profiles (H), targets (L), seen, candidates, users  per evaluation split
  metadata      protocol facts (cutoffs, profile rule, split statistics)
"""
import argparse
import gzip
import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = [
    "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID",
    "CREATE TABLE recipes(recipe_id TEXT PRIMARY KEY,name TEXT,ingredients TEXT NOT NULL,available_at TEXT) WITHOUT ROWID",
    "CREATE TABLE profiles(split TEXT,user_id TEXT,recipe_id TEXT,history_order INTEGER,rating INTEGER,timestamp TEXT,PRIMARY KEY(split,user_id,recipe_id)) WITHOUT ROWID",
    "CREATE TABLE targets(split TEXT,user_id TEXT,recipe_id TEXT,rating INTEGER,timestamp TEXT,PRIMARY KEY(split,user_id,recipe_id)) WITHOUT ROWID",
    "CREATE TABLE seen(split TEXT,user_id TEXT,recipe_id TEXT,PRIMARY KEY(split,user_id,recipe_id)) WITHOUT ROWID",
    "CREATE TABLE candidates(split TEXT,recipe_id TEXT,PRIMARY KEY(split,recipe_id)) WITHOUT ROWID",
    "CREATE TABLE users(split TEXT,user_id TEXT,PRIMARY KEY(split,user_id)) WITHOUT ROWID",
]
TRAIN_SCHEMA = "CREATE TABLE interactions(user_id TEXT,recipe_id TEXT,rating INTEGER,timestamp TEXT,partition TEXT)"
SCOPE = ("Only recipes used by frozen evaluation catalogs; source interactions table contains training records only. "
         "Fixed evaluation tables preserve valid/test inputs and targets.")


def read_recipes(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_db(dest, metadata, recipes, splits, train=None):
    """Write one SQLite file atomically (temp file, then rename).

    splits: {split_name: dict(h=DataFrame, l=DataFrame, seen=DataFrame, users=DataFrame, candidates=set)}
    Each H/L frame must have a 'time' column holding the timestamp (Allrecipes) or date (Food.com).
    """
    dest = Path(dest)
    temp = dest.with_name("." + dest.name + ".tmp")
    temp.unlink(missing_ok=True)
    with sqlite3.connect(temp) as db:
        for sql in SCHEMA:
            db.execute(sql)
        db.executemany("INSERT INTO metadata VALUES(?,?)",
                       [(k, json.dumps(v, ensure_ascii=False)) for k, v in sorted(metadata.items())])
        needed = set().union(*(s["candidates"] for s in splits.values()))
        db.executemany("INSERT INTO recipes VALUES(?,?,?,?)",
                       [(r["recipe_id"], r.get("name", ""), json.dumps(r["ingredients"], ensure_ascii=False, separators=(",", ":")),
                         r["available_at"]) for r in recipes if r["recipe_id"] in needed])
        for name, s in splits.items():
            db.executemany("INSERT INTO profiles VALUES(?,?,?,?,?,?)",
                           [(name, u, r, int(o), int(rt), t) for u, r, o, rt, t in s["h"][["user_id", "recipe_id", "history_order", "rating", "time"]].itertuples(index=False)])
            db.executemany("INSERT INTO targets VALUES(?,?,?,?,?)",
                           [(name, u, r, int(rt), t) for u, r, rt, t in s["l"][["user_id", "recipe_id", "rating", "time"]].itertuples(index=False)])
            db.executemany("INSERT INTO seen VALUES(?,?,?)", [(name, u, r) for u, r in s["seen"][["user_id", "recipe_id"]].itertuples(index=False)])
            db.executemany("INSERT INTO candidates VALUES(?,?)", [(name, r) for r in sorted(s["candidates"])])
            db.executemany("INSERT INTO users VALUES(?,?)", [(name, u) for u in s["users"].user_id])
        if train is not None:
            db.execute(TRAIN_SCHEMA)
            db.executemany("INSERT INTO interactions VALUES(?,?,?,?,?)",
                           [(u, r, int(rt), t, "train") for u, r, rt, t in train[["user_id", "recipe_id", "rating", "timestamp"]].itertuples(index=False)])
    # `with sqlite3.connect()` only commits; close explicitly so Windows can rename the file.
    db.close()
    temp.replace(dest)
    print(f"wrote {dest}")


def load_split(folder, time_column, candidates_file=None):
    folder = Path(folder)
    h = pd.read_csv(folder / "H.csv.gz", dtype={"user_id": str, "recipe_id": str})
    l = pd.read_csv(folder / "L.csv.gz", dtype={"user_id": str, "recipe_id": str})
    h["time"], l["time"] = h[time_column], l[time_column]
    return {"h": h, "l": l, "seen": pd.read_csv(folder / "seen.csv.gz", dtype=str),
            "users": pd.read_csv(folder / "evaluation_users.csv", dtype={"user_id": str}),
            "candidates": set(pd.read_csv(candidates_file or folder / "candidate_items.csv.gz", dtype=str).recipe_id)}


def build_allrecipes(prepared, dest):
    data = Path(prepared) / "data" / "allrecipes"
    report = json.loads((Path(prepared) / "reports" / "allrecipes_preparation" / "data_quality.json").read_text(encoding="utf-8"))
    recipes = read_recipes(data / "recipes.jsonl.gz")
    for r in recipes:  # Allrecipes has no publication date; first observed rating is the availability proxy.
        r["available_at"] = r.get("first_observed_at")
    splits = {name: load_split(data / f"{name}_evaluation", "timestamp") for name in ["valid", "test"]}
    train = pd.read_csv(data / "train.csv.gz", dtype={"user_id": str, "recipe_id": str, "timestamp": str})
    boundaries = report["split"]  # per-split counts and cutoffs from allrecipes.prepare_source
    metadata = {"boundaries": boundaries, "candidate_rule": "first_observed_at before evaluation cutoff", "dataset": "allrecipes",
                "ingredient_status": "already normalized with the same frozen shared rules", "package_scope": SCOPE,
                "positive_rating": 4, "profile_size": 5, "profile_rule": "first",
                "source": report.get("source_url", "https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1"),
                "splits": ["valid", "test"], "time_basis": "dateLastModified; not guaranteed original rating creation time", "top_k": 10}
    write_db(dest, metadata, recipes, splits, train=train)


def build_foodcom(prepared, dest):
    data = Path(prepared) / "data" / "foodcom"
    report = json.loads((Path(prepared) / "reports" / "foodcom_preparation" / "data_quality.json").read_text(encoding="utf-8"))
    recipes = read_recipes(data / "recipes.jsonl.gz")
    for r in recipes:
        r["available_at"] = r["submitted"]
    splits = {"test": load_split(data, "date", data / "candidate_items.csv.gz")}
    cutoff = report["protocol"]["cutoff"]
    metadata = {"candidate_rule": f"submitted before {cutoff}", "cutoff": cutoff, "dataset": "foodcom",
                "ingredient_status": "already normalized with the same frozen shared rules", "package_scope": SCOPE,
                "positive_rating": 4, "profile_size": report["protocol"]["profile_size"],
                "profile_rule": report["protocol"].get("profile_rule", "first"), "split_stats": report["split"],
                "source": "Food.com Recipes and Interactions (RAW_recipes.csv, RAW_interactions.csv)",
                "splits": ["test"], "time_basis": "date", "top_k": 10}
    write_db(dest, metadata, recipes, splits)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared", default=Path("."), type=Path, help="--output folder used by run_preprocessing.py")
    p.add_argument("--dest", default=Path("."), type=Path, help="folder for allrecipes.db and foodcom.db")
    p.add_argument("--only", choices=["allrecipes", "foodcom"], help="build just one database")
    a = p.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "allrecipes"):
        build_allrecipes(a.prepared, a.dest / "allrecipes.db")
    if a.only in (None, "foodcom"):
        build_foodcom(a.prepared, a.dest / "foodcom.db")


if __name__ == "__main__":
    main()
