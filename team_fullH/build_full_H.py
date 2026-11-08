"""Replace H (first 5 past positives) with ALL past positives in the teammate's databases.

Everything else in the databases is kept as it is: recipes (ingredient features), users, targets (L),
seen, candidates, interactions. Only the `profiles` table is rewritten.
The missing past ratings are read from the raw rating files (only to know which past recipes were rated >= 4).

Past period (same as the original protocol)
  Allrecipes valid : train ratings
  Allrecipes test  : train + valid ratings
  Food.com test    : ratings before 2013-02-01 (rating 0 and ratings dated before the recipe was submitted dropped)

Safety checks (the script stops if any fails)
  1. the first 5 of the new H are the same recipes as the teammate's H   -> same rules reproduced
  2. every new H recipe is in the teammate's `seen` table                -> same past period
  3. no new H recipe is in the target set L                               -> no future information
  4. every new H recipe is a candidate                                    -> CBF profile can be built

Run (mlteam env):
    python build_full_H.py --db-dir ../team_5H --out .
    (raw ratings are read from ../data by default; --allrecipes / --foodcom point to other folders)
Then use the patched cbf.py (accepts H with any length >= 1) in the output folder.
"""
import argparse
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

CUTOFF = pd.Timestamp("2013-02-01")
POS = 4
MAX_FIRST5_MISMATCH = 0.005   # allow tiny tie-order differences (same timestamp), report them
# Project-level raw data folder (ML_TermProject_1/data): core-data-*_rating.csv, RAW_recipes.csv, RAW_interactions.csv
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def find_file(folder, exact, pattern):
    """``folder/exact`` if it exists, otherwise the single file matching ``pattern``."""
    folder = Path(folder)
    if (folder / exact).exists():
        return folder / exact
    hits = sorted(folder.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"expected {exact} (or one file matching {pattern}) in {folder}, found {[h.name for h in hits]}")
    return hits[0]


def prefix_of(db, table, col):
    v = db.execute(f"SELECT {col} FROM {table} LIMIT 1").fetchone()[0]
    return v.rsplit(":", 1)[0] + ":" if ":" in v else ""


def load_allrecipes_past(folder):
    folder = Path(folder)
    parts = {p: pd.read_csv(find_file(folder, f"core-data-{p}_rating.csv", f"*{p}_rating*.csv")) for p in ["train", "valid"]}
    for k, d in parts.items():
        d.rename(columns={"dateLastModified": "timestamp"}, inplace=True)
        d["t"] = pd.to_datetime(d.timestamp, format="ISO8601")
    return {"valid": parts["train"], "test": pd.concat([parts["train"], parts["valid"]], ignore_index=True)}


def load_foodcom_past(folder):
    folder = Path(folder)
    r = pd.read_csv(find_file(folder, "RAW_interactions.csv", "*rating*.csv"), usecols=["user_id", "recipe_id", "date", "rating"])
    rec = pd.read_csv(find_file(folder, "RAW_recipes.csv", "*recipe*.csv"), usecols=["id", "submitted"])
    r = r.rename(columns={"date": "timestamp"})
    r["t"] = pd.to_datetime(r.timestamp)
    sub = dict(zip(rec.id, pd.to_datetime(rec.submitted)))
    r = r[(r.rating > 0) & (r.t >= r.recipe_id.map(sub)) & (r.t < CUTOFF)]
    return {"test": r}


def rebuild(db_path, past_by_split):
    with sqlite3.connect(db_path) as db:
        up, rp = prefix_of(db, "users", "user_id"), prefix_of(db, "profiles", "recipe_id")
        for split, past in past_by_split.items():
            users = [u for (u,) in db.execute("SELECT user_id FROM users WHERE split=?", (split,))]
            old = defaultdict(list)
            for u, r in db.execute("SELECT user_id, recipe_id FROM profiles WHERE split=? ORDER BY user_id, history_order", (split,)):
                old[u].append(r)
            seen, tgt = defaultdict(set), defaultdict(set)
            for u, r in db.execute("SELECT user_id, recipe_id FROM seen WHERE split=?", (split,)):
                seen[u].add(r)
            for u, r in db.execute("SELECT user_id, recipe_id FROM targets WHERE split=?", (split,)):
                tgt[u].add(r)
            cand = {r for (r,) in db.execute("SELECT recipe_id FROM candidates WHERE split=?", (split,))}

            p = past.copy()
            p["user_id"] = up + p.user_id.astype(str)
            p["recipe_id"] = rp + p.recipe_id.astype(str)
            p = p[p.user_id.isin(set(users)) & (p.rating >= POS)]
            p = p.sort_values(["user_id", "t", "recipe_id"], kind="mergesort").drop_duplicates(["user_id", "recipe_id"])
            new = p.groupby("user_id", sort=False)

            rows, mism, lengths = [], 0, []
            for u in users:
                g = new.get_group(u) if u in new.groups else p.iloc[0:0]
                h = list(g.recipe_id)
                # checks
                if not set(h) <= seen[u]:
                    raise SystemExit(f"[{db_path.name}/{split}] check 2 failed for {u}: H recipe not in seen")
                if set(h) & tgt[u]:
                    raise SystemExit(f"[{db_path.name}/{split}] check 3 failed for {u}: H overlaps targets")
                if not set(h) <= cand:
                    raise SystemExit(f"[{db_path.name}/{split}] check 4 failed for {u}: H recipe not a candidate")
                if set(h[:5]) != set(old[u]):
                    mism += 1
                    if not set(old[u]) <= set(h):
                        raise SystemExit(f"[{db_path.name}/{split}] check 1 failed for {u}: teammate H not inside new H")
                    h = old[u] + [r for r in h if r not in set(old[u])]   # keep teammate's first 5 exactly
                lengths.append(len(h))
                rows += [(split, u, r, i, int(rt), ts) for i, (r, rt, ts) in
                         enumerate(zip(h, g.set_index("recipe_id").reindex(h).rating.fillna(POS),
                                       g.set_index("recipe_id").reindex(h).timestamp.fillna("")))]
            rate = mism / len(users)
            if rate > MAX_FIRST5_MISMATCH:
                raise SystemExit(f"[{db_path.name}/{split}] check 1 failed: first-5 differs for {rate:.2%} of users")
            db.execute("DELETE FROM profiles WHERE split=?", (split,))
            db.executemany("INSERT INTO profiles VALUES (?,?,?,?,?,?)", rows)
            s = pd.Series(lengths)
            print(f"  {db_path.name}/{split}: {len(users):,} users | H length mean {s.mean():.1f}, median {s.median():.0f}, "
                  f"max {s.max():,} | first-5 tie-order differences {mism} ({rate:.2%}) | checks passed")
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('h_definition', '\"all past distinct positives (rating >= 4)\"')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-dir", required=True, help="folder with the teammate's allrecipes.db and foodcom.db")
    ap.add_argument("--allrecipes", default=DATA_DIR,
                    help="raw Allrecipes folder with core-data-{train,valid}_rating.csv (default: project data folder)")
    ap.add_argument("--foodcom", default=DATA_DIR,
                    help="raw Food.com folder with RAW_interactions.csv, RAW_recipes.csv (default: project data folder)")
    ap.add_argument("--out", required=True, help="new folder for the full-H databases")
    args = ap.parse_args()
    src, out = Path(args.db_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["allrecipes.db", "foodcom.db", "cbf_model.json"]:
        shutil.copy2(src / name, out / name)
    print("Allrecipes")
    rebuild(out / "allrecipes.db", load_allrecipes_past(args.allrecipes))
    print("Food.com")
    rebuild(out / "foodcom.db", load_foodcom_past(args.foodcom))
    print(f"done -> {out}/  (teammate's original files were not modified)")


if __name__ == "__main__":
    main()
