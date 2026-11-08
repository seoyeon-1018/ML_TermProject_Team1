"""Parse foodRecSys-V1 and preserve its verified chronological file boundaries.

Recipe publication dates are absent. Catalog availability uses first observed
rating as a conservative proxy, never an invented publication date.
"""
import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .io_utils import DATA_DIR, open_source, source_file, write_frame, write_json, write_jsonl
from .normalize import DEFAULT_ALIASES, IngredientNormalizer
from .prepare import sha256
from .split import split_h_l


def read_ratings(archive, name, role):
    """Read one foodRecSys rating CSV and convert it to the shared schema.

    ``archive`` is a folder that holds ``name`` (default: project data folder),
    the CSV itself, or a ZIP that contains it.

    Rows with a non-numeric ID, an unparsable dateLastModified or a rating outside
    1-5 are rejected and counted. IDs get a platform prefix ("allrecipes:u:", "allrecipes:r:")
    so they can never collide with Food.com IDs. Returns (frame, n_rejected, n_raw_rows).
    """
    with open_source(archive, name) as f:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
    if not {"user_id", "recipe_id", "rating", "dateLastModified"}.issubset(df.columns):
        raise ValueError(f"Missing rating columns in {name}")
    df["timestamp_raw"] = df.dateLastModified.str.strip()
    ts = pd.to_datetime(df.timestamp_raw, format="ISO8601", errors="coerce")
    rating = pd.to_numeric(df.rating, errors="coerce")
    valid = (df.user_id.str.fullmatch(r"[0-9]+") & df.recipe_id.str.fullmatch(r"[0-9]+")
             & ts.notna() & rating.isin([1, 2, 3, 4, 5]))
    rejected = int((~valid).sum())
    df = df.loc[valid].copy()
    df["timestamp"] = ts[valid].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    df["date"] = ts[valid].dt.strftime("%Y-%m-%d")
    df["rating"] = rating[valid].astype("int8")
    df["user_id"] = "allrecipes:u:" + df.user_id
    df["recipe_id"] = "allrecipes:r:" + df.recipe_id
    df["original_partition"] = role
    return df[["user_id", "recipe_id", "date", "rating", "timestamp", "original_partition"]], rejected, len(valid)


def prepare_source(recipe_archive, train_archive, valid_archive, test_archive, output):
    """Normalize Allrecipes recipes, clean ratings and build the valid/test H/L splits.

    The original train/valid/test files are kept as time partitions (asserted to be
    non-overlapping). The valid cutoff is the first valid timestamp, the test cutoff
    the first test timestamp. A recipe is a candidate if it was first rated before
    the cutoff (no publication date exists). H uses the "first" rule so valid/test
    profiles equal the profile built from train.
    """
    output = Path(output)
    data = output / "data" / "allrecipes"
    report = output / "reports" / "allrecipes_preparation"
    data.mkdir(parents=True, exist_ok=True)
    frames, counts, provenance = {}, {}, []
    for role, path in [("train", train_archive), ("valid", valid_archive), ("test", test_archive)]:
        name = f"core-data-{role}_rating.csv"
        df, invalid, n_raw = read_ratings(path, name, role)
        frames[role] = df
        counts[role] = {"raw_rows": n_raw, "invalid_rating_or_identifier_or_time": invalid,
                        "timestamp_min": df.timestamp.min(), "timestamp_max": df.timestamp.max()}
        provenance.append({"role": role, "archive": source_file(path, name).name, "sha256": sha256(source_file(path, name)), "member": name,
                           "origin": "user upload" if role == "test" else "Kaggle public dataset version 1"})
    assert frames["train"].timestamp.max() < frames["valid"].timestamp.min()
    assert frames["valid"].timestamp.max() < frames["test"].timestamp.min()
    valid_start = frames["valid"].timestamp.min()
    test_start = frames["test"].timestamp.min()
    normalizer, recipes, mapping, rejected, seen_recipe_ids = IngredientNormalizer(), [], Counter(), Counter(), set()
    raw_count = 0
    csv.field_size_limit(10000000)
    with open_source(recipe_archive, "core-data_recipe.csv") as raw_file:
        with io.TextIOWrapper(raw_file, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                raw_count += 1
                rid = row["recipe_id"].strip()
                if not rid.isdigit():
                    rejected["invalid_recipe_id"] += 1
                    continue
                if rid in seen_recipe_ids:
                    raise ValueError(f"Duplicate recipe ID: {rid}")
                raw = row["ingredients"].split("^")  # Allrecipes separates ingredients with ^
                ingredients = normalizer.normalize_list(raw)
                if not ingredients:
                    rejected["empty_ingredients"] += 1
                    continue
                seen_recipe_ids.add(rid)
                for s in raw:
                    mapping[(s, normalizer.normalize_one(s))] += 1
                recipes.append({"dataset": "allrecipes", "recipe_id": f"allrecipes:r:{rid}",
                                "raw_recipe_id": rid, "name": row.get("recipe_name", ""),
                                "submitted": None, "ingredients_raw": raw, "ingredients": ingredients})
    known = {r["recipe_id"] for r in recipes}
    for role, df in frames.items():
        counts[role]["unknown_recipe_rows"] = int((~df.recipe_id.isin(known)).sum())
        df = df[df.recipe_id.isin(known)].copy()
        duplicates = df.duplicated(["user_id", "recipe_id", "timestamp", "rating"])
        counts[role]["exact_duplicate_rows"] = int(duplicates.sum())
        df = df[~duplicates].sort_values(["timestamp", "user_id", "recipe_id"], kind="stable")
        counts[role]["kept"] = len(df)
        frames[role] = df
        write_frame(data / f"{role}.csv.gz", df)
    combined = pd.concat(list(frames.values()), ignore_index=True)
    duplicate_pairs = int(combined.duplicated(["user_id", "recipe_id"]).sum())
    if duplicate_pairs:
        raise ValueError("Repeated user-item pairs across partitions require an explicit policy")
    # Availability proxy: the first time anyone rated the recipe.
    first_seen = combined.groupby("recipe_id").timestamp.min().to_dict()
    for r in recipes:
        r["first_observed_at"] = first_seen.get(r["recipe_id"])
    write_jsonl(data / "recipes.jsonl.gz", recipes)
    write_frame(data / "ingredient_mapping.csv.gz", pd.DataFrame([
        {"raw_ingredient": a, "normalized_ingredient": b, "occurrences": n}
        for (a, b), n in sorted(mapping.items())]))
    split_stats = {}
    for name, start, end in [("valid", valid_start, test_start), ("test", test_start, None)]:
        catalog = {rid for rid, ts in first_seen.items() if ts < start}
        h, l, seen, users, stats = split_h_l(combined, catalog, cutoff=start, time_column="timestamp", target_end=end,
                                              profile_rule="first")  # keep the train-built profile (team plan, section 7)
        split_dir = data / f"{name}_evaluation"
        split_dir.mkdir(exist_ok=True)
        for filename, frame in [("H.csv.gz", h), ("L.csv.gz", l), ("seen.csv.gz", seen),
                                ("evaluation_users.csv", users), ("candidate_items.csv.gz", pd.DataFrame({"recipe_id": sorted(catalog)}))]:
            write_frame(split_dir / filename, frame)
        split_stats[name] = {**stats, "history_before": start, "target_end_exclusive": end,
                             "candidate_items": len(catalog), "catalog_availability": "first observed valid rating timestamp < history cutoff"}
    recipe_file = source_file(recipe_archive, "core-data_recipe.csv")
    provenance.append({"role": "recipes", "archive": recipe_file.name, "sha256": sha256(recipe_file),
                       "member": "core-data_recipe.csv", "origin": "Kaggle public dataset version 1"})
    report_obj = {"dataset": "allrecipes", "source_url": "https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1",
                  "source_version": 1, "files": provenance, "rating_partitions": counts,
                  "recipes": {"raw": raw_count, "kept": len(recipes), "removed": dict(rejected),
                              "raw_unique_ingredients": len(mapping),
                              "normalized_unique_ingredients": len({s for r in recipes for s in r["ingredients"]})},
                  "split": split_stats, "combined": {"ratings": len(combined), "users": int(combined.user_id.nunique()),
                                                       "rated_recipes": int(combined.recipe_id.nunique()), "duplicate_user_item_pairs": duplicate_pairs},
                  "limitations": ["dateLastModified is a modification timestamp, not a confirmed original rating time",
                                  "recipe publication dates absent; first observed rating used as conservative availability proxy",
                                  "historical metadata snapshots absent; current recipe text may differ from historical text"],
                  "checks": {"source_train_before_valid": True, "source_valid_before_test": True,
                             "allrecipes_train_end_before_foodcom_cutoff": bool(frames["train"].timestamp.max() < "2013-02-01"),
                             "source_original_partitions_preserved": True, "h_l_and_seen_l_overlap_zero": True},
                  "normalization_alias_sha256": sha256(DEFAULT_ALIASES),
                  "normalization_code_sha256": sha256(Path(__file__).with_name("normalize.py"))}
    write_json(report / "data_quality.json", report_obj)
    print(json.dumps(report_obj, ensure_ascii=False, indent=2), flush=True)
    return report_obj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=DATA_DIR, type=Path,
                   help="folder with core-data_recipe.csv and core-data-{train,valid,test}_rating.csv "
                        "(default: project data folder)")
    for n in ["recipe-archive", "train-archive", "valid-archive", "test-archive"]:
        p.add_argument("--" + n, type=Path, help="optional: this file (CSV or ZIP) instead of the one in --data-dir")
    p.add_argument("--output", default=Path("."), type=Path)
    a = p.parse_args()
    pick = lambda path: path if path is not None else a.data_dir
    prepare_source(pick(a.recipe_archive), pick(a.train_archive), pick(a.valid_archive), pick(a.test_archive), a.output)


if __name__ == "__main__":
    main()
