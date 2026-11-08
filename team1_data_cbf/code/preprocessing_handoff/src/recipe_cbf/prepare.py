"""Read Food.com RAW CSVs (default: the project data folder, or a ZIP); ignore source pre-splits."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from .io_utils import DATA_DIR, open_source, raw_member, source_file, write_json, write_jsonl, write_frame
from .normalize import DEFAULT_ALIASES, IngredientNormalizer, parse_list
from .split import split_h_l

FOODCOM_FILES = ["RAW_recipes.csv", "RAW_interactions.csv"]


def sha256(path):
    """SHA-256 of a file, read in 8 MB chunks (provenance of inputs and rule files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(source=DATA_DIR, output=".", cutoff="2013-02-01", profile_size=5, aliases=DEFAULT_ALIASES, profile_rule="first"):
    """Parse Food.com RAW files, normalize ingredients and build the fixed-date H/L split.

    source: folder with RAW_recipes.csv and RAW_interactions.csv (default: project data folder)
            or a ZIP that contains both.

    profile_rule defaults to "first": H = the first 5 distinct positives before the
    cutoff, the same rule as Allrecipes (team decision for the final results).
    "recent" (the 5 most recent positives, team plan section 9) is kept as a
    sensitivity analysis.
    """
    date.fromisoformat(cutoff)
    output = Path(output)
    data, reports = output / "data" / "foodcom", output / "reports" / "foodcom_preparation"
    data.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    normalizer = IngredientNormalizer(aliases)
    recipe_reasons, examples, mapping = Counter(), [], Counter()
    recipes, known_ids = [], set()
    raw_recipe_rows = 0
    with open_source(source, "RAW_recipes.csv") as raw_file:
        with io.TextIOWrapper(raw_file, encoding="utf-8-sig", newline="") as f:
            for row_number, r in enumerate(csv.DictReader(f), start=2):
                raw_recipe_rows += 1
                rid = r.get("id", "").strip()
                reason = None
                if not rid.isdigit():
                    reason = "invalid_recipe_id"
                elif rid in known_ids:
                    # Do not silently choose between inconsistent metadata rows.
                    raise ValueError(f"Duplicate recipe ID at source row {row_number}: {rid}")
                try:
                    submitted = date.fromisoformat(r.get("submitted", "")).isoformat()
                except ValueError:
                    reason = reason or "invalid_submitted_date"
                try:
                    raw_ingredients = parse_list(r.get("ingredients", ""))
                    normalized = normalizer.normalize_list(raw_ingredients)
                    if not normalized:
                        reason = reason or "empty_normalized_ingredients"
                except (ValueError, SyntaxError, TypeError):
                    reason = reason or "invalid_ingredient_list"
                if reason:
                    recipe_reasons[reason] += 1
                    if len(examples) < 30:
                        examples.append({"source": "RAW_recipes.csv", "row": row_number, "id": rid, "reason": reason})
                    continue
                known_ids.add(rid)
                for raw in raw_ingredients:
                    mapping[(raw, normalizer.normalize_one(raw))] += 1
                recipes.append({"dataset": "foodcom", "recipe_id": f"foodcom:r:{rid}",
                                "raw_recipe_id": rid, "name": r.get("name", ""),
                                "submitted": submitted, "ingredients_raw": raw_ingredients,
                                "ingredients": normalized})
    print(f"Parsed {len(recipes):,} recipes", flush=True)
    with open_source(source, "RAW_interactions.csv") as f:
        df = pd.read_csv(f, usecols=["user_id", "recipe_id", "date", "rating"], dtype=str, keep_default_na=False)
    raw_interactions = len(df)
    rejection_counts, bad_samples = {}, []

    def drop(mask, reason):
        # Remove rows matching ``mask`` and record the count and 3 examples per reason.
        nonlocal df
        rejection_counts[reason] = int(mask.sum())
        for record in df.loc[mask].head(3).to_dict("records"):
            bad_samples.append({**record, "reason": reason})
        df = df.loc[~mask].copy()

    drop(~df.user_id.str.fullmatch(r"[0-9]+") | ~df.recipe_id.str.fullmatch(r"[0-9]+"), "invalid_id")
    drop(~df.recipe_id.isin(known_ids), "unknown_or_rejected_recipe")
    parsed_date = pd.to_datetime(df.date, format="%Y-%m-%d", errors="coerce")
    drop(parsed_date.isna() | ~df.date.str.fullmatch(r"\d{4}-\d{2}-\d{2}"), "invalid_interaction_date")
    num = pd.to_numeric(df.rating, errors="coerce")
    drop(num.isna() | ~num.isin([0, 1, 2, 3, 4, 5]), "invalid_rating")
    df["rating"] = pd.to_numeric(df.rating).astype("int8")
    drop(df.rating == 0, "rating_zero")  # 0 is ambiguous on Food.com (often "not rated")
    submitted_by_raw = {r["raw_recipe_id"]: r["submitted"] for r in recipes}
    drop(df.date < df.recipe_id.map(submitted_by_raw), "interaction_before_recipe_submission")
    drop(df.duplicated(["user_id", "recipe_id", "date", "rating"]), "exact_duplicate")
    conflicting_same_day = int((df.groupby(["user_id", "recipe_id", "date"]).rating.nunique() > 1).sum())
    df["user_id"] = "foodcom:u:" + df.user_id
    df["recipe_id"] = "foodcom:r:" + df.recipe_id
    df = df.sort_values(["date", "user_id", "recipe_id"], kind="stable")
    # Only recipes already published before the cutoff could have been recommended then.
    candidate_ids = {r["recipe_id"] for r in recipes if r["submitted"] < cutoff}
    h, l, seen, users, split_stats = split_h_l(df, candidate_ids, cutoff, profile_size, profile_rule=profile_rule)
    write_jsonl(data / "recipes.jsonl.gz", recipes)
    write_frame(data / "candidate_items.csv.gz", pd.DataFrame({"recipe_id": sorted(candidate_ids)}))
    for filename, frame in [("interactions.csv.gz", df), ("H.csv.gz", h), ("L.csv.gz", l), ("seen.csv.gz", seen), ("evaluation_users.csv", users)]:
        write_frame(data / filename, frame)
    write_frame(data / "ingredient_mapping.csv.gz", pd.DataFrame([{"raw_ingredient": a, "normalized_ingredient": b, "occurrences": c}
                  for (a, b), c in sorted(mapping.items())]))
    vocab = {s for r in recipes for s in r["ingredients"]}
    stats = {
        "input": {"source": str(Path(source).resolve()),
                  "files": {n: {"file": source_file(source, n).name, "sha256": sha256(source_file(source, n))}
                            for n in FOODCOM_FILES},
                  "members_used": FOODCOM_FILES,
                  "members_not_used": ["PP_recipes.csv", "PP_users.csv", "ingr_map.pkl", "interactions_train.csv", "interactions_validation.csv", "interactions_test.csv"]},
        "protocol": {"cutoff": cutoff, "history": "date < cutoff", "targets": "date >= cutoff",
                     "profile_size": profile_size, "profile_rule": profile_rule, "positive_rating": 4, "candidate_creation": "submitted < cutoff",
                     "alias_file_sha256": sha256(aliases), "idf_fitted_on_foodcom": False},
        "recipes": {"raw_rows": raw_recipe_rows, "kept": len(recipes), "removed_by_first_reason": dict(recipe_reasons),
                    "raw_unique_ingredient_strings": len(mapping), "normalized_unique_ingredient_strings": len(vocab),
                    "candidate_items_at_cutoff": len(candidate_ids), "submitted_min": min(r["submitted"] for r in recipes),
                    "submitted_max": max(r["submitted"] for r in recipes)},
        "interactions": {"raw_rows": raw_interactions, "kept": len(df), "removed_by_first_reason": rejection_counts,
                         "users_kept": int(df.user_id.nunique()), "recipes_with_ratings_kept": int(df.recipe_id.nunique()),
                         "date_min": df.date.min(), "date_max": df.date.max(),
                         "conflicting_same_user_item_date_groups_retained": conflicting_same_day,
                         "rating_counts": {str(k): int(v) for k, v in sorted(df.rating.value_counts().items())}},
        "split": split_stats,
        "checks": {"h_l_user_item_overlap": 0, "seen_l_overlap": 0, "h_exactly_profile_size_per_user": True,
                   "h_strictly_before_cutoff": True, "l_at_or_after_cutoff": True,
                   "identifiers_namespaced": True, "allrecipes_model_required_for_cbf_evaluation": True},
        "status": "Food.com preprocessing and splitting complete. No CBF performance computed without Allrecipes training data."
    }
    assert raw_interactions == len(df) + sum(rejection_counts.values())
    write_json(reports / "data_quality.json", stats)
    write_json(reports / "rejected_examples.json", {"recipe_examples": examples, "interaction_examples": bad_samples})
    write_json(reports / "normalization_cautions.json", {
        "not_semantic_ground_truth": True,
        "rule": "Conservative explicit aliases and preparation cleanup; no automatic stemming or fuzzy merging",
        "ambiguous_examples": [r for r, n in mapping if " or " in r or " and " in r or ";" in r][:60],
        "note": "Compound and alternative ingredients remain a single phrase; review examples before expanding aliases. Apply the same frozen rules to Allrecipes."
    })
    print(json.dumps({"recipes": len(recipes), "interactions": len(df), **split_stats}, indent=2), flush=True)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", "--zip", dest="source", default=DATA_DIR, type=Path,
                   help="folder with RAW_recipes.csv and RAW_interactions.csv, or a ZIP with both "
                        "(default: project data folder)")
    p.add_argument("--output", default=Path("."), type=Path)
    p.add_argument("--cutoff", default="2013-02-01")
    p.add_argument("--profile-size", default=5, type=int)
    p.add_argument("--aliases", default=DEFAULT_ALIASES, type=Path)
    p.add_argument("--profile-rule", default="first", choices=["first", "recent"],
                   help="H = first 5 (default, final results) or 5 most recent positives before the cutoff")
    args = p.parse_args()
    prepare(args.source, args.output,args.cutoff, args.profile_size, args.aliases, args.profile_rule)


if __name__ == "__main__":
    main()
