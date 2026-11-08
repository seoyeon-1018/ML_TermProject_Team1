"""Split Allrecipes train ratings for LightFM training (full-H version).

Evaluation now uses ALL past positives as H, so training users are built the same way:
- Sort each user's ratings by timestamp (ties broken by recipe_id)
- Distinct positive (rating >= 4) recipes in time order
- Earlier part (H_FRAC, default 80%) -> H (used as user features)
- Later part -> training targets (at least 1 recipe each side)
- Users with fewer than 2 distinct positive recipes are dropped
- (Optional) negatives for logistic loss: ratings 1-3 of the kept users, excluding H recipes

Run: python split_train.py            (or: python split_train.py --h-frac 0.7)
Output: 3 CSV files in train_split/ + summary statistics
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = "allrecipes.db"
OUT_DIR = Path("train_split")
POSITIVE = 4


def to_sortable_time(col):
    if pd.api.types.is_numeric_dtype(col):
        return col
    return pd.to_datetime(col, format="ISO8601")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-frac", type=float, default=0.8, help="share of each user's positives used as H")
    args = ap.parse_args()

    with sqlite3.connect(DB_PATH) as conn:
        train = pd.read_sql_query(
            "SELECT user_id, recipe_id, rating, timestamp FROM interactions WHERE partition='train'", conn)
        recipe_ids = set(r[0] for r in conn.execute("SELECT recipe_id FROM recipes"))

    print(f"Train ratings: {len(train):,} / users: {train.user_id.nunique():,} / recipes: {train.recipe_id.nunique():,}")
    train["t"] = to_sortable_time(train["timestamp"])
    train = train.sort_values(["user_id", "t", "recipe_id"], kind="mergesort").reset_index(drop=True)

    # Distinct positive recipes only, in order of first appearance
    pos = train[train.rating >= POSITIVE].drop_duplicates(["user_id", "recipe_id"], keep="first").copy()
    pos["order"] = pos.groupby("user_id").cumcount()
    n_pos = pos.groupby("user_id").size()
    keep_users = n_pos[n_pos >= 2].index
    pos = pos[pos.user_id.isin(keep_users)]

    # Per-user cut: first round(n * h_frac) positives -> H, kept within [1, n - 1]
    n_h = np.clip(np.round(n_pos[keep_users] * args.h_frac).astype(int), 1, n_pos[keep_users] - 1)
    pos["n_h"] = pos.user_id.map(n_h)
    H = pos[pos.order < pos.n_h][["user_id", "recipe_id", "order"]].rename(columns={"order": "history_order"})
    targets = pos[pos.order >= pos.n_h][["user_id", "recipe_id", "rating", "timestamp"]]

    # Negatives for logistic loss (rating 1-3), excluding H recipes
    neg = train[(train.rating < POSITIVE) & train.user_id.isin(keep_users)]
    neg = neg.drop_duplicates(["user_id", "recipe_id"])
    h_pairs = set(zip(H.user_id, H.recipe_id))
    neg = neg[[p not in h_pairs for p in zip(neg.user_id, neg.recipe_id)]][["user_id", "recipe_id", "rating", "timestamp"]]

    # Sanity checks
    assert set(H.user_id) == set(targets.user_id)
    missing = (set(H.recipe_id) | set(targets.recipe_id)) - recipe_ids
    assert not missing, f"{len(missing)} recipes not found in recipes table"

    OUT_DIR.mkdir(exist_ok=True)
    H.to_csv(OUT_DIR / "train_H.csv", index=False)
    targets.to_csv(OUT_DIR / "train_targets.csv", index=False)
    neg.to_csv(OUT_DIR / "train_negatives.csv", index=False)

    h_len = H.groupby("user_id").size()
    per_user = targets.groupby("user_id").size()
    print(f"\n=== Split result (H = first {args.h_frac:.0%} of each user's positives) ===")
    print(f"Kept users: {len(keep_users):,} (dropped {train.user_id.nunique() - len(keep_users):,})")
    print(f"H rows: {len(H):,} | H per user: mean {h_len.mean():.1f}, median {h_len.median():.0f}, max {h_len.max():,}")
    print(f"Training targets: {len(targets):,} / recipes: {targets.recipe_id.nunique():,}")
    print(f"Targets per user: mean {per_user.mean():.1f}, median {per_user.median():.0f}, max {per_user.max():,}")
    print(f"Negatives (rating 1-3): {len(neg):,}")
    print(f"\nSaved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
