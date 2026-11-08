"""Split Allrecipes train ratings for LightFM training.

- Sort each user's ratings by timestamp (ties broken by recipe_id)
- First 5 distinct positive (rating >= 4) recipes -> H (used as user features)
- Later positive recipes -> training targets
- Users with fewer than 6 positive recipes are dropped
- (Optional) negatives for logistic loss: ratings 1-3 of the kept users, excluding H recipes

Run: python split_train.py
Output: 3 CSV files in train_split/ + summary statistics
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = "allrecipes.db"
OUT_DIR = Path("train_split")
H_SIZE = 5
POSITIVE = 4


def to_sortable_time(col):
    if pd.api.types.is_numeric_dtype(col):
        return col
    return pd.to_datetime(col, format="ISO8601")


def main():
    with sqlite3.connect(DB_PATH) as conn:
        train = pd.read_sql_query(
            "SELECT user_id, recipe_id, rating, timestamp FROM interactions WHERE partition='train'", conn)
        recipe_ids = set(r[0] for r in conn.execute("SELECT recipe_id FROM recipes"))

    print(f"Train ratings: {len(train):,} / users: {train.user_id.nunique():,} / recipes: {train.recipe_id.nunique():,}")
    print("Timestamp example:", train.timestamp.iloc[0])

    train["t"] = to_sortable_time(train["timestamp"])
    train = train.sort_values(["user_id", "t", "recipe_id"], kind="mergesort").reset_index(drop=True)

    # Distinct positive recipes only, in order of first appearance
    pos = train[train.rating >= POSITIVE].drop_duplicates(["user_id", "recipe_id"], keep="first").copy()
    pos["order"] = pos.groupby("user_id").cumcount()

    n_pos = pos.groupby("user_id").size()
    keep_users = n_pos[n_pos > H_SIZE].index
    pos = pos[pos.user_id.isin(keep_users)]

    H = pos[pos.order < H_SIZE][["user_id", "recipe_id", "order"]].rename(columns={"order": "history_order"})
    targets = pos[pos.order >= H_SIZE][["user_id", "recipe_id", "rating", "timestamp"]]

    # Negatives for logistic loss (rating 1-3), excluding H recipes
    neg = train[(train.rating < POSITIVE) & train.user_id.isin(keep_users)]
    neg = neg.drop_duplicates(["user_id", "recipe_id"])
    h_pairs = set(zip(H.user_id, H.recipe_id))
    neg = neg[[p not in h_pairs for p in zip(neg.user_id, neg.recipe_id)]][["user_id", "recipe_id", "rating", "timestamp"]]

    # Sanity checks
    assert (H.groupby("user_id").size() == H_SIZE).all()
    assert set(H.user_id) == set(targets.user_id)
    missing = (set(H.recipe_id) | set(targets.recipe_id)) - recipe_ids
    assert not missing, f"{len(missing)} recipes not found in recipes table"

    OUT_DIR.mkdir(exist_ok=True)
    H.to_csv(OUT_DIR / "train_H.csv", index=False)
    targets.to_csv(OUT_DIR / "train_targets.csv", index=False)
    neg.to_csv(OUT_DIR / "train_negatives.csv", index=False)

    per_user = targets.groupby("user_id").size()
    print("\n=== Split result ===")
    print(f"Positive rating ratio: {(train.rating >= POSITIVE).mean():.1%}")
    print(f"Kept users: {len(keep_users):,} (dropped {train.user_id.nunique() - len(keep_users):,})")
    print(f"H rows: {len(H):,}")
    print(f"Training targets: {len(targets):,} / recipes: {targets.recipe_id.nunique():,}")
    print(f"Targets per user: mean {per_user.mean():.1f}, median {per_user.median():.0f}, max {per_user.max():,}")
    print(f"Negatives (rating 1-3): {len(neg):,}")
    print(f"Recipes used (H + targets): {len(set(H.recipe_id) | set(targets.recipe_id)):,}")
    print(f"\nSaved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()