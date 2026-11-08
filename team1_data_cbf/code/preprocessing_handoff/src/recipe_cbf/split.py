"""Fixed-cutoff H/L split; target information never enters the profile.

Terminology
-----------
H     : profile input. ``profile_size`` distinct positive recipes rated before the cutoff.
L     : ground truth. Positive recipes rated at/after the cutoff that are in the
        cutoff catalog and were never rated before the cutoff.
seen  : every recipe the user rated (1-5) before the cutoff. Excluded from ranking.

Profile rules
------------------------------------------
"first"  : the earliest ``profile_size`` distinct positives before the cutoff.
           Used for every final result (Allrecipes valid/test and Food.com), so the
           evaluation H matches the H used for LightFM training.
"recent" : the latest ``profile_size`` distinct positives before the cutoff
           (team plan section 9). Final results use "first" on every dataset;
           "recent" is kept as a Food.com sensitivity analysis.
Either way H is strictly before the cutoff and L at/after it, so the rule changes
only which past items describe the user; L, seen and the evaluated cohort are identical.
"""
import pandas as pd

PROFILE_RULES = ("first", "recent")


def split_h_l(interactions, candidate_ids, cutoff="2013-02-01", profile_size=5, positive_rating=4,
              time_column="date", target_end=None, profile_rule="first"):
    """Split interactions at ``cutoff`` into H (profile), L (targets) and seen.

    Parameters
    ----------
    interactions : DataFrame with user_id, recipe_id, rating and ``time_column``.
    candidate_ids : set of recipe IDs available at the cutoff (the catalog).
    cutoff : ISO date/time string; history is ``< cutoff``, future is ``>= cutoff``.
    profile_size : number of distinct positive recipes in H.
    positive_rating : minimum rating counted as a positive (4 -> ratings 4 and 5).
    time_column : "date" (Food.com, day precision) or "timestamp" (Allrecipes).
    target_end : optional exclusive end of the target window (Allrecipes valid).
    profile_rule : "first" or "recent" (see module docstring).

    Returns (h, l, seen, users, stats).
    """
    if profile_size < 1:
        raise ValueError("profile_size must be positive")
    if profile_rule not in PROFILE_RULES:
        raise ValueError(f"profile_rule must be one of {PROFILE_RULES}")
    # Canonical dates are ISO strings, so string comparison equals time comparison.
    # Same-day records have no true event order; ties are broken by recipe_id only
    # to make the result deterministic.
    hist = interactions[interactions[time_column] < cutoff].copy()
    future_mask = interactions[time_column] >= cutoff
    if target_end is not None:
        future_mask &= interactions[time_column] < target_end
    future = interactions[future_mask].copy()
    # seen = all past ratings (1-5), not only positives and not only H.
    seen = hist[["user_id", "recipe_id"]].drop_duplicates()
    positive = hist[(hist.rating >= positive_rating) & hist.recipe_id.isin(candidate_ids)]
    positive = positive.sort_values(["user_id", time_column, "recipe_id"], kind="stable")
    # A recipe rated positively twice counts once, at its first positive rating.
    positive = positive.drop_duplicates(["user_id", "recipe_id"], keep="first")
    counts = positive.groupby("user_id").size()
    enough_history = set(counts[counts >= profile_size].index)
    grouped = positive[positive.user_id.isin(enough_history)].groupby("user_id", sort=False)
    # "first" keeps the oldest positives, "recent" the newest; rows stay in time order.
    h = (grouped.head(profile_size) if profile_rule == "first" else grouped.tail(profile_size)).copy()
    # history_order 1..profile_size runs from oldest to newest within H.
    h["history_order"] = h.groupby("user_id").cumcount() + 1
    l = future[(future.rating >= positive_rating) & future.recipe_id.isin(candidate_ids)].copy()
    # Drop future positives the user had already rated in the past (they are seen).
    l = l.merge(seen.assign(_seen=True), on=["user_id", "recipe_id"], how="left")
    l = l[l._seen.isna()].drop(columns="_seen")
    l = l.sort_values(["user_id", time_column, "recipe_id"], kind="stable").drop_duplicates(["user_id", "recipe_id"])
    # Evaluate only users with a full profile AND at least one target.
    eligible = enough_history.intersection(l.user_id.unique())
    h = h[h.user_id.isin(eligible)].copy()
    l = l[l.user_id.isin(eligible)].copy()
    seen = seen[seen.user_id.isin(eligible)].sort_values(["user_id", "recipe_id"])
    users = pd.DataFrame({"user_id": sorted(eligible)})
    users["n_h"] = users.user_id.map(h.groupby("user_id").size()).astype(int)
    users["n_seen"] = users.user_id.map(seen.groupby("user_id").size()).astype(int)
    users["n_l"] = users.user_id.map(l.groupby("user_id").size()).astype(int)
    stats = {
        "profile_rule": profile_rule,
        "history_interactions_before_cutoff": len(hist),
        "future_interactions_at_or_after_cutoff": len(future),
        "users_with_any_history": int(hist.user_id.nunique()),
        "users_with_at_least_profile_size_positive_items": len(enough_history),
        "eligible_users": len(eligible), "h_rows": len(h), "l_unique_pairs": len(l),
        "seen_unique_pairs_for_eligible_users": len(seen),
        "future_positive_rows_outside_cutoff_catalog": int(((future.rating >= positive_rating) & ~future.recipe_id.isin(candidate_ids)).sum()),
    }
    # Leakage and consistency checks: H/L/seen never overlap with L, times respect the cutoff.
    assert len(h) == profile_size * len(users)
    assert set(h.user_id) == set(l.user_id) == set(users.user_id)
    assert not len(h.merge(l, on=["user_id", "recipe_id"]))
    assert not len(seen.merge(l, on=["user_id", "recipe_id"]))
    assert (h[time_column] < cutoff).all() and (l[time_column] >= cutoff).all()
    if target_end is not None:
        assert (l[time_column] < target_end).all()
    return h, l, seen, users, stats
