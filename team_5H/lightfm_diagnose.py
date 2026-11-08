"""Why does LightFM fail on Food.com? Diagnose WHAT it recommends (no tuning, no model change).

Hypothesis: on Food.com many recipes have only 1-2 ingredients known to the Allrecipes vocabulary.
After L2 normalization all weight goes to those few ingredients -> item embedding becomes large
-> those recipes get extreme scores and fill everyone's top-10.

For Allrecipes test (works) and Food.com (fails), compare these recipe groups:
  All candidates / Target recipes (true answers) / CBF top-10 / LightFM top-10
on: known ingredients per recipe, item embedding norm, item bias, unique recipes recommended.
If the hypothesis is right, LightFM top-10 on Food.com should show few known ingredients and large norms.

Run (team_essentials folder, mlteam env):
    python lightfm_diagnose.py
"""
import os
from collections import Counter
from types import SimpleNamespace

import numpy as np

from cbf import load_evaluation, load_frozen
from lightfm_smoke import build_training, cbf_rankings, lightfm_scorer, prepare_eval, rank, train_one

CONFIG = SimpleNamespace(loss="warp", dim=32, lr=0.02, alpha=0.0, max_epochs=20, patience=5,
                         threads=os.cpu_count() or 1, db="allrecipes.db")
SEED = 42
DATASETS = [("Allrecipes test", "allrecipes.db"), ("Food.com zero-shot", "foodcom.db")]


def group_stats(rows, n_known, norm, bias):
    rows = np.asarray(rows)
    return {"recipes": len(set(rows.tolist())),
            "known": n_known[rows].mean(),
            "le2": (n_known[rows] <= 2).mean(),
            "norm": norm[rows].mean(),
            "bias": bias[rows].mean()}


def main():
    vec = load_frozen()
    inter, x_users, x_items = build_training(CONFIG, vec)
    valid = load_evaluation("allrecipes.db", "valid")
    model, best, _ = train_one(CONFIG, SEED, inter, x_users, x_items, valid,
                               prepare_eval(valid, vec, "allrecipes.db"))

    for name, db in DATASETS:
        ev = load_evaluation(db, "test")
        prep = prepare_eval(ev, vec, db)
        cand_ids = prep["cand_ids"]
        row_of = {r: i for i, r in enumerate(cand_ids)}
        batch = 64 if len(cand_ids) > 100_000 else 512

        n_known = np.diff(prep["x_cand"].indptr)                 # ingredients found in vocabulary
        bias, emb = model.get_item_representations(prep["x_cand"])
        norm = np.linalg.norm(emb, axis=1)

        lfm = rank(lightfm_scorer(model, prep), prep, batch)
        cbf = cbf_rankings(vec, ev, batch=64)
        lfm_rows = [row_of[r] for u in ev["users"] for r in lfm[u]]
        cbf_rows = [row_of[r] for u in ev["users"] for r in cbf[u]]
        tgt_rows = [row_of[r] for u in ev["users"] for r in ev["L"][u]]

        groups = {"All candidates": range(len(cand_ids)), "Target recipes (answers)": tgt_rows,
                  "CBF top-10": cbf_rows, "LightFM top-10": lfm_rows}

        print("\n" + "=" * 92)
        print(f"{name}: {len(ev['users']):,} users, {len(cand_ids):,} candidates")
        print("=" * 92)
        print(f"{'Group':<28}{'unique recipes':>15}{'known ingr.':>13}{'<=2 known':>12}{'emb norm':>12}{'item bias':>12}")
        print("-" * 92)
        for g, rows in groups.items():
            s = group_stats(list(rows), n_known, norm, bias)
            print(f"{g:<28}{s['recipes']:>15,}{s['known']:>13.2f}{s['le2']:>12.1%}{s['norm']:>12.3f}{s['bias']:>12.3f}")
        print("-" * 92)

        names = {r["recipe_id"]: r["name"] for r in ev["recipes"]}
        ingredients = {r["recipe_id"]: r["ingredients"] for r in ev["recipes"]}
        print("Most recommended by LightFM:")
        for rid, cnt in Counter(r for u in ev["users"] for r in lfm[u]).most_common(5):
            i = row_of[rid]
            known = [x for x in ingredients[rid] if x in vec.vocabulary]
            print(f"  {cnt:>6,} users | {str(names[rid])[:40]:<40} | known {n_known[i]}/{len(set(ingredients[rid]))} "
                  f"| norm {norm[i]:.2f} | {', '.join(known[:4])}")

    print("\nHow to read: if 'LightFM top-10' on Food.com has far fewer known ingredients and a larger emb norm")
    print("than 'All candidates' / 'Target recipes', the hypothesis is confirmed.")


if __name__ == "__main__":
    main()
