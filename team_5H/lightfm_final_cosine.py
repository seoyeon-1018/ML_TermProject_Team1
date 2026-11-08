"""Improvement (post-hoc, after diagnosis): LightFM with COSINE scoring.

Same as lightfm_final.py (same features, training, loss=warp, dim=32, lr=0.02, alpha=0, seeds 42/43/44).
Only the scoring rule changes:
    original: score = user bias + item bias + (user vector . item vector)
    cosine  : score = cos(user vector, item vector)      (biases removed, vector sizes equalized)
Why: diagnosis showed recipes with only 1-2 known ingredients get very large item vectors on Food.com
and fill the top-10. Cosine scoring removes the effect of vector size (same idea as CBF).

The best epoch is selected on Allrecipes valid using the cosine score itself.
Original results are read from lightfm_results/final_test.json for comparison (run lightfm_final.py first).

Run (team_essentials folder, mlteam env):
    python lightfm_final_cosine.py
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lightfm import LightFM

from cbf import load_evaluation, load_frozen
from lightfm_smoke import build_training, check_lightfm, metrics, prepare_eval, rank

CONFIG = SimpleNamespace(loss="warp", dim=32, lr=0.02, alpha=0.0, max_epochs=20, patience=5,
                         threads=os.cpu_count() or 1, db="allrecipes.db")
SEEDS = [42, 43, 44]
METRICS = ["precision@10", "recall@10", "ndcg@10"]
OUT = Path("lightfm_results")


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cosine_scorer(model, prep):
    _, iemb = model.get_item_representations(prep["x_cand"])
    _, uemb = model.get_user_representations(prep["x_users"])
    iemb, uemb = unit(iemb), unit(uemb)
    return lambda s, e: uemb[s:e] @ iemb.T


def train_seed(seed, data, valid, valid_prep):
    inter, x_users, x_items = data
    check_lightfm(CONFIG.loss)
    model = LightFM(no_components=CONFIG.dim, loss=CONFIG.loss, learning_rate=CONFIG.lr,
                    item_alpha=CONFIG.alpha, user_alpha=CONFIG.alpha, random_state=seed)
    best, best_state, bad = None, None, 0
    print(f"\n[seed {seed}]")
    for epoch in range(1, CONFIG.max_epochs + 1):
        model.fit_partial(inter, user_features=x_users, item_features=x_items, epochs=1, num_threads=CONFIG.threads)
        ndcg = metrics(valid, rank(cosine_scorer(model, valid_prep), valid_prep))["ndcg@10"]
        if best is None or ndcg > best[1]:
            best, bad = (epoch, ndcg), 0
            best_state = {k: getattr(model, k).copy() for k in
                          ["user_embeddings", "item_embeddings", "user_biases", "item_biases"]}
            print(f"  epoch {epoch:2d}  NDCG@10 {ndcg:.6f} *")
        else:
            bad += 1
            print(f"  epoch {epoch:2d}  NDCG@10 {ndcg:.6f}")
            if bad >= CONFIG.patience:
                break
    for k, v in best_state.items():
        setattr(model, k, v)
    print(f"  -> best epoch {best[0]}, NDCG@10 {best[1]:.6f}")
    return model, best[0]


def evaluate(models, ev, prep, batch):
    per_seed, uniques, short = [], [], []
    n_known = np.diff(prep["x_cand"].indptr)
    row_of = {r: i for i, r in enumerate(prep["cand_ids"])}
    for m in models:
        rk = rank(cosine_scorer(m, prep), prep, batch)
        per_seed.append(metrics(ev, rk))
        rows = [row_of[r] for u in ev["users"] for r in rk[u]]
        uniques.append(len(set(rows)))
        short.append(float((n_known[rows] <= 2).mean()))
    out = {k: float(np.mean([s[k] for s in per_seed])) for k in METRICS}
    out["ndcg@10_std"] = float(np.std([s["ndcg@10"] for s in per_seed]))
    out["unique_recipes"] = int(np.mean(uniques))
    out["le2_known"] = float(np.mean(short))
    return out


def main():
    vec = load_frozen()
    data = build_training(CONFIG, vec)
    valid = load_evaluation("allrecipes.db", "valid")
    valid_prep = prepare_eval(valid, vec, "allrecipes.db")
    trained = [train_seed(s, data, valid, valid_prep) for s in SEEDS]
    models = [m for m, _ in trained]

    results = {"Allrecipes valid": evaluate(models, valid, valid_prep, 512)}
    del valid_prep
    for name, db in [("Allrecipes test", "allrecipes.db"), ("Food.com zero-shot", "foodcom.db")]:
        print(f"\nEvaluating {name} ...")
        ev = load_evaluation(db, "test")
        prep = prepare_eval(ev, vec, db)
        results[name] = evaluate(models, ev, prep, 64 if len(prep["cand_ids"]) > 100_000 else 512)
        del prep

    orig = {}
    if (OUT / "final_test.json").exists():
        orig = json.loads((OUT / "final_test.json").read_text())["results"]

    print("\n" + "=" * 84)
    print(f"Cosine scoring vs original (mean over {len(SEEDS)} seeds, best valid epochs {[e for _, e in trained]})")
    print("=" * 84)
    print(f"{'Dataset':<22}{'Method':<18}{'P@10':>10}{'R@10':>10}{'NDCG@10':>11}{'unique recs':>13}")
    print("-" * 84)
    for name, r in results.items():
        if name in orig:
            for label in ["CBF", "LightFM"]:
                o = orig[name][label]
                print(f"{name:<22}{label + ' (orig)':<18}{o['precision@10']:>10.6f}{o['recall@10']:>10.6f}"
                      f"{o['ndcg@10']:>11.6f}{'':>13}")
        print(f"{name:<22}{'LightFM cosine':<18}{r['precision@10']:>10.6f}{r['recall@10']:>10.6f}"
              f"{r['ndcg@10']:>11.6f}{r['unique_recipes']:>13,}")
        print("-" * 84)
    print("unique recs = number of different recipes recommended across all users")
    print("Original LightFM unique recs (seed 42): Allrecipes test 61 / Food.com 35")
    for name, r in results.items():
        print(f"{name}: top-10 recipes with <=2 known ingredients {r['le2_known']:.1%}")

    OUT.mkdir(exist_ok=True)
    (OUT / "final_test_cosine.json").write_text(json.dumps(
        {"config": vars(CONFIG), "seeds": SEEDS, "best_valid_epochs": [e for _, e in trained],
         "results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT / 'final_test_cosine.json'}")


if __name__ == "__main__":
    main()