"""Final evaluation: Allrecipes TEST + Food.com zero-shot TEST. Run ONCE after tuning.

For each seed: train on Allrecipes train, pick the best epoch on Allrecipes valid (early stopping),
then evaluate that frozen model on Allrecipes test and Food.com test (no retraining, no tuning).
Reported LightFM numbers are mean +- std over seeds.

Baselines on the same cohorts:
  Random, Popularity (Allrecipes only: Food.com recipes have no train ratings), CBF (cbf.py)

Run (team_essentials folder, mlteam env):
    python lightfm_final.py
"""
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
import sqlite3

import numpy as np

from cbf import load_evaluation, load_frozen
from lightfm_smoke import (build_training, cbf_rankings, lightfm_scorer, metrics,
                           prepare_eval, rank, train_one)

TARGETS = [("Allrecipes test", "allrecipes.db"), ("Food.com zero-shot", "foodcom.db")]
METRICS = ["precision@10", "recall@10", "ndcg@10"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loss", default="warp")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = p.parse_args()
    t0 = time.time()
    print(f"Final config | loss={args.loss}, dim={args.dim}, lr={args.lr}, alpha={args.alpha}, seeds={args.seeds}")

    vec = load_frozen()
    train_args = SimpleNamespace(loss=args.loss, db="allrecipes.db")
    inter, x_users, x_items = build_training(train_args, vec)

    # ---- 1. train each seed, select epoch on Allrecipes valid ----
    valid = load_evaluation("allrecipes.db", "valid")
    valid_prep = prepare_eval(valid, vec, "allrecipes.db")
    models = []
    for seed in args.seeds:
        model, best, _ = train_one(args, seed, inter, x_users, x_items, valid, valid_prep)
        models.append((seed, model, best))
    del valid_prep

    # ---- 2. evaluate on test sets ----
    with sqlite3.connect("allrecipes.db") as conn:
        counts = dict(conn.execute("SELECT recipe_id, COUNT(*) FROM interactions "
                                   "WHERE partition='train' AND rating>=4 GROUP BY recipe_id"))
    results, info = {}, {}
    for name, db in TARGETS:
        print(f"\nEvaluating {name} ...")
        ev = load_evaluation(db, "test")
        prep = prepare_eval(ev, vec, db)
        n_cand = len(prep["cand_ids"])
        batch = 64 if n_cand > 100_000 else 512  # keep memory low on Food.com
        _, diag = vec.transform([r["ingredients"] for r in ev["recipes"]])
        info[name] = {"users": len(ev["users"]), "candidates": n_cand,
                      "oov_rate": diag["oov_rate"], "zero_vector_candidates": diag["zero_vector_documents"]}

        rng = np.random.default_rng(0)
        r = {"Random": metrics(ev, rank(lambda s, e: rng.random((e - s, n_cand)), prep, batch))}
        if db == "allrecipes.db":
            pop = np.array([counts.get(x, 0) for x in prep["cand_ids"]], dtype=np.float64)
            r["Popularity"] = metrics(ev, rank(lambda s, e: np.broadcast_to(pop, (e - s, n_cand)), prep, batch))
        r["CBF"] = metrics(ev, cbf_rankings(vec, ev, batch=64))
        for part, label in [("full", "LightFM"), ("bias", "  - bias only"), ("embedding", "  - embedding only")]:
            per_seed = [metrics(ev, rank(lightfm_scorer(m, prep, part), prep, batch)) for _, m, _ in models]
            r[label] = {k: float(np.mean([s[k] for s in per_seed])) for k in METRICS}
            r[label].update({f"{k}_std": float(np.std([s[k] for s in per_seed])) for k in METRICS})
        results[name] = r
        del prep

    # ---- 3. report ----
    a, f = (results[n] for n, _ in TARGETS)
    print("\n" + "=" * 86)
    print(f"Final test results (LightFM = mean over {len(models)} seeds, best valid epochs "
          f"{[b['epoch'] for _, _, b in models]})")
    print("=" * 86)
    print(f"{'':<22}{'------ Allrecipes test ------':^32}{'---- Food.com zero-shot ----':^32}")
    print(f"{'Method':<22}" + f"{'P@10':>10}{'R@10':>11}{'NDCG@10':>11}" * 2)
    print("-" * 86)
    for method in a:
        row = f"{method:<22}" + "".join(f"{a[method][k]:>{10 if i == 0 else 11}.6f}" for i, k in enumerate(METRICS))
        row += ("".join(f"{f[method][k]:>{10 if i == 0 else 11}.6f}" for i, k in enumerate(METRICS))
                if method in f else f"{'n/a':>10}{'n/a':>11}{'n/a':>11}")
        print(row)
    print("-" * 86)
    for name, _ in TARGETS:
        std = results[name]["LightFM"]["ndcg@10_std"]
        i = info[name]
        print(f"{name}: {i['users']:,} users, {i['candidates']:,} candidates, "
              f"ingredient OOV {i['oov_rate']:.2%}, LightFM NDCG@10 std {std:.6f}")
    print("Popularity n/a on Food.com: its recipes have no Allrecipes train ratings (not zero-shot transferable).")

    out = Path("lightfm_results")
    out.mkdir(exist_ok=True)
    report = {"config": vars(args), "best_valid_epochs": {s: b["epoch"] for s, _, b in models},
              "info": info, "results": results}
    (out / "final_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out / 'final_test.json'}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()