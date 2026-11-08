"""Last improvement attempt (post-hoc): change HOW LightFM learns, not how it scores.

Diagnosis: LightFM learned "popular ingredients" and recommended almost the same recipes to everyone.
  (1) Popularity down-weighting: each training interaction gets weight 1 / popularity^beta
      (popularity = how many train users liked that recipe). beta=0 means no change.
  (4) Stronger regularization: alpha (L2 penalty on feature embeddings).
Everything else is the same as lightfm_final.py (warp, dim=32, lr=0.02, original scoring).

Rule fixed BEFORE looking at any result (selection uses Allrecipes valid only):
  - A config/epoch is eligible only if its valid top-10 lists contain >= MIN_UNIQUE different recipes
    (original model: ~61 -> "same recipes for everyone").
  - Among eligible ones, pick the highest valid NDCG@10. If none is eligible, pick the highest NDCG@10 and say so.
  - Food.com is evaluated ONCE with the selected config. This is the final attempt.

Run (team_essentials folder, mlteam env; lightfm_final.py and lightfm_final_cosine.py already run):
    python lightfm_final_reweight.py
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lightfm import LightFM
from scipy import sparse

from cbf import load_evaluation, load_frozen
from lightfm_smoke import build_training, check_lightfm, lightfm_scorer, metrics, prepare_eval, rank

CONFIG = SimpleNamespace(loss="warp", dim=32, lr=0.02, max_epochs=20, patience=5,
                         threads=os.cpu_count() or 1, db="allrecipes.db")
BETAS = [0.0, 0.5, 1.0]
ALPHAS = [0.0, 1e-4, 1e-3]
MIN_UNIQUE = 500
GRID_SEED = 42
FINAL_SEEDS = [42, 43, 44]
METRICS = ["precision@10", "recall@10", "ndcg@10"]
OUT = Path("lightfm_results")
STATE = ["user_embeddings", "item_embeddings", "user_biases", "item_biases"]


def make_weights(inter, beta):
    """sample_weight with the same (row, col) order as inter; weight = 1/pop^beta, mean 1."""
    inter = inter.tocoo()
    pos = inter.data > 0
    pop = np.bincount(inter.col[pos], minlength=inter.shape[1]).astype(np.float64)
    w = 1.0 / np.power(np.maximum(pop[inter.col], 1.0), beta)
    w = (w / w.mean()).astype(np.float32)
    return sparse.coo_matrix((w, (inter.row, inter.col)), shape=inter.shape)


def better(cand, best):
    """Eligible (enough unique recipes) beats non-eligible; then higher NDCG wins."""
    if best is None:
        return True
    ce, be = cand["unique"] >= MIN_UNIQUE, best["unique"] >= MIN_UNIQUE
    return (ce and not be) or (ce == be and cand["ndcg@10"] > best["ndcg@10"])


def train(seed, beta, alpha, data, valid, prep, verbose=False):
    inter, x_users, x_items = data
    weights = make_weights(inter, beta)
    check_lightfm(CONFIG.loss)
    model = LightFM(no_components=CONFIG.dim, loss=CONFIG.loss, learning_rate=CONFIG.lr,
                    item_alpha=alpha, user_alpha=alpha, random_state=seed)
    best, state, bad = None, None, 0
    for epoch in range(1, CONFIG.max_epochs + 1):
        model.fit_partial(inter, user_features=x_users, item_features=x_items, sample_weight=weights,
                          epochs=1, num_threads=CONFIG.threads)
        rk = rank(lightfm_scorer(model, prep), prep)
        cand = {"epoch": epoch, "ndcg@10": metrics(valid, rk)["ndcg@10"],
                "unique": len({r for u in valid["users"] for r in rk[u]})}
        if better(cand, best):
            best, bad = cand, 0
            state = {k: getattr(model, k).copy() for k in STATE}
        else:
            bad += 1
        if verbose:
            print(f"  epoch {epoch:2d}  NDCG@10 {cand['ndcg@10']:.6f}  unique {cand['unique']:>6,}"
                  + ("  *" if bad == 0 else ""))
        if bad >= CONFIG.patience:
            break
    for k, v in state.items():
        setattr(model, k, v)
    return model, best


def evaluate(models, ev, prep, batch):
    per_seed, uniques = [], []
    for m in models:
        rk = rank(lightfm_scorer(m, prep), prep, batch)
        per_seed.append(metrics(ev, rk))
        uniques.append(len({r for u in ev["users"] for r in rk[u]}))
    out = {k: float(np.mean([s[k] for s in per_seed])) for k in METRICS}
    out["ndcg@10_std"] = float(np.std([s["ndcg@10"] for s in per_seed]))
    out["unique_recipes"] = int(np.mean(uniques))
    return out


def main():
    vec = load_frozen()
    data = build_training(SimpleNamespace(loss="warp", db="allrecipes.db"), vec)
    valid = load_evaluation("allrecipes.db", "valid")
    prep = prepare_eval(valid, vec, "allrecipes.db")

    # ---- step 1: choose beta/alpha on Allrecipes valid (seed 42) ----
    print(f"Step 1: grid on Allrecipes valid (eligible = unique recipes >= {MIN_UNIQUE})")
    print(f"{'beta':>6}{'alpha':>9}{'epoch':>7}{'NDCG@10':>11}{'unique':>9}  eligible")
    grid, chosen = [], None
    for beta in BETAS:
        for alpha in ALPHAS:
            _, b = train(GRID_SEED, beta, alpha, data, valid, prep)
            row = {"beta": beta, "alpha": alpha, **b}
            grid.append(row)
            ok = b["unique"] >= MIN_UNIQUE
            print(f"{beta:>6}{alpha:>9}{b['epoch']:>7}{b['ndcg@10']:>11.6f}{b['unique']:>9,}  {'yes' if ok else 'no'}")
            if better(row, chosen):
                chosen = row
    if chosen["unique"] < MIN_UNIQUE:
        print(f"No config reached {MIN_UNIQUE} unique recipes -> using the highest NDCG@10 instead.")
    print(f"-> selected: beta={chosen['beta']}, alpha={chosen['alpha']}")

    # ---- step 2: train final seeds with the selected config ----
    print(f"\nStep 2: final training, seeds {FINAL_SEEDS}")
    models, epochs = [], []
    for seed in FINAL_SEEDS:
        print(f"[seed {seed}]")
        m, b = train(seed, chosen["beta"], chosen["alpha"], data, valid, prep, verbose=True)
        models.append(m)
        epochs.append(b["epoch"])

    results = {"Allrecipes valid": evaluate(models, valid, prep, 512)}
    del prep
    for name, db in [("Allrecipes test", "allrecipes.db"), ("Food.com zero-shot", "foodcom.db")]:
        print(f"\nEvaluating {name} ...")
        ev = load_evaluation(db, "test")
        p = prepare_eval(ev, vec, db)
        results[name] = evaluate(models, ev, p, 64 if len(p["cand_ids"]) > 100_000 else 512)
        del p

    # ---- report ----
    def load(fname):
        f = OUT / fname
        return json.loads(f.read_text())["results"] if f.exists() else {}
    orig, cos = load("final_test.json"), load("final_test_cosine.json")

    print("\n" + "=" * 80)
    print(f"Final comparison (LightFM rows = mean over {len(FINAL_SEEDS)} seeds; reweight epochs {epochs})")
    print("=" * 80)
    print(f"{'Method':<26}{'Allrecipes test NDCG@10':>25}{'Food.com NDCG@10':>20}{'unique (Food)':>9}")
    print("-" * 80)
    rows = [("Random", orig, "Random", None), ("Popularity", orig, "Popularity", None),
            ("CBF", orig, "CBF", None), ("LightFM original", orig, "LightFM", None),
            ("LightFM cosine", cos, "LightFM cosine", "unique_recipes"),
            ("LightFM reweight", results, None, "unique_recipes")]
    for label, src, key, ukey in rows:
        def get(ds):
            if src is results:
                return results.get(ds)
            if key == "LightFM cosine":
                return src.get(ds)
            return src.get(ds, {}).get(key)
        a, f = get("Allrecipes test"), get("Food.com zero-shot")
        fa = f"{a['ndcg@10']:.6f}" if a else "-"
        ff = f"{f['ndcg@10']:.6f}" if f else "n/a"
        uu = f"{f[ukey]:,}" if (f and ukey and ukey in f) else ""
        print(f"{label:<26}{fa:>25}{ff:>20}{uu:>9}")
    print("-" * 80)
    print(f"Selected: beta={chosen['beta']}, alpha={chosen['alpha']} | "
          f"Food.com NDCG@10 std {results['Food.com zero-shot']['ndcg@10_std']:.6f}")
    print("Original LightFM unique recipes on Food.com: 35")

    OUT.mkdir(exist_ok=True)
    (OUT / "final_test_reweight.json").write_text(json.dumps(
        {"rule": {"min_unique": MIN_UNIQUE, "betas": BETAS, "alphas": ALPHAS}, "grid": grid,
         "selected": chosen, "final_seeds": FINAL_SEEDS, "best_valid_epochs": epochs,
         "results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT / 'final_test_reweight.json'}")


if __name__ == "__main__":
    main()