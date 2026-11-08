"""LightFM on Allrecipes valid: train with early stopping, compare with baselines.

Output:
  1. Epoch log (valid NDCG@10, * = new best)
  2. One comparison table: Random / Popularity / CBF / LightFM
     + LightFM decomposition (bias-only = popularity part, emb-only = personalized part)

Run (team_essentials folder, mlteam env, after split_train.py):
    python lightfm_smoke.py
    python lightfm_smoke.py --lr 0.01 --max-epochs 50 --skip-cbf
    python lightfm_smoke.py --seeds 42 43 44        # mean +- std over seeds
Test set and Food.com are NOT touched here (valid only).
"""
import argparse
import copy
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import lightfm
import numpy as np
import pandas as pd
from scipy import sparse
from lightfm import LightFM

from cbf import CandidateIndex, evaluate_rankings, l2_normalize, load_evaluation, load_frozen

SPLIT_DIR = Path("train_split")
K = 10
SUPPORTED_LIGHTFM = "1.16"  # see environments.txt


def check_lightfm(loss):
    """Stop with a clear message instead of a segfault.

    conda-forge LightFM 1.17 crashes on Windows when training with warp/bpr/warp-kos
    (positive-lookup losses); 1.16 (Python 3.9) works on every platform.
    """
    version = lightfm.__version__
    if version != SUPPORTED_LIGHTFM:
        print(f"[warning] LightFM {version} found; this project is tested with LightFM {SUPPORTED_LIGHTFM}.",
              file=sys.stderr)
    if sys.platform == "win32" and loss in ("warp", "bpr", "warp-kos") and version != SUPPORTED_LIGHTFM:
        raise SystemExit(
            f"LightFM {version} crashes on Windows with loss='{loss}'. Use LightFM {SUPPORTED_LIGHTFM}:\n"
            f"  conda create -n mlteam -c conda-forge python=3.9 lightfm={SUPPORTED_LIGHTFM} "
            f"\"numpy<2\" \"pandas<3\" scipy scikit-learn matplotlib")


# ---------------------------------------------------------------- features
def fetch_ingredients(db_path, recipe_ids):
    ids = sorted(set(recipe_ids))
    out = {}
    with sqlite3.connect(db_path) as conn:
        for start in range(0, len(ids), 900):  # SQLite parameter limit
            chunk = ids[start:start + 900]
            q = f"SELECT recipe_id, ingredients FROM recipes WHERE recipe_id IN ({','.join('?' * len(chunk))})"
            for rid, ing in conn.execute(q, chunk):
                out[rid] = json.loads(ing)
    return out


def user_features(item_x, row_of, h_lists):
    """Mean of H item vectors, then L2-normalize (same definition as CBF profile)."""
    rows, cols, vals = [], [], []
    for u, h in enumerate(h_lists):
        for rid in h:
            rows.append(u)
            cols.append(row_of[rid])
            vals.append(1.0 / len(h))
    w = sparse.csr_matrix((vals, (rows, cols)), shape=(len(h_lists), item_x.shape[0]), dtype=np.float32)
    return l2_normalize(w.dot(item_x)).astype(np.float32)


def build_training(args, vec):
    H = pd.read_csv(SPLIT_DIR / "train_H.csv").sort_values(["user_id", "history_order"])
    T = pd.read_csv(SPLIT_DIR / "train_targets.csv")
    users = sorted(H.user_id.unique())
    u_of = {u: i for i, u in enumerate(users)}

    item_ids = set(H.recipe_id) | set(T.recipe_id)
    neg = None
    if args.loss == "logistic":
        neg = pd.read_csv(SPLIT_DIR / "train_negatives.csv")
        pos_pairs = set(zip(T.user_id, T.recipe_id))
        neg = neg[[pr not in pos_pairs for pr in zip(neg.user_id, neg.recipe_id)]]
        item_ids |= set(neg.recipe_id)

    ing = fetch_ingredients(args.db, item_ids)
    item_ids = sorted(ing)
    i_of = {r: i for i, r in enumerate(item_ids)}
    x_items, _ = vec.transform([ing[r] for r in item_ids])
    x_items = x_items.astype(np.float32)
    h_lists = H.groupby("user_id").recipe_id.apply(list).reindex(users).tolist()
    x_users = user_features(x_items, i_of, h_lists)

    T = T[T.recipe_id.isin(i_of)]
    rows = [u_of[u] for u in T.user_id]
    cols = [i_of[r] for r in T.recipe_id]
    data = [1.0] * len(T)
    if neg is not None:
        neg = neg[neg.recipe_id.isin(i_of)]
        rows += [u_of[u] for u in neg.user_id]
        cols += [i_of[r] for r in neg.recipe_id]
        data += [-1.0] * len(neg)
    inter = sparse.coo_matrix((np.asarray(data, dtype=np.float32), (rows, cols)),
                              shape=(len(users), len(item_ids)))
    return inter, x_users, x_items


def prepare_eval(evaluation, vec, db_path):
    cand_ids = sorted(evaluation["candidate_ids"])
    cand_row = {r: i for i, r in enumerate(cand_ids)}
    ing = {r["recipe_id"]: r["ingredients"] for r in evaluation["recipes"]}
    x_cand, _ = vec.transform([ing[r] for r in cand_ids])

    users = evaluation["users"]
    h_ids = sorted({r for u in users for r in evaluation["H"][u]})
    h_ing = fetch_ingredients(db_path, h_ids)
    x_h, _ = vec.transform([h_ing[r] for r in h_ids])
    x_users = user_features(x_h, {r: i for i, r in enumerate(h_ids)}, [evaluation["H"][u] for u in users])

    seen_rows = [[cand_row[r] for r in evaluation["seen"][u] if r in cand_row] for u in users]
    return {"evaluation": evaluation, "cand_ids": cand_ids, "x_cand": x_cand.astype(np.float32),
            "x_users": x_users, "seen_rows": seen_rows}


# ---------------------------------------------------------------- ranking
def rank(score_fn, prep, batch=512):
    """score_fn(start, end) -> (end-start, n_candidates) scores. Seen items excluded."""
    users, cand_ids = prep["evaluation"]["users"], prep["cand_ids"]
    rankings = {}
    for start in range(0, len(users), batch):
        end = min(start + batch, len(users))
        scores = np.array(score_fn(start, end), dtype=np.float64, copy=True)
        for i, rows in enumerate(prep["seen_rows"][start:end]):
            scores[i, rows] = -np.inf
        top = np.argpartition(-scores, K, axis=1)[:, :K]
        for i in range(end - start):
            sel = top[i][np.isfinite(scores[i, top[i]])]
            sel = sel[np.lexsort((sel, -scores[i, sel]))]
            rankings[users[start + i]] = [cand_ids[j] for j in sel]
    return rankings


def lightfm_scorer(model, prep, part="full"):
    ib, iemb = model.get_item_representations(prep["x_cand"])
    ub, uemb = model.get_user_representations(prep["x_users"])

    def fn(s, e):
        if part == "bias":
            return np.broadcast_to(ib, (e - s, len(ib)))
        dot = uemb[s:e] @ iemb.T
        return dot if part == "embedding" else dot + ib[None, :] + ub[s:e, None]
    return fn


def cbf_rankings(vec, evaluation, batch=256):
    index = CandidateIndex(vec, evaluation["recipes"])
    users, rankings = evaluation["users"], {}
    for start in range(0, len(users), batch):
        b = users[start:start + batch]
        profiles = sparse.vstack([index.profile(evaluation["H"][u]) for u in b], format="csr")
        for u, ranked in zip(b, index.recommend_batch(profiles, [evaluation["seen"][u] for u in b], K)):
            rankings[u] = [rid for rid, _ in ranked]
    return rankings


def metrics(evaluation, rankings):
    summary, _ = evaluate_rankings(evaluation, rankings, K)
    return summary["macro_metrics"]


# ---------------------------------------------------------------- training
def train_one(args, seed, inter, x_users, x_items, valid, prep):
    check_lightfm(args.loss)
    model = LightFM(no_components=args.dim, loss=args.loss, learning_rate=args.lr,
                    item_alpha=args.alpha, user_alpha=args.alpha, random_state=seed)
    best, best_model, bad, history = None, None, 0, []
    print(f"\n[seed {seed}]")
    for epoch in range(1, args.max_epochs + 1):
        model.fit_partial(inter, user_features=x_users, item_features=x_items, epochs=1, num_threads=args.threads)
        ndcg = metrics(valid, rank(lightfm_scorer(model, prep), prep))["ndcg@10"]
        history.append({"epoch": epoch, "ndcg@10": ndcg})
        if best is None or ndcg > best["ndcg@10"]:
            best, best_model, bad = history[-1], copy.deepcopy(model), 0
            mark = " *"
        else:
            bad += 1
            mark = ""
        print(f"  epoch {epoch:2d}  NDCG@10 {ndcg:.6f}{mark}")
        if bad >= args.patience:
            break
    print(f"  -> best epoch {best['epoch']}, NDCG@10 {best['ndcg@10']:.6f}")
    return best_model, best, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="allrecipes.db")
    p.add_argument("--loss", choices=["warp", "logistic", "bpr"], default="warp")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.0, help="L2 penalty for user/item features")
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5, help="stop after N epochs without improvement")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--skip-cbf", action="store_true")
    p.add_argument("--output", default="lightfm_results")
    args = p.parse_args()
    t0 = time.time()

    config = f"loss={args.loss}, dim={args.dim}, lr={args.lr}, alpha={args.alpha}, seeds={args.seeds}"
    print(f"LightFM | {config}")

    vec = load_frozen()
    inter, x_users, x_items = build_training(args, vec)
    valid = load_evaluation(args.db, "valid")
    prep = prepare_eval(valid, vec, args.db)
    n_cand = len(prep["cand_ids"])
    print(f"Train: {inter.shape[0]:,} users, {inter.shape[1]:,} items, {inter.nnz:,} interactions")
    print(f"Valid: {len(valid['users']):,} users, {n_cand:,} candidates")

    # ---- LightFM ----
    runs = [train_one(args, s, inter, x_users, x_items, valid, prep) for s in args.seeds]
    best_model, best, _ = max(runs, key=lambda r: r[1]["ndcg@10"])

    # ---- comparison ----
    results = {}
    rng = np.random.default_rng(0)
    results["Random"] = metrics(valid, rank(lambda s, e: rng.random((e - s, n_cand)), prep))
    with sqlite3.connect(args.db) as conn:
        counts = dict(conn.execute("SELECT recipe_id, COUNT(*) FROM interactions "
                                   "WHERE partition='train' AND rating>=4 GROUP BY recipe_id"))
    pop = np.array([counts.get(r, 0) for r in prep["cand_ids"]], dtype=np.float64)
    results["Popularity"] = metrics(valid, rank(lambda s, e: np.broadcast_to(pop, (e - s, n_cand)), prep))
    if not args.skip_cbf:
        results["CBF"] = metrics(valid, cbf_rankings(vec, valid))
    results["LightFM"] = metrics(valid, rank(lightfm_scorer(best_model, prep), prep))
    results["  - bias only"] = metrics(valid, rank(lightfm_scorer(best_model, prep, "bias"), prep))
    results["  - embedding only"] = metrics(valid, rank(lightfm_scorer(best_model, prep, "embedding"), prep))

    print("\n" + "=" * 62)
    print(f"Valid results (LightFM: best epoch {best['epoch']})")
    print("=" * 62)
    print(f"{'Method':<22}{'P@10':>12}{'R@10':>12}{'NDCG@10':>12}")
    print("-" * 62)
    for name, m in results.items():
        print(f"{name:<22}{m['precision@10']:>12.6f}{m['recall@10']:>12.6f}{m['ndcg@10']:>12.6f}")
    print("-" * 62)
    if len(runs) > 1:
        ndcgs = [r[1]["ndcg@10"] for r in runs]
        print(f"LightFM over {len(runs)} seeds: NDCG@10 {np.mean(ndcgs):.6f} +- {np.std(ndcgs):.6f}, "
              f"best epochs {[r[1]['epoch'] for r in runs]}")
    print("bias only = popularity part / embedding only = personalized part")

    out = Path(args.output)
    out.mkdir(exist_ok=True)
    tag = f"valid_{args.loss}_d{args.dim}_lr{args.lr}_a{args.alpha}"
    report = {"config": vars(args), "best_epoch": best["epoch"], "results": results,
              "histories": [{"seed": s, "history": r[2]} for s, r in zip(args.seeds, runs)]}
    (out / f"{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out / (tag + '.json')}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()