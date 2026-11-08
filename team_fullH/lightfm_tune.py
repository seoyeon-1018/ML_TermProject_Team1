"""LightFM hyperparameter tuning on Allrecipes valid (NDCG@10).

Two stages (seed 42, early stopping on valid NDCG@10):
  Stage 1: loss x lr            (dim=32, alpha=0)            -> 3 x 4 = 12 runs
  Stage 2: dim x alpha          (best loss/lr from stage 1)  -> 3 x 3 = 9 runs
Results are appended to lightfm_results/tuning.csv after every run,
so if it stops midway, just run it again and finished runs are skipped.

Run (team_essentials folder, mlteam env; needs lightfm_smoke.py in the same folder):
    python lightfm_tune.py
Test set and Food.com are NOT touched here (valid only).
"""
import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from lightfm import LightFM

from cbf import load_evaluation, load_frozen
from lightfm_smoke import build_training, check_lightfm, lightfm_scorer, metrics, prepare_eval, rank

LOSSES = ["warp", "bpr", "logistic"]
LRS = [0.005, 0.01, 0.02, 0.05]
DIMS = [16, 32, 64]
ALPHAS = [0.0, 1e-5, 1e-4]
DEFAULT_DIM, DEFAULT_ALPHA = 32, 0.0
SEED = 42
RESULT_CSV = Path("lightfm_results") / "tuning.csv"


def train_eval(data, valid, prep, loss, dim, lr, alpha, max_epochs, patience, threads):
    inter, x_users, x_items = data
    check_lightfm(loss)
    model = LightFM(no_components=dim, loss=loss, learning_rate=lr,
                    item_alpha=alpha, user_alpha=alpha, random_state=SEED)
    best, bad, epoch = None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.fit_partial(inter, user_features=x_users, item_features=x_items, epochs=1, num_threads=threads)
        m = metrics(valid, rank(lightfm_scorer(model, prep), prep))
        if best is None or m["ndcg@10"] > best["ndcg@10"]:
            best, bad = {"best_epoch": epoch, **m}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    best["epochs_run"] = epoch
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="allrecipes.db")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = p.parse_args()
    t0 = time.time()

    vec = load_frozen()
    valid = load_evaluation(args.db, "valid")
    prep = prepare_eval(valid, vec, args.db)
    data_cache = {}

    RESULT_CSV.parent.mkdir(exist_ok=True)
    done = pd.read_csv(RESULT_CSV) if RESULT_CSV.exists() else pd.DataFrame()

    def run(stage, loss, dim, lr, alpha):
        nonlocal done
        key = (loss, dim, lr, alpha)
        if not done.empty:
            hit = done[(done.loss == loss) & (done.dim == dim) & (done.lr == lr) & (done.alpha == alpha)]
            if not hit.empty:
                r = hit.iloc[0]
                print(f"  {loss:<9}dim {dim:<4}lr {lr:<7}alpha {alpha:<8}"
                      f"NDCG@10 {r['ndcg@10']:.6f}  (epoch {int(r.best_epoch)})  [cached]")
                return r.to_dict()
        if loss not in data_cache:
            data_cache[loss] = build_training(SimpleNamespace(loss=loss, db=args.db), vec)
        t = time.time()
        best = train_eval(data_cache[loss], valid, prep, *key, args.max_epochs, args.patience, args.threads)
        row = {"stage": stage, "loss": loss, "dim": dim, "lr": lr, "alpha": alpha,
               "best_epoch": best["best_epoch"], "epochs_run": best["epochs_run"],
               "precision@10": best["precision@10"], "recall@10": best["recall@10"],
               "ndcg@10": best["ndcg@10"], "seconds": round(time.time() - t)}
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(RESULT_CSV, index=False)
        print(f"  {loss:<9}dim {dim:<4}lr {lr:<7}alpha {alpha:<8}"
              f"NDCG@10 {row['ndcg@10']:.6f}  (epoch {row['best_epoch']}, {row['seconds']}s)")
        return row

    print(f"Stage 1: loss x lr (dim={DEFAULT_DIM}, alpha={DEFAULT_ALPHA})")
    s1 = [run(1, loss, DEFAULT_DIM, lr, DEFAULT_ALPHA) for loss in LOSSES for lr in LRS]
    b1 = max(s1, key=lambda r: r["ndcg@10"])
    print(f"-> best: loss={b1['loss']}, lr={b1['lr']}\n")

    print(f"Stage 2: dim x alpha (loss={b1['loss']}, lr={b1['lr']})")
    for dim in DIMS:
        for alpha in ALPHAS:
            run(2, b1["loss"], dim, b1["lr"], alpha)

    # ---- summary ----
    table = done.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 78)
    print("Top 10 configurations (valid)")
    print("=" * 78)
    print(f"{'#':<4}{'loss':<10}{'dim':>5}{'lr':>8}{'alpha':>9}{'epoch':>7}{'P@10':>11}{'R@10':>11}{'NDCG@10':>11}")
    print("-" * 78)
    for i, r in table.head(10).iterrows():
        print(f"{i + 1:<4}{r.loss:<10}{int(r.dim):>5}{r.lr:>8}{r.alpha:>9}{int(r.best_epoch):>7}"
              f"{r['precision@10']:>11.6f}{r['recall@10']:>11.6f}{r['ndcg@10']:>11.6f}")
    print("-" * 78)
    print("Reference: Popularity 0.019298 / CBF 0.001414 / LightFM default (warp, lr 0.05) 0.011209")

    best = table.iloc[0]
    epochs = max(int(best.best_epoch) + 10, 20)
    print("\nBest config. Next, check it over 3 seeds:")
    print(f"  python lightfm_smoke.py --loss {best.loss} --dim {int(best.dim)} --lr {best.lr} "
          f"--alpha {best.alpha} --max-epochs {epochs} --seeds 42 43 44")
    print(f"\nSaved: {RESULT_CSV}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()