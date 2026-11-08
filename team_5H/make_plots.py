"""Make all figures for the report/PPT from saved results (no training, runs in seconds).

Inputs (lightfm_results/):
  final_test.json              original LightFM + baselines on Allrecipes test / Food.com
  final_test_cosine.json       improvement 1 (cosine scoring)
  final_test_reweight.json     improvement 2 (popularity re-weighting + regularization)
  tuning.csv                   hyperparameter tuning (valid)
  valid_warp_d32_lr0.02_a0.0.json   epoch curves of the final config (3 seeds)
Diagnosis numbers (fig 5) are copied from the lightfm_diagnose.py output (seed 42).

Output: figures/fig1 ~ fig8 (.png, 200 dpi). Missing input files are skipped.

Run (team_essentials folder, mlteam env):
    conda install -c conda-forge matplotlib -y     # once
    python make_plots.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path("lightfm_results")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
DS = ["Allrecipes test", "Food.com zero-shot"]
COLORS = {"Random": "#9e9e9e", "Popularity": "#f2a541", "CBF": "#4c78a8", "LightFM": "#e45756",
          "LightFM cosine": "#b279a2", "LightFM reweight": "#72b7b2"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    f = RES / name
    return json.loads(f.read_text()) if f.exists() else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)
    print(f"saved {FIG / name}")


def bar_labels(ax, bars, fmt="{:.4f}"):
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width() / 2, h * 1.08, fmt.format(h), ha="center", va="bottom", fontsize=7)


final, cosine, reweight = load("final_test.json"), load("final_test_cosine.json"), load("final_test_reweight.json")

# ---- Fig 1: main comparison (NDCG@10, log scale) ----
if final:
    r = final["results"]
    methods = ["Random", "Popularity", "CBF", "LightFM"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(DS))
    w = 0.2
    for i, m in enumerate(methods):
        vals = [r[d][m]["ndcg@10"] if m in r[d] else 0 for d in DS]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=COLORS[m])
        bar_labels(ax, bars)
    ax.text(x[1] - 0.5 * w, 1.2e-5, "n/a", ha="center", fontsize=8, color=COLORS["Popularity"])
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.1)
    ax.set_xticks(x, ["Allrecipes test\n(trained domain)", "Food.com\n(zero-shot)"])
    ax.set_ylabel("NDCG@10 (log scale)")
    ax.set_title("Fig 1. Recommendation quality: trained domain vs zero-shot transfer")
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    save(fig, "fig1_main_comparison.png")

    # ---- Fig 2: P@10 / R@10 / NDCG@10 per dataset ----
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, k, title in zip(axes, ["precision@10", "recall@10", "ndcg@10"], ["Precision@10", "Recall@10", "NDCG@10"]):
        for i, m in enumerate(methods):
            vals = [r[d][m][k] if m in r[d] else 0 for d in DS]
            ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=COLORS[m])
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 0.1)
        ax.set_xticks(x, ["Allrecipes", "Food.com"])
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.suptitle("Fig 2. All metrics (log scale; Popularity not applicable to Food.com)")
    save(fig, "fig2_all_metrics.png")

# ---- Fig 3: learning curves of the final config (valid) ----
curves = load("valid_warp_d32_lr0.02_a0.0.json")
if curves and "histories" in curves:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for h in curves["histories"]:
        ep = [e["epoch"] for e in h["history"]]
        nd = [e["ndcg@10"] for e in h["history"]]
        ax.plot(ep, nd, marker="o", ms=4, label=f"seed {h['seed']}")
    res = curves.get("results", {})
    for name, color in [("Popularity", COLORS["Popularity"]), ("CBF", COLORS["CBF"])]:
        if name in res:
            ax.axhline(res[name]["ndcg@10"], ls="--", color=color, lw=1, label=f"{name} (valid)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("valid NDCG@10")
    ax.set_title("Fig 3. LightFM learning curve (warp, dim 32, lr 0.02)\nbest at epoch 1-3, then no improvement")
    ax.legend(fontsize=8)
    save(fig, "fig3_learning_curve.png")

# ---- Fig 4: hyperparameter tuning (valid) ----
tune_f = RES / "tuning.csv"
if tune_f.exists():
    t = pd.read_csv(tune_f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    s1 = t[(t.dim == 32) & (t.alpha == 0.0)]
    for loss, g in s1.groupby("loss"):
        g = g.sort_values("lr")
        axes[0].plot(g.lr, g["ndcg@10"], marker="o", label=loss)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("learning rate")
    axes[0].set_ylabel("valid NDCG@10")
    axes[0].set_title("Stage 1: loss x learning rate (dim 32, alpha 0)")
    axes[0].legend()

    s2 = t[(t.loss == "warp") & (t.lr == 0.02)]
    piv = s2.pivot_table(index="dim", columns="alpha", values="ndcg@10")
    im = axes[1].imshow(piv.values, cmap="Reds", aspect="auto")
    axes[1].set_xticks(range(len(piv.columns)), [str(a) for a in piv.columns])
    axes[1].set_yticks(range(len(piv.index)), [str(d) for d in piv.index])
    axes[1].set_xlabel("alpha (L2 regularization)")
    axes[1].set_ylabel("dim")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[1].text(j, i, f"{piv.values[i, j]:.4f}", ha="center", va="center", fontsize=8)
    axes[1].set_title("Stage 2: dim x alpha (warp, lr 0.02)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.suptitle("Fig 4. Hyperparameter tuning on Allrecipes valid")
    save(fig, "fig4_tuning.png")

# ---- Fig 5: diagnosis (Food.com, seed 42; numbers from lightfm_diagnose.py output) ----
groups = ["All\ncandidates", "Target\nrecipes", "CBF\ntop-10", "LightFM\ntop-10"]
known = [8.18, 8.44, 7.71, 4.60]
le2 = [3.4, 2.6, 3.5, 32.4]
gcol = ["#bdbdbd", "#54a24b", COLORS["CBF"], COLORS["LightFM"]]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
bar_labels(axes[0], axes[0].bar(groups, known, color=gcol), "{:.2f}")
axes[0].set_ylabel("mean # known ingredients")
axes[0].set_title("Known ingredients per recipe")
bar_labels(axes[1], axes[1].bar(groups, le2, color=gcol), "{:.1f}%")
axes[1].set_ylabel("% of recipes")
axes[1].set_title("Recipes with <= 2 known ingredients")
fig.suptitle("Fig 5. Why LightFM fails on Food.com: it recommends recipes the model barely 'knows'")
save(fig, "fig5_diagnosis.png")

# ---- Fig 6: diversity (unique recipes recommended across all users) ----
if cosine and reweight:
    div = {"CBF": [15362, 24689],  # from lightfm_diagnose.py output
           "LightFM": [reweight["results"][d]["unique_recipes"] for d in DS],  # selected = original config
           "LightFM cosine": [cosine["results"][d]["unique_recipes"] for d in DS]}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(DS))
    w = 0.25
    for i, (m, vals) in enumerate(div.items()):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=m, color=COLORS[m])
        bar_labels(ax, bars, "{:,.0f}")
    ax.set_yscale("log")
    ax.set_ylim(10, 2e5)
    ax.set_xticks(x, ["Allrecipes test", "Food.com"])
    ax.set_ylabel("# different recipes recommended (log)")
    ax.set_title("Fig 6. Recommendation diversity: LightFM recommends ~the same recipes to everyone")
    ax.legend(fontsize=8)
    save(fig, "fig6_diversity.png")

# ---- Fig 7: improvement attempts ----
if final and cosine and reweight:
    r = final["results"]
    rows = {"CBF": [r[d]["CBF"]["ndcg@10"] for d in DS],
            "LightFM": [r[d]["LightFM"]["ndcg@10"] for d in DS],
            "LightFM cosine": [cosine["results"][d]["ndcg@10"] for d in DS],
            "LightFM reweight": [reweight["results"][d]["ndcg@10"] for d in DS]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, j, d in zip(axes, range(2), ["Allrecipes test", "Food.com zero-shot"]):
        names = list(rows)
        bars = ax.bar(names, [rows[n][j] for n in names], color=[COLORS[n] for n in names])
        bar_labels(ax, bars, "{:.5f}")
        ax.axhline(r[d]["Random"]["ndcg@10"], ls="--", color=COLORS["Random"], lw=1, label="Random")
        ax.set_title(d)
        ax.set_ylabel("NDCG@10")
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(fontsize=8)
    fig.suptitle("Fig 7. Improvement attempts (post-hoc): neither fixes zero-shot transfer")
    save(fig, "fig7_improvements.png")

    # ---- Fig 8: re-weighting trade-off (valid grid) ----
    g = pd.DataFrame(reweight["grid"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for beta, sub in g.groupby("beta"):
        ax.scatter(sub["unique"], sub["ndcg@10"], s=50, label=f"beta={beta}")
        for _, row in sub.iterrows():
            ax.annotate(f"a={row.alpha:g}", (row["unique"], row["ndcg@10"]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
    ax.axvline(reweight["rule"]["min_unique"], ls="--", color="gray", lw=1)
    ax.text(reweight["rule"]["min_unique"], ax.get_ylim()[1] * 0.97, " required (500)", fontsize=8, va="top")
    ax.set_xlabel("# different recipes recommended (valid)")
    ax.set_ylabel("valid NDCG@10")
    ax.set_title("Fig 8. Popularity re-weighting: more diversity = lower accuracy")
    ax.legend(fontsize=8)
    save(fig, "fig8_reweight_tradeoff.png")

print("done")