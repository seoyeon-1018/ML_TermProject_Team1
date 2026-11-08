"""Compare H = first 5 positives (team_5H) with H = all past positives (team_fullH).

Reads (folders next to this comparison/ folder):
  team_5H/lightfm_results/final_test.json                   original (H5): Random, Popularity, CBF, LightFM
  team_5H/lightfm_results/final_test_cosine.json            improvement 1 (H5)
  team_5H/lightfm_results/final_test_reweight.json          improvement 2 (H5)
  team_fullH/lightfm_results/final_test.json                additional experiment (H all): CBF, LightFM

Writes comparison_figures/:
  cmp1_ndcg_all.png        NDCG@10 of every setting, Allrecipes test vs Food.com
  cmp2_h_effect.png        H5 -> H all, CBF and LightFM (the effect of the user profile length)
  cmp3_all_metrics.png     P@10 / R@10 / NDCG@10
  comparison_table.csv     all numbers, with the kind of change (model vs input)

Run (from any folder; paths default to this script's location):
    python compare.py
    python compare.py --h5 <H5 folder> --hall <H all folder> --out <output folder>
    (once, if matplotlib is missing:  conda install -n mlteam -c conda-forge matplotlib -y)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent      # .../comparison
PROJECT = HERE.parent                       # project root with team_5H/ and team_fullH/
DS = ["Allrecipes test", "Food.com zero-shot"]
DS_LABEL = ["Allrecipes test\n(trained domain)", "Food.com\n(zero-shot)"]
METRICS = ["precision@10", "recall@10", "ndcg@10"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# name, H setting, kind of change, color, hatch
SETTINGS = [
    ("CBF (H5)",               "H5",    "baseline",          "#4c78a8", ""),
    ("LightFM (H5)",           "H5",    "baseline",          "#e45756", ""),
    ("LightFM cosine (H5)",    "H5",    "model change (1)",  "#b279a2", ""),
    ("LightFM reweight (H5)",  "H5",    "model change (2)",  "#72b7b2", ""),
    ("CBF (H all)",            "H all", "input change",      "#4c78a8", "///"),
    ("LightFM (H all)",        "H all", "input change",      "#e45756", "///"),
]


def load(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing: {p}  (run the matching lightfm_*.py first)")
    return json.loads(p.read_text())["results"]


def collect(base, full):
    h5, cos, rew = (load(base / "lightfm_results" / f) for f in
                    ["final_test.json", "final_test_cosine.json", "final_test_reweight.json"])
    hall = load(full / "lightfm_results" / "final_test.json")
    src = {"CBF (H5)": (h5, "CBF"), "LightFM (H5)": (h5, "LightFM"),
           "LightFM cosine (H5)": (cos, None), "LightFM reweight (H5)": (rew, None),
           "CBF (H all)": (hall, "CBF"), "LightFM (H all)": (hall, "LightFM")}
    rows = []
    for name, h, kind, _, _ in SETTINGS:
        res, key = src[name]
        for d in DS:
            m = res[d] if key is None else res[d][key]
            rows.append({"setting": name, "H": h, "change": kind, "dataset": d, **{k: m[k] for k in METRICS}})
    for d in DS:
        rows.append({"setting": "Random", "H": "-", "change": "reference", "dataset": d,
                     **{k: h5[d]["Random"][k] for k in METRICS}})
    rows.append({"setting": "Popularity", "H": "-", "change": "reference", "dataset": DS[0],
                 **{k: h5[DS[0]]["Popularity"][k] for k in METRICS}})
    return pd.DataFrame(rows)


def label_bars(ax, bars, fmt="{:.5f}"):
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width() / 2, h * 1.1, fmt.format(h), ha="center", va="bottom",
                    fontsize=6.5, rotation=90)


def get(t, setting, d, k="ndcg@10"):
    v = t[(t.setting == setting) & (t.dataset == d)][k]
    return float(v.iloc[0]) if len(v) else np.nan


def fig_all(t, out):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x, w = np.arange(len(DS)), 0.13
    for i, (name, _, _, color, hatch) in enumerate(SETTINGS):
        vals = [get(t, name, d) for d in DS]
        bars = ax.bar(x + (i - 2.5) * w, vals, w, color=color, hatch=hatch, edgecolor="white", label=name)
        label_bars(ax, bars)
    ax.hlines(get(t, "Random", DS[0]), x[0] - 0.45, x[0] + 0.45, colors="#7f7f7f", linestyles="--", lw=1, label="Random")
    ax.hlines(get(t, "Random", DS[1]), x[1] - 0.45, x[1] + 0.45, colors="#7f7f7f", linestyles="--", lw=1)
    ax.hlines(get(t, "Popularity", DS[0]), x[0] - 0.45, x[0] + 0.45, colors="#f2a541", linestyles=":", lw=1.5,
              label="Popularity (Allrecipes only)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.15)
    ax.set_xticks(x, DS_LABEL)
    ax.set_ylabel("NDCG@10 (log scale)")
    ax.set_title("NDCG@10 of all settings  (hatched = H = all past positives)")
    ax.legend(ncol=2, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "cmp1_ndcg_all.png", dpi=200)
    plt.close(fig)


def fig_h_effect(t, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d in zip(axes, DS):
        for model, color in [("CBF", "#4c78a8"), ("LightFM", "#e45756")]:
            a, b = get(t, f"{model} (H5)", d), get(t, f"{model} (H all)", d)
            ax.plot([0, 1], [a, b], marker="o", color=color, lw=2, label=model)
            ax.annotate(f"{a:.5f}", (0, a), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=8)
            ax.annotate(f"{b:.5f}  ({b / a:.2f}x)", (1, b), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8)
        ax.axhline(get(t, "Random", d), ls="--", color="#7f7f7f", lw=1, label="Random")
        if d == DS[0]:
            ax.axhline(get(t, "Popularity", d), ls=":", color="#f2a541", lw=1.5, label="Popularity")
        ax.set_yscale("log")
        ax.set_xticks([0, 1], ["H = first 5", "H = all past"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(d)
        ax.set_ylabel("NDCG@10 (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Effect of the user profile length (same data, only H changed + re-tuned LightFM)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "cmp2_h_effect.png", dpi=200)
    plt.close(fig)


def fig_metrics(t, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    names = [s[0] for s in SETTINGS]
    colors = [s[3] for s in SETTINGS]
    hatches = [s[4] for s in SETTINGS]
    for r, d in enumerate(DS):
        for c, (k, title) in enumerate(zip(METRICS, ["Precision@10", "Recall@10", "NDCG@10"])):
            ax = axes[r, c]
            vals = [get(t, n, d, k) for n in names]
            bars = ax.bar(range(len(names)), vals, color=colors, hatch=hatches, edgecolor="white")
            label_bars(ax, bars)
            ax.axhline(get(t, "Random", d, k), ls="--", color="#7f7f7f", lw=1)
            ax.set_yscale("log")
            ax.set_xticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=6.5)
            ax.set_title(f"{d} - {title}", fontsize=9)
    fig.suptitle("All metrics (log scale, dashed = Random, hatched = H all)")
    fig.tight_layout()
    fig.savefig(out / "cmp3_all_metrics.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=PROJECT / "team_5H", help="folder of the H = 5 experiments")
    ap.add_argument("--hall", default=PROJECT / "team_fullH", help="folder of the H = all experiment")
    ap.add_argument("--out", default=HERE / "comparison_figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    t = collect(Path(args.h5), Path(args.hall))
    t.to_csv(out / "comparison_table.csv", index=False)
    fig_all(t, out)
    fig_h_effect(t, out)
    fig_metrics(t, out)

    piv = t.pivot_table(index="setting", columns="dataset", values="ndcg@10", sort=False)
    print("NDCG@10")
    print(piv.to_string(float_format=lambda v: f"{v:.6f}"))
    print(f"\nSaved to {out.resolve()}/ (cmp1_ndcg_all.png, cmp2_h_effect.png, cmp3_all_metrics.png, comparison_table.csv)")


if __name__ == "__main__":
    main()
