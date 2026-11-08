"""Extra analysis of the CBF results and the figures used in the CBF report.

Inputs : result folders written by `cbf.py evaluate` (recommendations.csv, per_user_metrics.csv),
         allrecipes.db, foodcom.db and cbf_model.json, all next to this script.
Outputs: cbf_analysis/analysis.json and cbf_analysis/fig_c1..fig_c3 (PNG).

Run first:
  python cbf.py evaluate --db allrecipes.db --split test --output cbf_allrecipes_result
  python cbf.py evaluate --db foodcom.db   --split test --output cbf_foodcom_result
Then:
  python cbf_report_analysis.py
"""
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib
matplotlib.use('Agg')  # render to files only (no display needed)
import matplotlib.pyplot as plt

import cbf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'cbf_analysis'
OUT.mkdir(exist_ok=True)
# Korean-capable fonts, first one found wins (Windows: Malgun Gothic, macOS: AppleGothic)
plt.rcParams.update({'font.family': ['Malgun Gothic', 'AppleGothic', 'Noto Sans CJK JP', 'sans-serif'],
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': False})
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#b4b2a9'
# Random NDCG@10 from lightfm_final.py (same cohorts; quoted for figure 1)
RANDOM_NDCG = {'allrecipes': 0.000300, 'foodcom': 0.000085}

model = cbf.load_frozen(ROOT / 'cbf_model.json')
with sqlite3.connect(ROOT / 'allrecipes.db') as con:
    # Recipe popularity = number of positive (>= 4) Allrecipes train ratings
    popularity = Counter(r for (r,) in con.execute(
        "SELECT recipe_id FROM interactions WHERE partition='train' AND rating>=4"))


def n_known(ingredients):
    """Number of a recipe's ingredients that exist in the training vocabulary."""
    return sum(1 for i in set(ingredients) if i in model.vocabulary)


def analyze(db, folder, rng):
    ev = cbf.load_evaluation(ROOT / f'{db}.db', 'test')
    rec = pd.read_csv(ROOT / folder / 'recommendations.csv')
    pu = pd.read_csv(ROOT / folder / 'per_user_metrics.csv')
    ing = {r['recipe_id']: r['ingredients'] for r in ev['recipes']}
    n_cand = len(ev['candidate_ids'])
    # Expected Precision@10 of random ranking = mean over users of |L| / (candidates - seen)
    rand_p = np.mean([len(ev['L'][u]) / (n_cand - len(ev['seen'][u] & ev['candidate_ids'])) for u in ev['users']])
    hits = int(pu.hits.sum()); exp_hits = rand_p * 10 * len(pu)
    ndcg = pu['ndcg@10'].to_numpy()
    boot = [ndcg[rng.integers(0, len(ndcg), len(ndcg))].mean() for _ in range(2000)]  # user-level bootstrap
    counts = Counter(rec.recipe_id)
    targets = [r for u in ev['users'] for r in ev['L'][u]]
    le2 = lambda ids: float(np.mean([n_known(ing[r]) <= 2 for r in ids]))
    pu['hit'] = pu.hits > 0
    pu['L_bin'] = pd.cut(pu.n_targets, [0, 1, 3, 10, np.inf], labels=['1', '2-3', '4-10', '11+'])
    res = {
        'users': len(pu), 'candidates': n_cand, 'macro': pu[['precision@10', 'recall@10', 'ndcg@10']].mean().to_dict(),
        'hits': hits, 'users_hit': int(pu.hit.sum()), 'random_expected_hits': float(exp_hits),
        'poisson_p_vs_random': float(poisson.sf(hits - 1, exp_hits)),
        'ndcg_ci95': [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        'distinct_recommended': len(counts), 'coverage': len(counts) / n_cand, 'top1_recipe_users': counts.most_common(1)[0][1],
        'le2_share': {'candidates': le2(ing), 'cbf_recs': le2(rec.recipe_id)},
        'hit_by_L': pu.groupby('L_bin', observed=True).agg(n=('user_id', 'size'), hit=('hit', 'mean')).reset_index().to_dict('records'),
        'users_with_tied_scores_in_top10': float((rec.groupby('user_id').cosine_similarity.nunique() < 10).mean()),
    }
    if db == 'allrecipes':
        order = sorted(ing, key=lambda r: -popularity.get(r, 0))
        top1 = set(order[:int(n_cand * 0.01)])  # top 1% most popular recipes
        res['popularity'] = {
            'mean': {'candidates': float(np.mean([popularity.get(r, 0) for r in ing])),
                     'cbf_recs': float(np.mean([popularity.get(r, 0) for r in rec.recipe_id])),
                     'targets': float(np.mean([popularity.get(r, 0) for r in targets]))},
            'top1pct_share': {'candidates': 0.01, 'cbf_recs': float(rec.recipe_id.isin(top1).mean()),
                              'targets': float(np.mean([r in top1 for r in targets]))}}
    return res


rng = np.random.default_rng(0)
A = {'allrecipes': analyze('allrecipes', 'cbf_allrecipes_result', rng),
     'foodcom': analyze('foodcom', 'cbf_foodcom_result', rng)}
(OUT / 'analysis.json').write_text(json.dumps(A, ensure_ascii=False, indent=1, default=str), encoding='utf-8')


def label_bars(ax, xs, vals, fmt):
    for x, v in zip(xs, vals):
        ax.text(x, v, fmt.format(v), ha='center', va='bottom', fontsize=8)


# Figure 1: CBF vs Random NDCG@10 with 95% bootstrap interval
fig, ax = plt.subplots(figsize=(6.4, 3.6)); x = np.arange(2); w = 0.34
cb = [A[d]['macro']['ndcg@10'] for d in A]; ci = np.array([A[d]['ndcg_ci95'] for d in A])
ax.bar(x - w / 2, list(RANDOM_NDCG.values()), w, color=GRAY, label='Random')
ax.bar(x + w / 2, cb, w, color=BLUE, label='CBF (95% 부트스트랩 CI)',
       yerr=[np.array(cb) - ci[:, 0], ci[:, 1] - np.array(cb)], capsize=3)
label_bars(ax, x - w / 2, RANDOM_NDCG.values(), '{:.5f}'); label_bars(ax, x + w / 2, ci[:, 1], '')
ax.set_xticks(x, ['Allrecipes test', 'Food.com zero-shot']); ax.set_ylabel('NDCG@10'); ax.legend(frameon=False, loc='upper left')
ax.set_title('CBF와 Random의 NDCG@10', loc='left'); fig.tight_layout(); fig.savefig(OUT / 'fig_c1_cbf_vs_random.png', dpi=200); plt.close()

# Figure 2: popularity gap between candidates, CBF recommendations and targets (Allrecipes test)
p = A['allrecipes']['popularity']; cats = ['전체 후보', 'CBF 추천', '실제 정답(L)']
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
for ax, vals, title, fmt in [(axs[0], [v * 100 for v in p['top1pct_share'].values()], '인기 상위 1% 레시피 비율 (%)', '{:.1f}%'),
                             (axs[1], list(p['mean'].values()), 'train 긍정 평점 수 평균', '{:.0f}')]:
    ax.bar(cats, vals, 0.55, color=[GRAY, BLUE, ORANGE]); label_bars(ax, range(3), vals, fmt); ax.set_title(title, loc='left')
fig.tight_layout(); fig.savefig(OUT / 'fig_c2_popularity_gap.png', dpi=200); plt.close()

# Figure 3: Hit@10 by number of targets |L|
fig, ax = plt.subplots(figsize=(6.4, 3.4)); x = np.arange(4)
for off, d, col, lab in [(-w / 2, 'allrecipes', BLUE, 'Allrecipes test'), (w / 2, 'foodcom', ORANGE, 'Food.com zero-shot')]:
    vals = [r['hit'] * 100 for r in A[d]['hit_by_L']]; ax.bar(x + off, vals, w, color=col, label=lab); label_bars(ax, x + off, vals, '{:.2f}%')
ax.set_xticks(x, ['정답 1개', '정답 2-3개', '정답 4-10개', '정답 11+개']); ax.set_ylabel('적중 사용자 비율 (%)'); ax.legend(frameon=False)
ax.set_title('정답 수(|L|)별 Hit@10', loc='left'); fig.tight_layout(); fig.savefig(OUT / 'fig_c3_hit_by_L.png', dpi=200); plt.close()

print(json.dumps({d: {k: A[d][k] for k in ['macro', 'hits', 'users_hit', 'random_expected_hits', 'ndcg_ci95', 'distinct_recommended']} for d in A}, indent=1))
