# Additional experiment: H = all past positives

This folder repeats the LightFM experiment with a longer user profile.
In the main experiment (`team_5H`), each user is represented by **H = the first 5 distinct past positive recipes**.
Here, **H = all distinct past positive recipes** (rating ≥ 4) before the evaluation cut-off.

Everything else is identical to the main experiment: recipes and ingredient features, evaluation users,
candidates, ground truth (L), seen recipes, Allrecipes train ratings and the evaluation function.
Only the `profiles` table (H) of the databases was changed.

---

## Results

| Method | Allrecipes test NDCG@10 (H5 → H all) | Food.com NDCG@10 (H5 → H all) |
| --- | --- | --- |
| Random | 0.000300 | 0.000085 |
| Popularity | 0.016224 | n/a |
| CBF | 0.001420 → 0.001258 (0.89×) | **0.001086** → 0.000706 (0.65×) |
| LightFM | 0.005197 → **0.007646** (1.47×) | 0.000053 → 0.000178 (3.36×) |

H all: LightFM = mean over seeds 42, 43, 44 (NDCG@10 std 0.000159 on Allrecipes test, 0.000008 on Food.com).

| Method (H all) | Allrecipes test P@10 / R@10 | Food.com P@10 / R@10 |
| --- | --- | --- |
| CBF | 0.000689 / 0.001726 | 0.000237 / 0.000922 |
| LightFM | 0.004058 / 0.010454 | 0.000198 / 0.000018 |

**Findings**

1. A longer profile helps LightFM: +47% on Allrecipes test, and on Food.com it rises from below random to about 2× random.
2. A longer profile hurts CBF on both datasets. A plausible reason (not verified): averaging many recipes blurs the user's specific taste.
3. The best Food.com result is still CBF with H = 5 (0.001086). No method beats Popularity on Allrecipes.
4. The change is "H all + LightFM re-tuned for it" (see below), so the gain should not be attributed to H length alone.

---

## What changed compared with the main experiment

| Part | Main experiment (H5) | This experiment (H all) |
| --- | --- | --- |
| Evaluation H (`profiles` table) | first 5 distinct past positives | all distinct past positives |
| `cbf.py` | H must have exactly 5 recipes | H must have ≥ 1 recipe (one-line change in `load_evaluation`) |
| LightFM training users (`split_train.py`) | first 5 positives → user features, later ones → targets | first 80% of positives → user features, last 20% → targets |
| LightFM setting (tuned on Allrecipes valid) | WARP, dim 32, lr 0.02, alpha 0 | logistic, dim 16, lr 0.005, alpha 1e-4 |

H length after the change:

| Evaluation set | Users | H length mean / median / max |
| --- | ---: | --- |
| Allrecipes valid | 14,613 | 26.2 / 13 / 2,253 |
| Allrecipes test | 26,853 | 22.5 / 11 / 3,313 |
| Food.com test | 4,219 | 90.3 / 19 / 7,652 |

---

## How the databases were built

The shared databases from the CBF teammate do not contain the Allrecipes valid ratings or the Food.com ratings,
so the past ratings were read from the raw files only to know which past recipes were rated ≥ 4.

```bash
python build_full_H.py --db-dir ../team_5H --out .
```

| Input | Content |
| --- | --- |
| `../team_5H/` | teammate's `allrecipes.db`, `foodcom.db`, `cbf_model.json` (copied, not modified) |
| `../data/core-data-{train,valid}_rating.csv` | raw Allrecipes rating files (default of `--allrecipes`) |
| `../data/RAW_interactions.csv`, `../data/RAW_recipes.csv` | raw Food.com rating and recipe files (default of `--foodcom`) |

Past period, same as the original protocol:

- Allrecipes valid: train ratings
- Allrecipes test: train + valid ratings
- Food.com: ratings before 2013-02-01 (rating 0 and ratings dated before the recipe was submitted are dropped)

The script stops if any check fails. All checks passed:

1. The first 5 recipes of the new H are exactly the teammate's H (0 differences in all three sets)
2. Every new H recipe is in the teammate's `seen` table
3. No new H recipe is in the ground truth L
4. Every new H recipe is a candidate

---

## How to run

Use the Python of the `mlteam` environment (plain `python` may point to another Python).
The final setting uses `logistic` loss, so this experiment also runs on Windows.

```bash
cd team_fullH
conda activate mlteam
PY=python

$PY split_train.py                   # train users: first 80% -> H, rest -> targets
$PY lightfm_tune.py                  # tuning on Allrecipes valid
$PY lightfm_final.py --loss logistic --dim 16 --lr 0.005 --alpha 0.0001   # Allrecipes test + Food.com, once
```

Optional CBF results with H all:

```bash
$PY cbf.py evaluate --db allrecipes.db --split test --output cbf_allrecipes_fullH
$PY cbf.py evaluate --db foodcom.db --split test --output cbf_foodcom_fullH
```

---

## Files

| File | Role |
| --- | --- |
| `build_full_H.py` | builds the H-all databases from the teammate's databases + raw ratings |
| `allrecipes.db`, `foodcom.db` | databases with H = all past positives |
| `cbf_model.json` | same ingredient vocabulary and IDF as the main experiment |
| `cbf.py` | teammate's code with the H-length check relaxed |
| `split_train.py` | training split with H = first 80% of each user's positives (`--h-frac` to change) |
| `lightfm_smoke.py`, `lightfm_tune.py`, `lightfm_final.py` | same as the main experiment |
| `lightfm_results/` | `tuning.csv`, `final_test.json` |

---

## Notes and limitations

- This experiment was suggested by a teammate's exploration and run after the main results were known, so it is reported as an additional experiment, not as the main result.
- Food.com was evaluated once for this setting. Settings were chosen on Allrecipes valid only.
- The best learning rate (0.005) was the smallest value in the tuning grid; a smaller value was not tried.
- The 80% training split was not tuned (`--h-frac 0.5` is an option).
- With the 80% split, training targets drop from about 398k (H5) to about 124k.
