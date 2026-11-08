# Cross-Platform Recipe Recommendation (Allrecipes → Food.com, zero-shot)

Machine Learning Term Project · Fall 2026 · Team 1

Can a recipe recommender trained on one platform (**Allrecipes**) recommend recipes to users of another platform (**Food.com**) **without retraining**?
We compare a content-based filter (**CBF**) with a hybrid model (**LightFM**) that both use only ingredient features, so they can be applied to users and recipes never seen during training.

| | |
| --- | --- |
| Filtering methods | Content-based filtering (CBF), Hybrid filtering (LightFM) |
| Training data | Allrecipes (ratings + recipe ingredients) |
| Test data | Allrecipes test (same platform), Food.com (zero-shot, different platform) |
| Metrics | Precision@10, Recall@10, NDCG@10 (macro average over users) |

---

## Key results

| Method | Allrecipes test NDCG@10 | Food.com NDCG@10 |
| --- | ---: | ---: |
| Random | 0.000300 | 0.000085 |
| Popularity* | **0.016224** | n/a |
| CBF | 0.001420 | **0.001086** |
| LightFM (final) | 0.005197 | 0.000053 |
| LightFM + cosine scoring (post-hoc) | 0.001350 | 0.000073 |
| LightFM + popularity re-weighting (post-hoc) | 0.005197 | 0.000053 |

\* Popularity cannot be transferred: Food.com recipes have no Allrecipes ratings.
LightFM rows are the mean over 3 seeds (42, 43, 44).

**Findings**

1. On Allrecipes, LightFM is about 3.7× better than CBF, but its accuracy comes mostly from popularity: it recommended almost the same recipes to everyone (61 different recipes for 26,853 users; CBF: 15,362).
2. On Food.com, LightFM falls **below random**. Recipes with only 1–2 ingredients known to the Allrecipes vocabulary get extreme scores and fill the top-10 (32.4% of LightFM's recommendations vs. 3.4% of all candidates).
3. Neither cosine scoring nor popularity re-weighting fixed the transfer. Removing the popularity effect drops in-domain accuracy to the CBF level.
4. Only CBF transferred meaningfully: low accuracy, but stable across platforms (0.0014 → 0.0011).

Figures are in [`figures/`](figures/) (created by `make_plots.py`).

---

## Software architecture

```
allrecipes.db ─ split_train.py ─► train_split/ ─► lightfm_smoke.py ─► lightfm_tune.py ─► lightfm_final.py ─► lightfm_results/ ─► make_plots.py ─► figures/
foodcom.db ──────────────────────────────────────────────────────────────────────────────┘   (+ diagnose / cosine / reweight)
```

| Stage | File | Role |
| --- | --- | --- |
| Data | `allrecipes.db`, `foodcom.db` | SQLite databases: recipes (normalized ingredients), Allrecipes train ratings, fixed evaluation sets (H, L, seen, candidates, users) |
| Data | `cbf_model.json` | Ingredient vocabulary + IDF fitted on Allrecipes train recipes only |
| Shared | `cbf.py` | CBF model and the **common evaluation functions** (`load_evaluation`, `evaluate_rankings`) used by every model |
| Shared | `lightfm_smoke.py` | Feature building, LightFM training with early stopping, ranking, baselines (Random, Popularity, CBF). Its functions are imported by the other LightFM scripts |
| Pipeline | `split_train.py` | Splits Allrecipes train ratings into H (first 5 positive recipes → user features) and training targets |
| Pipeline | `lightfm_tune.py` | Hyperparameter search on Allrecipes valid (loss, learning rate, dim, alpha) |
| Pipeline | `lightfm_final.py` | Final evaluation on Allrecipes test and Food.com zero-shot |
| Analysis | `lightfm_diagnose.py` | Diagnoses why LightFM fails on Food.com |
| Analysis | `lightfm_final_cosine.py` | Improvement attempt 1: cosine scoring |
| Analysis | `lightfm_final_reweight.py` | Improvement attempt 2: popularity re-weighting + stronger regularization |
| Output | `make_plots.py` | Builds all figures from `lightfm_results/` |

---

## Repository structure

```
.
├── README.md
├── environments.txt          # CBF (numpy, scipy, pandas), conda environment (Python 3.9, LightFM 1.16 from conda-forge)
├── allrecipes.db, foodcom.db # prepared SQLite databases (from team1_data_cbf preprocessing)
├── cbf.py                    # CBF + common evaluation (teammate)
├── cbf_model.json
├── explore_data.py           # quick look at the database tables
├── split_train.py
├── lightfm_smoke.py
├── lightfm_tune.py
├── lightfm_final.py
├── lightfm_diagnose.py
├── lightfm_final_cosine.py
├── lightfm_final_reweight.py
├── make_plots.py
├── train_split/              # output of split_train.py
├── lightfm_results/          # result JSON/CSV files
└── figures/                  # fig1–fig8
```

The databases (`allrecipes.db`, `foodcom.db`) are large; if they are missing, rebuild them with
`team1_data_cbf/code/preprocessing_handoff` (see [Data](#data)).

---

## Setup

LightFM does not build with `pip` on recent Python versions, so we use conda.
The project uses **LightFM 1.16 on Python 3.9** (see `environments.txt`); WARP training with it was verified on Windows.

```bash
conda create -n mlteam -c conda-forge python=3.9 lightfm=1.16 "numpy<2" "pandas<3" scipy scikit-learn matplotlib
conda activate mlteam
python -c "import lightfm, numpy; print('ok', lightfm.__version__, numpy.__version__)"   # 1.16, numpy 1.x
```

Notes from our setup:

- Install LightFM from **conda-forge**, not pip.
- Keep **numpy < 2**. Installing packages with pip afterwards can upgrade numpy and break LightFM.
- LightFM may print `compiled without OpenMP support` and run single-threaded. This is only slower; results are the same.
- Why not LightFM 1.17 (the latest): on Windows it crashes (segmentation fault) when training with `warp` / `bpr` loss.
  `lightfm_smoke.check_lightfm` stops with an explanation in that case instead of crashing.
- The numbers in this README were produced with LightFM 1.17 on macOS. With 1.16 on Windows, `lightfm_final.py` gives
  LightFM NDCG@10 0.005786 (Allrecipes test) / 0.000070 (Food.com) vs. 0.005197 / 0.000053 above.
  The gap is within the seed std (0.00082 / 0.000038) and the conclusions are unchanged;
  WARP negative sampling is not bit-identical across versions/platforms.
- Always start Python from an activated conda environment (`conda activate ...` or `conda run -n ... python ...`),
  otherwise numpy may fail to load its DLLs on Windows.

---

## How to run

Run from the repository folder with the `mlteam` environment active, in this order.

```bash
python split_train.py              # 1. split train ratings -> train_split/
python lightfm_smoke.py            # 2. sanity check + baselines on valid
python lightfm_tune.py             # 3. hyperparameter tuning (~30 min)
python lightfm_final.py            # 4. final test: Allrecipes test + Food.com (run once)
python lightfm_diagnose.py         # 5. failure analysis
python lightfm_final_cosine.py     # 6. improvement attempt 1
python lightfm_final_reweight.py   # 7. improvement attempt 2
python make_plots.py               # 8. figures -> figures/
```

Useful options of `lightfm_smoke.py`:

```bash
python lightfm_smoke.py --loss warp --dim 32 --lr 0.02 --alpha 0.0 --max-epochs 20 --seeds 42 43 44
python lightfm_smoke.py --skip-cbf   # skip the CBF baseline (faster)
```

---

## Evaluation protocol

- Positive = rating ≥ 4. **H** = a user's first 5 distinct positive recipes; it is the only user information used at test time.
- **L** (ground truth) = later positive recipes that are candidates and not already rated.
- All candidates except already rated recipes (**seen**) are scored; Top-10 is evaluated.
- Food.com: ratings before 2013-02-01 are the past (H, seen); ratings from that date on are the future (L).
- Settings are chosen **only on Allrecipes valid**. Food.com is never used for training or tuning.
- Every model is evaluated with the same function, `cbf.evaluate_rankings`.

Final LightFM setting: WARP loss, 32 dimensions, learning rate 0.02, alpha 0, best epoch selected on valid (early stopping, patience 5).

---

## Data

| Dataset | Source | Use |
| --- | --- | --- |
| Allrecipes | <https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1> | training, validation, in-domain test |
| Food.com | Food.com Recipes and Interactions (RAW files) | zero-shot test |

The prepared databases (normalized ingredients, time splits, fixed evaluation sets) were built by the CBF teammate
with `team1_data_cbf/code/preprocessing_handoff` from the raw CSVs in the project `data/` folder
(see `team1_data_cbf/code/README_KO.md`). Place `allrecipes.db` and `foodcom.db` in this folder.

---

## Limitations

- Allrecipes timestamps are `dateLastModified`, not the time a rating was first written.
- User profiles use only the first 5 positive recipes.
- 9.33% of ingredient occurrences on Food.com are not in the Allrecipes vocabulary.
- Candidate sets are very large (32,889 on Allrecipes, 226,201 on Food.com), so all absolute scores are low.
- The two improvement attempts were designed after seeing the test results (post-hoc analysis). Their settings were still chosen on Allrecipes valid only.

---


## References

- LightFM: <https://github.com/lyst/lightfm> (Apache-2.0)
- M. Kula, "Metadata Embeddings for User and Item Cold-start Recommendations", CBRecSys Workshop at RecSys 2015.
- AI assistance: _to be completed according to the course policy._
