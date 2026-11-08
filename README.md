# Cross-Platform Recipe Recommendation (Allrecipes → Food.com, zero-shot)

Machine Learning Term Project · Team 1

Can a recipe recommender trained on **Allrecipes** recommend recipes to **Food.com** users without retraining?
We compare content-based filtering (**CBF**, ingredient TF-IDF + cosine) with a hybrid model (**LightFM**).
Both use only ingredient features, so they can be applied to unseen users and recipes.

| Folder | Content | Details |
| --- | --- | --- |
| [`data/`](data/) | raw CSVs (downloaded, not in the repository) | [data/README.md](data/README.md) |
| [`team1_data_cbf/`](team1_data_cbf/) | preprocessing (raw → SQLite DBs), CBF model, shared evaluation, CBF analysis | [README_KO.md](team1_data_cbf/code/README_KO.md) |
| [`team_5H/`](team_5H/) | main experiment: user profile H = first 5 positives; LightFM training, tuning, final test, diagnosis, improvement attempts, figures | [README.md](team_5H/README.md) |
| [`team_fullH/`](team_fullH/) | additional experiment: H = all past positives | [README.md](team_fullH/README.md) |
| [`comparison/`](comparison/) | H5 vs H-all comparison table and figures | `compare.py` |

Result files (`lightfm_results/`, `figures/`, `team1_data_cbf/results/`, `comparison/comparison_figures/`)
are included, so the reported numbers can be checked without re-running anything.

---

## 1. Setup

```bash
conda env create -f environment.yml      # Python 3.9 + LightFM 1.16 (conda-forge)
conda activate mlteam
```

- Use LightFM **1.16**. LightFM 1.17 crashes on Windows when training with `warp` / `bpr` loss
  (the scripts stop with a message in that case).
- Keep numpy < 2. Always run Python inside the activated environment.

## 2. Data

Download the six raw CSV files into `data/` (see [data/README.md](data/README.md)).

## 3. Build the databases (≈ 5 min)

The SQLite databases (`allrecipes.db`, `foodcom.db`) are generated, not stored in the repository.

```bash
cd team1_data_cbf/code/preprocessing_handoff
python run_preprocessing.py allrecipes --output out    # reads ../../../data by default
python run_preprocessing.py foodcom --output out
python build_db.py --prepared out --dest ../../../team_5H
cd ../../..

# databases with H = all past positives (team_fullH)
cd team_fullH
python build_full_H.py --db-dir ../team_5H --out .
cd ..

# optional: the CBF analysis script expects the databases next to it
cp team_5H/allrecipes.db team_5H/foodcom.db team1_data_cbf/code/
```

## 4. Run the experiments

Run each script from its own folder.

```bash
# CBF (team1_data_cbf/code)
python cbf.py evaluate --db allrecipes.db --split test --output cbf_allrecipes_result
python cbf.py evaluate --db foodcom.db   --split test --output cbf_foodcom_result
python cbf_report_analysis.py

# main experiment (team_5H)
python split_train.py
python lightfm_tune.py              # ~30 min or more (finished runs in lightfm_results/tuning.csv are skipped)
python lightfm_final.py             # Allrecipes test + Food.com zero-shot, 3 seeds
python lightfm_diagnose.py
python lightfm_final_cosine.py
python lightfm_final_reweight.py
python make_plots.py                # figures/

# additional experiment (team_fullH)
python split_train.py
python lightfm_final.py --loss logistic --dim 16 --lr 0.005 --alpha 0.0001

# comparison (any folder)
python comparison/compare.py
```

## 5. Key results (NDCG@10)

| Method | Allrecipes test | Food.com (zero-shot) |
| --- | ---: | ---: |
| Random | 0.000300 | 0.000085 |
| Popularity | 0.016224 | n/a |
| CBF (H5) | 0.001420 | **0.001086** |
| LightFM (H5, WARP) | 0.005197 | 0.000053 |
| LightFM (H all, logistic) | **0.007646** | 0.000178 |

LightFM rows: mean over seeds 42–44, produced with LightFM 1.17 on macOS.
Re-runs on Windows: Random, Popularity and CBF are reproduced exactly; the H-all LightFM row (logistic loss)
is reproduced exactly; the H5 LightFM row (WARP, LightFM 1.16) gives 0.005786 / 0.000070, within the seed
standard deviation, with the same conclusions. See the folder READMEs for the full analysis.
