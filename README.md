# 플랫폼 간 레시피 추천: Allrecipes → Food.com (zero-shot)

머신러닝 텀 프로젝트 · Team 1

**Allrecipes에서 학습한 레시피 추천 모델이, 재학습 없이 다른 플랫폼(Food.com) 사용자에게도 쓸 만한 추천을 할 수 있을까?**

두 가지 추천 방식을 비교한다.

| 방식 | 설명 |
| --- | --- |
| **CBF** (내용 기반 필터링) | 재료 이진 TF-IDF + 코사인 유사도. 사용자가 좋아한 레시피들의 재료 벡터 평균과 가까운 레시피를 추천 |
| **LightFM** (하이브리드) | 사용자·레시피를 재료 특징으로 표현하는 행렬분해 모델 (WARP / logistic loss) |

두 모델 모두 **재료 정보만** 특징으로 쓰기 때문에, 학습 때 보지 못한 사용자와 레시피(= Food.com)에도 그대로 적용할 수 있다.

---

## 목차

1. [핵심 결과](#1-핵심-결과)
2. [폴더 구조](#2-폴더-구조)
3. [설치](#3-설치)
4. [데이터 준비](#4-데이터-준비)
5. [실행 방법 (main.py)](#5-실행-방법-mainpy)
6. [출력물](#6-출력물)
7. [스크립트를 직접 실행하기](#7-스크립트를-직접-실행하기)
8. [평가 방법](#8-평가-방법)

---

## 1. 핵심 결과

NDCG@10 (사용자별 평균). LightFM은 seed 42·43·44 평균.

| 방법 | Allrecipes test (학습한 플랫폼) | Food.com (zero-shot) |
| --- | ---: | ---: |
| Random | 0.000300 | 0.000085 |
| Popularity (인기순) | **0.016224** | 적용 불가* |
| CBF (H = 첫 5개) | 0.001420 | **0.001086** |
| LightFM (H = 첫 5개, WARP) | 0.005197 | 0.000053 |
| LightFM + 코사인 점수 (사후 개선 1) | 0.001350 | 0.000073 |
| LightFM + 인기도 재가중 (사후 개선 2) | 0.005197 | 0.000053 |
| CBF (H = 과거 전체) | 0.001258 | 0.000706 |
| LightFM (H = 과거 전체, logistic) | **0.007646** | 0.000178 |

\* Food.com 레시피에는 Allrecipes 평점이 없으므로 인기순을 옮겨 쓸 수 없다.

**결론**

1. **같은 플랫폼(Allrecipes)에서는 LightFM이 CBF보다 약 3.7배 좋다.** 하지만 그 성능은 대부분 인기도에서 나온다.
   LightFM은 26,853명에게 단 61개의 서로 다른 레시피만 추천했다(CBF는 15,362개). 어떤 방법도 단순 인기순을 넘지 못했다.
2. **Food.com에서는 LightFM이 Random보다도 낮다.** Allrecipes 사전에 있는 재료가 1~2개뿐인 레시피가
   비정상적으로 높은 점수를 받아 top-10을 채운다(LightFM 추천의 32.4%, 전체 후보 중에서는 3.4%).
3. 코사인 점수와 인기도 재가중, 두 가지 개선 시도 모두 전이 문제를 해결하지 못했다.
4. **플랫폼 간에 의미 있게 옮겨 간 것은 CBF뿐이다.** 정확도는 낮지만 두 플랫폼에서 안정적이다(0.0014 → 0.0011).
5. 사용자 프로필을 과거 전체로 늘리면 LightFM은 좋아지고(+47%, Food.com에서도 Random의 약 2배) CBF는 나빠진다.
   그래도 Food.com 최고 성능은 여전히 CBF(H = 5)다.

그림: [`team_5H/figures/`](team_5H/figures/) (fig1~8), [`team1_data_cbf/results/figures/`](team1_data_cbf/results/figures/),
[`comparison/comparison_figures/`](comparison/comparison_figures/).

---

## 2. 폴더 구조

```
ML_TermProject_Team1/
├── main.py                  # 전체 파이프라인을 한 번에 실행 (5절)
├── environment.yml          # conda 환경 (Python 3.9 + LightFM 1.16)
├── data/                    # 원본 CSV를 넣는 곳 (저장소에는 없음, 4절)
├── team1_data_cbf/          # 데이터 전처리 + CBF + 공통 평가 함수
├── team_5H/                 # 본 실험: H = 첫 긍정 5개
├── team_fullH/              # 추가 실험: H = 과거 긍정 전체
└── comparison/              # 두 실험 비교
```

### `team1_data_cbf/` — 데이터 준비와 CBF

| 파일 | 역할 |
| --- | --- |
| `code/preprocessing_handoff/run_preprocessing.py` | 전처리 실행. `allrecipes`, `foodcom` 두 명령 |
| `code/preprocessing_handoff/src/recipe_cbf/normalize.py` | 두 플랫폼 공통 재료 정규화 규칙 (복수형·별칭 통일 등) |
| `code/preprocessing_handoff/src/recipe_cbf/allrecipes.py` | Allrecipes 평점·레시피 파싱, valid/test 시간 분할 |
| `code/preprocessing_handoff/src/recipe_cbf/prepare.py` | Food.com 파싱, 2013-02-01 기준 분할 |
| `code/preprocessing_handoff/src/recipe_cbf/split.py` | 공통 H / L / seen 분할 함수 |
| `code/preprocessing_handoff/build_db.py` | 전처리 결과를 `allrecipes.db`, `foodcom.db`(SQLite)로 묶음 |
| `code/preprocessing_handoff/make_foodcom_slim.py` | (선택) Food.com 원본에서 필요한 열만 남긴 작은 ZIP 생성 |
| `code/preprocessing_handoff/config/` | 재료 별칭표, 실험·분할 규칙 |
| `code/preprocessing_handoff/quality_checks/` | 원본 전처리 때의 데이터 품질 보고서 |
| `code/cbf.py` | CBF 모델 + **모든 모델이 같이 쓰는 평가 함수** (`load_evaluation`, `evaluate_rankings`) |
| `code/cbf_model.json` | Allrecipes train 레시피로만 학습한 재료 사전(14,418개)과 IDF |
| `code/cbf_report_analysis.py` | CBF 추가 분석(부트스트랩 신뢰구간, 인기도 편향, 정답 수별 적중률)과 그림 3개 |
| `results/` | CBF 지표·사용자별 결과·민감도 분석·그림 |

자세한 내용: [team1_data_cbf/code/README_KO.md](team1_data_cbf/code/README_KO.md)

### `team_5H/` — 본 실험 (H = 첫 긍정 5개)

| 파일 | 역할 |
| --- | --- |
| `split_train.py` | Allrecipes train 평점을 사용자 특징(H, 첫 5개)과 학습 정답으로 분리 → `train_split/` |
| `lightfm_smoke.py` | 특징 생성, LightFM 학습(조기 종료), 랭킹, 베이스라인(Random·Popularity·CBF). 다른 스크립트가 가져다 씀 |
| `lightfm_tune.py` | Allrecipes valid에서 하이퍼파라미터 탐색 (loss × lr, dim × alpha) |
| `lightfm_final.py` | 최종 평가: Allrecipes test + Food.com zero-shot, seed 3개 |
| `lightfm_diagnose.py` | Food.com에서 LightFM이 실패하는 원인 진단 |
| `lightfm_final_cosine.py` | 개선 시도 1: 코사인 점수 |
| `lightfm_final_reweight.py` | 개선 시도 2: 인기도 재가중 + 강한 정규화 |
| `make_plots.py` | 결과 JSON으로 그림 fig1~8 생성 |
| `explore_data.py` | DB 표 구성을 간단히 확인 |
| `cbf.py`, `cbf_model.json` | team1_data_cbf와 같은 파일 (같은 폴더에서 import하기 위해 복사) |
| `lightfm_results/`, `figures/` | 보고서에 쓴 결과와 그림 |

자세한 내용: [team_5H/README.md](team_5H/README.md)

### `team_fullH/` — 추가 실험 (H = 과거 긍정 전체)

| 파일 | 역할 |
| --- | --- |
| `build_full_H.py` | team_5H DB를 복사해 profiles(H) 표만 "과거 긍정 전체"로 다시 만듦. 안전 검사 4개 |
| `cbf.py` | H 길이 제한(정확히 5개)을 1개 이상으로 푼 버전 |
| `split_train.py` | 학습 사용자의 앞 80% 긍정을 H, 나머지를 정답으로 |
| `lightfm_smoke.py`, `lightfm_tune.py`, `lightfm_final.py` | team_5H와 같은 코드 |

자세한 내용: [team_fullH/README.md](team_fullH/README.md)

### `comparison/`

`compare.py`가 두 실험 결과를 모아 비교 표(`comparison_table.csv`)와 그림 3개를 만든다.

---

## 3. 설치

conda가 필요하다 (Anaconda 또는 Miniconda).

```bash
conda env create -f environment.yml
conda activate mlteam
python -c "import lightfm, numpy; print(lightfm.__version__, numpy.__version__)"   # 1.16, 1.x
```

- **LightFM은 1.16을 쓴다.** 최신 1.17은 Windows에서 WARP/BPR loss로 학습할 때 프로그램이 강제 종료(segmentation fault)된다.
  1.17이 설치되어 있으면 스크립트가 강제 종료 대신 안내 메시지를 출력하고 멈춘다.
- LightFM은 pip가 아니라 **conda-forge**에서 설치한다 (pip는 최신 Python에서 빌드 실패).
- **numpy는 2 미만**을 유지한다. 나중에 pip로 다른 패키지를 설치하면 numpy가 올라가 LightFM이 깨질 수 있다.
- Windows에서는 반드시 **conda 환경을 활성화한 상태**에서 Python을 실행한다.
  활성화하지 않고 환경의 python.exe를 직접 부르면 numpy가 DLL을 찾지 못해 계산 중에 종료될 수 있다.
- `compiled without OpenMP support` 경고는 단일 스레드로 돈다는 뜻이며, 느려질 뿐 결과는 같다.

---

## 4. 데이터 준비

원본 CSV(약 830 MB)는 용량 때문에 저장소에 없다. 아래 6개 파일을 받아 `data/` 폴더에 그대로 넣는다.

| 파일 | 데이터셋 | 출처 |
| --- | --- | --- |
| `core-data_recipe.csv` | Allrecipes (foodRecSys-V1) | <https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1> |
| `core-data-train_rating.csv` | 〃 | 〃 |
| `core-data-valid_rating.csv` | 〃 | 〃 |
| `core-data-test_rating.csv` | 〃 | 〃 |
| `RAW_recipes.csv` | Food.com Recipes and Interactions | <https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions> |
| `RAW_interactions.csv` | 〃 | 〃 |

전처리 스크립트는 기본으로 `data/` 폴더를 읽는다. 다른 위치에 두었다면 `python main.py --data <폴더>`.

---

## 5. 실행 방법 (main.py)

`main.py` 하나로 전처리부터 비교 그림까지 전부 실행한다. 저장소 최상위 폴더에서 실행한다.

```bash
python main.py --quick      # 빠른 동작 확인 (약 25분): seed 1개, epoch 2, 작은 탐색 범위
python main.py              # 보고서와 같은 설정으로 전체 실행 (약 2시간)
```

### 단계

| 단계 | 하는 일 | 소요 시간* |
| --- | --- | --- |
| `check` | `data/`의 CSV 6개와 Python·LightFM 버전 확인 | 즉시 |
| `db` | 원본 CSV → 재료 정규화 → 시간 분할 → `allrecipes.db`, `foodcom.db` | 약 3분 |
| `cbf` | CBF 평가(Allrecipes test, Food.com) + 분석 그림 | 약 2분 |
| `h5` | 학습 데이터 분할 → valid 학습 곡선 → 최종 평가 → 진단 → 개선 시도 2개 → 그림 | 약 1시간 20분 (최종 평가 12분, 진단 4분, 코사인 23분, 재가중 30분) |
| `fullh` | H 전체 DB 생성 → 학습 데이터 분할 → 최종 평가 | 약 22분 |
| `compare` | 두 실험 비교 표와 그림 | 즉시 |

\* Windows PC 1대, LightFM 1.16 단일 스레드 기준으로 측정한 대략적인 값. `--quick` 전체는 약 23분.

### 옵션

| 옵션 | 설명 |
| --- | --- |
| `--stages db cbf ...` | 일부 단계만 실행 (예: `--stages h5 compare`). `cbf`·`h5`·`fullh`는 `db` 단계 결과가 필요하다 |
| `--quick` | 빠른 동작 확인. 결과는 `outputs_quick/`에 쓰며, **보고서 수치가 아니다** |
| `--retune` | 하이퍼파라미터 탐색(`lightfm_tune.py`, 30분 이상)을 다시 실행. 기본은 저장된 `tuning.csv`를 복사 (같은 최종 설정이 선택됨) |
| `--force` | 이미 결과가 있는 단계도 다시 실행 |
| `--data <폴더>` | 원본 CSV 위치 (기본 `data/`) |
| `--out <폴더>` | 결과 폴더 (기본 `outputs/`, `--quick`이면 `outputs_quick/`) |
| `--list` | 단계와 세부 스텝 목록만 보여 주고 종료 |

### 동작 방식

- **저장소의 파일은 바꾸지 않는다.** 각 단계는 `outputs/<단계>/` 작업 폴더를 만들고, 해당 실험 폴더의 스크립트를
  거기에 복사해 실행한다. 그래서 결과는 모두 `outputs/`에 생기고, 저장소에 들어 있는 원본 결과(보고서 수치)는 그대로 남는다.
- DB 파일은 작업 폴더에 **하드링크**로 연결한다(같은 파일을 가리키므로 디스크를 더 쓰지 않음). 링크가 안 되면 복사한다.
- **이미 결과가 있는 스텝은 건너뛴다.** 중간에 멈춰도 다시 실행하면 이어서 진행한다.
- 스텝이 실패하면 그 자리에서 멈추고, 로그 마지막 25줄과 전체 로그 위치를 보여 준다.
- 모든 스텝의 출력은 `outputs/logs/`에 저장된다.

---

## 6. 출력물

`python main.py` 실행 후 `outputs/` 구성:

```
outputs/
├── logs/                         # 스텝별 실행 로그
├── db/                           # allrecipes.db, foodcom.db, 전처리 중간 결과(prep/)
├── cbf/
│   ├── cbf_allrecipes_result/    # metrics.json, recommendations.csv, per_user_metrics.csv
│   ├── cbf_foodcom_result/
│   └── cbf_analysis/             # analysis.json, fig_c1~c3
├── team_5H/
│   ├── lightfm_results/          # final_test.json, final_test_cosine.json, final_test_reweight.json, ...
│   └── figures/                  # fig1~fig8
├── team_fullH/
│   ├── allrecipes.db, foodcom.db # H = 과거 전체 버전 DB
│   └── lightfm_results/final_test.json
└── comparison/                   # comparison_table.csv, cmp1~cmp3
```

| 그림 | 내용 |
| --- | --- |
| fig1 | 학습 플랫폼 vs zero-shot 전이, NDCG@10 비교 |
| fig2 | P@10 / R@10 / NDCG@10 전체 |
| fig3 | LightFM valid 학습 곡선 |
| fig4 | 하이퍼파라미터 탐색 결과 |
| fig5 | Food.com 실패 원인 진단 (아는 재료 수) |
| fig6 | 추천 다양성 (서로 다른 추천 레시피 수) |
| fig7 | 개선 시도 결과 |
| fig8 | 인기도 재가중: 다양성과 정확도의 맞교환 |
| fig_c1~c3 | CBF vs Random 신뢰구간, 인기도 차이, 정답 수별 적중률 |
| cmp1~cmp3 | H5 vs H 전체 비교 |

---

## 7. 스크립트를 직접 실행하기

`main.py` 없이 단계별로 실행할 수도 있다. 각 스크립트는 **자기 폴더에서** 실행해야 한다
(같은 폴더의 `cbf.py`, `allrecipes.db` 등을 상대 경로로 찾는다). 이 경우 결과는 각 폴더 안에 생겨
저장소의 원본 결과를 덮어쓴다.

```bash
# 1) DB 생성
cd team1_data_cbf/code/preprocessing_handoff
python run_preprocessing.py allrecipes --output out
python run_preprocessing.py foodcom --output out
python build_db.py --prepared out --dest ../../../team_5H
cd ../../../team_fullH
python build_full_H.py --db-dir ../team_5H --out .

# 2) CBF (DB를 team1_data_cbf/code에 복사한 뒤)
cd ../team1_data_cbf/code
python cbf.py evaluate --db allrecipes.db --split test --output cbf_allrecipes_result
python cbf.py evaluate --db foodcom.db   --split test --output cbf_foodcom_result
python cbf_report_analysis.py

# 3) 본 실험
cd ../../team_5H
python split_train.py
python lightfm_smoke.py --loss warp --dim 32 --lr 0.02 --alpha 0.0 --max-epochs 20 --seeds 42 43 44
python lightfm_tune.py                 # 선택 (30분 이상)
python lightfm_final.py
python lightfm_diagnose.py
python lightfm_final_cosine.py
python lightfm_final_reweight.py
python make_plots.py

# 4) 추가 실험
cd ../team_fullH
python split_train.py
python lightfm_final.py --loss logistic --dim 16 --lr 0.005 --alpha 0.0001

# 5) 비교
cd ..
python comparison/compare.py
```

---

## 8. 평가 방법

| 항목 | 내용 |
| --- | --- |
| 긍정 | 평점 4 이상. Food.com의 평점 0은 의미가 모호해 제외 |
| **H** (사용자 프로필) | 기준 시점 이전, 서로 다른 긍정 레시피 중 **처음 5개**. 테스트 때 쓰는 사용자 정보는 이것뿐 |
| **L** (정답) | 기준 시점 이후의 긍정 레시피 중 후보에 있고 이전에 평가하지 않은 것 |
| seen | 기준 시점 이전에 평가한 모든 레시피 (H 포함). 추천에서 제외 |
| 후보 | Allrecipes: 기준 시점 전에 평점이 확인된 레시피 (test 32,889개). Food.com: 2013-02-01 전에 등록된 레시피 (226,201개) |
| 평가 사용자 | H 5개와 L 1개 이상을 가진 사용자. Allrecipes valid 14,613명, test 26,853명, Food.com 4,219명 |
| 시간 분할 | Allrecipes: 원본 train/valid/test 파일 경계. Food.com: 2013-02-01 이전은 과거, 이후는 미래 |
| 지표 | Precision@10, Recall@10, NDCG@10 (이진 relevance, log2 할인), 사용자별 단순 평균 |
| 공정성 | 모든 모델을 같은 함수 `cbf.evaluate_rankings`로 채점. 음성 샘플링 없이 전체 후보를 점수화 |
| 누설 방지 | 재료 사전·IDF는 Allrecipes train 레시피로만 추정. 설정 선택은 Allrecipes valid에서만. Food.com으로는 학습·튜닝하지 않음 |

LightFM 최종 설정: WARP loss, 32차원, learning rate 0.02, alpha 0, valid NDCG@10로 epoch 선택 (조기 종료, patience 5).
H 전체 실험은 다시 튜닝하여 logistic loss, 16차원, learning rate 0.005, alpha 1e-4.
