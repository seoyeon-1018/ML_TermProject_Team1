# 데이터 준비 · CBF 패키지 (Team 1)

데이터 파싱·재료 정규화, 시간순 분할(H/L), CBF 모델과 공통 평가 함수를 한 폴더에 모았다. LightFM도 같은 DB와 같은 평가 함수(`cbf.evaluate_rankings`)로 채점한다.

## 폴더 구성

| 파일 | 용도 |
| --- | --- |
| `allrecipes.db`* | 정규화된 레시피, Allrecipes train 평점, valid/test용 H·L·seen·후보·사용자 |
| `foodcom.db`* | 정규화된 Food.com 레시피, 외부 test용 H·L·seen·후보·사용자 |
| `cbf.py` | CBF 모델(binary TF-IDF + 코사인)과 모든 모델이 쓰는 공통 평가 함수 |
| `cbf_model.json` | 선택된 CBF의 재료 사전·IDF·학습 정보, valid 비교 결과 |
| `cbf_report_analysis.py` | CBF 보고서의 추가 분석과 그림 1~3 생성 |
| `preprocessing_handoff/` | 원본 → 정규화 → 분할 → DB 생성 코드와 설정 |
| `requirements.txt` | NumPy·SciPy·pandas (그림은 matplotlib) |

\* 두 DB는 용량이 커서 이 폴더에 두지 않았다. 같은 파일이 `../../team_5H/`에 있으므로 이 폴더로 복사하거나, 아래 재현 순서의 `build_db.py --dest ..`로 새로 만든다. `cbf_report_analysis.py`는 이 폴더의 DB를 읽는다.

`preprocessing_handoff/` 안의 구성은 다음과 같다.

| 파일 | 용도 |
| --- | --- |
| `run_preprocessing.py` | `allrecipes` / `foodcom` 전처리 실행 |
| `src/recipe_cbf/normalize.py` | 두 플랫폼 공통 재료 정규화 규칙 |
| `src/recipe_cbf/allrecipes.py` | Allrecipes 평점·레시피 파싱, valid/test 분할 |
| `src/recipe_cbf/prepare.py` | Food.com RAW 파일 파싱, 2013-02-01 기준 분할 |
| `src/recipe_cbf/split.py` | 공통 H/L/seen 분할 함수 (`profile_rule`: first / recent) |
| `build_db.py` | 전처리 결과를 `allrecipes.db`, `foodcom.db`로 묶음 |
| `make_foodcom_slim.py` | Food.com 원본에서 필요한 열만 남겨 용량을 줄임 |
| `config/` | 재료 별칭표, 실험·분할 규칙 |
| `quality_checks/` | 원본 전처리 실행 시의 데이터 품질 보고서 |

## 전체 재현 순서

```bash
cd preprocessing_handoff
# 원본 CSV는 기본적으로 프로젝트 최상위 data 폴더(ML_TermProject_1/data)에서 읽는다.
python run_preprocessing.py allrecipes --output out
python run_preprocessing.py foodcom --output out
python build_db.py --prepared out --dest ..
cd ..
python cbf.py train --min-df 1 --output new_cbf_model.json      # 선택: 사전·IDF 재학습
python cbf.py evaluate --db allrecipes.db --split test --output cbf_allrecipes_result
python cbf.py evaluate --db foodcom.db --split test --output cbf_foodcom_result
python cbf_report_analysis.py
```

원본 위치: `ML_TermProject_1/data/`에 `core-data_recipe.csv`, `core-data-{train,valid,test}_rating.csv`(Allrecipes), `RAW_recipes.csv`, `RAW_interactions.csv`(Food.com)를 둔다. 다른 위치를 쓰려면 allrecipes는 `--data-dir <폴더>`(또는 `--recipe-archive` 등에 CSV·ZIP 파일), foodcom은 `--source <폴더 또는 ZIP>`(기존 `--zip`도 동작)을 준다.

용량을 줄이려면 `python make_foodcom_slim.py`(기본 입력: data 폴더)로 필요한 열만 남긴 `foodcom_slim.zip`을 만들고 `run_preprocessing.py foodcom --source foodcom_slim.zip`으로 넣을 수 있다.

## 고정된 평가 조건

- 긍정은 평점 4 이상이다. 0점은 의미가 모호해 Food.com에서 제외한다.
- **H(프로필 입력) 규칙**
  - 최종 결과: 모든 평가 집합에서 기준 시점 이전 서로 다른 긍정 레시피 **최초 5개**(`profile_rule=first`, 기본값). LightFM 학습용 H(`split_train.py`)와 같은 정의이며, 팀 비교표도 이 기준이다.
  - 민감도 분석: Food.com에서 기준일 이전 **최근 5개**(`--profile-rule recent`, 팀 계획 9절의 원안).
  - H 전체(과거 긍정 전체) 실험은 LightFM 담당 팀원의 추가 실험 자료(`team_fullH`)를 따른다.
  - 같은 날짜의 기록은 recipe_id 순으로 순서를 정한다. 날짜 단위 자료라 실제 선후는 알 수 없다.
- seen은 H를 포함한 과거 평점 1~5의 모든 레시피이며 추천에서 제외한다.
- L은 미래 긍정 중 후보에 있고 seen에 없는 레시피다. H 5개와 L 1개 이상인 사용자를 평가한다. H 규칙을 바꿔도 L·seen·평가 사용자는 같다.
- Food.com 후보는 기준일 이전에 등록(submitted)된 레시피다. Allrecipes는 등록일이 없어 기준 시점 이전에 평점이 확인된 레시피만 후보로 넣는다.
- 재료 사전·IDF는 Allrecipes train 레시피로만 추정하고, 설정 선택은 Allrecipes valid에서만 한다. Food.com으로는 재학습·튜닝하지 않는다.
- 전체 후보에서 seen을 뺀 모든 레시피를 점수화한다. 음성 샘플링은 쓰지 않는다. Precision 분모는 10, NDCG는 이진 relevance와 log2 할인, 사용자별 단순 평균(macro)이다.

## 공통 평가 함수

```python
from cbf import load_evaluation, evaluate_rankings
evaluation = load_evaluation('foodcom.db', split='test')
# rankings = {user_id: [1위 recipe_id, ..., 10위 recipe_id], ...}
summary, per_user = evaluate_rankings(evaluation, rankings)
```

`evaluate_rankings`는 모든 평가 사용자의 추천이 있는지, 중복·seen·후보 밖 레시피가 없는지 확인한 뒤 P@10·R@10·NDCG@10을 계산한다.

## 확인한 사항

- `normalize.py`, `config/ingredient_aliases.json`의 SHA-256이 `cbf_model.json`에 기록된 값과 같다. 즉 CBF 학습에 쓴 정규화 코드와 같은 파일이다.
- Allrecipes 평점 원본 3개의 SHA-256이 품질 보고서와 같고, 이 원본으로 다시 분할·DB를 만들면 `allrecipes.db`의 7개 표가 모두 같다.
- DB에 저장된 정규화 재료(Allrecipes 16,528종, Food.com 14,598종)는 모두 고정점이다. 다시 정규화해도 값이 바뀌지 않는다.
- `cbf.py`의 재학습 결과(사전·IDF)와 Allrecipes 추천·지표 파일이 기존 결과와 바이트 단위로 같다.
- Food.com 원본(RAW_recipes.csv, RAW_interactions.csv)을 `profile_rule=first`로 다시 처리하면 기존 foodcom.db의 모든 표가 같다. 레시피 파싱·정규화부터 분할까지 원본에서 재현된다.
- 최종 foodcom.db는 기존 파일 그대로다(최초 5개). 기본 설정으로 전처리를 다시 돌리면 이 파일과 모든 표가 같다.
- `--profile-rule recent`로 만들면 H만 바뀌고 L·seen·후보·사용자는 같다. 4,219명 중 3,071명은 H 5개가 모두 바뀌고, 최근 5개 경계에 같은 날짜의 긍정이 더 있는 사용자가 632명(15.0%)이다.

## 출처

- Allrecipes: https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1 (core-data 파일, 이미지 미사용)
- Food.com: Food.com Recipes and Interactions의 RAW_recipes.csv, RAW_interactions.csv (팀 저장소 https://github.com/milin0808/food.com 에 올린 파일)
- 코드는 외부 블로그 코드를 복사하지 않고 작성했다.
