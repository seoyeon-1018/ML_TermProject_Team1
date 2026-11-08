# 원본 데이터

원본 CSV는 용량(약 830 MB) 때문에 저장소에 넣지 않았다. 아래 6개 파일을 받아 이 폴더에 그대로 넣는다.

| 파일 | 데이터셋 | 출처 |
| --- | --- | --- |
| `core-data_recipe.csv` | Allrecipes (foodRecSys-V1) | <https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1> |
| `core-data-train_rating.csv` | Allrecipes | 〃 |
| `core-data-valid_rating.csv` | Allrecipes | 〃 |
| `core-data-test_rating.csv` | Allrecipes | 〃 |
| `RAW_recipes.csv` | Food.com Recipes and Interactions | <https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions> |
| `RAW_interactions.csv` | Food.com Recipes and Interactions | 〃 |

위 CSV만 사용한다 (이미지, `PP_*`, `interactions_*` 파일은 쓰지 않음).
전처리 스크립트와 `main.py`는 기본으로 이 폴더를 읽는다.
