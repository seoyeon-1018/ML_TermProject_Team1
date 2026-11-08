# Raw data

The raw CSV files are not stored in the repository because of their size (about 830 MB).
Download them and put these six files directly in this folder:

| File | Dataset | Source |
| --- | --- | --- |
| `core-data_recipe.csv` | Allrecipes (foodRecSys-V1) | <https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1> |
| `core-data-train_rating.csv` | Allrecipes | same |
| `core-data-valid_rating.csv` | Allrecipes | same |
| `core-data-test_rating.csv` | Allrecipes | same |
| `RAW_recipes.csv` | Food.com Recipes and Interactions | <https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions> |
| `RAW_interactions.csv` | Food.com Recipes and Interactions | same |

Only the CSV files above are used (no images, no `PP_*` / `interactions_*` files).
The preprocessing scripts read this folder by default.
