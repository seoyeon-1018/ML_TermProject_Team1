import sqlite3

conn = sqlite3.connect('allrecipes.db')

tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
for t in tables:
    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(t, n)

import json
import pandas as pd

train = pd.read_sql_query(
    "SELECT user_id, recipe_id, rating, timestamp FROM interactions WHERE partition='train'", conn)
recipes = pd.read_sql_query("SELECT recipe_id, name, ingredients FROM recipes", conn)
recipes['ingredients'] = recipes['ingredients'].map(json.loads) 

print(train.head())
print(recipes.head())
