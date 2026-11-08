"""Binary ingredient TF-IDF + cosine similarity, implemented from equations.

An ingredient phrase is one feature. No splitting 'olive oil' into two words.
Only Allrecipes TRAIN recipe documents may determine vocabulary or IDF.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse


def l2_normalize(matrix):
    matrix = matrix.tocsr()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    scales = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms != 0)
    return sparse.diags(scales).dot(matrix).tocsr()


class IngredientCBF:
    def __init__(self, min_df=1):
        if not isinstance(min_df, int) or min_df < 1:
            raise ValueError("min_df must be a positive document count")
        self.min_df = min_df
        self.vocabulary = {}
        self.idf = None
        self.metadata = {}

    def fit(self, training_recipes, alias_sha256):
        recipes = list(training_recipes)
        if not recipes or any(r.get("dataset") != "allrecipes" or not r["recipe_id"].startswith("allrecipes:r:") for r in recipes):
            raise ValueError("fit requires nonempty Allrecipes training recipes only")
        ids = [r["recipe_id"] for r in recipes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate training recipe IDs")
        dfs = Counter()
        for r in recipes:
            dfs.update(set(r["ingredients"]))
        words = sorted(w for w, count in dfs.items() if count >= self.min_df)
        if not words:
            raise ValueError("No ingredient features survive min_df")
        self.vocabulary = {w: i for i, w in enumerate(words)}
        self.idf = np.array([np.log((1 + len(recipes)) / (1 + dfs[w])) + 1 for w in words], dtype=np.float32)
        self.metadata = {"training_dataset": "allrecipes", "training_recipe_count": len(recipes),
                         "training_recipe_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
                         "alias_sha256": alias_sha256, "n_features": len(words), "min_df": self.min_df,
                         "tf": "binary ingredient occurrence", "idf": "ln((1+n_train_docs)/(1+train_df))+1",
                         "norm": "l2", "foodcom_fit_performed": False}
        return self

    def transform(self, ingredient_lists):
        if self.idf is None:
            raise ValueError("Model is not fitted")
        indices, indptr, values = [], [0], []
        total_tokens = oov_tokens = zero_docs = 0
        for ingredients in ingredient_lists:
            columns = []
            for phrase in sorted(set(ingredients)):
                total_tokens += 1
                col = self.vocabulary.get(phrase)
                if col is None:
                    oov_tokens += 1
                else:
                    columns.append(col)
            columns.sort()
            if not columns:
                zero_docs += 1
            indices.extend(columns)
            values.extend(self.idf[columns])
            indptr.append(len(indices))
        x = sparse.csr_matrix((np.asarray(values, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
                              shape=(len(indptr) - 1, len(self.vocabulary)), dtype=np.float32)
        return l2_normalize(x), {"documents": len(indptr) - 1, "ingredient_occurrences": total_tokens,
                                 "oov_occurrences": oov_tokens,
                                 "oov_rate": oov_tokens / total_tokens if total_tokens else 0.0,
                                 "zero_vector_documents": zero_docs}



class CandidateIndex:
    """Build once; score sparse profiles in small batches, not an item-item matrix."""
    def __init__(self, model, recipes):
        recipes = sorted(recipes, key=lambda r: r["recipe_id"])
        self.ids = [r["recipe_id"] for r in recipes]
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Duplicate candidate IDs")
        self.row_of = {rid: i for i, rid in enumerate(self.ids)}
        self.x, self.diagnostics = model.transform([r["ingredients"] for r in recipes])
        self.xt = self.x.T.tocsr()

    def profile(self, h_recipe_ids):
        if not h_recipe_ids or len(h_recipe_ids) != len(set(h_recipe_ids)):
            raise ValueError("Profile must contain distinct history recipes")
        try:
            rows = [self.row_of[rid] for rid in h_recipe_ids]
        except KeyError as e:
            raise ValueError(f"H recipe not in candidate catalog: {e}") from e
        weights = sparse.csr_matrix((np.full(len(rows), 1 / len(rows), dtype=np.float32),
                                     (np.zeros(len(rows), dtype=np.int32), rows)), shape=(1, len(self.ids)))
        return l2_normalize(weights.dot(self.x))

    def recommend_batch(self, profiles, seen_per_user, k=10):
        if k < 1 or profiles.shape[0] != len(seen_per_user):
            raise ValueError("Invalid k or mismatched user batch")
        score_matrix = profiles.dot(self.xt).toarray()
        outputs = []
        for scores, seen in zip(score_matrix, seen_per_user):
            seen_rows = [self.row_of[r] for r in seen if r in self.row_of]
            scores[seen_rows] = -np.inf
            eligible = np.flatnonzero(np.isfinite(scores))
            n = min(k, len(eligible))
            if n == 0:
                outputs.append([])
                continue
            # Select ties deterministically by sorted recipe ID, including all-zero profiles.
            boundary = np.partition(scores[eligible], len(eligible) - n)[len(eligible) - n]
            better = eligible[scores[eligible] > boundary]
            ties = eligible[scores[eligible] == boundary]
            selected = np.concatenate([better, ties[:n - len(better)]])
            selected = selected[np.lexsort((selected, -scores[selected]))]
            outputs.append([(self.ids[int(i)], float(scores[i])) for i in selected])
        return outputs

# Dataset tables replace the many separate CSV/JSONL input files.
import argparse
import csv
import sqlite3
from collections import defaultdict

ROOT = Path(__file__).resolve().parent


def load_frozen(path=ROOT / 'cbf_model.json'):
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    meta = artifact['metadata']
    if meta.get('training_dataset') != 'allrecipes' or meta.get('foodcom_fit_performed') is not False:
        raise ValueError('Expected an Allrecipes-only frozen model')
    model = IngredientCBF(meta['min_df'])
    model.vocabulary = artifact['vocabulary']
    model.idf = np.asarray(artifact['idf'], dtype=np.float32)
    model.metadata = meta
    if sorted(model.vocabulary.values()) != list(range(len(model.idf))):
        raise ValueError('Vocabulary and IDF dimensions differ')
    if not np.isfinite(model.idf).all() or (model.idf <= 0).any():
        raise ValueError('Invalid IDF weights')
    return model


def load_evaluation(db_path, split='test'):
    """Return fixed users, candidates, H, L, seen and normalized recipe features.

    LightFM can use this same object and call evaluate_rankings with its ranking.
    Read ingredients from the prepared database; no new normalization is needed.
    """
    h, targets, seen = defaultdict(list), defaultdict(set), defaultdict(set)
    with sqlite3.connect(db_path) as db:
        metadata = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
        if split not in metadata['splits']:
            raise ValueError(f'Split {split} unavailable in {db_path}')
        users = [r[0] for r in db.execute('SELECT user_id FROM users WHERE split=? ORDER BY user_id', (split,))]
        for uid, rid in db.execute('SELECT user_id,recipe_id FROM profiles WHERE split=? ORDER BY user_id,history_order', (split,)):
            h[uid].append(rid)
        for uid, rid in db.execute('SELECT user_id,recipe_id FROM targets WHERE split=?', (split,)):
            targets[uid].add(rid)
        for uid, rid in db.execute('SELECT user_id,recipe_id FROM seen WHERE split=?', (split,)):
            seen[uid].add(rid)
        recipes = [{'dataset': metadata['dataset'], 'recipe_id': rid, 'name': name, 'ingredients': json.loads(ingredients)}
                   for rid, name, ingredients in db.execute('SELECT r.recipe_id,r.name,r.ingredients FROM recipes r JOIN candidates c ON r.recipe_id=c.recipe_id WHERE c.split=? ORDER BY r.recipe_id', (split,))]
    if not users or set(users) != set(h) or set(users) != set(targets):
        raise ValueError('Empty or inconsistent cohort')
    for uid in users:
        # H may hold any number (>= 1) of distinct past positives (was: exactly 5)
        if len(h[uid]) < 1 or len(set(h[uid])) != len(h[uid]) or not set(h[uid]).issubset(seen[uid]) or targets[uid] & seen[uid]:
            raise ValueError('Invalid profile/seen/target data')
    return {'dataset': metadata['dataset'], 'split': split, 'users': users, 'recipes': recipes,
            'candidate_ids': {r['recipe_id'] for r in recipes}, 'H': dict(h), 'L': dict(targets), 'seen': dict(seen)}


def evaluate_rankings(evaluation, rankings, k=10):
    """Evaluate any model's {user_id: [ranked recipe IDs]} on the fixed cohort.

    Missing users, duplicates, seen items and out-of-catalog items are errors.
    Precision uses denominator K; metrics are macro means across all users.
    """
    if k < 1 or set(rankings) != set(evaluation['users']):
        raise ValueError('Positive K and rankings for every fixed user are required')
    rows = []
    for uid in evaluation['users']:
        ranked = list(rankings[uid])[:k]
        if len(set(ranked)) != len(ranked) or not set(ranked).issubset(evaluation['candidate_ids']) or set(ranked) & evaluation['seen'][uid]:
            raise ValueError(f'Invalid recommendation list for {uid}')
        eligible_count = len(evaluation['candidate_ids']) - sum(rid in evaluation['candidate_ids'] for rid in evaluation['seen'][uid])
        expected_length = min(k, eligible_count)
        if len(ranked) != expected_length:
            raise ValueError(f'Incomplete recommendation list for {uid}')
        target = evaluation['L'][uid]
        hits = sum(rid in target for rid in ranked)
        dcg = sum(1 / np.log2(rank + 2) for rank, rid in enumerate(ranked) if rid in target)
        idcg = sum(1 / np.log2(rank + 2) for rank in range(min(k, len(target))))
        rows.append({'user_id': uid, 'hits': hits, 'n_targets': len(target), 'n_recommended': len(ranked),
                     f'precision@{k}': hits / k, f'recall@{k}': hits / len(target), f'ndcg@{k}': float(dcg / idcg)})
    summary = {'dataset': evaluation['dataset'], 'split': evaluation['split'], 'users': len(rows),
               'candidates': len(evaluation['candidate_ids']), 'k': k,
               'macro_metrics': {m: float(np.mean([row[m] for row in rows])) for m in [f'precision@{k}', f'recall@{k}', f'ndcg@{k}']}}
    return summary, rows


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def run_evaluate(db_path, split, model_path, output, batch_size=16):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    evaluation = load_evaluation(db_path, split)
    model = load_frozen(model_path)
    index = CandidateIndex(model, evaluation['recipes'])
    rankings = {}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    temp = output / '.recommendations.csv.tmp'
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'rank', 'recipe_id', 'cosine_similarity', 'is_target'])
        users = evaluation['users']
        for start in range(0, len(users), batch_size):
            batch = users[start:start + batch_size]
            profiles = sparse.vstack([index.profile(evaluation['H'][uid]) for uid in batch], format='csr')
            ranked_batch = index.recommend_batch(profiles, [evaluation['seen'][uid] for uid in batch], 10)
            for uid, ranked in zip(batch, ranked_batch):
                rankings[uid] = [rid for rid, score in ranked]
                writer.writerows((uid, rank, rid, score, int(rid in evaluation['L'][uid])) for rank, (rid, score) in enumerate(ranked, 1))
            if start % 256 == 0 or start + batch_size >= len(users):
                print(f'{min(start + batch_size, len(users))}/{len(users)} users', flush=True)
    temp.replace(output / 'recommendations.csv')
    summary, per_user = evaluate_rankings(evaluation, rankings)
    summary['ingredient_diagnostics'] = index.diagnostics
    save_json(output / 'metrics.json', summary)
    temp = output / '.per_user_metrics.csv.tmp'
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_user[0])); writer.writeheader(); writer.writerows(per_user)
    temp.replace(output / 'per_user_metrics.csv')
    print(json.dumps(summary, indent=2))
    return summary


def run_train(db_path, output, min_df=1):
    with sqlite3.connect(db_path) as db:
        dataset = json.loads(db.execute("SELECT value FROM metadata WHERE key='dataset'").fetchone()[0])
        if dataset != 'allrecipes':
            raise ValueError('Only Allrecipes train may fit CBF')
        rows = db.execute("SELECT recipe_id,ingredients FROM recipes WHERE recipe_id IN (SELECT recipe_id FROM interactions WHERE partition='train') ORDER BY recipe_id")
        recipes = [{'dataset': 'allrecipes', 'recipe_id': rid, 'ingredients': json.loads(ingredients)} for rid, ingredients in rows]
    original = load_frozen()
    model = IngredientCBF(min_df).fit(recipes, original.metadata['alias_sha256'])
    save_json(output, {'metadata': model.metadata, 'vocabulary': model.vocabulary, 'idf': model.idf.astype(float).tolist()})
    print(f'Saved {len(model.vocabulary)} ingredient features to {output}')


def main():
    parser = argparse.ArgumentParser(description='Minimal CBF and common ranking evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train'); train.add_argument('--db', default=str(ROOT / 'allrecipes.db')); train.add_argument('--min-df', type=int, default=1); train.add_argument('--output', default='new_cbf_model.json')
    evaluate = sub.add_parser('evaluate'); evaluate.add_argument('--db', required=True); evaluate.add_argument('--split', choices=['valid','test'], default='test'); evaluate.add_argument('--model', default=str(ROOT / 'cbf_model.json')); evaluate.add_argument('--output', default='cbf_results'); evaluate.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.command == 'train':
        run_train(args.db, args.output, args.min_df)
    else:
        run_evaluate(args.db, args.split, args.model, args.output, args.batch_size)


if __name__ == '__main__':
    main()
