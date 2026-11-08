"""Content-based filtering (CBF) with binary ingredient TF-IDF + cosine similarity.

Every formula is implemented directly with NumPy/SciPy (no scikit-learn).

Model
-----
* One normalized ingredient phrase = one feature ("olive oil" is NOT split into words).
* TF  : 1 if the recipe contains the ingredient, else 0 (binary).
* IDF : idf(w) = ln((1 + N) / (1 + df(w))) + 1, N = number of Allrecipes TRAIN recipes.
* Recipe vector  x_i = L2-normalize(TF * IDF). Out-of-vocabulary ingredients are ignored.
* User profile   p_u = L2-normalize(mean of the user's H recipe vectors).
* Score          score(u, i) = p_u . x_i  (= cosine similarity, both vectors have length 1).
* Ranking        seen recipes are masked with -inf; ties are broken by recipe_id (string order).

Only Allrecipes TRAIN recipe documents may determine the vocabulary or IDF.
The fitted model is applied to Food.com unchanged (zero-shot transfer).

This file also holds the shared evaluation used by every model in the project:
``load_evaluation`` reads the fixed cohort and ``evaluate_rankings`` scores any
{user_id: [recipe_id, ...]} ranking with Precision@K, Recall@K and NDCG@K.

Usage
-----
python cbf.py train    --db allrecipes.db --min-df 1 --output new_cbf_model.json
python cbf.py evaluate --db foodcom.db --split test --output cbf_foodcom_result
"""
import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent
# Alias table used by the shared ingredient normalizer (preprocessing_handoff/config).
DEFAULT_ALIASES = ROOT / "preprocessing_handoff" / "config" / "ingredient_aliases.json"


def l2_normalize(matrix):
    """Scale every row of a sparse matrix to unit L2 length.

    Rows whose norm is 0 (recipes with no known ingredient) stay all-zero instead
    of producing NaN. Implemented as D @ X with D = diag(1 / ||row||).
    """
    matrix = matrix.tocsr()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    scales = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms != 0)
    return sparse.diags(scales).dot(matrix).tocsr()


class IngredientCBF:
    """Vocabulary + IDF weights learned from Allrecipes training recipes.

    Parameters
    ----------
    min_df : int
        Minimum number of training recipes an ingredient must appear in to become
        a feature. Chosen on Allrecipes valid from {1, 2, 5}; 1 was selected.
    """

    def __init__(self, min_df=1):
        if not isinstance(min_df, int) or min_df < 1:
            raise ValueError("min_df must be a positive document count")
        self.min_df = min_df
        self.vocabulary = {}   # ingredient phrase -> column index
        self.idf = None        # float32 array aligned with the vocabulary columns
        self.metadata = {}

    def fit(self, training_recipes, alias_sha256):
        """Learn the vocabulary and IDF from Allrecipes TRAIN recipes only.

        training_recipes : iterable of dicts with 'dataset', 'recipe_id', 'ingredients'.
        alias_sha256     : hash of the alias file used for normalization (provenance only).
        """
        recipes = list(training_recipes)
        # Guard against leakage: refuse anything that is not an Allrecipes recipe.
        if not recipes or any(r.get("dataset") != "allrecipes" or not r["recipe_id"].startswith("allrecipes:r:") for r in recipes):
            raise ValueError("fit requires nonempty Allrecipes training recipes only")
        ids = [r["recipe_id"] for r in recipes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate training recipe IDs")
        # Document frequency: count each ingredient once per recipe.
        dfs = Counter()
        for r in recipes:
            dfs.update(set(r["ingredients"]))
        # Sorted vocabulary gives a fixed, reproducible column order.
        words = sorted(w for w, count in dfs.items() if count >= self.min_df)
        if not words:
            raise ValueError("No ingredient features survive min_df")
        self.vocabulary = {w: i for i, w in enumerate(words)}
        # Smoothed IDF; the +1 keeps every weight strictly positive.
        self.idf = np.array([np.log((1 + len(recipes)) / (1 + dfs[w])) + 1 for w in words], dtype=np.float32)
        self.metadata = {"training_dataset": "allrecipes", "training_recipe_count": len(recipes),
                         "training_recipe_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
                         "alias_sha256": alias_sha256, "n_features": len(words), "min_df": self.min_df,
                         "tf": "binary ingredient occurrence", "idf": "ln((1+n_train_docs)/(1+train_df))+1",
                         "norm": "l2", "foodcom_fit_performed": False}
        return self

    def transform(self, ingredient_lists):
        """Turn ingredient lists into L2-normalized binary TF-IDF rows.

        Returns (csr_matrix of shape [n_recipes, n_features], diagnostics dict).
        The diagnostics report how many ingredient occurrences were out of
        vocabulary (OOV) and how many recipes ended up as all-zero vectors.
        """
        if self.idf is None:
            raise ValueError("Model is not fitted")
        # Build the CSR arrays directly: indptr[k]..indptr[k+1] are recipe k's columns.
        indices, indptr, values = [], [0], []
        total_tokens = oov_tokens = zero_docs = 0
        for ingredients in ingredient_lists:
            columns = []
            for phrase in sorted(set(ingredients)):  # binary TF: duplicates count once
                total_tokens += 1
                col = self.vocabulary.get(phrase)
                if col is None:
                    oov_tokens += 1                  # unknown ingredient -> ignored
                else:
                    columns.append(col)
            columns.sort()
            if not columns:
                zero_docs += 1
            indices.extend(columns)
            values.extend(self.idf[columns])         # TF = 1, so the value is the IDF
            indptr.append(len(indices))
        x = sparse.csr_matrix((np.asarray(values, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
                              shape=(len(indptr) - 1, len(self.vocabulary)), dtype=np.float32)
        return l2_normalize(x), {"documents": len(indptr) - 1, "ingredient_occurrences": total_tokens,
                                 "oov_occurrences": oov_tokens,
                                 "oov_rate": oov_tokens / total_tokens if total_tokens else 0.0,
                                 "zero_vector_documents": zero_docs}


class CandidateIndex:
    """Candidate recipe matrix built once and reused for every user.

    Users are scored in small batches (profile batch x candidates^T) instead of
    materializing an item-item similarity matrix, which would not fit in memory
    for 226,201 Food.com candidates.
    """

    def __init__(self, model, recipes):
        # Rows are sorted by recipe_id, so a lower row index = earlier in ID order.
        # This is what makes the tie-break below deterministic.
        recipes = sorted(recipes, key=lambda r: r["recipe_id"])
        self.ids = [r["recipe_id"] for r in recipes]
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Duplicate candidate IDs")
        self.row_of = {rid: i for i, rid in enumerate(self.ids)}
        self.x, self.diagnostics = model.transform([r["ingredients"] for r in recipes])
        self.xt = self.x.T.tocsr()  # transposed once for fast profile @ X^T

    def profile(self, h_recipe_ids):
        """User profile = L2-normalized equal-weight mean of the H recipe vectors.

        h_recipe_ids : the user's distinct H recipe IDs (5 in this project).
        The 1/len(H) weight does not change the direction; it is kept so the
        intermediate vector is a true mean before normalization.
        """
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
        """Return the top-k (recipe_id, cosine) list for each profile row.

        profiles      : csr_matrix [batch, n_features] of user profiles.
        seen_per_user : list of sets; these recipes are never recommended.
        k             : list length (10).
        """
        if k < 1 or profiles.shape[0] != len(seen_per_user):
            raise ValueError("Invalid k or mismatched user batch")
        score_matrix = profiles.dot(self.xt).toarray()  # dense [batch, n_candidates] cosine scores
        outputs = []
        for scores, seen in zip(score_matrix, seen_per_user):
            seen_rows = [self.row_of[r] for r in seen if r in self.row_of]
            scores[seen_rows] = -np.inf                 # exclude already-rated recipes
            eligible = np.flatnonzero(np.isfinite(scores))
            n = min(k, len(eligible))
            if n == 0:
                outputs.append([])
                continue
            # Deterministic tie-break, including all-zero profiles:
            # 1) np.partition finds the k-th largest score (the boundary) in O(n);
            # 2) keep every score strictly above it, then fill with boundary ties
            #    in row order (= recipe_id order);
            # 3) sort the k picks by score desc, then row index asc (np.lexsort uses
            #    the LAST key as the primary key).
            boundary = np.partition(scores[eligible], len(eligible) - n)[len(eligible) - n]
            better = eligible[scores[eligible] > boundary]
            ties = eligible[scores[eligible] == boundary]
            selected = np.concatenate([better, ties[:n - len(better)]])
            selected = selected[np.lexsort((selected, -scores[selected]))]
            outputs.append([(self.ids[int(i)], float(scores[i])) for i in selected])
        return outputs


def load_frozen(path=ROOT / 'cbf_model.json'):
    """Load a saved CBF model (vocabulary + IDF) and validate it.

    The JSON format avoids pickle, so loading a model never executes code.
    """
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
    """Return the fixed evaluation cohort of one split as plain Python objects.

    db_path : allrecipes.db (split 'valid' or 'test') or foodcom.db (split 'test').
    Returned keys:
      users         sorted list of user IDs that MUST all be evaluated
      H             {user: [5 profile recipe IDs, oldest -> newest]}
      L             {user: set of future positive recipe IDs (ground truth)}
      seen          {user: set of recipes rated before the cutoff (excluded)}
      recipes       candidate recipes with normalized ingredient lists
      candidate_ids set of all candidate recipe IDs
    LightFM uses the same object and calls evaluate_rankings with its ranking.
    Ingredients are already normalized in the database; no new normalization is needed.
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
    # Consistency checks: same users in users/H/L, exactly 5 distinct H, H inside seen, L outside seen.
    if not users or set(users) != set(h) or set(users) != set(targets):
        raise ValueError('Empty or inconsistent cohort')
    for uid in users:
        if len(h[uid]) != 5 or len(set(h[uid])) != 5 or not set(h[uid]).issubset(seen[uid]) or targets[uid] & seen[uid]:
            raise ValueError('Invalid profile/seen/target data')
    return {'dataset': metadata['dataset'], 'split': split, 'users': users, 'recipes': recipes,
            'candidate_ids': {r['recipe_id'] for r in recipes}, 'H': dict(h), 'L': dict(targets), 'seen': dict(seen)}


def evaluate_rankings(evaluation, rankings, k=10):
    """Evaluate any model's {user_id: [ranked recipe IDs]} on the fixed cohort.

    Missing users, duplicates, seen items and out-of-catalog items are errors, so
    every model is scored on exactly the same users and candidates.
    Per user (binary relevance, rel_r = 1 if the rank-r item is in L):
      Precision@K = hits / K               (denominator is always K)
      Recall@K    = hits / |L|
      NDCG@K      = DCG / IDCG, DCG = sum_r rel_r / log2(r + 1), IDCG = best possible DCG
    The summary is the macro mean: every user has the same weight.
    Returns (summary dict, list of per-user rows).
    """
    if k < 1 or set(rankings) != set(evaluation['users']):
        raise ValueError('Positive K and rankings for every fixed user are required')
    rows = []
    for uid in evaluation['users']:
        ranked = list(rankings[uid])[:k]
        if len(set(ranked)) != len(ranked) or not set(ranked).issubset(evaluation['candidate_ids']) or set(ranked) & evaluation['seen'][uid]:
            raise ValueError(f'Invalid recommendation list for {uid}')
        # A list may be shorter than K only if fewer than K unseen candidates exist.
        eligible_count = len(evaluation['candidate_ids']) - sum(rid in evaluation['candidate_ids'] for rid in evaluation['seen'][uid])
        expected_length = min(k, eligible_count)
        if len(ranked) != expected_length:
            raise ValueError(f'Incomplete recommendation list for {uid}')
        target = evaluation['L'][uid]
        hits = sum(rid in target for rid in ranked)
        # enumerate() gives rank0 = 0 for rank 1, so the discount is log2(rank0 + 2).
        dcg = sum(1 / np.log2(rank + 2) for rank, rid in enumerate(ranked) if rid in target)
        idcg = sum(1 / np.log2(rank + 2) for rank in range(min(k, len(target))))
        rows.append({'user_id': uid, 'hits': hits, 'n_targets': len(target), 'n_recommended': len(ranked),
                     f'precision@{k}': hits / k, f'recall@{k}': hits / len(target), f'ndcg@{k}': float(dcg / idcg)})
    summary = {'dataset': evaluation['dataset'], 'split': evaluation['split'], 'users': len(rows),
               'candidates': len(evaluation['candidate_ids']), 'k': k,
               'macro_metrics': {m: float(np.mean([row[m] for row in rows])) for m in [f'precision@{k}', f'recall@{k}', f'ndcg@{k}']}}
    return summary, rows


def save_json(path, value):
    """Write JSON atomically: write a temp file, then rename it over the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def run_evaluate(db_path, split, model_path, output, batch_size=16):
    """Recommend top-10 for every cohort user and write the results.

    Outputs in ``output``: recommendations.csv (user, rank, recipe, cosine, is_target),
    per_user_metrics.csv and metrics.json (macro metrics + ingredient diagnostics).
    batch_size users are scored at once; 16 keeps the dense score block small.
    """
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


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_train(db_path, output, min_df=1, aliases=None):
    """Fit the CBF vocabulary/IDF on recipes that appear in Allrecipes TRAIN ratings.

    aliases : optional path to ingredient_aliases.json; only its hash is stored.
              Falls back to preprocessing_handoff/config, then to the hash in an
              existing cbf_model.json, so training no longer requires that file.
    """
    with sqlite3.connect(db_path) as db:
        dataset = json.loads(db.execute("SELECT value FROM metadata WHERE key='dataset'").fetchone()[0])
        if dataset != 'allrecipes':
            raise ValueError('Only Allrecipes train may fit CBF')
        rows = db.execute("SELECT recipe_id,ingredients FROM recipes WHERE recipe_id IN (SELECT recipe_id FROM interactions WHERE partition='train') ORDER BY recipe_id")
        recipes = [{'dataset': 'allrecipes', 'recipe_id': rid, 'ingredients': json.loads(ingredients)} for rid, ingredients in rows]
    if aliases is not None:
        alias_sha256 = file_sha256(aliases)
    elif DEFAULT_ALIASES.exists():
        alias_sha256 = file_sha256(DEFAULT_ALIASES)
    elif (ROOT / 'cbf_model.json').exists():
        alias_sha256 = load_frozen().metadata['alias_sha256']
    else:
        alias_sha256 = None
    model = IngredientCBF(min_df).fit(recipes, alias_sha256)
    save_json(output, {'metadata': model.metadata, 'vocabulary': model.vocabulary, 'idf': model.idf.astype(float).tolist()})
    print(f'Saved {len(model.vocabulary)} ingredient features to {output}')


def main():
    parser = argparse.ArgumentParser(description='Minimal CBF and common ranking evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train', help='fit vocabulary and IDF on Allrecipes train')
    train.add_argument('--db', default=str(ROOT / 'allrecipes.db'))
    train.add_argument('--min-df', type=int, default=1, help='minimum training recipes per ingredient feature')
    train.add_argument('--aliases', default=None, help='ingredient_aliases.json used in preprocessing (hash is recorded)')
    train.add_argument('--output', default='new_cbf_model.json')
    evaluate = sub.add_parser('evaluate', help='recommend and score one evaluation split')
    evaluate.add_argument('--db', required=True)
    evaluate.add_argument('--split', choices=['valid', 'test'], default='test')
    evaluate.add_argument('--model', default=str(ROOT / 'cbf_model.json'))
    evaluate.add_argument('--output', default='cbf_results')
    evaluate.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.command == 'train':
        run_train(args.db, args.output, args.min_df, args.aliases)
    else:
        run_evaluate(args.db, args.split, args.model, args.output, args.batch_size)


if __name__ == '__main__':
    main()
