import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.ir_evaluation import format_test_collection, IREvaluator
import config

def evaluate_tfidf(test_query_path, product_collection_path, qrels_path,
                   analyzer='char_wb', ngram_range=(3, 5),
                   lowercase=True, max_features=None, top_k=50):
    queries, product_collection, qrels_df, qrels_binary_strict = format_test_collection(
        test_query_path=test_query_path,
        product_collection_path=product_collection_path,
        qrels_path=qrels_path
    )
    test_query_df = pd.read_csv(test_query_path, index_col=None)
    product_df = pd.read_parquet(product_collection_path)

    query_texts = test_query_df['query'].astype(str).tolist()
    name_texts  = product_df['name'].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        lowercase=lowercase,
        max_features=max_features,
        norm='l2'
    )
    X_corpus = vectorizer.fit_transform(name_texts)
    X_query  = vectorizer.transform(query_texts)

    sim = X_query @ X_corpus.T
    top_k = min(top_k, X_corpus.shape[0])

    query_result_lists = []
    for i in range(sim.shape[0]):
        row = sim.getrow(i).toarray().ravel()
        if top_k < row.size:
            idx = np.argpartition(-row, top_k-1)[:top_k]
            idx = idx[np.argsort(-row[idx])]
        else:
            idx = np.argsort(-row)

        result_df = product_df.iloc[idx].copy()
        result_df['score'] = row[idx]
        query_result_lists.append(
            [{'corpus_id': r['item_id'], 'score': r['score']} for _, r in result_df.iterrows()]
        )

    evaluator = IREvaluator(
        queries=queries,
        corpus=product_collection,
        relevant_docs=qrels_binary_strict,
        relevant_docs_3lv=qrels_df,
        mrr_at_k=[1,5,10,20,50],
        ndcg_at_k=[1,5,10,20,50],
        ndcg_at_k_3lv=[1,5,10,20,50],
        accuracy_at_k=[1,5,10,20,50],
        precision_recall_at_k=[1,5,10,20,50],
        map_at_k=[1,5,10,20,50],
    )
    metrics = evaluator.compute_metrics(query_result_lists)
    print(pd.DataFrame(metrics).T)

if __name__ == '__main__':
    test_collection_name = "round1"
    test_query_path = config.round1_test_query_path
    product_collection_path = config.round1_product_collection_sm_path
    qrels_path = config.round1_qrels_path

    evaluate_tfidf(test_query_path, product_collection_path, qrels_path)
