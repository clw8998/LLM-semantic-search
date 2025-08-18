import numpy as np
import pandas as pd
from sentence_transformers import util, SentenceTransformer
import torch
from utils.ir_evaluation import format_test_collection, IREvaluator
import config

def evaluate(test_query_path, product_collection_path, qrels_path):
    # To access the model, please visit https://huggingface.co/clw8998/ABRSS and submit a request for access
    model = SentenceTransformer(
        model_name_or_path="clw8998/ABRSS",
        trust_remote_code=True,
        device=config.device,
        truncate_dim=768,
    )

    model.eval()

    queries, product_collection, qrels_df, qrels_binary_strict = format_test_collection(
        test_query_path=test_query_path,
        product_collection_path=product_collection_path,
        qrels_path=qrels_path
    )

    batch_size = config.eval_batch_size
    print("evaluation batch_size:", batch_size)

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
        batch_size=batch_size,
        score_functions={'cos_sim': util.cos_sim},
        main_score_function='cos_sim',
    )

   
    test_query_df = pd.read_csv(test_query_path, index_col=None)
    query_sentences = test_query_df['query'].to_list()
    
    query_embeddings = model.encode(
        sentences=query_sentences,
        batch_size=batch_size,
        normalize_embeddings=True,
        prompt_name="query" if model.prompts and 'query' in model.prompts else None,
    )

    product_collection_sm_df = pd.read_parquet(product_collection_path)
    name_sentences = product_collection_sm_df['name'].to_list()
    
    product_embeddings = model.encode(
        sentences=name_sentences,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        prompt_name="passage" if model.prompts and 'passage' in model.prompts else None,
    )

    scores = np.dot(query_embeddings, product_embeddings.T)
    top_50_indices = np.argsort(-scores)[:,:50]
    query_result_lists = []
    for i in range(len(top_50_indices)):
        query_result_list = product_collection_sm_df[product_collection_sm_df.index.isin(top_50_indices[i])].reindex(top_50_indices[i])
        query_result_list['score'] = scores[i][top_50_indices[i]]
        query_result_lists.append([{'corpus_id': r['item_id'], 'score': r['score']} for r in query_result_list.iloc])

    metrics = evaluator.compute_metrics(query_result_lists)
    print(pd.DataFrame(metrics).T)

    del query_embeddings, product_embeddings, scores, top_50_indices, query_result_lists, metrics
    del evaluator

    model.to('cpu')
    del model

    torch.cuda.empty_cache()
    import gc; gc.collect()

if __name__ == '__main__':
    test_collection_name = "round1"
    test_query_path=config.round1_test_query_path
    product_collection_path=config.round1_product_collection_sm_path
    qrels_path=config.round1_qrels_path
    
    evaluate(test_query_path, product_collection_path, qrels_path)