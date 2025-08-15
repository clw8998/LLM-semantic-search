import os
import logging
from typing import List, Tuple, Dict, Set, Callable
from tqdm import tqdm, trange

import torch
from torch import Tensor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import cos_sim, dot_score

logger = logging.getLogger(__name__)

# -------------------------------------------------------
#   Format test collection
# -------------------------------------------------------
def format_test_collection(test_query_path, product_collection_path, qrels_path, selected_query_ids=None):
    print('\n# -------------------------------------------------------')
    print('#    Format test collection')
    print('# -------------------------------------------------------')
    # -------------------------------------------------------
    #   1. Queries
    # -------------------------------------------------------
    test_query_df = pd.read_csv(test_query_path, index_col=None)
    test_query_df = test_query_df[['query_id', 'query']]
    queries = {r['query_id']: r['query'] for r in test_query_df.iloc}

    if selected_query_ids is not None:
        queries = {qid: queries[qid] for qid in selected_query_ids if qid in queries}
    
    print('Queries:', len(queries))

    # -------------------------------------------------------
    #   2. Product collection
    # -------------------------------------------------------
    product_collection_df = pd.read_parquet(product_collection_path)
    collection = {r['item_id']: r['name'] for r in product_collection_df.iloc}
    print('Product collection:', len(collection))

    # -------------------------------------------------------
    #   3. Qrels
    # -------------------------------------------------------
    # relevant docs: qrels, default: 3-levels
    qrels_df = pd.read_parquet(qrels_path)
    print('Qrels:', len(qrels_df))
    
    # convert to binary
    qrels_binary_df = qrels_df.copy()
    qrels_binary_df['relevance'] = qrels_binary_df['relevance'].map(lambda x: 1.0 if x == 2.0 else 0.0)
    qrels_binary = {q_i: set(group[group['relevance'] == 1.0]['item_id'].to_list()) for q_i, group in qrels_binary_df.groupby('query_id')}

    if selected_query_ids is not None:
        qrels_binary = {qid: qrels_binary[qid] for qid in selected_query_ids if qid in qrels_binary}
        qrels_df = qrels_df[qrels_df['query_id'].isin(selected_query_ids)]

    return queries, collection, qrels_df, qrels_binary

# -------------------------------------------------------
#   Information retrieval evaluator
#   ---
#   Note: 由 sentence transformer > sentence evaluator 
#       繼承而來，加入了 3 levels ndcg 指標，並用其作為
#       model 挑選的依據
# -------------------------------------------------------
class IREvaluator(SentenceEvaluator):
    def __init__(self,
        # test collection
        queries: Dict[str, str],  # qid => query
        corpus: Dict[str, str],  # cid => doc
        relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]
        relevant_docs_3lv,

        # chunk size
        corpus_chunk_size: int = 50000,
        
        # metrics
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        ndcg_at_k_3lv: List[int] = [1, 5, 10, 20, 50],
        accuracy_at_k: List[int] = [1, 3, 5, 10],
        precision_recall_at_k: List[int] = [1, 3, 5, 10],
        map_at_k: List[int] = [100],
        
        # options
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = '',
        write_csv: bool = True,
        score_functions: List[Callable[[Tensor, Tensor], Tensor] ] = {'cos_sim': cos_sim, 'dot_score': dot_score},       #Score function, higher=more similar
        main_score_function: str = None
    ):
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.relevant_docs_3lv = relevant_docs_3lv.set_index('item_id')
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.ndcg_at_k_3lv = ndcg_at_k_3lv
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in accuracy_at_k:
                self.csv_headers.append("{}-Accuracy@{}".format(score_name, k))
            for k in precision_recall_at_k:
                self.csv_headers.append("{}-Precision@{}".format(score_name, k))
                self.csv_headers.append("{}-Recall@{}".format(score_name, k))
            for k in mrr_at_k:
                self.csv_headers.append("{}-MRR@{}".format(score_name, k))
            for k in ndcg_at_k:
                self.csv_headers.append("{}-NDCG@{}".format(score_name, k))
            for k in ndcg_at_k_3lv:
                self.csv_headers.append("{}-3-levels-NDCG@{}".format(score_name, k))
            for k in map_at_k:
                self.csv_headers.append("{}-MAP@{}".format(score_name, k))

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"
        logger.info("Information Retrieval Evaluation on " + self.name + " dataset" + out_txt)
        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")
            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")
            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]['accuracy@k'][k])
                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]['precision@k'][k])
                    output_data.append(scores[name]['recall@k'][k])
                for k in self.mrr_at_k:
                    output_data.append(scores[name]['mrr@k'][k])
                for k in self.ndcg_at_k:
                    output_data.append(scores[name]['ndcg@k'][k])
                for k in self.ndcg_at_k_3lv:
                    output_data.append(scores[name]['3-levels-ndcg@k'][k])
                for k in self.map_at_k:
                    output_data.append(scores[name]['map@k'][k])
            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if self.main_score_function is None:
            return max([scores[name]['3-levels-ndcg@k'][max(self.ndcg_at_k_3lv)] for name in self.score_function_names])
        else:
            # ndcg_at_k_3lv_scores = [scores[self.main_score_function]['3-levels-ndcg@k'][k] for k in self.ndcg_at_k_3lv]
            # mean_score = np.array(ndcg_at_k_3lv_scores).mean()
            # return mean_score

            # accuracy_at_k_score = scores[self.main_score_function]['accuracy@k'][max(self.accuracy_at_k)]
            # ndcg_at_k_score = scores[self.main_score_function]['ndcg@k'][max(self.ndcg_at_k)]
            # ndcg_at_k_3lv_score = scores[self.main_score_function]['3-levels-ndcg@k'][max(self.ndcg_at_k_3lv)]
            # precision_at_k_score = scores[self.main_score_function]['precision@k'][max(self.precision_recall_at_k)]
            # recall_at_k_score = scores[self.main_score_function]['recall@k'][max(self.precision_recall_at_k)]
            # mrr_at_k_score = scores[self.main_score_function]['mrr@k'][max(self.mrr_at_k)]
            # map_at_k_score = scores[self.main_score_function]['map@k'][max(self.map_at_k)]
            # # return ndcg_at_k_3lv_score, precision_at_k_score, recall_at_k_score
            # return {
            #     f"{self.name}accuracy@50": accuracy_at_k_score,
            #     f"{self.name}ndcg@50": ndcg_at_k_score,
            #     f"{self.name}3-levels-NDCG@50": ndcg_at_k_3lv_score,
            #     f"{self.name}precision@50": precision_at_k_score,
            #     f"{self.name}recall@50": recall_at_k_score,
            #     f"{self.name}mrr@50": mrr_at_k_score,
            #     f"{self.name}map@50": map_at_k_score,
            # }

            metrics = {
                'accuracy@k':      self.accuracy_at_k,
                'ndcg@k':          self.ndcg_at_k,
                '3-levels-ndcg@k': self.ndcg_at_k_3lv,
                'precision@k':     self.precision_recall_at_k,
                'recall@k':        self.precision_recall_at_k,
                'mrr@k':           self.mrr_at_k,
                'map@k':           self.map_at_k,
            }

            result = {}
            for metric_name, k_list in metrics.items():
                for k in k_list:
                    score_val = scores[self.main_score_function][metric_name][k]
                    out_key = f"{self.name}{metric_name.replace('@k', f'@{k}')}"
                    result[out_key] = score_val

            return result


    def compute_metrices(self, model, corpus_model = None, corpus_embeddings: Tensor = None) -> Dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k), max(self.precision_recall_at_k), max(self.map_at_k))

        # Compute embedding for the queries
        query_embeddings = model.encode(
            self.queries, 
            show_progress_bar=self.show_progress_bar, 
            batch_size=self.batch_size, 
            convert_to_tensor=True,
            prompt_name="query" if model.prompts and 'query' in model.prompts else None,
        )

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        #Iterate over chunks of the corpus
        for corpus_start_idx in trange(0, len(self.corpus), self.corpus_chunk_size, desc='Corpus Chunks', disable=not self.show_progress_bar):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            #Encode chunk of corpus
            if corpus_embeddings is None:
                sub_corpus_embeddings = corpus_model.encode(
                    self.corpus[corpus_start_idx:corpus_end_idx], 
                    show_progress_bar=False, 
                    batch_size=self.batch_size, 
                    convert_to_tensor=True,
                    prompt_name="passage" if model.prompts and 'passage' in model.prompts else None,
                )
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            #Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                #Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False)
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
                        corpus_id = self.corpus_ids[corpus_start_idx+sub_corpus_id]
                        queries_result_list[name][query_itr].append({'corpus_id': corpus_id, 'score': score})

        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        #Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        #Output
        for name in self.score_function_names:
            logger.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])
        return scores

    def compute_metrics(self, queries_result_list: List[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        ndcg_3lv = {k: [] for k in self.ndcg_at_k_3lv}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]
            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1
                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [1 if top_hit['corpus_id'] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * len(query_relevant_docs)
                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)

            # 3 levels NDCG@k
            for k_val in self.ndcg_at_k_3lv:
                ndcg_value, _, _ = self.q_ndcg_at_k_3lv_score(query_id, k_val, top_hits, form='exp')
                ndcg_3lv[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)
                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)
        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])
        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])
        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])
        for k in ndcg_3lv:
            ndcg_3lv[k] = np.mean(ndcg_3lv[k])
        for k in MRR:
            MRR[k] /= len(self.queries)
        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {
            'accuracy@k': num_hits_at_k, 
            'precision@k': precisions_at_k, 
            'recall@k': recall_at_k, 
            'ndcg@k': ndcg, 
            '3-levels-ndcg@k': ndcg_3lv,
            'mrr@k': MRR, 
            'map@k': AveP_at_k
        }

    def output_scores(self, scores):
        for k in scores['accuracy@k']:
            logger.info("Accuracy@{}: {:.2f}%".format(k, scores['accuracy@k'][k]*100))
        for k in scores['precision@k']:
            logger.info("Precision@{}: {:.2f}%".format(k, scores['precision@k'][k]*100))
        for k in scores['recall@k']:
            logger.info("Recall@{}: {:.2f}%".format(k, scores['recall@k'][k]*100))
        for k in scores['mrr@k']:
            logger.info("MRR@{}: {:.4f}".format(k, scores['mrr@k'][k]))
        for k in scores['ndcg@k']:
            logger.info("NDCG@{}: {:.4f}".format(k, scores['ndcg@k'][k]))
        for k in scores['3-levels-ndcg@k']:
            logger.info("3-levels-NDCG@{}: {:.4f}".format(k, scores['3-levels-ndcg@k'][k]))
        for k in scores['map@k']:
            logger.info("MAP@{}: {:.4f}".format(k, scores['map@k'][k]))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  #+2 as we start our idx at 0
        return dcg
    
    # score for 3 levels ndcg of single query at k
    def q_ndcg_at_k_3lv_score(self, query_id, k, query_result_list, form='exp'):
        # filter: query and relevance item
        result_list = pd.Series([d['corpus_id'] for d in query_result_list])

        # idea score ranking
        q_relevant_docs_3lv = self.relevant_docs_3lv[self.relevant_docs_3lv['query_id'] == query_id]
        ideal_score_rank = q_relevant_docs_3lv['relevance'].to_numpy()
        ideal_score_rank = -np.sort(-ideal_score_rank)[:k]
        if len(ideal_score_rank < k): ideal_score_rank = np.pad(ideal_score_rank, (0, k-len(ideal_score_rank)), 'constant', constant_values=0)
        
        # predict score ranking
        q_relevant_docs_3lv = self.relevant_docs_3lv.loc[(self.relevant_docs_3lv['query_id'] == query_id) & (self.relevant_docs_3lv.index.isin(result_list))].reindex(result_list)
        q_relevant_docs_3lv['relevance'] = q_relevant_docs_3lv['relevance'].map(lambda x: 0.0 if np.isnan(x) else x) # replace nan relevance to 0.0
        predict_score_rank = q_relevant_docs_3lv['relevance'].to_numpy()[:k]
        if len(predict_score_rank) < k: predict_score_rank = np.pad(predict_score_rank, (0, k-len(predict_score_rank)), 'constant', constant_values=0)

        # compute ndcg score
        discount = 1 / (np.log2(np.arange(k) + 2))
        if form == 'linear':
            idcg = np.sum(ideal_score_rank * discount)
            dcg = np.sum(predict_score_rank * discount)
        elif form == 'exp':
            idcg = np.sum([2**x - 1 for x in ideal_score_rank] * discount)
            dcg = np.sum([2**x - 1 for x in predict_score_rank] * discount)
        ndcg = dcg / idcg

        return ndcg, dcg, idcg

# -------------------------------------------------------
#   Averaged precision recall curve 
# -------------------------------------------------------
# interpolation (11 points)
def interpolate_11_points(
        queries,
        relevant_docs,
        query_result_lists=[],
        k=50,
    ):
    precision_recall_at_k = [i+1 for i in range(k)]

    # init queries
    queries_ids = []
    for qid in queries:
        if qid in relevant_docs and len(relevant_docs[qid]) > 0:
            queries_ids.append(qid)
    queries = [qid for qid in queries_ids]
    
    # init all queries' precisions
    all_query_interp_precisions = []

    # loop all queries
    for q_i in range(len(query_result_lists)):
        query_id = queries_ids[q_i]

        # Sort scores
        top_hits = sorted(query_result_lists[q_i], key=lambda x: x['score'], reverse=True)
        query_relevant_docs = relevant_docs[query_id]
        
        # Precision and Recall@k
        precisions = []
        recalls = []
        for k_val in precision_recall_at_k:
            num_correct = 0
            for hit in top_hits[0:k_val]:
                if hit['corpus_id'] in query_relevant_docs:
                    num_correct += 1
            precisions.append(num_correct / k_val)
            recalls.append(num_correct / len(query_relevant_docs))

        # 11 points interpolation
        interp_recalls = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
        interp_precisions = []
        r2p = {r: p for r, p in zip(recalls[::-1], precisions[::-1])}
        r2p = dict(sorted([(r,p) for r, p in r2p.items()], key=lambda x: x[0]))
        for r_lv in interp_recalls:
            interp_precisions.append(max([0]+[p for r, p in r2p.items() if r >= r_lv]))
        all_query_interp_precisions.append(interp_precisions)

        # # interpolated line
        # interp_df = pd.DataFrame({
        #     'precision': list(interp_precisions),
        #     'recall': list(interp_recalls),
        # })
        # sns.lineplot(data=interp_df, x='recall', y='precision', estimator=None)
        # sns.scatterplot(data=interp_df, x='recall', y='precision')
        
        # # original graph
        # df = pd.DataFrame({
        #     'precision': list(precisions),
        #     'recall': list(recalls),
        # })
        # sns.lineplot(data=df, x='recall', y='precision', estimator=None)
        # sns.scatterplot(data=df, x='recall', y='precision')
        
    # mean of all queries interp precisions
    mean_interp_precisions = np.array(all_query_interp_precisions).mean(axis=0).tolist()
    return mean_interp_precisions

# plot avg pr curve
def plot_avg_pr_curve(precisions_dict, image_save_path=''):
    # init plot
    fig = plt.figure(figsize=(8,6), dpi=100)
    fig.add_subplot(1,1,1)
    plt.title('Averaged Precision/Recall Curve')
    
    interp_recalls = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

    # loop precision lists
    for name, precisions in precisions_dict.items():
        mean_interp_df = pd.DataFrame({
            'precision': list(precisions),
            'recall': list(interp_recalls),
        })
        sns.lineplot(data=mean_interp_df, x='recall', y='precision', label=name, estimator=None)
        sns.scatterplot(data=mean_interp_df, x='recall', y='precision')

    plt.grid()
    plt.xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    plt.yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    if image_save_path != '': plt.savefig(image_save_path)

# -------------------------------------------------------
#   Plot IDCG and DCG curve
# -------------------------------------------------------
# get idcg and dcg points
def get_idcg_and_dcg_points(query_ids, evaluator, query_result_lists, k=50):
    print('Calculate @1 ~ @50 mean of DCG and mean of IDCG ...')
    queries_dcg = []
    queries_idcg = []
    for i, query_id in tqdm(enumerate(query_ids)): # loop query id
        query_result_list = query_result_lists[i]
        query_dcg = []
        query_idcg = []
        for kk in range(1, k+1): # @1 ~ @50
            _, dcg, idcg = evaluator.q_ndcg_at_k_3lv_score(query_id=query_id, k=kk, query_result_list=query_result_list)
            query_dcg.append(dcg)
            query_idcg.append(idcg)
        queries_dcg.append(query_dcg)
        queries_idcg.append(query_idcg)
    return {
        'idcg': np.array(queries_idcg).mean(axis=0).tolist(),
        'dcg': np.array(queries_dcg).mean(axis=0).tolist()
    }

# plot idcg and dcg curve
def plot_idcg_and_dcg_curve(idcg_and_dcgs_dict, image_save_path=''):
    # init plot
    fig = plt.figure(figsize=(10,3), dpi=100)
    fig.add_subplot(1,1,1)
    plt.title('IDCG and DCG curve')

    # loop dcgs lists
    for name, dcgs in idcg_and_dcgs_dict.items():
        dcgs_df = pd.DataFrame({
            'dcg': dcgs,
            'k': [i for i in range(1, 51)]
        })
        sns.lineplot(data=dcgs_df, x='k', y='dcg', label=name, estimator=None)
        sns.scatterplot(data=dcgs_df, x='k', y='dcg')

    plt.grid()
    # plt.xticks([i for i in range(1, 51)])
    plt.xticks([1, 5, 10, 20, 50])
    if image_save_path != '': plt.savefig(image_save_path)

# -------------------------------------------------------
#   NDCG boxplot
# -------------------------------------------------------
# plot boxplot
def plot_ndcg_boxplot(ndcg_scores, image_save_path=''):
    fig = plt.figure(figsize=(8,8), dpi=100)
    fig.add_subplot(1,1,1)
    plt.title('nDCG@50')
    sns.boxplot(x='run', y='ndcg', data=ndcg_scores, width=0.5)
    if image_save_path != '': plt.savefig(image_save_path)

# -------------------------------------------------------
#   Get ratio of relevant and partially relevant
# -------------------------------------------------------
def get_ratio_of_relevant(qrels_df):
    rel_count = {}
    part_rel_count = {}
    not_rel_count = {}
    for query_id, df in qrels_df.groupby('query_id'):
        rel_count[query_id] = len(df[df['relevance'] == 2])
        part_rel_count[query_id] = len(df[df['relevance'] == 1])
        not_rel_count[query_id] = len(df[df['relevance'] == 0])
    ratio_of_rel = {}
    ratio_of_part_rel = {}
    for i in range(len(not_rel_count.keys())):
        all_count = rel_count[i] + part_rel_count[i] + not_rel_count[i]
        ratio_of_rel[i] = rel_count[i] / all_count
        ratio_of_part_rel[i] = part_rel_count[i] / all_count
    return {'relevant': np.array(list(ratio_of_rel.values())).mean(), 'partially relevant': np.array(list(ratio_of_part_rel.values())).mean()}

# --------------------------------------------------
#   Main
# --------------------------------------------------
if __name__ == '__main__':
    pass