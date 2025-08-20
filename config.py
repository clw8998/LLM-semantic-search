import random
import numpy as np
import torch
SEED = 2022

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# VALID/TEST SET
round1_test_query_path = './data/round1_test_query_250.csv'
round1_product_collection_sm_path ='./data/round1_product_collection_sm.parquet'
round1_qrels_path = './data/round1_qrels.parquet'

valid_query_path = round1_test_query_path
valid_product_collection_path = round1_product_collection_sm_path
valid_qrels_path = round1_qrels_path

# test dataset
test_query_path = round1_test_query_path
test_product_collection_path = round1_product_collection_sm_path
test_qrels_path = round1_qrels_path

train_data_raw_path = "./data/train_df.parquet"
distillation_train_data_path = "./datasets/Qwen3_embedding_8B_distillation_dataset"

llm_prompts = {
    "query": (
        "Instruct: Given an e-commerce search query, your goal is to maximize all retrieval metrics. "
        "Retrieve the most relevant product titles that match the query\n"
        "Query:"
    ),
    "passage": "Product Title:"
}

def add_llm_prompts(example):
    example["anchor"] = llm_prompts["query"] + example["anchor"]
    example["positive"] = llm_prompts["passage"] + example["positive"]
    example["negative"] = [llm_prompts["passage"] + neg for neg in example["negative"]]
    return example

use_llm_prompt = True

loss = 'MSELoss'
epochs = 50
batch_size = 32
eval_batch_size = 32
gradient_accumulation_steps = 1
learning_rate = 2e-5
lr_scheduler_type = "linear"

# distillation config
teacher_model_name = "Qwen/Qwen3-Embedding-8B"
student_model_name = "ckiplab/bert-base-chinese"
use_non_pretrained_model = False

layers_to_keep = [0, 5, 11]
# layers_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# CONFIG
experiments_folder = './experiments/'
exp_name = f"{'_'.join(student_model_name.split('/')[-2:])}_L{len(layers_to_keep)}_distill_from_QWen3_sm_anc_pos_2neg_input_no_prompt_{loss}"

