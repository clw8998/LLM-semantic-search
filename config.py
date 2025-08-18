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
round1_test_query_path = './data/test_query_250.csv'
round1_product_collection_sm_path ='./data/round1_product_collection_sm.parquet'
round1_qrels_path = './data/round1_qrels.parquet'

valid_query_path = round1_test_query_path
valid_product_collection_path = round1_product_collection_sm_path
valid_qrels_path = round1_qrels_path

# test dataset
test_query_path = round1_test_query_path
test_product_collection_path = round1_product_collection_sm_path
test_qrels_path = round1_qrels_path
