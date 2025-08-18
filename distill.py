import torch
from datasets import Dataset
import os
from utils.ir_evaluation import format_test_collection, IREvaluator
import config
from sentence_transformers import util, SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, models
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers
from utils.loss.loss_mse import MSELoss

os.environ["WANDB_DISABLED"] = "true" # Set to "false" to enable Weights & Biases logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config.set_seed()
print('distill.py ...')

model_kwargs = None

student_model = SentenceTransformer(
    config.student_model_name,
    trust_remote_code=True,
    device=config.device,
    model_kwargs=model_kwargs,
)

normalize = models.Normalize()
modules = list(student_model._modules.values())
modules.append(normalize)     

student_model = SentenceTransformer(
    modules=modules,
    trust_remote_code=True,
    device=config.device,
    model_kwargs=model_kwargs,
)
    
for name, module in student_model._modules.items():
    print(f"name: {name}, type: {type(module)}")

# bert:
auto_model = student_model._first_module().auto_model
layers_to_keep = config.layers_to_keep
new_layers = torch.nn.ModuleList(
    [layer_module for i, layer_module in enumerate(auto_model.encoder.layer) if i in layers_to_keep]
)
auto_model.encoder.layer = new_layers
auto_model.config.num_hidden_layers = len(layers_to_keep)


model = student_model.to(torch.device(config.device))

if config.use_llm_prompt and (model.prompts == None or model.prompts == {}):
    model.prompts = config.llm_prompts
    print("Using prompts:", model.prompts)

train_dataset = Dataset.load_from_disk(config.distillation_train_data_path)

queries, product_collection, qrels_df, qrels_binary = format_test_collection(
    test_query_path=config.valid_query_path,
    product_collection_path=config.valid_product_collection_path,
    qrels_path=config.valid_qrels_path,
)

evaluator = IREvaluator(
    queries=queries,
    corpus=product_collection,
    relevant_docs=qrels_binary,
    relevant_docs_3lv=qrels_df,
    mrr_at_k=[1,5,10,20,50],
    ndcg_at_k=[1,5,10,20,50],
    ndcg_at_k_3lv=[1,5,10,20,50],
    accuracy_at_k=[1,5,10,20,50],
    precision_recall_at_k=[1,5,10,20,50],
    map_at_k=[1,5,10,20,50],
    batch_size=128,
    score_functions={'cos_sim': util.cos_sim},
    main_score_function='cos_sim',
)

if config.loss == 'MSELoss':
    loss = MSELoss(model, 1)

run_name = config.exp_name

print("student model dimension:", model.get_sentence_embedding_dimension())

args = SentenceTransformerTrainingArguments(
    output_dir=f"experiments/{run_name}",
    num_train_epochs=config.epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    learning_rate=config.learning_rate,
    lr_scheduler_type=config.lr_scheduler_type,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_only_model=True,
    save_total_limit=100,
    logging_steps=100,
    logging_first_step=False,
    report_to="none",
    run_name=run_name,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=evaluator,
)

trainer.train()
