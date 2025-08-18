import torch
from datasets import load_dataset, Dataset
import numpy as np
import config
from sentence_transformers import SentenceTransformer

config.set_seed()

print('create_distillation_dataset.py ...')

ds = load_dataset(
    "parquet",
    data_files=config.train_data_raw_path,
    split="train",
    keep_in_memory=False
)

ds = ds.rename_column("pos", "positive") \
    .rename_column("neg", "negative")

df = ds.to_pandas()
df_grouped = (
    df
    .groupby(["anchor", "positive"], as_index=False, sort=False)
    .agg({"negative": list})
)

train_dataset = Dataset.from_pandas(df_grouped)

def handle_one_neg(example):
    neg_list = example["negative"]
    return {"negative": neg_list[0] if len(neg_list) == 1 else neg_list}

train_dataset = train_dataset.map(handle_one_neg)

all_rows = []
for example in train_dataset:
    all_rows.append({"sentence": example["anchor"]})
    all_rows.append({"sentence": example["positive"]})
    all_rows.extend([{"sentence": neg} for neg in example["negative"]])


if config.use_llm_prompt:
    train_dataset = train_dataset.map(config.add_llm_prompts)

print("Train dataset size:", len(train_dataset))

prompted_all_rows = []
for example in train_dataset:
    prompted_all_rows.append({"prompted_sentence": example["anchor"]})
    prompted_all_rows.append({"prompted_sentence": example["positive"]})
    prompted_all_rows.extend([{"prompted_sentence": neg} for neg in example["negative"]])

assert len(all_rows) == len(prompted_all_rows)

merged_rows = [
    {
        "sentence": orig["sentence"],
        "prompted_sentence": prom["prompted_sentence"]
    }
    for orig, prom in zip(all_rows, prompted_all_rows)
]

train_dataset = Dataset.from_list(merged_rows[:])

print("Train dataset size after merging:", len(train_dataset))

model_kwargs = {"torch_dtype": torch.bfloat16}

teacher_model = SentenceTransformer(
    model_name_or_path=config.teacher_model_name,
    trust_remote_code=True,
    device=config.device,
    truncate_dim=1024,
    model_kwargs=model_kwargs,
)

teacher_model.eval()

num_samples = len(train_dataset)
all_labels = [] 
batch_size = 64
import math
from tqdm.auto import tqdm
batch_count = math.ceil(num_samples / batch_size)

for i in tqdm(range(0, num_samples, batch_size), total=batch_count, desc="Embedding Progress"):
    batch_sentences = train_dataset[i : i + batch_size]["prompted_sentence"]
    embeddings = teacher_model.encode(
        batch_sentences,
        batch_size=batch_size,
        show_progress_bar=False 
    )

    embeddings = np.array(embeddings, dtype=np.float32)
    all_labels.extend(embeddings)
    del embeddings
    torch.cuda.empty_cache()

assert len(all_labels) == num_samples

train_dataset = train_dataset.add_column("label", all_labels)
train_dataset = train_dataset.remove_columns(["prompted_sentence"])
train_dataset.save_to_disk(config.distillation_train_data_path)

del teacher_model
import gc
gc.collect()
torch.cuda.empty_cache()

