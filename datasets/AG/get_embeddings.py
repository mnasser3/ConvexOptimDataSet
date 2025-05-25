import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm.auto import tqdm
import pandas as pd

# 1. Load AG News
ds = load_dataset("ag_news")
train = ds["train"]
test  = ds["test"]

# 2. Balanced downsample to 60k train (15k per class)
n_per_class_train = 60000 // 4
train_dfs = []
for label in range(4):
    sub = train.filter(lambda ex, lbl=label: ex["label"] == lbl)
    train_dfs.append(sub.shuffle(seed=42).select(range(n_per_class_train)))
train_bal = Dataset.from_dict(pd.concat([pd.DataFrame(t) for t in train_dfs], ignore_index=True).to_dict(orient="list"))

# 3. Balanced downsample to ~4k val (1000 per class) from test split
n_per_class_val = 4000 // 4
val_dfs = []
for label in range(4):
    sub = test.filter(lambda ex, lbl=label: ex["label"] == lbl)
    val_dfs.append(sub.shuffle(seed=42).select(range(n_per_class_val)))
val_bal = Dataset.from_dict(pd.concat([pd.DataFrame(v) for v in val_dfs], ignore_index=True).to_dict(orient="list"))

# 4. Save to CSV & reload via `csv` loader
train_bal.to_pandas().to_csv("agnews_train_60k.csv", index=False)
val_bal.to_pandas().to_csv("agnews_val_4k.csv",   index=False)

# reload so that your pipeline is identical
csv_ds = load_dataset(
    "csv",
    data_files={"train": "agnews_train_60k.csv", "validation": "agnews_val_4k.csv"},
)

# 5. Compute embeddings with tqdm
model_name = "prajjwal1/bert-tiny"
tokenizer  = BertTokenizer.from_pretrained(model_name)
model      = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

def compute_embeddings(split, batch_size=32):
    loader = DataLoader(split, batch_size=batch_size, shuffle=False)
    all_emb = []
    for batch in tqdm(loader, desc="Batches", leave=False):
        inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model.bert(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_emb.append(cls_emb)
    return np.vstack(all_emb)

X_train = compute_embeddings(csv_ds["train"])
X_val   = compute_embeddings(csv_ds["validation"])

# Save them
np.save("X_train.npy", X_train)
np.save("X_val.npy",   X_val)

print(f"X_train.npy: {X_train.shape}")
print(f"X_val.npy:   {X_val.shape}")
