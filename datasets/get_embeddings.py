import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification

dataset = load_dataset('csv', data_files={
    'train': 'sst2_train.csv',
    'validation': 'sst2_validation.csv'
})

model_name = "prajjwal1/bert-tiny"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

def compute_embeddings(split, batch_size=16):
    loader = DataLoader(split, batch_size=batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            inputs = tokenizer(batch['sentence'], padding=True, truncation=True, return_tensors='pt')
            outputs = model.bert(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    return np.vstack(embeddings)

X_train = compute_embeddings(dataset['train'])
X_val = compute_embeddings(dataset['validation'])

np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)

print(f"Saved X_train.npy with shape {X_train.shape}")
print(f"Saved X_val.npy with shape {X_val.shape}")
