import time
import psutil
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import pandas as pd
from sklearn.metrics import accuracy_score
import os

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

model_name = "prajjwal1/bert-tiny"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

dataset = load_dataset("glue", "sst2")
dataset["train"].to_csv("sst2_train.csv", index=False)
dataset["validation"].to_csv("sst2_validation.csv", index=False)
dataset["test"].to_csv("sst2_test.csv", index=False)

dataset = load_dataset('csv', data_files={
    'train': 'sst2_train.csv',
    'validation': 'sst2_validation.csv'
})

def tokenize_function(example):
    return tokenizer(example['sentence'], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

os.environ["WANDB_DISABLED"] = "true"

subset_sizes = [25, 50, 100]
results = []

for size in subset_sizes:
    train_dataset = tokenized_datasets['train'].shuffle(seed=42)
    n_total = len(train_dataset)
    subset = train_dataset.select(range(int(n_total * size / 100)))

    training_args = TrainingArguments(
        output_dir=f'./results_subset_{size}',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=10,
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=subset,
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 ** 2)
    start_time = time.time()

    train_output = trainer.train()
    eval_output = trainer.evaluate()

    time_elapsed = time.time() - start_time
    mem_after = process.memory_info().rss / (1024 ** 2)
    train_eval = trainer.evaluate(subset)

    results.append({
        'subset_size': size,
        'train_time_sec': round(time_elapsed, 2),
        'memory_before_mb': round(mem_before, 2),
        'memory_after_mb': round(mem_after, 2),
        'train_accuracy': round(train_eval.get('eval_accuracy', 0), 4),
        'eval_accuracy': round(eval_output.get('eval_accuracy', 0), 4)
    })

df_results = pd.DataFrame(results)
df_results.to_csv("subset_selection_results.csv", index=False)
