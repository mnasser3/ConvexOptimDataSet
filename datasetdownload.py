from datasets import load_dataset

dataset = load_dataset("glue", "sst2")
dataset["train"].to_csv("sst2_train.csv", index=False)
dataset["validation"].to_csv("sst2_validation.csv", index=False)
dataset["test"].to_csv("sst2_test.csv", index=False)
