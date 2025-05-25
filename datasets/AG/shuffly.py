import numpy as np
import pandas as pd

def save_labels(csv_path, npy_path):
    # Load the shuffled CSV
    df = pd.read_csv(csv_path)
    # Extract labels as a NumPy array
    labels = df['label'].to_numpy()
    # Save to .npy
    np.save(npy_path, labels)
    print(f"Saved {npy_path} with shape {labels.shape}")

# Train labels
save_labels(
    "/Users/mn/Desktop/EE364B/Project/ConvexOptimDataSet/datasets/AG/agnews_train.csv",
    "/Users/mn/Desktop/EE364B/Project/ConvexOptimDataSet/datasets/AG/Y_train.npy"
)

# Validation labels
save_labels(
    "/Users/mn/Desktop/EE364B/Project/ConvexOptimDataSet/datasets/AG/agnews_val.csv",
    "/Users/mn/Desktop/EE364B/Project/ConvexOptimDataSet/datasets/AG/Y_val.npy"
)
