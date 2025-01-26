import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import os

# Define the file path
file_path = r"E:\Thesis\Max Temp\11111\2d_ssa_reconstructed_11111_max_temp.xlsx"
output_dir = r"E:\Thesis\Max Temp\11111\splits"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
data = pd.read_excel(file_path)

# Initialize the TimeSeriesSplit for expanding window cross-validation
ts_split = TimeSeriesSplit(n_splits=5)

# Perform the split and save the train/test sets
fold = 1
for train_index, test_index in ts_split.split(data):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    # Save the train and test datasets for each fold
    train_file = os.path.join(output_dir, f"train_fold_{fold}.xlsx")
    test_file = os.path.join(output_dir, f"test_fold_{fold}.xlsx")

    train_data.to_excel(train_file, index=False)
    test_data.to_excel(test_file, index=False)

    print(f"Fold {fold}: Training data saved to {train_file}, Testing data saved to {test_file}")
    fold += 1
