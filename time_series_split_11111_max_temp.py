import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Load dataset
file_path = r"E:\Thesis\2d_ssa_reconstructed_11111_max_temp.xlsx"
data = pd.read_excel(file_path)

# Separate metadata and weather data
metadata_columns = ['Station_Index', 'Year', 'Month']
metadata = data[metadata_columns]
data_values = data.drop(columns=metadata_columns)

# Time-series split (80:20 split using TimeSeriesSplit)
tscv = TimeSeriesSplit(n_splits=5)  # Specify the number of splits
split_indices = list(tscv.split(data_values))[-1]  # Get the indices for the last fold (80:20 split)

train_indices, test_indices = split_indices
train_metadata = metadata.iloc[train_indices]
test_metadata = metadata.iloc[test_indices]

train_data = data_values.iloc[train_indices]
test_data = data_values.iloc[test_indices]

# Combine metadata and data
train_output = pd.concat([train_metadata.reset_index(drop=True), train_data.reset_index(drop=True)], axis=1)
test_output = pd.concat([test_metadata.reset_index(drop=True), test_data.reset_index(drop=True)], axis=1)

# Save results to Excel
train_output_path = "train_data_11111_timeseries.xlsx"
test_output_path = "test_data_11111_timeseries.xlsx"

train_output.to_excel(train_output_path, index=False)
test_output.to_excel(test_output_path, index=False)

print(f"Files saved:\n- Training data: {train_output_path}\n- Testing data: {test_output_path}")
