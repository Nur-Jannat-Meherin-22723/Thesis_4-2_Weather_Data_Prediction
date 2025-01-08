import pandas as pd
import numpy as np
from sklearn.utils.extmath import randomized_svd

# Load the dataset
file_path = r"D:\4-1\4-2\Thesis\code\DataPreprocessing\DataPreprocessing\outlier_processed_max_temp_station_11111.xlsx"
data = pd.read_excel(file_path)

# Extract metadata columns (Station_Index, Year, Month)
metadata = data.iloc[:, :3]  # First three columns assumed to be metadata

# Extract daily columns (Day_1, Day_2, ..., Day_31)
daily_data = data.iloc[:, 3:]  # Columns from Day_1 onwards

# Flatten daily data into a 1D array while preserving NaNs for empty cells
time_series = daily_data.values.flatten()

# Handle NaNs (for SSA purposes only, leave them NaN in the final output)
nan_mask = np.isnan(time_series)
mean_value = np.nanmean(time_series)  # Calculate mean excluding NaNs
time_series_filled = np.where(nan_mask, mean_value, time_series)

# SSA Functions
def create_trajectory_matrix(series, window_length):
    n = len(series)
    k = n - window_length + 1
    trajectory_matrix = np.column_stack([series[i:i + window_length] for i in range(k)])
    return trajectory_matrix

def perform_ssa(series, window_length, components_to_reconstruct):
    trajectory_matrix = create_trajectory_matrix(series, window_length)
    U, Sigma, VT = randomized_svd(trajectory_matrix, n_components=min(window_length, trajectory_matrix.shape[1]))
    reconstructed = np.zeros_like(trajectory_matrix)
    for comp in components_to_reconstruct:
        reconstructed += Sigma[comp] * np.outer(U[:, comp], VT[comp, :])
    n, m = reconstructed.shape
    reconstructed_series = np.zeros(n + m - 1)
    for i in range(n + m - 1):
        values = []
        for j in range(max(0, i - m + 1), min(n, i + 1)):
            values.append(reconstructed[j, i - j])
        reconstructed_series[i] = np.mean(values)
    return reconstructed_series

# SSA Parameters
window_length = 30  # Choose a window length (e.g., one month of data)
components_to_reconstruct = [0, 1]  # Use the first two components for trend and seasonality

# Apply SSA
reconstructed_series = perform_ssa(time_series_filled, window_length, components_to_reconstruct)

# Retain NaNs in their original positions
reconstructed_series[nan_mask] = np.nan

# Reshape the reconstructed series back to the original 2D format
reconstructed_data = reconstructed_series.reshape(daily_data.shape)

# Create the final DataFrame with metadata and reconstructed data
output_df = pd.concat([metadata, pd.DataFrame(reconstructed_data, columns=daily_data.columns)], axis=1)

# Save the output to a new Excel file
output_file_path = r"D:\4-1\4-2\Thesis\code\DataPreprocessing\DataPreprocessing\2d_ssa_reconstructed_11111_max_temp.xlsx"
output_df.to_excel(output_file_path, index=False)

print(f"SSA reconstructed data saved to {output_file_path}")
