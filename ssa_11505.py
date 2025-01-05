import pandas as pd
import numpy as np
from sklearn.utils.extmath import randomized_svd

# Load the dataset
file_path = r"D:\4-1\4-2\Thesis\code\DataPreprocessing\DataPreprocessing\outlier_processed_max_temp_station_11505.xlsx"
data = pd.read_excel(file_path)

# Reshape the data into a 1D time series (e.g., maximum temperatures)
# Assuming the dataset has columns like 'Day_1', 'Day_2', ..., 'Day_31'
time_series = data.iloc[:, 3:].values.flatten()  # Flattening daily data into 1D array

# Handle NaN and infinite values
time_series = np.nan_to_num(time_series, nan=np.nanmean(time_series), posinf=np.nanmax(time_series), neginf=np.nanmin(time_series))

# Function to create trajectory matrix
def create_trajectory_matrix(series, window_length):
    n = len(series)
    k = n - window_length + 1
    trajectory_matrix = np.column_stack([series[i:i + window_length] for i in range(k)])
    return trajectory_matrix

# Perform SSA
def perform_ssa(series, window_length, components_to_reconstruct):
    # Create the trajectory matrix
    trajectory_matrix = create_trajectory_matrix(series, window_length)
    
    # Perform Singular Value Decomposition (SVD)
    U, Sigma, VT = randomized_svd(trajectory_matrix, n_components=min(window_length, trajectory_matrix.shape[1]))
    
    # Reconstruct the series using selected components
    reconstructed = np.zeros_like(trajectory_matrix)
    for comp in components_to_reconstruct:
        reconstructed += Sigma[comp] * np.outer(U[:, comp], VT[comp, :])
    
    # Average the anti-diagonals to obtain the final reconstructed series
    n, m = reconstructed.shape
    reconstructed_series = np.zeros(n + m - 1)
    for i in range(n + m - 1):
        values = []
        for j in range(max(0, i - m + 1), min(n, i + 1)):
            values.append(reconstructed[j, i - j])
        reconstructed_series[i] = np.mean(values)
    
    return reconstructed_series

# Parameters for SSA
window_length = 30  # Choose a window length (e.g., one month of data)
components_to_reconstruct = [0, 1]  # Use the first two components for trend and seasonality

# Apply SSA to the time series
reconstructed_series = perform_ssa(time_series, window_length, components_to_reconstruct)

# Save the reconstructed series back to a DataFrame
output_df = pd.DataFrame({
    'Original': time_series,
    'Reconstructed': reconstructed_series[:len(time_series)]  # Match length to original series
})

# Save the output to a new Excel file
output_file_path = r"D:\4-1\4-2\Thesis\code\DataPreprocessing\DataPreprocessing\ssa_11505_reconstructed_max_temp.xlsx"
output_df.to_excel(output_file_path, index=False)

print(f"SSA processed data saved to {output_file_path}")
