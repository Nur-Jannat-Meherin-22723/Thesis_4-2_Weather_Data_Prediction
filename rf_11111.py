import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# File paths
file_path = r"E:\Thesis\2d_ssa_reconstructed_11111_max_temp.xlsx"
output_file = r"E:\Thesis\time_rf_11111_max_temp_prediction_output.xlsx"

# Load the dataset
data = pd.read_excel(file_path)

# Extract metadata and daily data
metadata = data.iloc[:, :3]  # Metadata columns (Station_Index, Year, Month)
daily_data = data.iloc[:, 3:]  # Daily columns (Day_1, Day_2, ..., Day_31)

# Flatten the daily data into a 1D time series for model training
time_series = daily_data.values.flatten()

# Track null (empty) cells to preserve them in the output
nan_mask = np.isnan(time_series)

# Handle NaN values for training purposes (replace with mean)
mean_value = np.nanmean(time_series)
time_series_filled = np.where(nan_mask, mean_value, time_series)

# Function to create lagged features
def create_lagged_features(series, lag):
    X, y = [], []
    for i in range(lag, len(series)):
        X.append(series[i - lag:i])  # Lagged features
        y.append(series[i])          # Target value
    return np.array(X), np.array(y)

# Parameters
lag = 5  # Number of lagged steps to use as features

# Create lagged features
X, y = create_lagged_features(time_series_filled, lag)

# Initialize the TimeSeriesSplit with 5 splits
tscv = TimeSeriesSplit(n_splits=5)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Prepare to store predictions for comparison
predicted_data = np.full(time_series.shape, np.nan)  # Start with all NaNs

# Perform time series split for training and testing
for train_index, test_index in tscv.split(X):
    # Get the training and testing data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the Random Forest model
    rf_model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = rf_model.predict(X_test)

    # Insert predictions into the full predicted data array
    predicted_data[test_index + lag] = y_pred  # Add lag offset to predictions

# Reshape the predicted values back to the 2D format to match the original data
predicted_data_2d = predicted_data.reshape(daily_data.shape)

# Extract the last part of the data (test data for comparison)
test_data_2d = daily_data.iloc[int(0.8 * daily_data.shape[0]):, :]

# Create the output DataFrame with test data and predicted columns
output_df = metadata.iloc[int(0.8 * metadata.shape[0]):, :].reset_index(drop=True)

for day in range(daily_data.shape[1]):  # Loop over daily columns
    day_column = daily_data.columns[day]
    predicted_column = f"Predicted_{day + 1}"
    
    # Keep original nulls in the predicted data
    predicted_col_data = np.where(np.isnan(test_data_2d.iloc[:, day]), np.nan, predicted_data_2d[int(0.8 * daily_data.shape[0]):, day])
    
    output_df[day_column] = test_data_2d.iloc[:, day].values
    output_df[predicted_column] = predicted_col_data

# Save the output to an Excel file
output_df.to_excel(output_file, index=False)

# Print completion message
print(f"Comparison output saved to: {output_file}")
