import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# File paths
file_path = r"E:\Thesis\Relative Humidity\11505\2d_ssa_reconstructed_11505_rh.xlsx"
output_folder = r"E:\Thesis\Relative Humidity\11505\rf_prediction_output_11505_rh"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load the dataset
data = pd.read_excel(file_path)

# Extract metadata and daily data
metadata = data.iloc[:, :3]  # Metadata columns (Station_Index, Year, Month)
daily_data = data.iloc[:, 3:]  # Daily columns (Day_1, Day_2, ..., Day_31)

# Convert daily data to NumPy array
time_series = daily_data.values.flatten()

# Handle NaN values by filling with column mean for training purposes
nan_mask = np.isnan(time_series)
mean_value = np.nanmean(time_series)
time_series_filled = np.where(nan_mask, mean_value, time_series)

# Function to create lagged features
def create_lagged_features(series, lag):
    X, y, indices = [], [], []
    for i in range(lag, len(series)):
        X.append(series[i - lag:i])  # Lagged features
        y.append(series[i])  # Target value
        indices.append(i)  # Keep track of indices
    return np.array(X), np.array(y), np.array(indices)

# Parameters
lag = 5  # Number of lagged steps to use as features
k = 7  # Number of folds for time series cross-validation

# Create lagged features
X, y, indices = create_lagged_features(time_series_filled, lag)

# Initialize time series cross-validation
tscv = TimeSeriesSplit(n_splits=k)

# Lists to store metrics
mse_list, mae_list, rmse_list, r2_list = [], [], [], []
fold = 1

for train_index, test_index in tscv.split(X):
    # Ensure 80:20 split within the fold
    train_size = int(len(train_index) * 0.8)
    train_idx, test_idx = train_index[:train_size], test_index
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Initialize and train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = rf_model.predict(X_test)
    
    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Store metrics
    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)
    r2_list.append(r2)
    
    # 🔹 **Fix: Map test indices back to the correct DataFrame rows**
    test_actual_indices = indices[test_idx]  # Get original indices from time_series
    
    # Find corresponding row indices in the original dataset
    row_indices = np.unique(test_actual_indices // daily_data.shape[1])

    # Select only the test rows in metadata and daily_data
    output_df = metadata.iloc[row_indices].reset_index(drop=True)
    test_actual_values = daily_data.iloc[row_indices].reset_index(drop=True)
    
    # Assign predicted values correctly
    test_predictions = np.full(test_actual_values.shape, np.nan)
    
    # Find column positions for each test index
    col_positions = test_actual_indices % daily_data.shape[1]

    # Assign predictions to the correct row and column
    for i, col_idx in enumerate(col_positions):
        row_idx = np.where(row_indices == (test_actual_indices[i] // daily_data.shape[1]))[0][0]
        test_predictions[row_idx, col_idx] = y_pred[i]

    # Create alternating actual/predicted column structure
    new_columns = []
    output_values = []

    for row_actual, row_pred in zip(test_actual_values.values, test_predictions):
        new_row = []
        for actual, pred in zip(row_actual, row_pred):
            new_row.append(actual)  # Actual value
            new_row.append(pred if not np.isnan(actual) else np.nan)  # Predicted value (keep NaN structure)
        output_values.append(new_row)

    # Create alternating columns (Day_X, Predicted_X)
    for col in daily_data.columns:
        new_columns.append(col)
        new_columns.append(f'Predicted_{col.split("_")[1]}')

    output_df = pd.concat([output_df, pd.DataFrame(output_values, columns=new_columns)], axis=1)
    
    # Save output file for this fold (only test dataset)
    output_file_path = os.path.join(output_folder, f"RF_Prediction_Fold_{fold}.xlsx")
    output_df.to_excel(output_file_path, index=False)

    # Print metrics for the current fold
    print(f"Fold {fold} Results Saved to: {output_file_path}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}\n")
    
    fold += 1

# Compute and print average metrics across all folds
print("Average Metrics Over 7 Splits:")
print(f"Average MSE: {np.mean(mse_list)}")
print(f"Average MAE: {np.mean(mae_list)}")
print(f"Average RMSE: {np.mean(rmse_list)}")
print(f"Average R^2: {np.mean(r2_list)}")
