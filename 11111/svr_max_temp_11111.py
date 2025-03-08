import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress the InconsistentVersionWarning if you want to ignore it
warnings.simplefilter("ignore", InconsistentVersionWarning)

# File paths
file_path = r"E:\Thesis\Max Temp\11111\normalized_max_temp_station_11111.xlsx"
scaler_file = r"E:\Thesis\Max Temp\11111\scaler.pkl"
output_dir = r"E:\Thesis\Max Temp\11111\svr_prediction"
os.makedirs(output_dir, exist_ok=True)

# Load data and scaler
data = pd.read_excel(file_path)

# Load the scaler
with open(scaler_file, "rb") as f:
    scalers = pickle.load(f)

# Define TimeSeriesSplit for cross-validation with 7 splits
ts_split = TimeSeriesSplit(n_splits=7)

# Lists to store metrics for each fold
mse_list = []
mae_list = []
rmse_list = []
r2_list = []

# Loop through the splits
fold = 1
for train_index, test_index in ts_split.split(data):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    metadata_cols = ['station index', 'Year', 'Month']
    day_cols = [col for col in data.columns if col.startswith('Day_')]

    output_test_data = test_data.copy()

    for day_col in day_cols:
        y_train = train_data[day_col].dropna().values
        y_test = test_data[day_col].values

        if len(y_train) > 0:
            # Fit the SVR model with the training data
            #svr = SVR(kernel='rbf')
            svr = SVR(kernel='rbf', C=1e-100, epsilon=1e-100, gamma='scale')
            svr.fit(np.arange(len(y_train)).reshape(-1, 1), y_train)

            # Predict the values for the test set
            predictions = svr.predict(np.arange(len(y_test)).reshape(-1, 1))

            # Handle missing values by setting predictions to NaN where y_test is NaN
            predictions[np.isnan(y_test)] = np.nan
        else:
            predictions = np.full_like(y_test, np.nan)

        # Denormalize actual and predicted values using the preloaded scalers
        scaler = scalers[day_col]  
        denormalized_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        denormalized_predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        # Handle NaN values in actual and predicted values
        denormalized_actual[np.isnan(y_test)] = np.nan
        denormalized_predictions[np.isnan(predictions)] = np.nan

        # Insert predicted values next to actual values in the output DataFrame
        predicted_col = f'Predicted_{day_col.split("_")[1]}'
        output_test_data[day_col] = denormalized_actual
        output_test_data.insert(output_test_data.columns.get_loc(day_col) + 1, predicted_col, denormalized_predictions)

    # Save output for this fold
    output_file = os.path.join(output_dir, f'svr_predictions_fold_{fold}.xlsx')
    output_test_data.to_excel(output_file, index=False)

    # Evaluate model performance
    actual_values = output_test_data[day_cols].values.flatten()
    predicted_values = output_test_data[[f'Predicted_{col.split("_")[1]}' for col in day_cols]].values.flatten()

    # Remove NaN values before calculating metrics
    valid_mask = ~np.isnan(actual_values) & ~np.isnan(predicted_values)
    actual_values = actual_values[valid_mask]
    predicted_values = predicted_values[valid_mask]

    mse = mean_squared_error(actual_values, predicted_values)
    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values, predicted_values)

    # Store metrics for the current fold
    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)
    r2_list.append(r2)

    # Print evaluation metrics for the current fold
    print(f"Fold {fold}:")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    print(f"Output saved to {output_file}\n")

    fold += 1

# Calculate and print the average of the metrics across all folds
avg_mse = np.mean(mse_list)
avg_mae = np.mean(mae_list)
avg_rmse = np.mean(rmse_list)
avg_r2 = np.mean(r2_list)

print(f"Average MSE: {avg_mse}")
print(f"Average MAE: {avg_mae}")
print(f"Average RMSE: {avg_rmse}")
print(f"Average R2: {avg_r2}")
