import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Define file paths
file_path = r"E:\Thesis\Max Temp\11505\normalized_max_temp_station_11505.xlsx"
output_dir = r"E:\Thesis\Max Temp\11505\svr_prediction_max_temp_11505"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
data = pd.read_excel(file_path)

# Function to denormalize data
def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

# Get min and max values for denormalization (ignoring NaNs)
min_val = data[[col for col in data.columns if col.startswith('Day_')]].min().min()
max_val = data[[col for col in data.columns if col.startswith('Day_')]].max().max()

# Initialize the TimeSeriesSplit for expanding window cross-validation
ts_split = TimeSeriesSplit(n_splits=5)

# Perform the split and train/test the SVR model
fold = 1
for train_index, test_index in ts_split.split(data):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    # Separate metadata and day columns
    metadata_cols = ['station index', 'Year', 'Month']
    day_cols = [col for col in data.columns if col.startswith('Day_')]

    # Create output DataFrame
    output_test_data = test_data.copy()

    # Iterate through each day column to train and predict
    for i, day_col in enumerate(day_cols):
        y_train = train_data[day_col].dropna().values
        y_test = test_data[day_col].values

        # Train only if there are valid training data points
        if len(y_train) > 0:
            svr = SVR(kernel='rbf')
            svr.fit(np.arange(len(y_train)).reshape(-1, 1), y_train)

            # Predict the testing data
            predictions = svr.predict(np.arange(len(y_test)).reshape(-1, 1))

            # Handle empty cells in the test data
            predictions[np.isnan(y_test)] = np.nan
        else:
            predictions = np.full_like(y_test, np.nan)

        # Denormalize day data and predictions
        denormalized_day = np.where(~np.isnan(test_data[day_col].values), denormalize(test_data[day_col].values, min_val, max_val), np.nan)
        denormalized_predictions = np.where(~np.isnan(predictions), denormalize(predictions, min_val, max_val), np.nan)

        # Add predictions to the output DataFrame in the correct position
        predicted_col = f'Predicted_{day_col.split("_")[1]}'
        output_test_data[day_col] = denormalized_day
        output_test_data.insert(output_test_data.columns.get_loc(day_col) + 1, predicted_col, denormalized_predictions)

    # Save the output file for this fold
    output_file = os.path.join(output_dir, f'svr_predictions_fold_{fold}.xlsx')
    output_test_data.to_excel(output_file, index=False)

    # Calculate evaluation metrics for this fold
    actual_values = test_data[day_cols].fillna(0).values.flatten()
    predicted_values = output_test_data[[f'Predicted_{col.split("_")[1]}' for col in day_cols]].fillna(0).values.flatten()

    mse = mean_squared_error(actual_values, predicted_values)
    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values, predicted_values)

    # Print the metrics
    print(f"Fold {fold}:")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")
    print(f"Output saved to {output_file}\n")

    fold += 1
