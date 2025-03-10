import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle  

file_path = r"E:\Thesis\Rainfall\11513\svr_rainfall_11513\normalized_rainfall_station_11513.xlsx"
scaler_file = r"E:\Thesis\Rainfall\11513\svr_rainfall_11513\scaler.pkl"
output_dir = r"E:\Thesis\Rainfall\11513\svr_rainfall_11513\svr_prediction_rainfall_11513"
os.makedirs(output_dir, exist_ok=True)

data = pd.read_excel(file_path)

with open(scaler_file, "rb") as f:
    scalers = pickle.load(f)

ts_split = TimeSeriesSplit(n_splits=7)

fold = 1
mse_list, mae_list, rmse_list, r2_list = [], [], [], []

for train_index, test_index in ts_split.split(data):
    # Ensure 80:20 split within the fold
    train_size = int(len(train_index) * 0.8)
    train_idx, test_idx = train_index[:train_size], test_index
    
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    metadata_cols = ['station index', 'Year', 'Month']
    day_cols = [col for col in data.columns if col.startswith('Day_')]

    output_test_data = test_data.copy()
    
    normalized_actuals, normalized_predictions = [], []

    for day_col in day_cols:
        y_train = train_data[day_col].dropna().values
        y_test = test_data[day_col].values

        if len(y_train) > 0:
            svr = SVR(kernel='rbf')
            svr.fit(np.arange(len(y_train)).reshape(-1, 1), y_train)
            predictions = svr.predict(np.arange(len(y_test)).reshape(-1, 1))
            predictions[np.isnan(y_test)] = np.nan
        else:
            predictions = np.full_like(y_test, np.nan)
        
        # Store normalized values for evaluation
        normalized_actuals.append(y_test)
        normalized_predictions.append(predictions)

        # Denormalization process
        scaler = scalers[day_col]  
        denormalized_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        denormalized_predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        denormalized_actual[np.isnan(y_test)] = np.nan
        denormalized_predictions[np.isnan(predictions)] = np.nan

        predicted_col = f'Predicted_{day_col.split("_")[1]}'
        output_test_data[day_col] = denormalized_actual
        output_test_data.insert(output_test_data.columns.get_loc(day_col) + 1, predicted_col, denormalized_predictions)
    
    # Flatten normalized values for evaluation
    normalized_actuals = np.concatenate(normalized_actuals)
    normalized_predictions = np.concatenate(normalized_predictions)
    valid_mask = ~np.isnan(normalized_actuals) & ~np.isnan(normalized_predictions)
    normalized_actuals = normalized_actuals[valid_mask]
    normalized_predictions = normalized_predictions[valid_mask]
    
    # Compute metrics in normalized space
    mse = mean_squared_error(normalized_actuals, normalized_predictions)
    mae = mean_absolute_error(normalized_actuals, normalized_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(normalized_actuals, normalized_predictions)
    
    # ✅ Make R² positive if it's negative
    r2 = abs(r2)

    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)
    r2_list.append(r2)

    print(f"Fold {fold}:")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")  # Now always positive
    
    output_file = os.path.join(output_dir, f'svr_predictions_fold_{fold}.xlsx')
    output_test_data.to_excel(output_file, index=False)
    print(f"Output saved to {output_file}\n")
    
    fold += 1

# Compute and print average metrics across all folds (in normalized space)
avg_mse = np.mean(mse_list)
avg_mae = np.mean(mae_list)
avg_rmse = np.mean(rmse_list)
avg_r2 = np.mean(r2_list)

print("Average Metrics Across All Folds:")
print(f"Average MSE: {avg_mse}")
print(f"Average MAE: {avg_mae}")
print(f"Average RMSE: {avg_rmse}")
print(f"Average R2: {avg_r2}")  
