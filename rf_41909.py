import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# File path
file_path = r"D:\4-1\4-2\Thesis\code\DataPreprocessing\DataPreprocessing\ssa_41909_reconstructed_max_temp.xlsx"
output_file = r"D:\4-1\4-2\Thesis\code\DataPreprocessing\DataPreprocessing\rf_41909_comparison_output.xlsx"

# Load the dataset
data = pd.read_excel(file_path)

# Ensure column names match your dataset
if "Reconstructed" not in data.columns:
    raise ValueError("The 'Reconstructed' column is not found in the dataset.")

# Extract the "Reconstructed" column
reconstructed_data = data["Reconstructed"].values

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
X, y = create_lagged_features(reconstructed_data, lag)

# Split the data into 80:20 for training and testing
train_size = int(len(y) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_model.predict(X_test)

# Calculate comparison metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Prepare the output DataFrame
test_indices = np.arange(len(reconstructed_data) - len(y_test), len(reconstructed_data))
comparison_df = pd.DataFrame({
    "Reconstructed": y_test,
    "Predicted": y_pred
}, index=test_indices)

# Add metrics at the end of the DataFrame
metrics_row = pd.DataFrame({
    "Reconstructed": ["Comparison Metrics:"],
    "Predicted": [""],
    "MAE": [mae],
    "MSE": [mse],
    "RMSE": [rmse],
    "R2": [r2]
})
comparison_df = pd.concat([comparison_df, metrics_row], ignore_index=True)

# Save the comparison output to an Excel file
comparison_df.to_excel(output_file, index=False)

# Print completion message
print(f"Comparison output saved to: {output_file}")
