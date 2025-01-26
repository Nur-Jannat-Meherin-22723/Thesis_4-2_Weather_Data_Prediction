import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Load the dataset
file_path = r"E:\Thesis\Max Temp\11111\2d_ssa_reconstructed_11111_max_temp.xlsx"
data = pd.read_excel(file_path)

# Verify column names
print("Column Names in Dataset:", data.columns)

# Identify metadata columns and day columns
metadata_columns = ['Station_Index', 'Year', 'Month']  # Adjust based on your dataset
day_columns = [col for col in data.columns if col.startswith('Day_')]

# Extract metadata and day data
metadata = data[metadata_columns]
day_data = data[day_columns]

# Flatten the day data for processing
day_data_flattened = day_data.stack().reset_index(level=1)  # Convert to 1D with index
day_data_flattened.columns = ['Day', 'Temperature']  # Rename columns
day_data_flattened['Temperature'] = day_data_flattened['Temperature'].fillna(method='ffill')

# Create lagged features
def create_lagged_features(series, lags=7):
    lagged_data = pd.DataFrame()
    for lag in range(1, lags + 1):
        lagged_data[f"Lag_{lag}"] = series.shift(lag)
    lagged_data['Target'] = series
    lagged_data = lagged_data.dropna()  # Drop rows with NaN values
    return lagged_data

# Generate lagged features
lagged_data = create_lagged_features(day_data_flattened['Temperature'], lags=7)

# Split features and target
X = lagged_data.iloc[:, :-1]  # Lagged features
y = lagged_data['Target']    # Target (current temperature)

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Initialize TimeSeriesSplit
ts_split = TimeSeriesSplit(n_splits=5)

# Train and test SVR using Expanding Window CV
predictions = []  # Store predictions
fold = 1

for train_index, test_index in ts_split.split(X_scaled):
    # Split the data
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train SVR model
    svr = SVR(kernel='rbf', C=10.0, gamma=0.1)
    svr.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = svr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Fold {fold} - Mean Squared Error: {mse}")
    
    # Save predictions with corresponding index
    predictions.extend(zip(test_index, y_pred))
    fold += 1

# Create a DataFrame for predictions
predictions_df = pd.DataFrame(predictions, columns=['Index', 'Predicted'])
predictions_df = predictions_df.set_index('Index')

# Merge predictions back to the original day data
day_data_flattened['Predicted'] = day_data_flattened.index.map(predictions_df['Predicted'])
day_data_flattened['Predicted'] = day_data_flattened['Predicted'].fillna(day_data_flattened['Temperature'])

# Reshape back to the original 2D structure
predicted_day_data = day_data_flattened.pivot(columns='Day', values='Predicted')
predicted_day_data.columns.name = None  # Remove hierarchical column name

# Combine metadata with predicted day data
final_output = pd.concat([metadata.reset_index(drop=True), predicted_day_data.reset_index(drop=True)], axis=1)

# Save the final output
output_path = r"E:\Thesis\Max Temp\11111\SVR_2D_Predictions.xlsx"
final_output.to_excel(output_path, index=False)
print(f"2D predictions saved to: {output_path}")
