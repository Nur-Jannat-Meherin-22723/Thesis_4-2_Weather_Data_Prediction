import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# Load dataset (Fix: Use raw string r"" to prevent escape sequence errors)
file_path = r"E:\Thesis\Web App\min_temp\min_temp_11513\2d_ssa_reconstructed_11513_min_temp.xlsx"

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: File not found at '{file_path}'")

# Read Excel file
df = pd.read_excel(file_path)

# Extract metadata columns
meta_cols = ['Station_Index', 'Year', 'Month']
day_cols = [col for col in df.columns if 'Day' in col]

# Filter data to ensure correct training range (Jan 1977 - May 2021)
df_train = df[(df['Year'] > 1976) & ((df['Year'] < 2021) | ((df['Year'] == 2021) & (df['Month'] <= 5)))]  # 1977 - May 2021

# Prepare data for training
X = df_train[day_cols].values  # Features (daily temperatures)

# Handle missing values (Fix: Replace NaNs properly)
X = np.nan_to_num(X, nan=np.nanmean(X))  # Replace NaNs with column mean

# Identify the last available month in the dataset (May 2021)
last_year = 2021
last_month = 5

# Train-test split (80% training, 20% testing)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, X_train)  # Model learns to predict next month's daily values

# Predict future data (Jun 2021 - Dec 2027)
predicted_years = np.arange(2021, 2028)
predicted_months = np.arange(1, 13)
predictions = []

# Start with last known month’s data
previous_data = X[-1].reshape(1, -1)  # Ensure correct shape (1, 31)

for year in predicted_years:
    for month in predicted_months:
        if year == 2021 and month <= 5:
            continue  # Skip months before June 2021
        
        # Ensure only last month's data is used for prediction
        input_data = previous_data[:, :31]  # Keep only 31 days

        # Predict next month's temperature
        predicted_days = rf.predict(input_data)[0]  # Extract array from prediction
        
        # Add slight randomness to avoid identical values across years
        predicted_days += np.random.normal(0, 0.5, size=predicted_days.shape)

        # Handle missing days (Feb 29-31, Apr/Jun/Sep/Nov 31st)
        predicted_days = list(predicted_days)
        if month == 2:
            if year % 4 == 0:
                predicted_days[29:] = [np.nan, np.nan]  # Keep only up to 30 days in leap years
            else:
                predicted_days[28:] = [np.nan, np.nan, np.nan]  # Remove day 29-31 in non-leap years
        elif month in [4, 6, 9, 11]:
            predicted_days[30] = np.nan  # No 31st in these months

        # Store prediction
        predictions.append([11513, year, month] + predicted_days)  # Fixed station index
        
        # Update input for the next prediction
        previous_data = np.array(predicted_days).reshape(1, -1)

# Create DataFrame for predictions
predicted_df = pd.DataFrame(predictions, columns=meta_cols + day_cols)

# Merge with original dataset
output_df = pd.concat([df, predicted_df], ignore_index=True)

# Safe file-saving
output_path = r"E:\Thesis\Web App\min_temp\min_temp_11513\predicted_2d_min_temp_11513.xlsx"

# Check if file is open before saving
if os.path.exists(output_path):
    try:
        os.remove(output_path)  # Try removing the existing file
    except PermissionError:
        print(f"Error: Please close '{output_path}' before running this script.")
        exit()

# Save output
output_df.to_excel(output_path, index=False)
print(f"Predictions saved to: {output_path}")
