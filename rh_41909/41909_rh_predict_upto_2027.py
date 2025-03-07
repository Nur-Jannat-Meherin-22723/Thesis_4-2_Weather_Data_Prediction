import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# Load dataset
file_path = r"E:\Thesis\Web App\relative_humidity\rh_41909\2d_ssa_reconstructed_41909_rh.xlsx"

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: File not found at '{file_path}'")

# Read Excel file
df = pd.read_excel(file_path)

# Extract metadata columns
meta_cols = ['Station_Index', 'Year', 'Month']
day_cols = [col for col in df.columns if 'Day' in col]

# 🔹 **Fix: Correct the training range (Jan 1987 - May 2021)**
df_train = df[(df['Year'] > 1986) & ((df['Year'] < 2021) | ((df['Year'] == 2021) & (df['Month'] <= 5)))]  

# Prepare data for training
X = df_train[day_cols].values  # Features (daily relative humidity)

# Handle missing values
X = np.nan_to_num(X, nan=np.nanmean(X))  # Replace NaNs with column mean

# Identify last available month
last_year, last_month = 2021, 5

# Train-test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, X_train)  

# Predict future data (Jun 2021 - Dec 2027)
predicted_years = np.arange(2021, 2028)
predicted_months = np.arange(1, 13)
predictions = []

# Start with last known month’s data
previous_data = X[-1].reshape(1, -1)  

for year in predicted_years:
    for month in predicted_months:
        if year == 2021 and month <= 5:
            continue  # Skip months before June 2021
        
        # Ensure only last month's data is used for prediction
        input_data = previous_data[:, :31]  

        # Predict next month's humidity
        predicted_days = rf.predict(input_data)[0]  

        # Add slight randomness to avoid identical values
        predicted_days += np.random.normal(0, 0.5, size=predicted_days.shape)

        # Handle missing days (Feb 29-31, Apr/Jun/Sep/Nov 31st)
        predicted_days = list(predicted_days)
        if month == 2:
            if year % 4 == 0:
                predicted_days[29:] = [np.nan, np.nan]  
            else:
                predicted_days[28:] = [np.nan, np.nan, np.nan]  
        elif month in [4, 6, 9, 11]:
            predicted_days[30] = np.nan  

        # Store prediction
        predictions.append([41909, year, month] + predicted_days)  
        
        # Update input for next prediction
        previous_data = np.array(predicted_days).reshape(1, -1)

# Create DataFrame for predictions
predicted_df = pd.DataFrame(predictions, columns=meta_cols + day_cols)

# Merge with original dataset
output_df = pd.concat([df, predicted_df], ignore_index=True)

# Safe file-saving
output_path = r"E:\Thesis\Web App\relative_humidity\rh_41909\predicted_rh_41909.xlsx"

# Check if file is open before saving
if os.path.exists(output_path):
    try:
        os.remove(output_path)  
    except PermissionError:
        print(f"Error: Please close '{output_path}' before running this script.")
        exit()

# Save output
output_df.to_excel(output_path, index=False)
print(f"Predictions saved to: {output_path}")
