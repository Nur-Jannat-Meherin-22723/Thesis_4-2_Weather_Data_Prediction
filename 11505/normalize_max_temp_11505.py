import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Input and Output File Paths
input_file = r"E:\Thesis\Max Temp\11505\2d_ssa_reconstructed_11505_max_temp.xlsx"
output_file = os.path.join(os.path.dirname(input_file), "normalized_max_temp_station_11505.xlsx")

# Read the Excel file
data = pd.read_excel(input_file)

# Outlier Capping using IQR method
def cap_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

# Normalize the dataset using Min-Max Scaling while keeping empty cells unchanged
def normalize_data(df, columns):
    scaler = MinMaxScaler()
    for col in columns:
        original_col = df[col]
        mask = original_col.isna()  # Identify missing values
        temp_col = original_col.fillna(0)  # Temporarily replace NaN with 0 for scaling
        scaled_col = scaler.fit_transform(temp_col.values.reshape(-1, 1)).flatten()
        scaled_col[mask] = np.nan  # Restore NaN where they originally existed
        df[col] = scaled_col
    return df

# Identify columns to process (Day_1 to Day_31)
columns_to_process = [col for col in data.columns if col.startswith('Day_')]

# Apply outlier detection and normalization only on selected columns
data = cap_outliers(data, columns_to_process)
data = normalize_data(data, columns_to_process)

# Save the processed data to a new Excel file
data.to_excel(output_file, index=False)

print(f"Processed file saved to: {output_file}")
