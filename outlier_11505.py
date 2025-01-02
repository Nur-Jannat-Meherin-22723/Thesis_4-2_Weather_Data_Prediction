import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Input and Output File Paths
input_file = r"D:\\4-1\\4-2\\Thesis\\code\\DataPreprocessing\\DataPreprocessing\\max_temp_station_11505.xlsx"
output_file = os.path.join(os.path.dirname(input_file), "outlier_processed_max_temp_station_11505.xlsx")

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

# Normalize the dataset using Min-Max Scaling
'''def normalize_data(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
    '''

# Identify columns to process (Day_1 to Day_31)
columns_to_process = [col for col in data.columns if col.startswith('Day_')]

# Apply outlier detection and normalization only on selected columns
data = cap_outliers(data, columns_to_process)
#data = normalize_data(data, columns_to_process)

# Save the processed data to a new Excel file
data.to_excel(output_file, index=False)

print(f"Processed file saved to: {output_file}")
