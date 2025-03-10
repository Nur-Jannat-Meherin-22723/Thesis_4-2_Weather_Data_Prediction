import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle  

# Use raw string to prevent escape sequence issues
input_file = r"E:\Thesis\Relative Humidity\11505\svr_rh_11505\2d_ssa_reconstructed_11505_rh.xlsx"
output_file = os.path.join(os.path.dirname(input_file), "normalized_rh_station_11505.xlsx")
scaler_file = os.path.join(os.path.dirname(input_file), "scaler.pkl")  

data = pd.read_excel(input_file)

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

scalers = {}  
columns_to_process = [col for col in data.columns if col.startswith('Day_')]

for col in columns_to_process:
    scaler = MinMaxScaler()
    original_col = data[col]
    mask = original_col.isna()  
    temp_col = original_col.fillna(0).values.reshape(-1, 1)  
    scaled_col = scaler.fit_transform(temp_col).flatten()  
    scaled_col[mask] = np.nan  
    data[col] = scaled_col
    scalers[col] = scaler  

with open(scaler_file, "wb") as f:
    pickle.dump(scalers, f)

data.to_excel(output_file, index=False)
print(f"Processed file saved to: {output_file}")
print(f"Scaler saved to: {scaler_file}")
