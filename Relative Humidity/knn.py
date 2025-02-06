import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np


file_path = "/home/hp/Downloads/Data Preprocessing till knn/Relative Humidity/RH% 1953-2021 (Aug).xlsx"
original_data = pd.read_excel(file_path, header=0)  


original_data = original_data.dropna(axis=1, how="all")


original_data.columns = original_data.columns.map(lambda x: str(x).strip())


humidity_columns = [col for col in original_data.columns if col not in ['Station', 'Year', 'Month']]


humidity_columns = [int(col) if col.isdigit() else col for col in humidity_columns]


original_data.columns = ['Station', 'Year', 'Month'] + humidity_columns

def impute_station_data(station_df):

    imputation_mask = station_df[humidity_columns] == -9999
    
    station_df[humidity_columns] = station_df[humidity_columns].replace(-9999, np.nan)
    
    imputer = KNNImputer(n_neighbors=3)
    

    imputed_data = imputer.fit_transform(station_df[humidity_columns])
    

    imputed_df = pd.DataFrame(imputed_data, columns=humidity_columns, index=station_df.index)
    

    station_df[humidity_columns] = station_df[humidity_columns].where(~imputation_mask, imputed_df)
    
    return station_df


imputed_data = original_data.groupby('Station', group_keys=False).apply(impute_station_data)


output_path = "/home/hp/Downloads/Data Preprocessing till knn/Relative Humidity/Processed_Humidity_Data_Stationwise.xlsx"
imputed_data.to_excel(output_path, index=False)

print(f"Processed data saved to {output_path}")