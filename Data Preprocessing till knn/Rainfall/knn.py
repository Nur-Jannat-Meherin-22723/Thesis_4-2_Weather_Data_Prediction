import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

#file_path = '/home/hp/Downloads/Data Preprocessing till knn/Rainfall/RAINFALL 1953-2021 (May).xlsx'
file_path = '/home/hp/Downloads/Data Preprocessing till knn/Relative Humidity/RH% 1953-2021 (Aug).xlsx'
data = pd.ExcelFile(file_path)

rainfall_data = data.parse(data.sheet_names[0])

rainfall_columns = [col for col in rainfall_data.columns if col not in ['Station', 'Year', 'Month']]


imputation_mask = rainfall_data[rainfall_columns] == -9999
rainfall_data[rainfall_columns] = rainfall_data[rainfall_columns].replace(-9999, float('NaN'))

def impute_station_data(station_df):
    scaler = MinMaxScaler()
    imputer = KNNImputer(n_neighbors=3)
    

    normalized_data = scaler.fit_transform(station_df[rainfall_columns])
    

    imputed_data = imputer.fit_transform(normalized_data)
    

    imputed_data_original_scale = scaler.inverse_transform(imputed_data)
    

    imputed_df = pd.DataFrame(imputed_data_original_scale, columns=rainfall_columns)
    station_df[rainfall_columns] = station_df[rainfall_columns].where(~imputation_mask.loc[station_df.index], imputed_df)
    
    return station_df


rainfall_data = rainfall_data.groupby('Station', group_keys=False).apply(impute_station_data)


output_path = '/home/hp/Downloads/Data Preprocessing till knn/Relative Humidity/Processed_Humidity_Data_Stationwise.xlsx'
rainfall_data.to_excel(output_path, index=False)

print(f"Station-wise processed data saved to {output_path}")
