import pandas as pd
from sklearn.impute import KNNImputer


file_path = '/home/hp/Downloads/Data Preprocessing till knn/Temperature/Processed_MIN_TEMP_Data.xlsx'
data = pd.read_excel(file_path)


def knn_impute_station(data, station_index):

    station_data = data[data['Station_Index'] == station_index].copy()


    temp_data = station_data.replace(-9999, float('NaN'))


    day_columns = [col for col in station_data.columns if col.startswith('Day_')]


    imputer = KNNImputer(n_neighbors=3)
    imputed_values = imputer.fit_transform(temp_data[day_columns])


    for col in day_columns:
        station_data[col] = station_data[col].where(station_data[col] != -9999, imputed_values[:, day_columns.index(col)])

    return station_data


station_indices = data['Station_Index'].unique()

imputed_data = pd.DataFrame()
for station in station_indices:
    station_imputed = knn_impute_station(data, station)
    imputed_data = pd.concat([imputed_data, station_imputed], ignore_index=True)


output_path = '/home/hp/Downloads/Data Preprocessing till knn/Temperature/Processed_MIN_TEMP_Data_Imputed.xlsx'
imputed_data.to_excel(output_path, index=False)

print(f"Imputed data saved to {output_path}")