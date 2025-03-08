import pandas as pd
from sklearn.impute import KNNImputer

# Load the dataset
file_path = '/home/student/Downloads/DataPreprocessing/Processed_MAX_TEMP_Data.xlsx'
data = pd.read_excel(file_path)

# Function to perform KNN imputation for a single station
def knn_impute_station(data, station_index):
    # Filter data for the specific station
    station_data = data[data['Station_Index'] == station_index].copy()

    # Replace -9999 with NaN for KNN processing
    temp_data = station_data.replace(-9999, float('NaN'))

    # Extract numeric columns (Days)
    day_columns = [col for col in station_data.columns if col.startswith('Day_')]

    # Perform KNN imputation only on columns with -9999
    imputer = KNNImputer(n_neighbors=3)
    imputed_values = imputer.fit_transform(temp_data[day_columns])

    # Retain original NaN values (for days that do not exist in certain months)
    for col in day_columns:
        station_data[col] = station_data[col].where(station_data[col] != -9999, imputed_values[:, day_columns.index(col)])

    return station_data

# Get unique station indices
station_indices = data['Station_Index'].unique()

# Apply KNN imputation station-wise
imputed_data = pd.DataFrame()
for station in station_indices:
    station_imputed = knn_impute_station(data, station)
    imputed_data = pd.concat([imputed_data, station_imputed], ignore_index=True)

# Save the imputed dataset to a new Excel file
output_path = '/home/student/Downloads/DataPreprocessing/Processed_MAX_TEMP_Data_Imputed.xlsx'
imputed_data.to_excel(output_path, index=False)

print(f"Imputed data saved to {output_path}")
