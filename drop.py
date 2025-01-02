import os
import pandas as pd

# File path
file_path = "D:\4-1\4-2\Thesis\code\DataPreprocessing\DataPreprocessing\MAX_TEMP.xlsx"

# Step 1: Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

# Load the Excel file, skipping the first 6 rows of metadata
df = pd.read_excel(file_path, sheet_name='Sheet1', skiprows=6)

# Rename columns for easier reference
df.columns = ['Station_Index', 'Year', 'Month'] + [f'Day_{i}' for i in range(1, 32)]

# Define station-specific year and month ranges to delete
delete_info = {
    'Dhaka': {'station_index': 11111, 'year': 2021, 'months': [9, 10, 11, 12]},
    'Tangail': {'station_index': 41909, 'year': 1987, 'months': [1, 2, 3]},
    'Tangail_2021': {'station_index': 41909, 'year': 2021, 'months': [6, 7, 8, 9, 10, 11, 12]},
    'Faridpur_2021': {'station_index': 11505, 'year': 2021, 'months': [6, 7, 8, 9, 10, 11, 12]},
    'Madaripur_2021': {'station_index': 11513, 'year': 2021, 'months': [6, 7, 8, 9, 10, 11, 12]}
}

# Initialize an empty list to store filtered data
filtered_data = []

# Step 2: Apply deletion based on year and month for each station
for station, info in delete_info.items():
    print(f"Processing station: {station}")
    
    # Filter the rows for the given station, excluding the specified year and months
    condition = (df['Station_Index'] == info['station_index']) & \
                ~((df['Year'] == info['year']) & df['Month'].isin(info['months']))
    
    # Apply the condition to get the filtered DataFrame
    station_df = df[condition]
    
    # Append the filtered DataFrame to the list
    filtered_data.append(station_df)

# Step 3: Concatenate all filtered station data into one DataFrame
final_df = pd.concat(filtered_data, ignore_index=True)

# Step 4: Remove duplicates from final_df, if any
final_df.drop_duplicates(inplace=True)

# Step 5: Save the resulting data with the deleted rows
output_path = '/home/hp/Downloads/DataPreprocessing/Processed_MAX_TEMP_Data.xlsx'
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    final_df.to_excel(writer, index=False)

print(f"Processed data saved to {output_path}")
