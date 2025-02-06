import os
import pandas as pd


file_path = '/home/hp/Downloads/Data Preprocessing till knn/Temperature/MIN TEMP 1953 - 2021 (Aug).xlsx'


if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()


df = pd.read_excel(file_path, sheet_name='Sheet1', skiprows=6)


df.columns = ['Station_Index', 'Year', 'Month'] + [f'Day_{i}' for i in range(1, 32)]


delete_info = {
    'Dhaka': {'station_index': 11111, 'year': 2021, 'months': [9, 10, 11, 12]},
    'Tangail': {'station_index': 41909, 'years': [(1987, [1, 2, 3]), (2021, [6, 7, 8, 9, 10, 11, 12])]},
    'Faridpur_2021': {'station_index': 11505, 'year': 2021, 'months': [6, 7, 8, 9, 10, 11, 12]},
    'Madaripur_2021': {'station_index': 11513, 'year': 2021, 'months': [6, 7, 8, 9, 10, 11, 12]}
}


for station, info in delete_info.items():
    #print(f"Processing station: {station}")

    if 'years' in info:
        for year, months in info['years']:
            
            df.drop(df[(df['Station_Index'] == info['station_index']) & 
                    (df['Year'] == year) & 
                    (df['Month'].isin(months))].index, inplace=True)
    else:
        df.drop(df[(df['Station_Index'] == info['station_index']) & 
                (df['Year'] == info['year']) & 
                (df['Month'].isin(info['months']))].index, inplace=True)


df.reset_index(drop=True, inplace=True)


output_path = '/home/hp/Downloads/Data Preprocessing till knn/Temperature/Processed_MIN_TEMP_Data.xlsx'
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False)

#print(f"Processed data saved to {output_path}")
