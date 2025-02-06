import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


file_path = "/home/hp/Downloads/Data Preprocessing till knn/Temperature/Processed_MAX_TEMP_Data_Imputed.xlsx"  # Replace with your file path
data = pd.read_excel(file_path)


station_indexes = data['Station_Index'].unique()


for station_index in station_indexes:
    
    station_data = data[data['Station_Index'] == station_index]

    
    dates = []
    temperatures = []

    for index, row in station_data.iterrows():
        year = int(row['Year'])
        month = int(row['Month'])
        days = [f"Day_{i}" for i in range(1, 32)]
        
        for day_idx, temp in enumerate(row[days], start=1):
            if not pd.isna(temp):  
                try:
                    
                    date = datetime(year, month, int(day_idx))
                    dates.append(date)
                    temperatures.append(temp)
                except ValueError:
                    
                    continue

    
    plt.figure(figsize=(14, 8))
    plt.plot(dates, temperatures, label=f"Station {station_index} Temperature", linewidth=0.8)

    
    plt.title(f"Original Time Series Plot - Station {station_index}", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Temperature (°C)", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    
    plt.tight_layout()
    plt.show()