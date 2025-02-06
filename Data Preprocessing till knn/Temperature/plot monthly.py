import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


file_path = "/home/hp/Downloads/Data Preprocessing till knn/Temperature/Processed_MAX_TEMP_Data_Imputed.xlsx"  # Replace with your file path
data = pd.read_excel(file_path)


station_indexes = data['Station_Index'].unique()


for station_index in station_indexes:
    
    station_data = data[data['Station_Index'] == station_index]

    
    days_columns = [f"Day_{i}" for i in range(1, 32)]
    station_data_melted = station_data.melt(
        id_vars=["Station_Index", "Year", "Month"], 
        value_vars=days_columns, 
        var_name="Day", 
        value_name="Temperature"
    )

    
    station_data_melted['Day'] = station_data_melted['Day'].str.extract('(\d+)').astype(float)

    
    station_data_melted = station_data_melted.dropna(subset=['Temperature'])
    station_data_melted['Year-Month'] = pd.to_datetime(station_data_melted[['Year', 'Month']].assign(Day=1))
    monthly_avg = station_data_melted.groupby('Year-Month')['Temperature'].mean().reset_index()

    
    plt.figure(figsize=(14, 8))
    plt.plot(monthly_avg['Year-Month'], monthly_avg['Temperature'], label=f"Station {station_index} Monthly Average", linewidth=1.5)


    plt.title(f"Monthly Average Temperature - Station {station_index}", fontsize=16)
    plt.xlabel("Year-Month", fontsize=14)
    plt.ylabel("Temperature (°C)", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)


    plt.tight_layout()
    plt.show()
