import pandas as pd

# Define file path
file_path = r"E:\Thesis\Web App\min_temp\min_temp_11111\predicted_2d_min_temp_11111.xlsx"

# Load the dataset
df = pd.read_excel(file_path)

# Identify day columns (assuming they start with 'Day_')
day_columns = [col for col in df.columns if col.startswith('Day_')]

# Format day columns to four decimal places
df[day_columns] = df[day_columns].applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

# Save the modified file
output_file = r"E:\Thesis\Web App\min_temp\min_temp_11111\formatted_predicted_2d_min_temp_11111.xlsx"
df.to_excel(output_file, index=False)

print(f"File has been successfully saved at: {output_file}")
