import pandas as pd

file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\preprocessing\2023_latitude_32_dataset.csv'

df = pd.read_csv(file_path)

df['avg_temp'] = df['avg_temp'].round(2)

df.to_csv(file_path, index=False)

print(f"'avg_temp' column values rounded to 2 decimal places and file updated.")
