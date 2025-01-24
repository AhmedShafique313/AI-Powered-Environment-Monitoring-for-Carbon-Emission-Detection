import pandas as pd

file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\preprocessing\2023_latitude_32_dataset.csv'

df = pd.read_csv(file_path)

df.drop(columns=['Unnamed: 14'], inplace=True)

df.to_csv(file_path, index=False)

print(f"Column 'Unnamed: 14' dropped and file updated.")
