import pandas as pd

dataset_file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\preprocessing\2023_latitude_32_dataset.csv'
temp_file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\preprocessing\fool.csv'

dataset_df = pd.read_csv(dataset_file_path)
temp_df = pd.read_csv(temp_file_path)

dataset_df['avg_temp'] = temp_df['Average_temp']

dataset_df.to_csv(dataset_file_path, index=False)

print(f"Final dataset updated and saved at: {dataset_file_path}")
