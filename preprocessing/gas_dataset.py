import pandas as pd

file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\preprocessing\2022_latitude_32_dataset.csv'
dataset = pd.read_csv(file_path)

dataset[['date', 'time_of_day']] = dataset['time'].str.split(' ', expand=True)
dataset.to_csv(file_path, index=False)
print(dataset)