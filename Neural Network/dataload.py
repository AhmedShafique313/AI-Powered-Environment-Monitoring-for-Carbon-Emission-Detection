import pandas as pd

df = pd.read_csv(r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\refined datasets\2023_dataset_32.csv')
print(df.head())
print('-----------------------------')
print(df.tail())
print('-----------------------------')
print(df.info())