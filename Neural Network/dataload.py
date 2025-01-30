import pandas as pd

df = pd.read_csv(r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\refined datasets\2023_dataset_32.csv')
# print(df.head())
# print('-----------------------------')
# print(df.tail())
# print('-----------------------------')
# print(df.info())

features = ['pm2_5', 'no', 'no2', 'o3', 'co', 'so2', 'pm10', 'ch4', 'time_sin', 'time_cos']
target_pm25 = 'PM2.5_AQI'
target_temp = 'avg_temp'

X = df[features].values
y_pm25 = df[target_pm25].values
y_temp = df[target_temp].values