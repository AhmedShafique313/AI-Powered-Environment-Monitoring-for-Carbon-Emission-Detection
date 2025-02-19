import pandas as pd
import numpy as np

file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\preprocessing\2023_latitude_32_dataset.csv'
df = pd.read_csv(file_path)

df['time'] = pd.to_datetime(df['time_of_day'])

df['hour'] = df['time'].dt.hour
df['minute'] = df['time'].dt.minute
df['second'] = df['time'].dt.second

df['seconds_in_day'] = df['hour'] * 3600 + df['minute'] * 60 + df['second']
SECONDS_IN_DAY = 24 * 60 * 60

df['time_sin'] = np.sin(2 * np.pi * df['seconds_in_day'] / SECONDS_IN_DAY)
df['time_cos'] = np.cos(2 * np.pi * df['seconds_in_day'] / SECONDS_IN_DAY)

df = df.drop(['hour', 'minute', 'second', 'seconds_in_day'], axis=1)

df.to_csv(file_path, index=False)
print(df.head())
