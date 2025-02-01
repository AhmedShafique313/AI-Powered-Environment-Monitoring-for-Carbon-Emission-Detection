import pandas as pd

breakpoints = {
    "PM2.5": [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200),
              (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)],
    "PM10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200),
             (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500)],
    "CO": [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200),
           (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)],
    "SO2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200),
            (305, 604, 201, 300), (605, 804, 301, 400), (805, 1004, 401, 500)],
    "NO2": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200),
            (650, 1249, 201, 300), (1250, 1649, 301, 400), (1650, 2049, 401, 500)],
    "O3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200),
           (106, 200, 201, 300), (201, 300, 301, 400), (301, 400, 401, 500)],
}

def calculate_aqi(concentration, breakpoints):
    for bp in breakpoints:
        if bp[0] <= concentration <= bp[1]:
            return ((bp[3] - bp[2]) / (bp[1] - bp[0])) * (concentration - bp[0]) + bp[2]
    return None

file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\Datasets\data_transformed_31_6.csv'
df = pd.read_csv(file_path)

df['PM2.5_AQI'] = df['pm2_5'].apply(lambda x: calculate_aqi(x, breakpoints["PM2.5"]))
df['PM10_AQI'] = df['pm10'].apply(lambda x: calculate_aqi(x, breakpoints["PM10"]))
df['CO_AQI'] = df['co'].apply(lambda x: calculate_aqi(x, breakpoints["CO"]))
df['SO2_AQI'] = df['so2'].apply(lambda x: calculate_aqi(x, breakpoints["SO2"]))
df['NO2_AQI'] = df['no2'].apply(lambda x: calculate_aqi(x, breakpoints["NO2"]))
df['O3_AQI'] = df['o3'].apply(lambda x: calculate_aqi(x, breakpoints["O3"]))

df['Final_AQI'] = df[['PM2.5_AQI', 'PM10_AQI', 'CO_AQI', 'SO2_AQI', 'NO2_AQI', 'O3_AQI']].max(axis=1)

df.to_csv(file_path, index=False)

print(df[['time', 'PM2.5_AQI', 'PM10_AQI', 'CO_AQI', 'SO2_AQI', 'NO2_AQI', 'O3_AQI', 'Final_AQI']])
