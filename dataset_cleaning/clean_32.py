import pandas as pd

df = pd.read_csv(r'C:\Users\Ahmed Shafique\Documents\Projects\FYP\Datasets\latitude_32.csv')
# print(df.head())
df = df.drop_duplicates()
df = df.dropna()
df['time'] = pd.to_datetime(df['time'])
print(df.info())