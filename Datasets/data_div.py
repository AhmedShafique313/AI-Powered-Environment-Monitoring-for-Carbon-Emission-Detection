import pandas as pd

df = pd.read_csv(r'C:\Users\Ahmed Shafique\Documents\Projects\FYP\Datasets\latitude_32.csv')
df = df.drop_duplicates()
df = df.dropna()
df['time'] = pd.to_datetime(df['time'])
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['hour'] = df['time'].dt.hour

df['year'] = df['year'].astype(str)
unique_years = df['year'].unique()
for year in unique_years:
    year_data = df[df['year'] == year]
    output_path = f'{year}_latitude_32_dataset.csv'
    year_data.to_csv(output_path, index=False)
    print(f"Saved {year} data to {output_path}")