import pandas as pd

# Load the existing CSV file
existing_csv_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\22_temperature_dataset.csv'
df_existing = pd.read_csv(existing_csv_path)

# Load the new text file data
new_text_file = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\Datasets\22_year_min_temp.txt'  # Replace with the path to your text file
df_new = pd.read_csv(new_text_file, delim_whitespace=True)

# Merge the two DataFrames on the 'Day' column
df_combined = pd.merge(df_existing, df_new, on='Day', suffixes=('_Old', '_New'))

# Save the combined DataFrame into the same CSV file
updated_csv_path = 'updated_temperature_data.csv'
df_combined.to_csv(updated_csv_path, index=False)

print(f"Updated CSV file saved as {updated_csv_path}")
