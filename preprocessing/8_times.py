import pandas as pd

# File path to the source dataset
file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\22temprature_dataset.csv'
df = pd.read_csv(file_path)

# List of columns for months (adjust based on your dataset)
month_columns = ['Avg_Jan', 'Avg_Feb', 'Avg_Mar', 'Avg_Apr', 'Avg_May', 'Avg_Jun', 
                 'Avg_Jul', 'Avg_Aug', 'Avg_Sep', 'Avg_Oct', 'Avg_Nov', 'Avg_Dec']

# Create an empty DataFrame to store the repeated values for all months
new_df = pd.DataFrame()

# Loop through each month column and add the repeated values as a new column
for month in month_columns:
    new_df[month + '_R'] = df[month].repeat(8).reset_index(drop=True)

# Define the path for the updated CSV file
update_file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\preprocessing\22temprature_final.csv'

# Save the new DataFrame to the specified path
new_df.to_csv(update_file_path, index=False)

print(f"New CSV file with repeated values for all months created and saved at: {update_file_path}")
