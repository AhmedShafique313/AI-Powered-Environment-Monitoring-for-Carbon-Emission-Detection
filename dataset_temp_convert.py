import pandas as pd

# Load the text file
file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\Datasets\23max_temp.txt'  # Path to your saved text file
df = pd.read_csv(file_path, delim_whitespace=True)

# Save as a CSV file
output_csv_path = '23_temperature_dataset.csv'
df.to_csv(output_csv_path, index=False)
print(f"CSV file saved as {output_csv_path}")
