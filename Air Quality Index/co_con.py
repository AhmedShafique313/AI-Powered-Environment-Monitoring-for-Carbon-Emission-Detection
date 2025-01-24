import pandas as pd

# Function to convert CO concentration from µg/m³ to ppm
def convert_co_to_ppm(co_concentration):
    M_CO = 28.01  # Molar mass of CO in g/mol
    return co_concentration / (M_CO * 100)

file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\refined datasets\2023_dataset_32.csv'
df = pd.read_csv(file_path)

# Convert CO concentration from µg/m³ to ppm
df['CO_C'] = df['co'].apply(convert_co_to_ppm)

# Save the updated DataFrame back to the same CSV file
df.to_csv(file_path, index=False)

# Display the updated DataFrame
print(df[['time', 'CO_C']])
