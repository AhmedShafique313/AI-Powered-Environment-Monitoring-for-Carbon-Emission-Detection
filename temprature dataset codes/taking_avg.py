import pandas as pd

csv_file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\23temperature_dataset.csv'  
df = pd.read_csv(csv_file_path)

column1 = 'Dec_Old'  
column2 = 'Dec_New'  

df['Avg_Dec'] = df[[column1, column2]].mean(axis=1)

df.to_csv(csv_file_path, index=False)

print("Dataset is updated")
