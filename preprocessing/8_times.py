import pandas as pd

file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\preprocessing\23temperature_dataset.csv'
df = pd.read_csv(file_path)

avg_jan_repeat = df['Avg_Dec'].repeat(8).reset_index(drop=True)
new_df = pd.DataFrame({'Avg_Dec_R': avg_jan_repeat})

update_file_path = r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\preprocessing\23temprature.csv'
new_df.to_csv(update_file_path, index=False)

print(f"Step 1 complete: Avg_Jan repeated and saved at {update_file_path}")
