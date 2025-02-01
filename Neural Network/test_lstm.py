
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the trained model
from lstm_model import LSTMModel  # Import your LSTM model class

# Load the dataset
data = pd.read_csv(r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\refined datasets\5years_32_4_dataset.csv')

# Features used during training
features = ['pm2_5', 'no', 'no2', 'o3', 'co', 'so2', 'pm10', 'ch4', 'time_sin', 'time_cos']
target_pm25 = 'pm2_5'  # Now predicting PM2.5 instead of AQI

# Extract features and target
X_new = data[features].values
y_actual = data[target_pm25].values  # True PM2.5 values

# Standardize the new dataset using the same scaler as training
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)  # Fit using only the features

# Convert to PyTorch tensor
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

# Load the trained model
input_size = len(features)
hidden_size = 64
output_size = 1

model = LSTMModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\Neural Network\pm25_lstm_model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    predictions_pm25 = model(X_new_tensor).numpy().flatten()

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_actual - predictions_pm25) / y_actual)) * 100

# Calculate Accuracy
accuracy = 100 - mape

print(f"Model Test Accuracy: {accuracy:.2f}%")

# Plot Actual vs. Predicted PM2.5 values
plt.figure(figsize=(10, 5))
plt.plot(y_actual, label="Actual PM2.5", color="blue", linestyle="dashed")
plt.plot(predictions_pm25, label="Predicted PM2.5", color="red")
plt.xlabel("Sample Index")
plt.ylabel("PM2.5 Value")
plt.title("Actual vs Predicted PM2.5 Levels")
plt.legend()
plt.show()



