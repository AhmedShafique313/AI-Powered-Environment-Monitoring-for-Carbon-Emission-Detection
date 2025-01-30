import pandas as pd

df = pd.read_csv(r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\refined datasets\2023_dataset_32.csv')
# print(df.head())
# print('-----------------------------')
# print(df.tail())
# print('-----------------------------')
# print(df.info())

features = ['pm2_5', 'no', 'no2', 'o3', 'co', 'so2', 'pm10', 'ch4', 'time_sin', 'time_cos']
target_pm25 = 'PM2.5_AQI'
target_temp = 'avg_temp'

X = df[features].values
y_pm25 = df[target_pm25].values
y_temp = df[target_temp].values

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train_pm25, y_test_pm25, y_train_temp, y_test_temp = train_test_split(X_scaled, y_pm25, y_temp, test_size=0.2, random_state=42)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_pm25_tensor = torch.tensor(y_train_pm25, dtype=torch.float32).view(-1, 1)
y_test_pm25_tensor = torch.tensor(y_test_pm25, dtype=torch.float32).view(-1, 1)
y_train_temp_tensor = torch.tensor(y_train_temp, dtype=torch.float32).view(-1, 1)
y_test_temp_tensor = torch.tensor(y_test_temp, dtype=torch.float32).view(-1, 1)

train_data = TensorDataset(X_train_tensor, y_train_pm25_tensor, y_train_temp_tensor)
test_data = TensorDataset(X_test_tensor, y_test_pm25_tensor, y_test_temp_tensor)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=3)  # Adjusted in_channels to 10 for 10 features
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16 * 3, 128)  # Adjusted input size for fully connected layer
        self.fc2_pm25 = nn.Linear(128, 1)  # For PM2.5 AQI prediction
        self.fc2_temp = nn.Linear(128, 1)  # For temperature prediction
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        pm25_output = self.fc2_pm25(x)
        temp_output = self.fc2_temp(x)
        return pm25_output, temp_output

model = CNN_Model()
criterion_pm25 = nn.MSELoss()
criterion_temp = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
train_losses_pm25 = []
train_losses_temp = []

for epoch in range(epochs):
    model.train()
    running_loss_pm25 = 0.0
    running_loss_temp = 0.0
    
    for inputs, labels_pm25, labels_temp in train_loader:
        # Reshape inputs for CNN: (batch_size, channels, length)
        inputs = inputs.view(inputs.size(0), 10, 1)  # 10 features as channels
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs_pm25, outputs_temp = model(inputs)
        
        # Compute loss
        loss_pm25 = criterion_pm25(outputs_pm25, labels_pm25)
        loss_temp = criterion_temp(outputs_temp, labels_temp)
        
        # Backward pass and optimization
        loss_pm25.backward()
        loss_temp.backward()
        optimizer.step()
        
        running_loss_pm25 += loss_pm25.item()
        running_loss_temp += loss_temp.item()

    # Append losses for visualization
    train_losses_pm25.append(running_loss_pm25 / len(train_loader))
    train_losses_temp.append(running_loss_temp / len(train_loader))

    print(f"Epoch {epoch+1}/{epochs}, Loss PM2.5: {running_loss_pm25 / len(train_loader):.4f}, Loss Temp: {running_loss_temp / len(train_loader):.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses_pm25, label='PM2.5 AQI Loss')
plt.title('Training Loss for PM2.5 AQI')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses_temp, label='Temperature Loss')
plt.title('Training Loss for Temperature')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

import numpy as np
import math
from sklearn.metrics import mean_squared_error

model.eval()
predictions_pm25 = []
predictions_temp = []
true_pm25 = []
true_temp = []

with torch.no_grad():
    for inputs, labels_pm25, labels_temp in test_loader:
        inputs = inputs.view(inputs.size(0), 10, 1)  # Reshape inputs
        outputs_pm25, outputs_temp = model(inputs)
        predictions_pm25.append(outputs_pm25.numpy())
        predictions_temp.append(outputs_temp.numpy())
        true_pm25.append(labels_pm25.numpy())
        true_temp.append(labels_temp.numpy())

predictions_pm25 = np.concatenate(predictions_pm25, axis=0)
predictions_temp = np.concatenate(predictions_temp, axis=0)
true_pm25 = np.concatenate(true_pm25, axis=0)
true_temp = np.concatenate(true_temp, axis=0)

rmse_pm25 = math.sqrt(mean_squared_error(true_pm25, predictions_pm25))
rmse_temp = math.sqrt(mean_squared_error(true_temp, predictions_temp))

print(f"Test RMSE for PM2.5 AQI: {rmse_pm25:.4f}")
print(f"Test RMSE for Temperature: {rmse_temp:.4f}")