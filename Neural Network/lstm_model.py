# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader, TensorDataset
# import matplotlib.pyplot as plt

# # Step 1: Load the dataset
# data = pd.read_csv(r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\refined datasets\2023_dataset_32.csv')

# # Step 2: Feature selection
# features = ['pm2_5', 'no', 'no2', 'o3', 'co', 'so2', 'pm10', 'ch4', 'time_sin', 'time_cos']
# target_pm25 = 'PM2.5_AQI'

# # Step 3: Data Preprocessing
# # Extract features and target
# X = data[features].values
# y_pm25 = data[target_pm25].values

# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Step 4: Train-test split (80% for training, 20% for testing)
# X_train, X_test, y_train_pm25, y_test_pm25 = train_test_split(X_scaled, y_pm25, test_size=0.2, random_state=42)

# # Step 5: Convert data to PyTorch tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_train_pm25_tensor = torch.tensor(y_train_pm25, dtype=torch.float32).view(-1, 1)
# y_test_pm25_tensor = torch.tensor(y_test_pm25, dtype=torch.float32).view(-1, 1)

# # Step 6: Create DataLoader for batching
# train_data = TensorDataset(X_train_tensor, y_train_pm25_tensor)
# test_data = TensorDataset(X_test_tensor, y_test_pm25_tensor)

# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# # Step 7: Define the LSTM model
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         # LSTM expects input in shape (batch_size, seq_len, input_size)
#         x = x.unsqueeze(1)  # Add a sequence dimension (seq_len = 1)
#         lstm_out, (h_n, c_n) = self.lstm(x)
#         out = self.fc(lstm_out[:, -1, :])  # Get the output from the last time step
#         return out

# # Step 8: Initialize the model, loss function, and optimizer
# input_size = X_train.shape[1]  # Number of features
# hidden_size = 64
# output_size = 1

# model = LSTMModel(input_size, hidden_size, output_size)
# criterion = nn.MSELoss()  # For PM2.5 AQI
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Step 9: Train the model
# epochs = 50
# train_losses = []

# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
    
#     for inputs, labels_pm25 in train_loader:
#         # Zero the parameter gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         outputs_pm25 = model(inputs)
        
#         # Compute loss
#         loss = criterion(outputs_pm25, labels_pm25)
        
#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()

#     # Append loss for visualization
#     train_losses.append(running_loss / len(train_loader))

#     print(f"Epoch {epoch+1}/{epochs}, Loss PM2.5: {running_loss / len(train_loader):.4f}")

# # Step 10: Plot the training loss
# plt.figure(figsize=(8, 6))
# plt.plot(train_losses, label='PM2.5 AQI Loss')
# plt.title('Training Loss for PM2.5 AQI')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # Step 11: Testing the model
# model.eval()
# predictions_pm25 = []
# true_pm25 = []

# with torch.no_grad():
#     for inputs, labels_pm25 in test_loader:
#         outputs_pm25 = model(inputs)
#         predictions_pm25.append(outputs_pm25.numpy())
#         true_pm25.append(labels_pm25.numpy())

# # Convert to numpy arrays for evaluation
# predictions_pm25 = np.concatenate(predictions_pm25, axis=0)
# true_pm25 = np.concatenate(true_pm25, axis=0)

# # Calculate RMSE for PM2.5 predictions
# from sklearn.metrics import mean_squared_error
# import math

# rmse_pm25 = math.sqrt(mean_squared_error(true_pm25, predictions_pm25))

# print(f"Test RMSE for PM2.5 AQI: {rmse_pm25:.4f}")



import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv(r'C:\Users\Ahmeds Gaming Laptop\Documents\Projects\AI-Powered-Environment-Monitoring-for-Carbon-Emission-Detection\refined datasets\2023_dataset_32.csv')

# Step 2: Feature selection
features = ['pm2_5', 'no', 'no2', 'o3', 'co', 'so2', 'pm10', 'ch4', 'time_sin', 'time_cos']
target_pm25 = 'pm2_5'  # Changed this to 'pm2_5'

# Step 3: Data Preprocessing
# Extract features and target
X = data[features].values
y_pm25 = data[target_pm25].values  # Changed to use 'pm2_5'

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-test split (80% for training, 20% for testing)
X_train, X_test, y_train_pm25, y_test_pm25 = train_test_split(X_scaled, y_pm25, test_size=0.2, random_state=41)

# Step 5: Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_pm25_tensor = torch.tensor(y_train_pm25, dtype=torch.float32).view(-1, 1)
y_test_pm25_tensor = torch.tensor(y_test_pm25, dtype=torch.float32).view(-1, 1)

# Step 6: Create DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_pm25_tensor)
test_data = TensorDataset(X_test_tensor, y_test_pm25_tensor)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# Step 7: Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # LSTM expects input in shape (batch_size, seq_len, input_size)
        x = x.unsqueeze(1)  # Add a sequence dimension (seq_len = 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Get the output from the last time step
        return out

# Step 8: Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]  # Number of features
hidden_size = 64
output_size = 1  # Single output for pm2_5

model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # For pm2_5 prediction
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 9: Train the model
epochs = 100
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels_pm25 in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs_pm25 = model(inputs)
        
        # Compute loss
        loss = criterion(outputs_pm25, labels_pm25)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Append loss for visualization
    train_losses.append(running_loss / len(train_loader))

    print(f"Epoch {epoch+1}/{epochs}, Loss PM2.5: {running_loss / len(train_loader):.4f}")

# Step 10: Plot the training loss
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='PM2.5 Loss')
plt.title('Training Loss for PM2.5 Prediction')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

torch.save(model.state_dict(), "pm25_lstm_model.pth")
print("Model saved successfully as pm25_lstm_model.pth")

# Step 11: Testing the model
model.eval()
predictions_pm25 = []
true_pm25 = []

with torch.no_grad():
    for inputs, labels_pm25 in test_loader:
        outputs_pm25 = model(inputs)
        predictions_pm25.append(outputs_pm25.numpy())
        true_pm25.append(labels_pm25.numpy())

# Convert to numpy arrays for evaluation
predictions_pm25 = np.concatenate(predictions_pm25, axis=0)
true_pm25 = np.concatenate(true_pm25, axis=0)

# Calculate RMSE for PM2.5 predictions
from sklearn.metrics import mean_squared_error
import math

rmse_pm25 = math.sqrt(mean_squared_error(true_pm25, predictions_pm25))

print(f"Test RMSE for PM2.5: {rmse_pm25:.4f}")

