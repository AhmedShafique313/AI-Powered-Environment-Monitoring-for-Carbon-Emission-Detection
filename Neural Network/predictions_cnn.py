from plotting import *
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