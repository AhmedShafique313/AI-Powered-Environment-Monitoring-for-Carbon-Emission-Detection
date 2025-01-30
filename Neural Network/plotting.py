from cnn_model import *
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