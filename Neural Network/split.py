from dataload import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train_pm25, y_test_pm25, y_train_temp, y_test_temp = train_test_split(X_scaled, y_pm25, y_temp, test_size=0.2, random_state=42)