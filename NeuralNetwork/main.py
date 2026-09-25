import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Import your custom C-wrapper
from model import NeuralNetwork
print("1. Fetching California Housing Dataset...")
X, y = fetch_california_housing(return_X_y=True)

# The C engine expects a 2D matrix (N, 1) for the targets, not a 1D array (N,)
y = y.reshape(-1, 1)

print("2. Splitting and Scaling Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Networks require normalized inputs (mean 0, variance 1)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scaling targets helps the MSE loss stay stable during C backprop
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

print(f"Data ready. X_train shape: {X_train_scaled.shape}, y_train shape: {y_train_scaled.shape}")

# 3. Initialize and Train the Custom C Framework
print("\n--- Booting C Engine ---")
model = NeuralNetwork(layer_sizes=[8, 256, 128, 64, 1], l2lambda=0.001)

# This drops into C, crunches via OpenBLAS, and returns instantly
model.train(X_train_scaled, y_train_scaled, epochs=1000, lr=0.001, batch_size=64)

# 4. Run Inference and Evaluate
print("\n--- Running Inference ---")
predictions_scaled = model.predict(X_test_scaled)

# Inverse transform to get actual house prices (in $100,000s)
predictions = scaler_y.inverse_transform(predictions_scaled)

# Calculate final metrics
mse = mean_squared_error(y_test, predictions)
print(f"Final Test MSE: {mse:.4f}")
print(f"Final Test RMSE: {np.sqrt(mse):.4f}")

# (Optional) Test the blazing fast binary I/O
model.save("california_model.bin")
