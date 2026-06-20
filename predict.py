import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocess import preprocess_data
from model import CustomAttention   #  IMPORTANT

# Load data
X_train, y_train, X_test, y_test, scaler = preprocess_data()

# 🔹 LOAD MODEL WITH CUSTOM OBJECT
model = load_model(
    "stock_lstm_attention.h5",
    custom_objects={"CustomAttention": CustomAttention},
    compile=False
)

# Predict
predictions = model.predict(X_test)

# Inverse scale
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Trend detection
trend = ["UP" if predictions[i] > actual[i-1] else "DOWN"
         for i in range(1, len(predictions))]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title("Stock Price Prediction using LSTM + Attention")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

print("Latest Trend:", trend[-1])
