import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from preprocess import preprocess_data
from model import build_model

# -----------------------------------
# Load Data
# -----------------------------------

X_train, y_train, X_test, y_test, scaler = preprocess_data()

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# -----------------------------------
# Build Model
# -----------------------------------

model = build_model(
    (X_train.shape[1], X_train.shape[2])
)

model.summary()

# -----------------------------------
# Callbacks
# -----------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath="best_stock_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# -----------------------------------
# Training
# -----------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ],
    verbose=1
)

# -----------------------------------
# Save Final Model
# -----------------------------------

model.save("stock_lstm_attention.keras")

print("\nModel saved successfully")

# -----------------------------------
# Predictions
# -----------------------------------

predictions = model.predict(
    X_test,
    verbose=0
)

# -----------------------------------
# Evaluation Metrics
# -----------------------------------

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

mae = mean_absolute_error(
    y_test,
    predictions
)

print("\nModel Evaluation")
print("-----------------------")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")

# -----------------------------------
# Training Loss Graph
# -----------------------------------

plt.figure(figsize=(10,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()