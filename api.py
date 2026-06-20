from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np

from preprocess import preprocess_data
from model import CustomAttention
from data_fetch import fetch_latest

# ---------------------------------------
# FastAPI App
# ---------------------------------------

app = FastAPI(
    title="AI Stock Prediction API",
    description="Bi-LSTM + Attention Stock Forecasting API",
    version="2.0"
)

# ---------------------------------------
# Load Model Once
# ---------------------------------------

try:

    model = load_model(
        "best_stock_model.keras",
        custom_objects={
            "CustomAttention": CustomAttention
        },
        compile=False
    )

except Exception as e:

    print(f"Model loading failed: {e}")
    model = None

# ---------------------------------------
# Helper Function
# ---------------------------------------

def inverse_close(predictions, scaler):

    dummy = np.zeros(
        (len(predictions), 9)
    )

    dummy[:, 3] = predictions.flatten()

    restored = scaler.inverse_transform(dummy)

    return restored[:, 3]

# ---------------------------------------
# Root Endpoint
# ---------------------------------------

@app.get("/")
def home():

    return {
        "message": "AI Stock Prediction API Running",
        "version": "2.0"
    }

# ---------------------------------------
# Health Check
# ---------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# ---------------------------------------
# Prediction Endpoint
# ---------------------------------------

@app.get("/predict")
def predict_stock(symbol: str = "AAPL"):

    try:

        fetch_latest(symbol=symbol)

        (
            X_train,
            y_train,
            X_test,
            y_test,
            scaler
        ) = preprocess_data()

        preds = model.predict(
            X_test,
            verbose=0
        )

        pred_prices = inverse_close(
            preds,
            scaler
        )

        actual_prices = inverse_close(
            y_test.reshape(-1, 1),
            scaler
        )

        trend = (
            "UP"
            if pred_prices[-1] >
               actual_prices[-1]
            else "DOWN"
        )

        return {
            "symbol": symbol,
            "latest_prediction":
                float(pred_prices[-1]),
            "actual_price":
                float(actual_prices[-1]),
            "trend": trend
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ---------------------------------------
# Metrics Endpoint
# ---------------------------------------

@app.get("/metrics")
def metrics(symbol: str = "AAPL"):

    try:

        fetch_latest(symbol=symbol)

        (
            X_train,
            y_train,
            X_test,
            y_test,
            scaler
        ) = preprocess_data()

        preds = model.predict(
            X_test,
            verbose=0
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                preds
            )
        )

        mae = mean_absolute_error(
            y_test,
            preds
        )

        mape = np.mean(
            np.abs(
                (y_test - preds.flatten())
                / y_test
            )
        ) * 100

        accuracy = max(
            0,
            100 - mape
        )

        return {
            "symbol": symbol,
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "mape": round(mape, 2),
            "accuracy": round(
                accuracy,
                2
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ---------------------------------------
# 7-Day Forecast
# ---------------------------------------

@app.get("/forecast7")
def forecast7(symbol: str = "AAPL"):

    try:

        fetch_latest(symbol=symbol)

        (
            X_train,
            y_train,
            X_test,
            y_test,
            scaler
        ) = preprocess_data()

        current_sequence = (
            X_test[-1].copy()
        )

        future = []

        for _ in range(7):

            pred = model.predict(
                current_sequence.reshape(
                    1,
                    current_sequence.shape[0],
                    current_sequence.shape[1]
                ),
                verbose=0
            )

            future.append(
                pred[0][0]
            )

            new_row = (
                current_sequence[-1].copy()
            )

            new_row[3] = pred[0][0]

            current_sequence = np.vstack(
                [
                    current_sequence[1:],
                    new_row
                ]
            )

        forecast_prices = inverse_close(
            np.array(future).reshape(-1, 1),
            scaler
        )

        return {
            "symbol": symbol,
            "forecast_7_days":
                [
                    round(float(x), 2)
                    for x in forecast_prices
                ]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )