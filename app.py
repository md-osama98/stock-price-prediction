import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

from tensorflow.keras.models import load_model
from preprocess import preprocess_data
from model import CustomAttention
from data_fetch import fetch_latest

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def inverse_close(values, scaler):
    """
    Recover Close Price from scaler
    fitted on multiple features.
    """

    dummy = np.zeros((len(values), 9))

    dummy[:, 3] = values.flatten()

    restored = scaler.inverse_transform(dummy)

    return restored[:, 3]

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📈 AI Stock Price Prediction Dashboard")

st.markdown(
    "### Bi-LSTM + Attention Based Stock Forecasting System"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Controls")

symbol = st.sidebar.selectbox(
    "Select Stock",
    [
        "AAPL",
        "MSFT",
        "TSLA",
        "GOOGL",
        "AMZN",
        "META"
    ]
)

refresh = st.sidebar.button(
    "🔄 Refresh Live Data"
)

# --------------------------------------------------
# DATA REFRESH
# --------------------------------------------------

if refresh:

    with st.spinner(
        "Fetching latest stock data..."
    ):

        try:

            fetch_latest(symbol=symbol)

            time.sleep(1)

            st.sidebar.success(
                "Data Updated Successfully!"
            )

        except Exception as e:

            st.sidebar.error(
                f"Error: {str(e)}"
            )

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_trained_model():

    return load_model(
        "best_stock_model.keras",
        custom_objects={
            "CustomAttention":
            CustomAttention
        },
        compile=False
    )

try:

    model = load_trained_model()

except Exception as e:

    st.error(
        f"Model Loading Failed: {e}"
    )

    st.stop()

# --------------------------------------------------
# PREPROCESS DATA
# --------------------------------------------------

try:

    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler
    ) = preprocess_data()

except Exception as e:

    st.error(
        f"Preprocessing Failed: {e}"
    )

    st.stop()

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

predictions = model.predict(
    X_test,
    verbose=0
)

predictions = inverse_close(
    predictions,
    scaler
)

actual = inverse_close(
    y_test.reshape(-1, 1),
    scaler
)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

rmse = np.sqrt(
    np.mean(
        (actual - predictions) ** 2
    )
)

mae = np.mean(
    np.abs(
        actual - predictions
    )
)

mape = np.mean(
    np.abs(
        (actual - predictions)
        / actual
    )
) * 100

accuracy = max(
    0,
    100 - mape
)

next_price = predictions[-1]

trend = (
    "📈 Bullish"
    if predictions[-1] >
       actual[-1]
    else "📉 Bearish"
)

# --------------------------------------------------
# METRIC CARDS
# --------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Trend",
        trend
    )

with col2:
    st.metric(
        "Next Price",
        f"${next_price:.2f}"
    )

with col3:
    st.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

with col4:
    st.metric(
        "MAE",
        f"{mae:.2f}"
    )

with col5:
    st.metric(
        "Accuracy",
        f"{accuracy:.2f}%"
    )

# --------------------------------------------------
# CHART
# --------------------------------------------------

st.subheader(
    "📊 Actual vs Predicted Prices"
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        y=actual,
        mode="lines",
        name="Actual Price"
    )
)

fig.add_trace(
    go.Scatter(
        y=predictions,
        mode="lines",
        name="Predicted Price"
    )
)

fig.update_layout(
    height=600,
    xaxis_title="Time",
    yaxis_title="Stock Price",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# TABLE
# --------------------------------------------------

st.subheader(
    "📋 Prediction Results"
)

results_df = pd.DataFrame(
    {
        "Actual Price":
            actual,

        "Predicted Price":
            predictions
    }
)

st.dataframe(
    results_df.tail(20),
    use_container_width=True
)

# --------------------------------------------------
# DOWNLOAD CSV
# --------------------------------------------------

csv = results_df.to_csv(
    index=False
)

st.download_button(
    label="📥 Download Predictions",
    data=csv,
    file_name="stock_predictions.csv",
    mime="text/csv"
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.success(
    "🚀 AI Powered Stock Prediction using Bi-LSTM + Attention"
)