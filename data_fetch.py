import yfinance as yf
import pandas as pd
import time
import logging
from datetime import datetime

# ---------------------------------------
# Logging Configuration
# ---------------------------------------

logging.basicConfig(
    filename="stock_fetch.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------
# Fetch Latest Stock Data
# ---------------------------------------

def fetch_latest(
    symbol="AAPL",
    period="2y",
    interval="1d",
    retries=3
):

    print(f"\nFetching latest data for {symbol}...")

    for attempt in range(retries):

        try:

            df = yf.download(
                tickers=symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )

            if df.empty:
                raise ValueError(
                    "Yahoo Finance returned empty data."
                )

            df.reset_index(inplace=True)

            # Ensure required columns exist
            required_cols = [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]

            missing = [
                col for col in required_cols
                if col not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing columns: {missing}"
                )

            df.to_csv(
                "stock_data.csv",
                index=False
            )

            print(
                f"Data updated successfully at "
                f"{datetime.now()}"
            )

            logging.info(
                f"{symbol} data fetched successfully."
            )

            return df

        except Exception as e:

            logging.error(
                f"Attempt {attempt+1} failed: {e}"
            )

            print(
                f"Attempt {attempt+1} failed."
            )

            if attempt < retries - 1:
                time.sleep(2)

    print("Failed to fetch data.")

    return None

# ---------------------------------------
# Multiple Stocks Fetch
# ---------------------------------------

def fetch_multiple_stocks(symbols):

    all_data = {}

    for symbol in symbols:

        data = fetch_latest(symbol)

        if data is not None:
            all_data[symbol] = data

    return all_data

# ---------------------------------------
# Main
# ---------------------------------------

if __name__ == "__main__":

    fetch_latest(
        symbol="AAPL",
        period="2y",
        interval="1d"
    )