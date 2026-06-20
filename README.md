AI-Powered Stock Market Forecasting System
Overview

AI-Powered Stock Market Forecasting System is an end-to-end Machine Learning application designed to predict stock price trends using Deep Learning techniques. The project leverages a Bi-LSTM (Bidirectional Long Short-Term Memory) network with a custom Attention Mechanism to learn complex temporal patterns from historical stock market data.

The system fetches real-time stock data from Yahoo Finance, performs feature engineering using technical indicators, and provides stock price predictions through an interactive Streamlit dashboard and FastAPI backend.

Features
Real-time stock data acquisition using Yahoo Finance API
Bi-LSTM + Attention Mechanism for time-series forecasting
Technical Indicators:
RSI (Relative Strength Index)
MACD (Moving Average Convergence Divergence)
SMA (Simple Moving Average)
EMA (Exponential Moving Average)
Interactive Streamlit Dashboard
FastAPI REST API for model serving
Trend Detection (Bullish / Bearish)
Performance Metrics:
RMSE
MAE
MAPE
Accuracy
Interactive Plotly Visualizations
Download Prediction Results as CSV
Modular and Scalable Architecture
Technology Stack
Programming Language
Python
Machine Learning & Deep Learning
TensorFlow
Keras
Scikit-Learn
NumPy
Pandas
Backend
FastAPI
Uvicorn
Frontend & Visualization
Streamlit
Plotly
Matplotlib
Data Source
Yahoo Finance (yfinance)
Development Tools
Git
GitHub
VS Code
Project Architecture

Stock Market Data
↓
Yahoo Finance API
↓
Data Preprocessing
↓
Feature Engineering
(Open, High, Low, Close, Volume)
↓
Technical Indicators
(RSI, MACD, SMA, EMA)
↓
Bi-LSTM + Attention Model
↓
Prediction Engine
↓
FastAPI Backend
↓
Streamlit Dashboard

Project Structure
stock_price_prediction/
│
├── app.py
├── api.py
├── model.py
├── preprocess.py
├── data_fetch.py
├── train.py
│
├── stock_data.csv
├── best_stock_model.keras
│
├── requirements.txt
├── README.md
│
└── screenshots/
Installation
Clone Repository
git clone https://github.com/your-username/AI-Stock-Market-Forecasting-System.git
cd AI-Stock-Market-Forecasting-System
Create Virtual Environment
python -m venv venv

Activate:

Windows

venv\Scripts\activate

Linux/Mac

source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Training the Model
python train.py

The trained model will be saved as:

best_stock_model.keras
Run Streamlit Dashboard
streamlit run app.py
Run FastAPI Server
uvicorn api:app --reload

API Documentation:

http://127.0.0.1:8000/docs
Model Performance Metrics

The model is evaluated using:

Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
Mean Absolute Percentage Error (MAPE)
Prediction Accuracy
Future Enhancements
7-Day Stock Price Forecasting
News Sentiment Analysis using FinBERT
Buy / Sell / Hold Recommendation System
Multi-Stock Portfolio Analysis
Docker Deployment
Cloud Deployment (AWS/Azure/GCP)
Explainable AI using SHAP
