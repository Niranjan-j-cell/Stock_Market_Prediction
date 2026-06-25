import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

# Import training and feature engineering functions from train.py
from train import train_model, engineer_features, download_data

def predict_stock(ticker):
    """
    Predict the next day's stock price for a given ticker.
    Loads a saved model from models/ directory, or trains a new one if it doesn't exist.
    """
    ticker = ticker.upper().strip()
    # Automatically append Indian NSE suffix (.NS) for standard equity tickers
    if ticker and not ticker.startswith('^') and not (ticker.endswith('.NS') or ticker.endswith('.BO')):
        ticker = f"{ticker}.NS"
        
    model_path = f"models/{ticker}_model.joblib"
    
    # 1. Check if model exists, if not train a new one
    if not os.path.exists(model_path):
        print(f"No pre-trained model found for {ticker}. Training model now...")
        try:
            train_model(ticker, period="5y")
        except Exception as e:
            print(f"Error training model for {ticker}: {e}")
            # Return a fallback prediction using simple moving average if training fails
            return get_fallback_prediction(ticker)
            
    # 2. Load the trained model and helper objects
    try:
        checkpoint = joblib.load(model_path)
        model = checkpoint['model']
        scaler = checkpoint['scaler']
        feature_cols = checkpoint['features']
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}. Re-training model...")
        try:
            checkpoint = train_model(ticker, period="5y")
            model = checkpoint['model']
            scaler = checkpoint['scaler']
            feature_cols = checkpoint['features']
        except Exception as e_train:
            print(f"Failed to re-train model: {e_train}")
            return get_fallback_prediction(ticker)

    # 3. Fetch latest data (120 days is needed to compute SMA_50 and ATR)
    try:
        raw_data = download_data(ticker, period="120d")
        # Align index timezone to Indian Standard Time (IST) if timezone aware
        if raw_data.index.tz is not None:
            raw_data.index = raw_data.index.tz_convert('Asia/Kolkata')
    except Exception as e:
        print(f"Error downloading recent data for prediction: {e}")
        return get_fallback_prediction(ticker)
        
    # Inject latest live price from yfinance fast_info for real-time accuracy
    try:
        ticker_obj = yf.Ticker(ticker)
        fast_info = ticker_obj.fast_info
        live_price = float(fast_info.last_price)
        
        if live_price > 0:
            last_date = raw_data.index[-1].date()
            tz = raw_data.index.tz
            today = pd.Timestamp.now(tz=tz).date() if tz else pd.Timestamp.now().date()
            
            if today > last_date:
                # Append a new row for today with the live price
                today_timestamp = pd.Timestamp(today, tz=tz) if tz else pd.Timestamp(today)
                new_row = pd.DataFrame({
                    'Open': [live_price],
                    'High': [live_price],
                    'Low': [live_price],
                    'Close': [live_price],
                    'Adj Close': [live_price],
                    'Volume': [0]
                }, index=[today_timestamp])
                raw_data = pd.concat([raw_data, new_row])
                print(f"Injected live price ₹{live_price:.2f} as a new row for {today}")
            elif today == last_date:
                # Update today's last recorded candle with the live price
                raw_data.loc[raw_data.index[-1], 'Close'] = live_price
                raw_data.loc[raw_data.index[-1], 'Adj Close'] = live_price
                print(f"Updated today's close price with live price ₹{live_price:.2f}")
    except Exception as e:
        print(f"Warning: Could not inject live price, using historical daily close: {e}")
        
    # 4. Engineer features (is_training=False ensures we keep the last row)
    try:
        data_features = engineer_features(raw_data, is_training=False)
        
        if data_features.empty:
            print("Feature dataframe is empty after feature engineering.")
            return get_fallback_prediction(ticker)
            
        # Get the most recent row of features
        latest_row = data_features.iloc[-1:]
        
        # Verify all required features exist in the dataframe
        missing_features = [col for col in feature_cols if col not in latest_row.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns for prediction: {missing_features}")
            
        # Extract features and scale
        X_latest = latest_row[feature_cols]
        X_latest_scaled = scaler.transform(X_latest)
        
        # Predict ratio and multiply by the last close price to get next close price
        predicted_ratio = model.predict(X_latest_scaled)[0]
        last_close = float(raw_data['Close'].iloc[-1])
        predicted_price = last_close * predicted_ratio
        return float(predicted_price)
        
    except Exception as e:
        print(f"Error preparing features or predicting: {e}")
        return get_fallback_prediction(ticker)

def get_fallback_prediction(ticker):
    """
    Fallback method to predict next stock price if the machine learning model fails.
    Uses the last close price and a simple short-term momentum factor.
    """
    print("Using fallback prediction method...")
    try:
        data = yf.download(ticker, period="10d")
        if data.empty:
            return 0.0
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        last_close = float(data['Close'].iloc[-1])
        # Calculate a simple 3-day momentum
        if len(data) >= 3:
            momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-3]) - 1.0
            # Cap momentum effect to +/- 2% for safety
            momentum = max(min(momentum, 0.02), -0.02)
            predicted = last_close * (1.0 + momentum)
        else:
            predicted = last_close
        return predicted
    except Exception as e:
        print(f"Fallback prediction failed: {e}")
        return 0.0
