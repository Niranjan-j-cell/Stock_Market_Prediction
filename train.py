import os
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def download_data(ticker, period="5y"):
    """Download historical stock data and ensure clean column names."""
    print(f"Downloading data for {ticker} (period: {period})...")
    data = yf.download(ticker, period=period)
    
    if data.empty:
        raise ValueError(f"No data found for ticker symbol: {ticker}")
        
    # Handle MultiIndex columns that newer yfinance versions might return
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    return data

def engineer_features(df, is_training=True):
    """Engineer technical indicators and lag features for prediction."""
    df = df.copy()
    
    # Cast essential columns to float
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    # 1. Moving Averages & Trend
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 2. Volatility (Bollinger Bands & ATR)
    df['BB_Middle'] = df['SMA_20']
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # 3. Momentum (RSI & Stochastic Oscillator)
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator (%K and %D)
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-9))
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
    
    # Returns & Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
    
    # 4. Lag Features
    for i in range(1, 6):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        
    if is_training:
        # Target: Next day's Close price ratio and absolute Close price (for evaluation)
        df['Target_Ratio'] = df['Close'].shift(-1) / df['Close']
        df['Target_Close'] = df['Close'].shift(-1)
        # Drop all rows with NaN (including the last row since shift(-1) is NaN)
        df = df.dropna()
    else:
        # For prediction, we need to preserve the last row which does NOT have a target yet
        feature_cols = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'ATR', 'RSI', 'Stochastic_K', 'Stochastic_D',
            'Daily_Return', 'Volatility_5', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
            'Close_Lag_4', 'Close_Lag_5'
        ]
        df = df.dropna(subset=feature_cols)
        
    return df

def train_model(ticker, period="5y"):
    """Train and evaluate the Random Forest model, then save it."""
    # Download and engineer features
    raw_data = download_data(ticker, period)
    data = engineer_features(raw_data, is_training=True)
    
    if len(data) < 100:
        raise ValueError(f"Insufficient data to train the model ({len(data)} rows after filtering).")
        
    # Define features and target
    feature_cols = [
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
        'BB_Upper', 'BB_Lower', 'BB_Middle', 'ATR', 'RSI', 'Stochastic_K', 'Stochastic_D',
        'Daily_Return', 'Volatility_5', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
        'Close_Lag_4', 'Close_Lag_5'
    ]
    
    X = data[feature_cols]
    y = data['Target_Ratio']
    y_abs = data['Target_Close']
    
    # Time-series split: train on the first 80%, test on the last 20%
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    y_train_abs, y_test_abs = y_abs.iloc[:split_idx], y_abs.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Regressor
    print(f"Training Random Forest Regressor for {ticker}...")
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Predict ratio and reconstruct absolute prices
    y_pred_ratio = model.predict(X_test_scaled)
    
    current_close_test = data.iloc[split_idx:]['Close']
    y_pred_abs = current_close_test * y_pred_ratio
    
    # Evaluate on absolute prices
    mae = mean_absolute_error(y_test_abs, y_pred_abs)
    rmse = np.sqrt(mean_squared_error(y_test_abs, y_pred_abs))
    r2 = r2_score(y_test_abs, y_pred_abs)
    
    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): ₹{mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): ₹{rmse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    
    # Save the trained model, scaler, and features metadata
    os.makedirs('models', exist_ok=True)
    model_path = f"models/{ticker.upper()}_model.joblib"
    checkpoint = {
        'model': model,
        'scaler': scaler,
        'features': feature_cols,
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2}
    }
    joblib.dump(checkpoint, model_path)
    print(f"Model saved successfully to {model_path}\n")
    
    return checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock prediction model.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--period", type=str, default="5y", help="Data period (default: 5y)")
    
    args = parser.parse_args()
    try:
        train_model(args.ticker.upper(), args.period)
    except Exception as e:
        print(f"Error during training: {e}")
