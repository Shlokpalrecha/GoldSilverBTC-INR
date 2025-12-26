# src/ml_models.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from hmmlearn.hmm import GaussianHMM
from prophet import Prophet

def train_hmm(df, column='Spread'):
    """Hidden Markov Model for market regime classification."""
    series = df[column].dropna().values.reshape(-1, 1)
    model = GaussianHMM(n_components=2, n_iter=200, covariance_type="diag")
    model.fit(series)
    df['Regime'] = model.predict(series)
    return df, model

def forecast_prophet(df, column='Gold', days=10):
    """Forecast next `days` values using Prophet."""
    temp = df[[column]].reset_index()
    temp.columns = ['ds', 'y']
    model = Prophet()
    model.fit(temp)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def train_lstm(df, target_col='Spread', lookback=10):
    """Train a simple LSTM model on spread data."""
    data = df[target_col].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=25, batch_size=16, verbose=0)
    
    df['LSTM_Pred'] = np.nan
    df.iloc[-len(y):, df.columns.get_loc('LSTM_Pred')] = scaler.inverse_transform(model.predict(X))
    
    return df, model, scaler
