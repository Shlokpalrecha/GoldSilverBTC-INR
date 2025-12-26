# src/features.py
import pandas as pd

def compute_features(df):
    df = df.copy()
    df['Gold_Returns'] = df['Gold'].pct_change()
    df['Silver_Returns'] = df['Silver'].pct_change()
    df['BTC_Returns'] = df['Bitcoin'].pct_change()
    df['Volatility'] = df['Spread'].rolling(10).std()
    df['Corr_GS'] = df['Gold_Returns'].rolling(20).corr(df['Silver_Returns'])
    return df.dropna()
