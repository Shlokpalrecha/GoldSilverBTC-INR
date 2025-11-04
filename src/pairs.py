# src/pairs.py
import pandas as pd
import statsmodels.api as sm

def fit_spread_and_z(df):
    """Fits OLS regression between Gold and Silver, then computes the Z-score."""
    if not {'Gold', 'Silver'}.issubset(df.columns):
        raise KeyError("DataFrame must contain 'Gold' and 'Silver' columns.")

    # Linear regression (Gold ~ Silver)
    x = sm.add_constant(df['Silver'])
    y = df['Gold']
    model = sm.OLS(y, x).fit()

    df['Spread'] = y - model.predict(x)
    df['Zscore'] = (df['Spread'] - df['Spread'].mean()) / df['Spread'].std()
    return df, model
