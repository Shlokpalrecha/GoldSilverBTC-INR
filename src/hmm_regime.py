# src/hmm_regime.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def fit_hmm(series, n_states=2):
    """Fits a Hidden Markov Model to the Z-score series."""
    X = series.values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
    model.fit(X)
    states = model.predict(X)
    return states, model

def add_regime_to_df(df):
    """Adds a regime label to the dataframe using HMM on Z-score."""
    if 'Zscore' not in df.columns:
        raise KeyError("DataFrame must contain 'Zscore' before adding HMM regimes.")

    series = df['Zscore'].dropna()
    states, model = fit_hmm(series)
    df.loc[series.index, 'Regime'] = states
    return df
