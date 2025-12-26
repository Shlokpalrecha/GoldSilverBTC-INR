# src/api.py
import os
from src.data_fetch import fetch_data
from src.pairs import fit_spread_and_z
from src.features import compute_features
from src.ml_models import train_hmm, train_lstm, forecast_prophet


def full_pipeline():
    """
    Complete live data + ML pipeline for Gold, Silver, and BTC-INR.
    Fetches data → computes spread/z-score → adds engineered features →
    trains ML models (HMM, LSTM, Prophet) → saves outputs.
    """
    print("🚀 Starting full AI-driven pipeline...")

    # Step 1 — Fetch live data
    df = fetch_data()
    if df is None or df.empty:
        raise ValueError("⚠️ No data fetched. Check your internet connection or API tickers.")

    # Step 2 — Statistical pair analysis
    df, model = fit_spread_and_z(df)

    # Step 3 — Feature engineering
    df = compute_features(df)

    # Step 4 — Hidden Markov Model for regimes
    try:
        df, hmm_model = train_hmm(df)
    except Exception as e:
        print(f"⚠️ HMM training warning: {e}")

    # Step 5 — LSTM spread prediction
    try:
        df, lstm_model, scaler = train_lstm(df)
    except Exception as e:
        print(f"⚠️ LSTM training warning: {e}")

    # Step 6 — Prophet forecast for Gold
    try:
        forecast = forecast_prophet(df, 'Gold', days=7)
    except Exception as e:
        print(f"⚠️ Prophet forecast warning: {e}")
        forecast = None

    # Step 7 — Cache results
    os.makedirs("data/cache", exist_ok=True)
    df.to_csv("data/cache/full_pipeline_output.csv", index=True)
    if forecast is not None:
        forecast.to_csv("data/cache/gold_forecast.csv", index=False)

    print("✅ AI pipeline completed successfully!")
    return df, forecast
