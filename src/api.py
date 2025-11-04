# src/api.py
import os
import pandas as pd
from src.data_fetch import fetch_data
from src.pairs import fit_spread_and_z
from src.hmm_regime import add_regime_to_df


def full_pipeline():
    """Complete live data pipeline for Gold, Silver, and BTC-INR with ML features."""
    print("🚀 Starting full pipeline...")

    # Step 1: Fetch live data
    df = fetch_data()
    if df is None or df.empty:
        raise ValueError("⚠️ No data fetched. Check your internet connection or API tickers.")

    # Step 2: Fit spread and z-score
    try:
        df, model = fit_spread_and_z(df)
    except Exception as e:
        raise ValueError(f"Error fitting spread/z-score: {e}")

    # Step 3: Add Hidden Markov Model regimes
    try:
        df = add_regime_to_df(df)
    except Exception as e:
        print(f"⚠️ Warning: Could not add HMM regimes - {e}")

    # Step 4: Cache data
    os.makedirs("data/cache", exist_ok=True)
    df.to_csv("data/cache/full_pipeline_output.csv")

    print("✅ Full pipeline completed successfully!")
    return df
