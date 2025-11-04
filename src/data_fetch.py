# src/data_fetch.py
import pandas as pd
import yfinance as yf
import investpy

def fetch_data():
    """Fetch live data for Gold (INR/kg), Silver (INR/kg), and Bitcoin (INR)."""
    try:
        # --- Gold and Silver from Investpy ---
        gold = investpy.get_commodity_historical_data(
            commodity='gold',
            from_date='01/10/2024',
            to_date='04/11/2025'
        )
        silver = investpy.get_commodity_historical_data(
            commodity='silver',
            from_date='01/10/2024',
            to_date='04/11/2025'
        )

        gold = gold.rename(columns={'Close': 'Gold'})[['Gold']]
        silver = silver.rename(columns={'Close': 'Silver'})[['Silver']]

        # Convert ounces to kilograms
        gold['Gold'] *= 32.1507
        silver['Silver'] *= 32.1507

        # --- Bitcoin from Yahoo Finance ---
        btc = yf.download("BTC-INR", period="1y", interval="1d", progress=False)
        btc = btc[['Close']].rename(columns={'Close': 'BTC-INR'})

        # Combine all
        df = pd.concat([gold, silver, btc], axis=1).dropna()
        return df

    except Exception as e:
        print(f"⚠️ Error fetching live data: {e}")
        return pd.DataFrame()
