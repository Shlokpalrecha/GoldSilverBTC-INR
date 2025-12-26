# src/data_fetch.py
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

METALPRICE_API_KEY = "f17d668dca7049d639045209308f359b"

def fetch_data(days=90):
    """
    Fetch live historical data for Gold, Silver (INR/kg), and BTC-INR.
    Uses yfinance for historical prices + live API for latest rates.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # --- Fetch historical BTC-INR from yfinance ---
        print(f"🔗 Fetching {days} days of BTC-INR history from Yahoo Finance...")
        btc = yf.download("BTC-INR", start=start_date, end=end_date, progress=False, auto_adjust=True)
        if btc.empty:
            raise ValueError("No BTC-INR data from yfinance")
        # Handle multi-level columns from yfinance
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = btc.columns.get_level_values(0)
        btc = btc[['Close']].copy()
        btc.columns = ['Bitcoin']
        btc.index = pd.to_datetime(btc.index).tz_localize(None)
        
        # --- Fetch historical Gold & Silver (using GC=F and SI=F in USD, convert to INR) ---
        print("🔗 Fetching Gold & Silver futures from Yahoo Finance...")
        gold_usd = yf.download("GC=F", start=start_date, end=end_date, progress=False, auto_adjust=True)
        silver_usd = yf.download("SI=F", start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        # Get USD-INR rate
        print("🔗 Fetching USD-INR exchange rate...")
        usdinr = yf.download("INR=X", start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if gold_usd.empty or silver_usd.empty or usdinr.empty:
            raise ValueError("Missing Gold/Silver/USDINR data from yfinance")
        
        # Handle multi-level columns
        for data in [gold_usd, silver_usd, usdinr]:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
        
        gold_usd = gold_usd[['Close']].copy()
        gold_usd.columns = ['Gold_USD']
        gold_usd.index = pd.to_datetime(gold_usd.index).tz_localize(None)
        
        silver_usd = silver_usd[['Close']].copy()
        silver_usd.columns = ['Silver_USD']
        silver_usd.index = pd.to_datetime(silver_usd.index).tz_localize(None)
        
        usdinr = usdinr[['Close']].copy()
        usdinr.columns = ['USDINR']
        usdinr.index = pd.to_datetime(usdinr.index).tz_localize(None)
        
        # Merge all data
        df = btc.join(gold_usd, how='outer')
        df = df.join(silver_usd, how='outer')
        df = df.join(usdinr, how='outer')
        df = df.ffill().dropna()
        
        # Convert to INR per kg (1 troy oz = 0.0311035 kg)
        df['Gold'] = df['Gold_USD'].values * df['USDINR'].values / 0.0311035
        df['Silver'] = df['Silver_USD'].values * df['USDINR'].values / 0.0311035
        
        # Keep only needed columns
        df = df[['Gold', 'Silver', 'Bitcoin']]
        
        # --- Update with latest live prices ---
        print("\n🔗 Fetching latest live prices from MetalpriceAPI...")
        metal_url = f"https://api.metalpriceapi.com/v1/latest?api_key={METALPRICE_API_KEY}&base=INR&currencies=XAU,XAG"
        metals = requests.get(metal_url).json()
        
        if "rates" in metals:
            gold_inr_kg = (1 / metals["rates"]["XAU"]) * 32.1507
            silver_inr_kg = (1 / metals["rates"]["XAG"]) * 32.1507
            print(f"✅ Live Gold (INR/kg): {gold_inr_kg:,.2f}")
            print(f"✅ Live Silver (INR/kg): {silver_inr_kg:,.2f}")
        
        print("🔗 Fetching latest BTC-INR from CoinGecko...")
        btc_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=inr"
        btc_resp = requests.get(btc_url).json()
        btc_inr = btc_resp["bitcoin"]["inr"]
        print(f"✅ Live BTC-INR: ₹{btc_inr:,.2f}")
        
        # Append today's live data
        today = pd.Timestamp(datetime.now().date())
        if today not in df.index:
            live_row = pd.DataFrame({
                'Gold': [gold_inr_kg if "rates" in metals else df['Gold'].iloc[-1]],
                'Silver': [silver_inr_kg if "rates" in metals else df['Silver'].iloc[-1]],
                'Bitcoin': [btc_inr]
            }, index=[today])
            df = pd.concat([df, live_row])
        
        df = df.sort_index()
        print(f"\n📊 Loaded {len(df)} days of data from {df.index[0].date()} to {df.index[-1].date()}")
        
        # Cache the data
        df.to_csv("data/cache/live_data.csv")
        
        return df

    except Exception as e:
        print(f"\n❌ Error fetching live data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


if __name__ == "__main__":
    df = fetch_data()
    print(df.tail())
