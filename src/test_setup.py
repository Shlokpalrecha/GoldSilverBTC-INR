import pandas as pd, yfinance as yf
print("Environment OK")

# quick fetch test (daily)
df = yf.download(["GOLDBEES.NS","SILVERETF.NS","BTC-INR"], period="5d")
print(df['Adj Close'].tail())
