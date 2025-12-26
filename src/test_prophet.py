# test_prophet.py
from prophet import Prophet
import pandas as pd

# Dummy gold price data for 30 days
data = pd.DataFrame({
    "ds": pd.date_range("2025-10-01", periods=30),
    "y": [65000 + i * 10 for i in range(30)]
})

model = Prophet()
model.fit(data)
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

forecast[['ds', 'yhat']].to_csv("data/cache/test_forecast.csv", index=False)
print("✅ Prophet forecast generated successfully!")
