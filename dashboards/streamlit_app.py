import streamlit as st
import pandas as pd
import plotly.express as px
import os, sys, time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.api import full_pipeline

st.set_page_config(page_title="AI Quant Dashboard", page_icon="💰", layout="wide")
st.title("🤖 AI-Powered Gold–Silver–Bitcoin (INR) Dashboard")

placeholder = st.empty()
while True:
    with placeholder.container():
        try:
            df, forecast = full_pipeline()
            st.success("✅ Pipeline executed successfully.")

            latest = df.iloc[-1]
            col1, col2, col3 = st.columns(3)
            col1.metric("Gold (INR/kg)", f"{latest['Gold']:,.2f}")
            col2.metric("Silver (INR/kg)", f"{latest['Silver']:,.2f}")
            col3.metric("BTC (INR)", f"{latest['Bitcoin']:,.2f}")

            st.subheader("📊 Spread and Regimes")
            st.line_chart(df['Spread'])

            st.subheader("🔮 Prophet Forecast (Gold)")
            st.line_chart(forecast.set_index('ds')['yhat'])
        except Exception as e:
            st.error(f"❌ {e}")
        time.sleep(120)
        st.rerun()
