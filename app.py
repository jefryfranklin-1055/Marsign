import streamlit as st
import pandas as pd
import datetime as dt
import time
import yfinance as yf

st.title("Nifty Options Signal Analyzer")
st.write("Click 'Update Signal' for fresh data. For educational use only.")

# Session state
if "last_run" not in st.session_state:
    st.session_state.last_run = 0
    st.session_state.signal = "Click Update to start"
    st.session_state.details = {}

def run_analysis():
    st.session_state.last_run = time.time()
    try:
        st.info("Fetching data...")
        nifty = yf.Ticker("^NSEI")
        vix_ticker = yf.Ticker("^INDIAVIX")
        
        spot_hist = nifty.history(period="1d", timeout=30)  # 30s timeout
        if spot_hist.empty:
            raise ValueError("No spot data (market closed?)")
        spot = spot_hist['Close'].iloc[-1]
        
        vix_hist = vix_ticker.history(period="1d", timeout=30)
        if vix_hist.empty:
            raise ValueError("No VIX data")
        vix = vix_hist['Close'].iloc[-1]
        
        # Historical 5-min
        hist = nifty.history(period="1d", interval="5m", timeout=30)
        df = hist.reset_index()[['Datetime', 'Open', 'High', 'Low', 'Close']]
        df.rename(columns={'Datetime': 'date'}, inplace=True)
        
        # OI neutral (disabled for cloud)
        ce_oi = pe_oi = 0
        pcr = 1.0
        bullish_oi = bearish_oi = False
        expiry = atm_strike = "N/A"
        st.warning("OI disabled (use local/Indian host for NSE). Neutral scores.")
        
        # Candlesticks if data available
        if len(df) >= 3:
            last_candle = df.iloc[-1]
            bullish_candle = last_candle['Close'] > last_candle['Open']
            
            prev_candle = df.iloc[-2]
            bullish_engulfing = (prev_candle['Close'] < prev_candle['Open']) and \
                                (last_candle['Open'] < prev_candle['Close']) and \
                                (last_candle['Close'] > prev_candle['Open'])
            
            up_trend = (df.iloc[-1]['Close'] > df.iloc[-2]['Close']) and \
                       (df.iloc[-2]['Close'] > df.iloc[-3]['Close'])
        else:
            bullish_candle = bullish_engulfing = up_trend = False
        
        low_vol = vix < 15
        high_vol = vix > 20
        
        bullish_score = int(bullish_candle) + int(bullish_engulfing) + int(up_trend) + int(low_vol) + int(bullish_oi)
        bearish_score = int(not bullish_candle) + int(not bullish_engulfing) + int(not up_trend) + int(high_vol) + int(bearish_oi)
        
        if bullish_score > bearish_score:
            signal = "Buy CE - Market likely to go UP"
        elif bearish_score > bullish_score:
            signal = "Buy PE - Market likely to go DOWN"
        else:
            signal = "Neutral - No clear signal"
        
        st.session_state.signal = signal
        st.session_state.details = {
            "Spot": f"{spot:.2f}",
            "VIX": f"{vix:.2f}",
            "Expiry": expiry,
            "ATM Strike": atm_strike,
            "CE OI": ce_oi,
            "PE OI": pe_oi,
            "PCR": round(pcr, 2),
            "Bullish Score": bullish_score,
            "Bearish Score": bearish_score
        }
        st.success("Update complete!")
    
    except Exception as e:
        st.session_state.signal = f"Error: {str(e)}"
        st.error(f"Details: {e}")

# Update button (manual to avoid loops)
if st.button("Update Signal (every 5 min recommended)"):
    run_analysis()

# Display
st.header("Latest Signal")
st.subheader(st.session_state.signal)

st.header("Details")
for key, value in st.session_state.details.items():
    st.write(f"**{key}:** {value}")

st.write(f"**Last updated:** {dt.datetime.fromtimestamp(st.session_state.last_run).strftime('%Y-%m-%d %H:%M:%S IST') if st.session_state.last_run > 0 else 'Not run yet'}")
