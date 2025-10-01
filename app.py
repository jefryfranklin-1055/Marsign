import streamlit as st
import pandas as pd
import datetime as dt
import time
import yfinance as yf

# Streamlit app setup
st.title("Nifty Options Signal Analyzer")
st.write("Fetches data from yfinance (NSE fallback disabled due to blocks) and analyzes every ~5 minutes. Signal: Buy CE (up) or PE (down). For educational use only.")

# Session state for timing
if "last_run" not in st.session_state:
    st.session_state.last_run = time.time()  # Initialize to now
    st.session_state.signal = "Initializing..."
    st.session_state.details = {}

# Main function to run analysis
def run_analysis():
    st.session_state.last_run = time.time()  # Always update timestamp
    try:
        # Fetch spot and VIX via yfinance (primary, no NSE needed)
        st.info("Fetching via yfinance...")
        nifty = yf.Ticker("^NSEI")
        vix_ticker = yf.Ticker("^INDIAVIX")
        
        spot_hist = nifty.history(period="1d")
        if spot_hist.empty:
            raise ValueError("No spot data available (market closed?)")
        spot = spot_hist['Close'].iloc[-1]
        
        vix_hist = vix_ticker.history(period="1d")
        if vix_hist.empty:
            raise ValueError("No VIX data available")
        vix = vix_hist['Close'].iloc[-1]
        
        st.success(f"Spot: {spot:.2f}, VIX: {vix:.2f}")
        
        # Historical 5-min data via yfinance
        hist = nifty.history(period="1d", interval="5m")
        df = hist.reset_index()[['Datetime', 'Open', 'High', 'Low', 'Close']]
        df.rename(columns={'Datetime': 'date'}, inplace=True)
        
        # OI fallback: Disabled due to NSE blocks; neutral impact
        ce_oi = 0
        pe_oi = 0
        expiry = "N/A"
        atm_strike = round(spot / 50) * 50
        pcr = 1.0  # Neutral PCR
        bullish_oi = False
        bearish_oi = False
        st.warning("OI analysis disabled (NSE cloud blocks). Using neutral scores.")
        
        # Candlestick and pattern analysis (if enough data)
        if len(df) >= 3:
            last_candle = df.iloc[-1]
            bullish_candle = last_candle['Close'] > last_candle['Open']
            
            prev_candle = df.iloc[-2]
            bullish_engulfing = (prev_candle['Close'] < prev_candle['Open']) and \
                                (last_candle['Open'] < prev_candle['Close']) and \
                                (last_candle['Close'] > prev_candle['Open'])
            
            up_trend = (df.iloc[-1]['Close'] > df.iloc[-2]['Close']) and \
                       (df.iloc[-2]['Close'] > df.iloc[-3]['Close'])
            st.success("Candlestick analysis complete.")
        else:
            st.warning(f"Insufficient 5-min data ({len(df)} bars). Skipping candlesticks; neutral scores.")
            bullish_candle = False
            bullish_engulfing = False
            up_trend = False
        
        low_vol = vix < 15
        high_vol = vix > 20
        
        # Scores and signal (OI neutral, adjust for missing candles)
        bullish_score = int(bullish_candle) + int(bullish_engulfing) + int(up_trend) + int(low_vol) + int(bullish_oi)
        bearish_score = int(not bullish_candle) + int(not bullish_engulfing) + int(not up_trend) + int(high_vol) + int(bearish_oi)
        
        if bullish_score > bearish_score:
            signal = "Buy CE - Market likely to go UP"
        elif bearish_score > bullish_score:
            signal = "Buy PE - Market likely to go DOWN"
        else:
            signal = "Neutral - No clear signal"
        
        # Store results
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
    
    except Exception as e:
        st.session_state.signal = f"Error: {str(e)}"
        st.error(f"Detailed error: {e}")

# Run analysis if 5 mins passed or first time
current_time = time.time()
if current_time - st.session_state.last_run >= 300:
    run_analysis()

# Display results
st.header("Latest Signal")
st.subheader(st.session_state.signal)

st.header("Details")
for key, value in st.session_state.details.items():
    st.write(f"**{key}:** {value}")

st.write(f"**Last updated:** {dt.datetime.fromtimestamp(st.session_state.last_run).strftime('%Y-%m-%d %H:%M:%S IST')}")
st.write("Refreshing in ~5 minutes... (Auto-updates on new data)")

# Auto-rerun after delay
time.sleep(1)
if current_time - st.session_state.last_run >= 300:
    st.rerun()
