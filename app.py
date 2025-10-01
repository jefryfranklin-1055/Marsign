import streamlit as st
import requests
import json
import pandas as pd
import datetime as dt
import time
import yfinance as yf

# Function to initialize session for NSE (kept for option chain attempt)
def get_nse_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.nseindia.com/',
    })
    session.get('https://www.nseindia.com')
    return session

# Updated: Fetch Nifty spot using yfinance (fallback if NSE fails)
def get_nifty_spot(session):
    try:
        url = 'https://www.nseindia.com/api/quote-equity?symbol=NIFTY%2050'
        response = session.get(url)
        if response.status_code == 200:
            data = response.json()
            return data['priceInfo']['lastPrice']
        else:
            st.warning(f"NSE spot fetch failed (code: {response.status_code}). Using yfinance fallback.")
    except:
        pass
    # Fallback to yfinance
    nifty = yf.Ticker("^NSEI")
    hist = nifty.history(period="1d")
    if not hist.empty:
        return hist['Close'].iloc[-1]
    raise ValueError("Failed to fetch Nifty spot from all sources")

# Updated: Fetch VIX using yfinance (fallback if NSE fails)
def get_vix(session):
    try:
        url = 'https://www.nseindia.com/api/quote-equity?symbol=INDIA%20VIX'
        response = session.get(url)
        if response.status_code == 200:
            data = response.json()
            return data['priceInfo']['lastPrice']
        else:
            st.warning(f"NSE VIX fetch failed (code: {response.status_code}). Using yfinance fallback.")
    except:
        pass
    # Fallback to yfinance
    vix_ticker = yf.Ticker("^INDIAVIX")
    hist = vix_ticker.history(period="1d")
    if not hist.empty:
        return hist['Close'].iloc[-1]
    raise ValueError("Failed to fetch VIX from all sources")

# Function to fetch option chain (keep NSE, but handle failure gracefully)
def get_option_data(session, spot):
    try:
        url = 'https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY'
        response = session.get(url)
        if response.status_code == 200:
            data = response.json()
            records = data['records']
            expiry = records['expiryDates'][0]
            atm_strike = round(spot / 50) * 50
            ce_oi = 0
            pe_oi = 0
            for opt in records['data']:
                if opt['expiryDate'] == expiry and opt['strikePrice'] == atm_strike:
                    if 'CE' in opt:
                        ce_oi = opt['CE']['openInterest']
                    if 'PE' in opt:
                        pe_oi = opt['PE']['openInterest']
            return expiry, atm_strike, ce_oi, pe_oi
        else:
            st.warning(f"Option chain fetch failed (code: {response.status_code}). Disabling OI analysis.")
    except:
        st.warning("Option chain fetch failed. Disabling OI analysis.")
    # Fallback: No OI data
    return None, None, 0, 0

# Streamlit app setup
st.title("Nifty Options Signal Analyzer")
st.write("Fetches data from NSE/yfinance and analyzes every ~5 minutes. Signal: Buy CE (up) or PE (down). For educational use only.")

# Session state for timing
if "last_run" not in st.session_state:
    st.session_state.last_run = 0
    st.session_state.signal = "Initializing..."
    st.session_state.details = {}

# Main function to run analysis
def run_analysis():
    st.session_state.last_run = time.time()  # Update timestamp always, even on partial failure
    try:
        session = get_nse_session()
        
        # Fetch data
        spot = get_nifty_spot(session)
        vix = get_vix(session)
        expiry, atm_strike, ce_oi, pe_oi = get_option_data(session, spot)
        
        # OI Analysis (skip if OI is 0)
        pcr = pe_oi / ce_oi if ce_oi > 0 else 0
        bullish_oi = pcr < 0.5 if ce_oi > 0 else False
        bearish_oi = pcr > 1.0 if ce_oi > 0 else False
        
        # Historical data via yfinance
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="1d", interval="5m")
        df = hist.reset_index()[['Datetime', 'Open', 'High', 'Low', 'Close']]
        df.rename(columns={'Datetime': 'date'}, inplace=True)
        if len(df) < 3:
            raise ValueError("Insufficient historical data")
        
        # Candlestick and pattern analysis
        last_candle = df.iloc[-1]
        bullish_candle = last_candle['Close'] > last_candle['Open']
        
        prev_candle = df.iloc[-2]
        bullish_engulfing = (prev_candle['Close'] < prev_candle['Open']) and \
                            (last_candle['Open'] < prev_candle['Close']) and \
                            (last_candle['Close'] > prev_candle['Open'])
        
        up_trend = (df.iloc[-1]['Close'] > df.iloc[-2]['Close']) and \
                   (df.iloc[-2]['Close'] > df.iloc[-3]['Close'])
        
        low_vol = vix < 15
        high_vol = vix > 20
        
        # Scores and signal
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
            "Spot": spot,
            "VIX": vix,
            "Expiry": expiry if expiry else "N/A",
            "ATM Strike": atm_strike if atm_strike else "N/A",
            "CE OI": ce_oi,
            "PE OI": pe_oi,
            "PCR": round(pcr, 2) if ce_oi > 0 else "N/A",
            "Bullish Score": bullish_score,
            "Bearish Score": bearish_score
        }
    
    except Exception as e:
        st.session_state.signal = f"Error: {str(e)} - Retrying soon..."

# Run analysis if 5 mins passed or first time
current_time = time.time()
if current_time - st.session_state.last_run >= 300:
    run_analysis()

# Display results
st.header("Latest Signal")
st.subheader(st.session_state.signal)

st.header("Details")
for key, value in st.session_state.details.items():
    st.write(f"{key}: {value}")

st.write(f"Last updated: {dt.datetime.fromtimestamp(st.session_state.last_run).strftime('%Y-%m-%d %H:%M:%S')}")
st.write("Refreshing in ~5 minutes...")

# Auto-rerun after delay
time.sleep(1)
if current_time - st.session_state.last_run >= 300:
    st.rerun()
