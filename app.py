import streamlit as st
import pandas as pd
import datetime as dt
import time
import yfinance as yf
from datetime import timedelta

st.title("Nifty Options Signal Analyzer")
st.write("Monitors trading session (9:15 AM - 3:30 PM IST) and notifies on solid moves only.")

# Session state
if "last_run" not in st.session_state:
    st.session_state.last_run = 0
    st.session_state.signal = "Monitoring trading session..."
    st.session_state.details = {}
    st.session_state.monitoring = False

def is_trading_session(current_time):
    """Check if within NSE trading hours (9:15 AM - 3:30 PM IST, Mon-Fri)"""
    if current_time.weekday() > 4:  # Sat/Sun
        return False
    start = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
    end = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= current_time <= end

def detect_patterns(df):
    """Expanded pattern recognition"""
    patterns = {}
    
    # Basic candles
    last = df.iloc[-1]
    prev = df.iloc[-2]
    patterns['bullish_candle'] = last['Close'] > last['Open']
    patterns['bearish_candle'] = last['Close'] < last['Open']
    
    # Engulfing
    patterns['bullish_engulfing'] = (prev['Close'] < prev['Open']) and (last['Open'] < prev['Close']) and (last['Close'] > prev['Open'])
    patterns['bearish_engulfing'] = (prev['Close'] > prev['Open']) and (last['Open'] > prev['Close']) and (last['Close'] < prev['Open'])
    
    # Trend
    patterns['up_trend'] = (df.iloc[-1]['Close'] > df.iloc[-2]['Close']) and (df.iloc[-2]['Close'] > df.iloc[-3]['Close'])
    patterns['down_trend'] = (df.iloc[-1]['Close'] < df.iloc[-2]['Close']) and (df.iloc[-2]['Close'] < df.iloc[-3]['Close'])
    
    # Additional patterns
    body = abs(last['Close'] - last['Open'])
    lower_shadow = last['Open'] - last['Low'] if last['Close'] > last['Open'] else last['Close'] - last['Low']
    upper_shadow = last['High'] - last['Close'] if last['Close'] > last['Open'] else last['High'] - last['Open']
    total_range = last['High'] - last['Low']
    
    patterns['hammer'] = (lower_shadow > 2 * body) and (upper_shadow < body) and (body > 0)  # Bullish reversal
    patterns['shooting_star'] = (upper_shadow > 2 * body) and (lower_shadow < body) and (body > 0)  # Bearish reversal
    patterns['doji'] = body <= 0.1 * total_range  # Indecision
    
    # Double bottom/top (simple 3-bar check with tolerance)
    tolerance = 0.005  # 0.5% tolerance for approximate equality
    patterns['double_bottom'] = (abs(df.iloc[-3]['Low'] - df.iloc[-1]['Low']) / df.iloc[-1]['Low'] < tolerance) and (df.iloc[-2]['Low'] > df.iloc[-1]['Low'])
    patterns['double_top'] = (abs(df.iloc[-3]['High'] - df.iloc[-1]['High']) / df.iloc[-1]['High'] < tolerance) and (df.iloc[-2]['High'] < df.iloc[-1]['High'])
    
    return patterns

def run_analysis():
    current_time = dt.datetime.now()
    if not is_trading_session(current_time):
        st.session_state.signal = "Outside trading hours (9:15 AM - 3:30 PM IST). Monitoring paused."
        return
    
    st.session_state.last_run = current_time
    try:
        st.info("Fetching data...")
        nifty = yf.Ticker("^NSEI")
        vix_ticker = yf.Ticker("^INDIAVIX")
        
        spot_hist = nifty.history(period="1d")
        if spot_hist.empty:
            raise ValueError("No spot data available")
        spot = spot_hist['Close'].iloc[-1]
        
        vix_hist = vix_ticker.history(period="1d")
        if vix_hist.empty:
            raise ValueError("No VIX data available")
        vix = vix_hist['Close'].iloc[-1]
        
        # Historical 5-min data (continuous monitoring – last 1h for recent patterns)
        hist = nifty.history(period="1d", interval="5m")
        df = hist.reset_index()[['Datetime', 'Open', 'High', 'Low', 'Close']]
        df.rename(columns={'Datetime': 'date'}, inplace=True)
        if len(df) < 5:  # Need more bars for advanced patterns
            raise ValueError("Insufficient historical data")
        
        # OI/Greeks neutral (yfinance limits)
        ce_oi = pe_oi = 0
        ce_oi_change = pe_oi_change = 0
        ce_delta = pe_delta = ce_theta = pe_theta = 0
        expiry = atm_strike = "N/A (Weekly Tuesday)"
        pcr = 1.0
        bullish_oi = bearish_oi = False
        st.warning("OI/Greeks disabled (yfinance limits). Neutral scores for those factors.")
        
        # Expanded Pattern Detection
        patterns = detect_patterns(df)
        
        # Scoring (weighted: patterns 60%, VIX 40% – continuous monitoring amplifies trends)
        bullish_score = 0
        bearish_score = 0
        
        # Patterns (expanded)
        if patterns['bullish_candle']:
            bullish_score += 1
        if patterns['bullish_engulfing']:
            bullish_score += 2
        if patterns['up_trend']:
            bullish_score += 1
        if patterns['hammer']:
            bullish_score += 2
        if patterns['double_bottom']:
            bullish_score += 1.5
        
        if patterns['bearish_candle']:
            bearish_score += 1
        if patterns['bearish_engulfing']:
            bearish_score += 2
        if patterns['down_trend']:
            bearish_score += 1
        if patterns['shooting_star']:
            bearish_score += 2
        if patterns['double_top']:
            bearish_score += 1.5
        
        # Skip doji for scoring (indecision – neutral)
        
        # VIX (high VIX = bigger moves, continuous monitoring)
        if vix < 15:
            bullish_score += 0.5
            bearish_score += 0.5
        elif vix > 20:
            bullish_score *= 1.3  # Amplify for solid moves
            bearish_score *= 1.3
        
        # Determine Direction & Magnitude (only signal on solid moves)
        net_score = bullish_score - bearish_score
        if net_score > 8:  # Higher threshold for solid moves
            signal = "Market likely to move UP"
            potential = "Big move"
        elif net_score > 5:
            signal = "Market likely to move UP"
            potential = "75+"
        elif net_score > 3:
            signal = "Market likely to move UP"
            potential = "50+"
        elif net_score < -8:
            signal = "Market likely to move DOWN"
            potential = "Big move"
        elif net_score < -5:
            signal = "Market likely to move DOWN"
            potential = "75+"
        elif net_score < -3:
            signal = "Market likely to move DOWN"
            potential = "50+"
        else:
            return  # No signal if no solid move
        
        # Set signal only if solid move detected
        st.session_state.signal = f"{signal} - Give signal as {potential}"
        
        # Notification (visual toast + sidebar alert)
        st.toast(f"Solid Move Detected: {st.session_state.signal}", icon="🚨")
        st.sidebar.success(f"🚨 ALERT: {st.session_state.signal}")
        
        st.session_state.details = {
            "Spot": f"{spot:.2f}",
            "VIX": f"{vix:.2f}",
            "Expiry": expiry,
            "ATM Strike": atm_strike,
            "CE OI": ce_oi,
            "PE OI": pe_oi,
            "CE OI Change": ce_oi_change,
            "PE OI Change": pe_oi_change,
            "CE Delta": ce_delta,
            "CE Theta": ce_theta,
            "Bullish Score": bullish_score,
            "Bearish Score": bearish_score,
            "Patterns": patterns  # Show detected patterns
        }
        st.success("Analysis complete – monitoring continues.")
    
    except Exception as e:
        st.session_state.signal = f"Error: {str(e)}"
        st.error(f"Details: {e}")

# Continuous Monitoring Button (runs loop during session)
if st.button("Start Session Monitoring (Entire Trading Day)"):
    st.session_state.monitoring = True
    st.info("Monitoring started – checks every minute for solid moves.")
    while st.session_state.monitoring and is_trading_session(dt.datetime.now()):
        run_analysis()
        time.sleep(60)  # Check every minute for solid moves
    st.session_state.monitoring = False
    st.success("Monitoring stopped (end of session or manual).")

# Manual Update Button
if st.button("Manual Update (Check Now)"):
    run_analysis()

# Display results
st.header("Latest Signal")
st.subheader(st.session_state.signal)

st.header("Details")
for key, value in st.session_state.details.items():
    st.write(f"**{key}:** {value}")

st.write(f"**Last updated:** {st.session_state.last_run.strftime('%Y-%m-%d %H:%M:%S IST') if st.session_state.last_run else 'Not run yet'}")

# Stop Monitoring Button
if st.button("Stop Monitoring"):
    st.session_state.monitoring = False
    st.info("Monitoring stopped.")
