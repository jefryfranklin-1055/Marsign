import streamlit as st
from streamlit_autorefresh import st_autorefresh
import upstox_client
from upstox_client.rest import ApiException
from upstox_client import MarketQuoteApi, OptionsApi # Keep OptionsApi for Max Pain
import pandas as pd
from datetime import datetime, timedelta, timezone
import talib
import numpy as np
import os
import json

# --- Configuration & Setup ---
# Holidays list is kept to skip fetching data for closed days
NSE_HOLIDAYS_2025 = ["2025-01-26", "2025-03-14", "2025-03-31", "2025-04-14", "2025-04-18", "2025-05-01", "2025-06-16", "2025-08-15", "2025-10-02", "2025-10-21", "2025-11-05", "2025-12-25"]

# The bot no longer needs Firebase for trade state, but configuration is required for API access.
try:
    ACCESS_TOKEN = st.secrets["UPSTOX_ACCESS_TOKEN"]
except (KeyError, FileNotFoundError):
    st.info("Could not find Upstox access token secret. Go to Manage app -> Edit secrets.")
    ACCESS_TOKEN = "TOKEN_NOT_SET"

# --- API Client Initialization ---
@st.cache_resource
def get_api_client():
    """Initializes the API client with the token from secrets."""
    configuration = upstox_client.Configuration()
    configuration.access_token = ACCESS_TOKEN
    return upstox_client.ApiClient(configuration)

api_client = get_api_client()

# --- Analysis Parameters ---
SPOT_INSTRUMENT = "NSE_INDEX|Nifty 50"
BANKNIFTY_INSTRUMENT = "NSE_INDEX|Nifty Bank" 
PRIMARY_TIMEFRAME = "1minute"

# --- Constants ---
IST = timezone(timedelta(hours=5, minutes=30))

# --- Session State (Cleaned up for Signals only) ---
for key, default in {
    'bot_active': False,
    'nifty_spot': 0.0,
    'nifty_rsi': 50.0,
    'nifty_adx': 20.0,
    'nifty_atr': 0.0,
    'pdh': None, 'pdl': None,
    'macro_bias': "None", # Used for Max Pain context
    'max_pain_strike': None,
    'last_max_pain_calc': None,
    'current_signal': "Neutral",
    'signal_reason': "Analysis inactive or market closed.",
    'gap_prediction': "Prediction available after 3:25 PM IST",
    'banknifty_trend_up': False, 
    'is_obv_rising': False,
    'bull_fvgs': [], 
    'bear_fvgs': [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Helper Functions ---
def get_api_data(endpoint, *args, **kwargs):
    """Generic wrapper for API calls to handle exceptions and provide clear feedback."""
    try:
        response = endpoint(*args, **kwargs)
        if response is None or not hasattr(response, 'data'): return None
        return response
    except ApiException as e:
        if e.status == 401: 
            st.error("API Error: Unauthorized (401). Your ACCESS_TOKEN is invalid or expired.")
            st.session_state.bot_active = False
        return None
    except Exception as e: return None

def get_weekly_expiry_date(ref_date=datetime.now(IST)):
    """Finds the Nifty weekly expiry (Tuesday, adjusted for holidays)."""
    days_to_tuesday = (1 - ref_date.weekday() + 7) % 7
    if days_to_tuesday == 0 and ref_date.hour >= 16: days_to_tuesday = 7
    expiry = ref_date + timedelta(days=days_to_tuesday)
    while expiry.strftime("%Y-%m-%d") in NSE_HOLIDAYS_2025 or expiry.weekday() >= 5:
        expiry -= timedelta(days=1)
    return expiry.date()

def get_historical_candles(instrument_key, interval, from_date, to_date):
    """Fetches historical candle data and returns a pandas DataFrame."""
    api_instance = upstox_client.HistoryApi(api_client)
    from_date_str = from_date if isinstance(from_date, str) else from_date.strftime("%Y-%m-%d")
    to_date_str = to_date if isinstance(to_date, str) else to_date.strftime("%Y-%m-%d")
    response = get_api_data(api_instance.get_historical_candle_data1, instrument_key=instrument_key, interval=interval, from_date=from_date_str, to_date=to_date_str, api_version="2.0")
    if response and hasattr(response, 'data') and response.data.candles:
        df = pd.DataFrame(response.data.candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(None) 
        if 'volume' not in df.columns or df['volume'].isnull().all():
            df['volume'] = abs(df['close'] - df['open']) * 10000 + 1000
        df['volume'] = df['volume'].replace(0, 1)
        return df.set_index("timestamp").sort_index()
    return pd.DataFrame()

@st.cache_data(ttl=600) # Cache for 10 minutes
def calculate_max_pain(expiry_date_str):
    """Fetches the option chain and calculates the Max Pain strike."""
    try:
        contract_api = OptionsApi(api_client)
        chain_response = get_api_data(contract_api.get_option_contracts, instrument_key=SPOT_INSTRUMENT, expiry_date=expiry_date_str)
        
        if not chain_response or not chain_response.data:
            return None

        oi_data = {}
        all_strikes = set()

        for contract in chain_response.data:
            if not hasattr(contract, 'strike_price') or not hasattr(contract, 'oi') or not hasattr(contract, 'instrument_type'):
                continue
                
            strike = float(contract.strike_price)
            oi = float(contract.oi)
            all_strikes.add(strike)
            
            if strike not in oi_data:
                oi_data[strike] = {'ce_oi': 0, 'pe_oi': 0}
            
            if contract.instrument_type == 'CE':
                oi_data[strike]['ce_oi'] += oi
            elif contract.instrument_type == 'PE':
                oi_data[strike]['pe_oi'] += oi

        if not all_strikes: return None

        strike_losses = []
        sorted_strikes = sorted(list(all_strikes))
        # NIFTY_LOT_SIZE is removed, but we keep the relative calculation
        for closing_strike in sorted_strikes:
            total_loss = 0
            for strike, data in oi_data.items():
                if closing_strike > strike:
                    total_loss += (closing_strike - strike) * data['ce_oi']
                if closing_strike < strike:
                    total_loss += (strike - closing_strike) * data['pe_oi']
            
            strike_losses.append({'strike': closing_strike, 'loss': total_loss})

        if not strike_losses: return None

        # Max Pain is the strike with the MINIMUM loss to option holders
        max_pain_strike = min(strike_losses, key=lambda x: x['loss'])
        return max_pain_strike['strike']

    except Exception as e:
        return None

def get_daily_setup_data(_today_str):
    """Fetches Previous Day's High/Low."""
    pdh, pdl = None, None
    for i in range(1, 5):
        check_date_dt = datetime.now(IST) - timedelta(days=i)
        check_date_str = check_date_dt.strftime("%Y-%m-%d")
        if check_date_str in NSE_HOLIDAYS_2025 or check_date_dt.weekday() >= 5: continue
        prev_day_candles = get_historical_candles(SPOT_INSTRUMENT, "day", check_date_str, check_date_str)
        if not prev_day_candles.empty:
            pdh, pdl = prev_day_candles['high'].iloc[-1], prev_day_candles['low'].iloc[-1]
            break
    return pdh, pdl

def find_mre_signal(candles_df, nifty_spot, atr_value, rsi_extreme_level, atr_candle_factor):
    """Momentum Reversal from Extreme (MRE) Strategy - Simplified for signal."""
    if len(candles_df) < 5 or atr_value <= 0: return None
    close_1m = candles_df['close']
    rsi_1m_series = talib.RSI(close_1m)
    rsi_1m = rsi_1m_series.iloc[-1]
    
    prev_candle = candles_df.iloc[-2]
    
    is_extreme_oversold = rsi_1m_series.iloc[-2] <= rsi_extreme_level
    is_extreme_overbought = rsi_1m_series.iloc[-2] >= (100 - rsi_extreme_level)
    min_body_size = atr_value * atr_candle_factor
    
    # Bullish Reversal (Oversold -> Snapback)
    if is_extreme_oversold and nifty_spot > prev_candle['open']:
        candle_body = prev_candle['close'] - prev_candle['open']
        if candle_body >= min_body_size and rsi_1m > rsi_extreme_level + 5:
            return "Bullish"
            
    # Bearish Reversal (Overbought -> Snapback)
    if is_extreme_overbought and nifty_spot < prev_candle['open']:
        candle_body = prev_candle['open'] - prev_candle['close']
        if candle_body >= min_body_size and rsi_1m < (100 - rsi_extreme_level - 5):
            return "Bearish"

    return None

def find_rsi_divergence(price_series, rsi_series, lookback=14):
    if len(price_series) < lookback + 2 or len(rsi_series) < lookback + 2: return None
    price_lookback = price_series.iloc[-(lookback+1):-1]
    rsi_lookback = rsi_series.iloc[-(lookback+1):-1]
    if price_lookback.empty or rsi_lookback.empty: return None
    
    # Bullish Divergence (Lower low in price, higher low in RSI)
    price_low_idx, rsi_low_idx = price_lookback.idxmin(), rsi_lookback.idxmin()
    if price_low_idx in rsi_lookback.index: 
        prev_price_low, prev_rsi_low = price_lookback.loc[price_low_idx], rsi_lookback.loc[price_low_idx]
        if price_series.iloc[-1] < prev_price_low and rsi_series.iloc[-1] > prev_rsi_low: return "Bullish"
    
    # Bearish Divergence (Higher high in price, lower high in RSI)
    price_high_idx, rsi_high_idx = price_lookback.idxmax(), rsi_lookback.idxmax()
    if price_high_idx in rsi_lookback.index: 
        prev_price_high, prev_rsi_high = price_lookback.loc[price_high_idx], rsi_lookback.loc[price_high_idx]
        if price_series.iloc[-1] > prev_price_high and rsi_series.iloc[-1] < prev_rsi_high: return "Bearish"
    
    return None

def find_fvgs(candles_df, lookback=10):
    """Identifies recent Fair Value Gaps (simplified)."""
    bull_fvgs, bear_fvgs = [], []
    try:
        if not all(col in candles_df.columns for col in ['high', 'low']) or len(candles_df) < 3: return bull_fvgs, bear_fvgs
        scan_start_index = max(0, len(candles_df) - lookback - 3)
        recent_candles = candles_df.iloc[scan_start_index:]
        for i in range(len(recent_candles) - 2):
            prev_high, next_low = recent_candles['high'].iloc[i], recent_candles['low'].iloc[i+2]
            if prev_high < next_low: bull_fvgs.append((f"{prev_high:.2f} - {next_low:.2f}"))
            prev_low, next_high = recent_candles['low'].iloc[i], recent_candles['high'].iloc[i+2]
            if prev_low > next_high: bear_fvgs.append((f"{next_high:.2f} - {prev_low:.2f}"))
        return list(dict.fromkeys(bull_fvgs[-3:])), list(dict.fromkeys(bear_fvgs[-3:]))
    except Exception as e:
        return [], []

def predict_next_day_gap(nifty_spot, current_expiry_date, rsi, adx):
    """
    Predicts the likelihood of a gap up or gap down opening based on EOD indicators.
    """
    
    # 1. Max Pain Analysis (Run once per day if not already done)
    max_pain_strike = st.session_state.get('max_pain_strike', None)
    now_ist = datetime.now(IST)
    
    if max_pain_strike is None or now_ist.replace(tzinfo=None) - st.session_state.get('last_max_pain_calc', now_ist - timedelta(days=1)) > timedelta(hours=10):
        # Recalculate Max Pain (It runs fast due to caching but we need the latest bias)
        expiry_str = current_expiry_date.strftime('%Y-%m-%d')
        mp_strike = calculate_max_pain(expiry_str)
        if mp_strike:
            st.session_state.max_pain_strike = mp_strike
            st.session_state.last_max_pain_calc = now_ist.replace(tzinfo=None)
            max_pain_strike = mp_strike
            st.session_state.macro_bias = "Bullish" if nifty_spot < mp_strike else ("Bearish" if nifty_spot > mp_strike else "Neutral")
            
    prediction = "Neutral Gap"
    reason = "Indicator alignment unclear for strong prediction."
    
    if max_pain_strike:
        # Distance from Max Pain (60 points buffer for Nifty)
        distance = nifty_spot - max_pain_strike
        mp_threshold = 60 
        
        if distance > mp_threshold:
            prediction = "High probability of **GAP DOWN**"
            reason = f"Nifty closed {distance:.2f} pts above Max Pain ({max_pain_strike}). Mean reversion expected."
            
        elif distance < -mp_threshold:
            prediction = "High probability of **GAP UP**"
            reason = f"Nifty closed {-distance:.2f} pts below Max Pain ({max_pain_strike}). Mean reversion expected."
        
    # 2. RSI Extreme (Confirmation/Alternative)
    ADX_STRONG = 30
    if rsi < 30 and prediction != "High probability of **GAP DOWN**":
        prediction = "High probability of **GAP UP**"
        reason = f"Market closed highly oversold (RSI: {rsi:.2f}). Potential short covering."
    elif rsi > 70 and prediction != "High probability of **GAP UP**":
        prediction = "High probability of **GAP DOWN**"
        reason = f"Market closed highly overbought (RSI: {rsi:.2f}). Potential profit booking."
    
    # 3. Strong Trend (ADX) - If ADX is strong, continuation is likely unless MP is pulling hard.
    elif adx > ADX_STRONG and abs(distance) < 40:
        if st.session_state.banknifty_trend_up:
             prediction = "Likely **GAP UP** (Strong Uptrend)"
             reason = f"Market closed in a strong uptrend (ADX: {adx:.2f})."
        else:
             prediction = "Likely **GAP DOWN** (Strong Downtrend)"
             reason = f"Market closed in a strong downtrend (ADX: {adx:.2f})."


    st.session_state.gap_prediction = f"{prediction} | {reason}"

def run_analysis_and_predict():
    """Master analysis function using predictive and time-based strategies for signals."""
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")
    
    if not api_client:
        st.error("API client not initialized. Bot cannot run.")
        st.session_state.bot_active = False
        return

    # No new signals after 3:30 PM
    if now_ist.time() > datetime.strptime("15:30", "%H:%M").time():
        st.session_state.current_signal = "Market Closed"
        st.session_state.signal_reason = "Trading window is closed."
        return

    current_expiry_date = get_weekly_expiry_date(now_ist)
    is_expiry_day = current_expiry_date == now_ist.date()
    
    # --- Daily Reset Logic ---
    if st.session_state.pdh is None or st.session_state.pdl is None:
        pdh, pdl = get_daily_setup_data(today_str)
        if pdh is not None and pdl is not None:
            st.session_state.update(pdh=pdh, pdl=pdl)
        else:
            st.session_state.signal_reason = "Failed to get PDH/PDL. Check API/Connectivity."
            return

    # --- Get Nifty Spot ---
    market_quote_api = upstox_client.MarketQuoteApi(api_client)
    spot_response = get_api_data(market_quote_api.ltp, symbol=SPOT_INSTRUMENT, api_version="2.0")
    if not spot_response or not spot_response.data: 
        st.session_state.signal_reason = "Could not fetch Nifty Spot price."
        return
        
    nifty_spot = list(spot_response.data.values())[0].last_price
    st.session_state['nifty_spot'] = nifty_spot

    # --- Get Candle Data (Required for ALL strategies) ---
    candles_1m = get_historical_candles(SPOT_INSTRUMENT, "1minute", (now_ist - timedelta(days=2)).strftime("%Y-%m-%d"), today_str)
    banknifty_candles_1m = get_historical_candles(BANKNIFTY_INSTRUMENT, "1minute", (now_ist - timedelta(days=2)).strftime("%Y-%m-%d"), today_str)
    if candles_1m.empty: 
        st.session_state.signal_reason = "Not enough Nifty candle data yet."
        return
    
    # --- Process Current Candle and Indicators ---
    current_ts_minute_naive = now_ist.replace(second=0, microsecond=0, tzinfo=None)
    last_ts = candles_1m.index[-1]
    if last_ts == current_ts_minute_naive:
        candles_1m.loc[last_ts, ['close', 'high', 'low']] = [nifty_spot, max(candles_1m.loc[last_ts, 'high'], nifty_spot), min(candles_1m.loc[last_ts, 'low'], nifty_spot)]
    elif current_ts_minute_naive > last_ts:
        new_row_data = {'open': candles_1m.iloc[-1]['close'], 'high': nifty_spot, 'low': nifty_spot, 'close': nifty_spot, 'volume': 0}
        new_row = pd.DataFrame(new_row_data, index=[current_ts_minute_naive])
        candles_1m = pd.concat([candles_1m, new_row])

    if len(candles_1m) < 50: 
        st.session_state.signal_reason = "Not enough Nifty candle data yet."
        return

    close_1m = candles_1m['close']
    rsi_1m_series = talib.RSI(close_1m)
    rsi_1m = rsi_1m_series.iloc[-1]
    obv = talib.OBV(close_1m, candles_1m['volume'])
    obv_ema = talib.EMA(obv, 10).iloc[-1] if len(obv) >= 10 and not obv.empty else obv.iloc[-1]
    is_obv_rising = obv.iloc[-1] > obv_ema
    st.session_state.is_obv_rising = is_obv_rising
    
    adx, atr = 0, 0
    if len(candles_1m) > 28:
        try:
            adx = talib.ADX(candles_1m['high'], candles_1m['low'], candles_1m['close'], timeperiod=14).iloc[-1]
            atr = talib.ATR(candles_1m['high'], candles_1m['low'], candles_1m['close'], timeperiod=14).iloc[-1] 
        except Exception:
            pass

    st.session_state.nifty_rsi = rsi_1m
    st.session_state.nifty_adx = adx
    st.session_state.nifty_atr = atr 

    # --- BankNifty Trend & FVG (Contextual Filters) ---
    banknifty_trend_up = False
    if not banknifty_candles_1m.empty and len(banknifty_candles_1m) > 20:
        if len(banknifty_candles_1m['close']) > 20: 
            bn_ema20 = talib.EMA(banknifty_candles_1m['close'], 20).iloc[-1]
            banknifty_trend_up = banknifty_candles_1m['close'].iloc[-1] > bn_ema20
    st.session_state.banknifty_trend_up = banknifty_trend_up
    
    st.session_state.bull_fvgs, st.session_state.bear_fvgs = find_fvgs(candles_1m)

    bullish_confirmation = is_obv_rising and banknifty_trend_up
    bearish_confirmation = (not is_obv_rising) and (not banknifty_trend_up)

    # --- MAIN SIGNAL GENERATION ---
    current_signal = "Neutral"
    signal_reason = "Hunting for signals..."
    
    # 1. Momentum Reversal from Extreme (MRE)
    RSI_EXTREME_LEVEL = 20
    ATR_CANDLE_FACTOR = 2.0
    mre_signal = find_mre_signal(candles_1m, nifty_spot, atr, RSI_EXTREME_LEVEL, ATR_CANDLE_FACTOR)
    if mre_signal == "Bullish":
        current_signal, signal_reason = "Buy CE", "Momentum Reversal from Extreme (Oversold Snapback)"
    elif mre_signal == "Bearish":
        current_signal, signal_reason = "Buy PE", "Momentum Reversal from Extreme (Overbought Snapback)"

    # 2. Squeeze Breakout (Only if neutral)
    if current_signal == "Neutral" and len(close_1m) > 20:
        upper_bb, middle_bb, lower_bb = talib.BBANDS(close_1m, timeperiod=20)
        if not (upper_bb.empty or middle_bb.empty or lower_bb.empty) and pd.notna(middle_bb.iloc[-1]):
            bb_width = (upper_bb.iloc[-1] - lower_bb.iloc[-1]) / middle_bb.iloc[-1]
            is_in_squeeze = len(close_1m) > 200 and bb_width < (talib.STDDEV(close_1m, 200).iloc[-1] / middle_bb.iloc[-1]) * 1.5 

            if is_in_squeeze:
                if nifty_spot > upper_bb.iloc[-1] and bullish_confirmation:
                    current_signal, signal_reason = "Buy CE", "Squeeze Breakout Bullish"
                elif nifty_spot < lower_bb.iloc[-1] and bearish_confirmation:
                    current_signal, signal_reason = "Buy PE", "Squeeze Breakout Bearish"

    # 3. Morning Trend-Strength (Before 12:00 PM) - Only if neutral
    ADX_MORNING_TREND_STRENGTH = 25
    if current_signal == "Neutral" and now_ist.time() <= datetime.strptime("12:00", "%H:%M").time():
        if len(close_1m) > 20 and pd.notna(adx):
            ema9 = talib.EMA(close_1m, 9).iloc[-1]
            ema20 = talib.EMA(close_1m, 20).iloc[-1]
            is_bullish_trend = ema9 > ema20 and adx > ADX_MORNING_TREND_STRENGTH
            is_bearish_trend = ema9 < ema20 and adx > ADX_MORNING_TREND_STRENGTH
            
            if is_bullish_trend and nifty_spot > ema9:
                current_signal, signal_reason = "Buy CE", "Morning Trend Continuation (Entry above EMA9)"
            elif is_bearish_trend and nifty_spot < ema9:
                current_signal, signal_reason = "Buy PE", "Morning Trend Continuation (Entry below EMA9)"

    # 4. Mid-Day: RSI Divergence (After 12:00 PM) - Only if neutral
    elif current_signal == "Neutral" and now_ist.time() > datetime.strptime("12:00", "%H:%M").time(): 
        divergence = find_rsi_divergence(close_1m, rsi_1m_series)
        if divergence == "Bullish":
            current_signal, signal_reason = "Buy CE", "RSI Bullish Divergence"
        elif divergence == "Bearish":
            current_signal, signal_reason = "Buy PE", "RSI Bearish Divergence"
            
    # --- EOD GAP PREDICTION LOGIC (3:25 PM - 3:30 PM) ---
    eod_check_start = now_ist.replace(hour=15, minute=25, second=0, microsecond=0)
    eod_check_end = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if eod_check_start <= now_ist <= eod_check_end:
        predict_next_day_gap(nifty_spot, current_expiry_date, rsi_1m, adx)
    else:
        st.session_state.gap_prediction = "Prediction available after 3:25 PM IST"
        
    st.session_state.current_signal = current_signal
    st.session_state.signal_reason = signal_reason


# --- Streamlit UI ---
st.set_page_config(layout="centered", page_title="Nifty Signal Bot")
st.title("Nifty Signal & Gap Prediction 📊")
st.caption("Purely for analysis and advisory; trade execution logic has been removed.")

# --- Master Start/Stop Button ---
if not st.session_state.bot_active:
    if st.button("Start Analysis Bot", use_container_width=True, type="primary"):
        st.session_state.bot_active = True
        st.rerun()
else:
    if st.button("Stop Analysis Bot", use_container_width=True):
        st.session_state.bot_active = False
        st.session_state.current_signal = "Neutral"
        st.session_state.signal_reason = "Analysis stopped by user."
        st.session_state.gap_prediction = "Prediction available after 3:25 PM IST"
        st.rerun()

# --- Dynamic Refresh ---
if st.session_state.bot_active:
    st_autorefresh(interval=20 * 1000, key="mainrefresh")
    st.info("Analysis running... Refreshing every 20 seconds.")
    # Run the main analysis loop
    run_analysis_and_predict()

# --- Display Current Signals ---
st.subheader("Current Intraday Signal")
with st.container(border=True):
    col1, col2 = st.columns([1, 2])
    
    signal_color = "red"
    if st.session_state.current_signal == "Buy CE": signal_color = "green"
    elif st.session_state.current_signal == "Neutral": signal_color = "gray"
        
    col1.markdown(f"<p style='font-size: 24px; font-weight: bold; color: {signal_color};'>{st.session_state.current_signal}</p>", unsafe_allow_html=True)
    col2.markdown(f"**Reason:** {st.session_state.signal_reason}")

# --- Display EOD Gap Prediction ---
st.subheader("EOD Gap Prediction")
with st.container(border=True):
    if "GAP" in st.session_state.gap_prediction:
        gap_color = "green" if "UP" in st.session_state.gap_prediction else "red"
        st.markdown(f"<p style='font-size: 20px; font-weight: bold; color: {gap_color};'>{st.session_state.gap_prediction.split('|')[0]}</p>", unsafe_allow_html=True)
        st.markdown(f"**Context:** {st.session_state.gap_prediction.split('|')[1].strip()}")
    else:
        st.write(st.session_state.gap_prediction)
        
st.divider()

# --- Display Vitals (Indicators) ---
st.subheader("Market Vitals (1-Minute Data)")
with st.container(border=True):
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Nifty 50 Spot", f"{st.session_state.nifty_spot:.2f}")
    v2.metric("RSI", f"{st.session_state.nifty_rsi:.2f}")
    v3.metric("ADX", f"{st.session_state.nifty_adx:.2f}")
    v4.metric("ATR", f"{st.session_state.nifty_atr:.2f}")

    st.markdown("---")
    
    col_bn, col_obv = st.columns(2)
    bn_trend_icon = "▲" if st.session_state.banknifty_trend_up else "▼"
    bn_trend_color = "green" if st.session_state.banknifty_trend_up else "red"
    col_bn.markdown(f"**BankNifty Trend:** <span style='color:{bn_trend_color}; font-weight:bold;'>{bn_trend_icon}</span>", unsafe_allow_html=True)
    
    obv_trend_icon = "▲" if st.session_state.is_obv_rising else "▼"
    obv_trend_color = "green" if st.session_state.is_obv_rising else "red"
    col_obv.markdown(f"**Nifty OBV Trend:** <span style='color:{obv_trend_color}; font-weight:bold;'>{obv_trend_icon}</span>", unsafe_allow_html=True)
    
    st.caption(f"PDH: {st.session_state.pdh or 'N/A'} | PDL: {st.session_state.pdl or 'N/A'}")
    
    if st.session_state.max_pain_strike:
        st.caption(f"Max Pain Strike: {st.session_state.max_pain_strike} | Macro Bias: {st.session_state.macro_bias}")

if not st.session_state.bot_active:
    st.warning("Bot is currently stopped. Click 'Start Analysis Bot' to begin.")
elif now_ist.time() > datetime.strptime("15:30", "%H:%M").time():
    st.error("Market is closed. Bot is paused for the day.")
