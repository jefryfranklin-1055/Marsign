import streamlit as st
from streamlit_autorefresh import st_autorefresh
import upstox_client
from upstox_client.rest import ApiException
import pandas as pd
from datetime import datetime, timedelta, timezone
import talib
import numpy as np
import os
import json

# --- Configuration & Setup ---
NSE_HOLIDAYS_2025 = ["2025-01-26", "2025-03-14", "2025-03-31", "2025-04-14", "2025-04-18", "2025-05-01", "2025-06-16", "2025-08-15", "2025-10-02", "2025-10-21", "2025-11-05", "2025-12-25"]

try:
    ACCESS_TOKEN = st.secrets["UPSTOX_ACCESS_TOKEN"]
except (KeyError, FileNotFoundError):
    st.info("Could not find Upstox access token secret. Please set it in Streamlit Cloud.")
    ACCESS_TOKEN = "TOKEN_NOT_SET"

# --- API Client Initialization ---
@st.cache_resource
def get_api_client():
    """Initializes the Upstox API client."""
    configuration = upstox_client.Configuration()
    configuration.access_token = ACCESS_TOKEN
    return upstox_client.ApiClient(configuration)

api_client = get_api_client()

# --- Trading Parameters ---
SPOT_INSTRUMENT = "NSE_INDEX|Nifty 50"
VIX_INSTRUMENT = "NSE_INDEX|India VIX"
BANKNIFTY_INSTRUMENT = "NSE_INDEX|Nifty Bank" 
PRIMARY_TIMEFRAME = "1minute"

# --- Constants ---
IST = timezone(timedelta(hours=5, minutes=30))

# --- Session State ---
# Initialize all session state variables
for key, default in {
    'last_run_day': None,
    'signal_count': 0,
    'signal_log': [],
    'active_strategy_msg': "Initializing...",
    'pdh': None, 'pdl': None,
    'last_fvg_scan_time': None, 'bull_fvgs': [], 'bear_fvgs': [],
    'banknifty_trend_up': False, 'is_obv_rising': False,
    # 'bot_active' state removed - bot is always on
    'strategy_squeeze_enabled': True,
    'strategy_breakout_enabled': True,
    'strategy_divergence_enabled': True,
    'strategy_trendcont_enabled': True,
    'strategy_fibonacci_enabled': True,
    'strategy_hero_enabled': True,
    'nifty_spot': 0.0,
    'nifty_rsi': 50.0,
    'nifty_adx': 20.0,
    'last_signal_time': None,
    'last_signal_key': None, # To prevent signal spam
    'hoz_range': None,
    'hoz_signal_given': False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Helper Functions ---
def get_api_data(endpoint, *args, **kwargs):
    """Generic wrapper for API calls to handle exceptions."""
    try:
        response = endpoint(*args, **kwargs)
        if response is None or not hasattr(response, 'data'): return None
        return response
    except ApiException as e:
        if e.status == 401: st.error("API Error: Unauthorized (401). Token expired?")
        return None
    except Exception as e: return None

def get_weekly_expiry_date(ref_date=datetime.now(IST)):
    """Calculates the nearest weekly expiry date (Tuesday for Nifty)."""
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

# --- Core Logic ---
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

def generate_signal(signal_type, strategy_name):
    """Generates and logs a signal, with a cool-down to prevent spam."""
    now = datetime.now(IST)
    
    # Create a unique key for this signal type and strategy
    signal_key = f"{signal_type}_{strategy_name}"
    
    # Check if we're in a cool-down period for this *specific* signal
    if st.session_state.last_signal_key == signal_key:
        if st.session_state.last_signal_time and (now - st.session_state.last_signal_time) < timedelta(minutes=3):
            # st.info(f"Signal '{strategy_name}' in 3-min cool-down. Ignoring.")
            return # In cool-down, do nothing

    st.session_state.signal_count += 1
    st.session_state.last_signal_time = now
    st.session_state.last_signal_key = signal_key
    
    signal_time = now.strftime('%I:%M %p')
    log_entry = {
        'Time': signal_time, 
        'Strategy': strategy_name, 
        'Signal': f"BUY {signal_type}",
        'Nifty Spot': f"{st.session_state.nifty_spot:.2f}"
    }
    
    st.session_state.signal_log.insert(0, log_entry) # Add to top of the log
    
    if signal_type == "CE":
        st.success(f"✅ SIGNAL: BUY CE ({strategy_name}) at {signal_time}")
    else:
        st.error(f"🔻 SIGNAL: BUY PE ({strategy_name}) at {signal_time}")

def find_rsi_divergence(price_series, rsi_series, lookback=14):
    """Looks for bullish or bearish RSI divergence."""
    if len(price_series) < lookback + 2 or len(rsi_series) < lookback + 2: return None
    price_lookback = price_series.iloc[-(lookback+1):-1]
    rsi_lookback = rsi_series.iloc[-(lookback+1):-1]
    if price_lookback.empty or rsi_lookback.empty: return None
    
    try:
        price_low_idx, rsi_low_idx = price_lookback.idxmin(), rsi_lookback.idxmin()
        if price_low_idx not in rsi_lookback.index: return None
        prev_price_low, prev_rsi_low = price_lookback.loc[price_low_idx], rsi_lookback.loc[price_low_idx]
        if price_series.iloc[-1] < prev_price_low and rsi_series.iloc[-1] > prev_rsi_low: return "Bullish"
        
        price_high_idx, rsi_high_idx = price_lookback.idxmax(), rsi_lookback.idxmax()
        if price_high_idx not in rsi_lookback.index: return None
        prev_price_high, prev_rsi_high = price_lookback.loc[price_high_idx], rsi_lookback.loc[price_high_idx]
        if price_series.iloc[-1] > prev_price_high and rsi_series.iloc[-1] < prev_rsi_high: return "Bearish"
    except Exception as e:
        # st.warning(f"Error in divergence calc: {e}")
        pass
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

def analyze_for_entry_signal():
    """Master analysis function using predictive and time-based strategies."""
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")
    current_expiry_date = get_weekly_expiry_date(now_ist)
    is_expiry_day = current_expiry_date == now_ist.date()

    # --- Daily Reset Logic ---
    if st.session_state.pdh is None or st.session_state.pdl is None or st.session_state.last_run_day != today_str:
        if st.session_state.last_run_day != today_str:
            st.session_state.update(
                last_run_day=today_str, signal_count=0, signal_log=[],
                hoz_range=None, hoz_signal_given=False
            )
        
        pdh, pdl = get_daily_setup_data(today_str)
        if pdh is not None and pdl is not None:
            st.session_state.update(pdh=pdh, pdl=pdl)
        else:
            st.error("Failed to get PDH/PDL. Breakout strategy will be disabled.")
            return

    # --- Get Nifty Spot ---
    market_quote_api = upstox_client.MarketQuoteApi(api_client)
    spot_response = get_api_data(market_quote_api.ltp, symbol=SPOT_INSTRUMENT, api_version="2.0")
    if not spot_response or not spot_response.data: 
        st.session_state.active_strategy_msg = "Waiting for Nifty Spot price..."
        return
    nifty_spot = list(spot_response.data.values())[0].last_price
    st.session_state['nifty_spot'] = nifty_spot

    # --- Get Candle Data ---
    candles_1m = get_historical_candles(SPOT_INSTRUMENT, "1minute", (now_ist - timedelta(days=2)).strftime("%Y-%m-%d"), today_str)
    banknifty_candles_1m = get_historical_candles(BANKNIFTY_INSTRUMENT, "1minute", (now_ist - timedelta(days=2)).strftime("%Y-%m-%d"), today_str)
    if candles_1m.empty: 
        st.session_state.active_strategy_msg = "Waiting for 1-min candle data..."
        return
    
    # --- Process Current Candle ---
    last_ts = candles_1m.index[-1]
    current_ts_minute_naive = now_ist.replace(second=0, microsecond=0, tzinfo=None)
    
    if last_ts == current_ts_minute_naive:
        candles_1m.loc[last_ts, ['close', 'high', 'low']] = [nifty_spot, max(candles_1m.loc[last_ts, 'high'], nifty_spot), min(candles_1m.loc[last_ts, 'low'], nifty_spot)]
    elif current_ts_minute_naive > last_ts:
        new_row = pd.DataFrame({'open': candles_1m.iloc[-1]['close'], 'high': nifty_spot, 'low': nifty_spot, 'close': nifty_spot, 'volume': 0, 'oi': 0}, index=[current_ts_minute_naive])
        candles_1m = pd.concat([candles_1m, new_row])

    if len(candles_1m) < 50: 
        st.session_state.active_strategy_msg = "Not enough candle data yet..."
        return

    # --- Calculate Indicators ---
    close_1m = candles_1m['close']
    rsi_1m_series = talib.RSI(close_1m)
    rsi_1m = rsi_1m_series.iloc[-1]
    obv = talib.OBV(close_1m, candles_1m['volume'])
    obv_ema = talib.EMA(obv, 10).iloc[-1]
    is_obv_rising = obv.iloc[-1] > obv_ema
    st.session_state.is_obv_rising = is_obv_rising
    
    adx = 0
    if len(candles_1m) > 14 * 2:
        adx = talib.ADX(candles_1m['high'], candles_1m['low'], candles_1m['close'], timeperiod=14).iloc[-1]
    st.session_state.nifty_rsi = rsi_1m
    st.session_state.nifty_adx = adx

    # --- BankNifty Trend ---
    banknifty_trend_up = False
    if not banknifty_candles_1m.empty and len(banknifty_candles_1m) > 20:
        bn_spot_response = get_api_data(market_quote_api.ltp, symbol=BANKNIFTY_INSTRUMENT, api_version="2.0")
        if bn_spot_response and bn_spot_response.data:
            bn_spot = list(bn_spot_response.data.values())[0].last_price
            bn_last_ts = banknifty_candles_1m.index[-1]
            if bn_last_ts == current_ts_minute_naive:
                banknifty_candles_1m.loc[bn_last_ts, 'close'] = bn_spot
            elif current_ts_minute_naive > bn_last_ts:
                bn_new_row = pd.DataFrame({'open': banknifty_candles_1m.iloc[-1]['close'], 'high': bn_spot, 'low': bn_spot, 'close': bn_spot, 'volume': 0, 'oi': 0}, index=[current_ts_minute_naive])
                banknifty_candles_1m = pd.concat([banknifty_candles_1m, bn_new_row])
        
        if len(banknifty_candles_1m['close']) > 20: 
            bn_ema20 = talib.EMA(banknifty_candles_1m['close'], 20).iloc[-1]
            banknifty_trend_up = banknifty_candles_1m['close'].iloc[-1] > bn_ema20
    st.session_state.banknifty_trend_up = banknifty_trend_up

    # --- FVG Scan ---
    if st.session_state.last_fvg_scan_time is None or (now_ist.replace(tzinfo=None) - st.session_state.last_fvg_scan_time) >= timedelta(minutes=5):
        st.session_state.bull_fvgs, st.session_state.bear_fvgs = find_fvgs(candles_1m)
        st.session_state.last_fvg_scan_time = now_ist.replace(tzinfo=None)
    
    # --- Contextual Confirmations ---
    bullish_confirmation = is_obv_rising or banknifty_trend_up
    bearish_confirmation = not is_obv_rising or not banknifty_trend_up
    
    # --- Hero or Zero Logic ---
    if is_expiry_day and now_ist.time() >= datetime.strptime("14:00", "%H:%M").time() and st.session_state.strategy_hero_enabled and not st.session_state.hoz_signal_given:
        st.session_state.active_strategy_msg = "Expiry: Hunting Hero or Zero"
        range_end_time_dt = now_ist.replace(hour=14, minute=15, second=0, microsecond=0)

        if now_ist >= range_end_time_dt and st.session_state.hoz_range is None:
            try:
                afternoon_candles = candles_1m.between_time("13:00", "14:15")
                if not afternoon_candles.empty:
                    range_high, range_low = afternoon_candles['high'].max(), afternoon_candles['low'].min()
                    range_width = range_high - range_low
                    
                    if range_width > 60 or range_width < 10:
                        st.session_state.hoz_signal_given = True # Disable further checks
                        return
                    else:
                        st.session_state.hoz_range = {'high': range_high, 'low': range_low}
                else:
                    st.session_state.hoz_signal_given = True
                    return
            except Exception as e:
                st.session_state.hoz_signal_given = True
                return

        if st.session_state.hoz_range is not None:
            st.session_state.active_strategy_msg = "Expiry: Stalking HoZ Breakout"
            if len(candles_1m) < 3: return
            
            range_high = st.session_state.hoz_range['high']
            range_low = st.session_state.hoz_range['low']
            prev_close = candles_1m['close'].iloc[-2]
            prev_prev_close = candles_1m['close'].iloc[-3]
            
            is_bullish_breakout = (prev_close > range_high) and (prev_prev_close <= range_high)
            is_bearish_breakout = (prev_close < range_low) and (prev_prev_close >= range_low)
            
            if is_bullish_breakout and rsi_1m > 60:
                generate_signal("CE", "Hero or Zero")
                st.session_state.hoz_signal_given = True
                return
            
            elif is_bearish_breakout and rsi_1m < 40:
                generate_signal("PE", "Hero or Zero")
                st.session_state.hoz_signal_given = True
                return
        
        return # Prevent other strategies in HoZ window

    # --- Squeeze Strategy ---
    upper_bb, middle_bb, lower_bb = talib.BBANDS(close_1m, timeperiod=20)
    bb_width = (upper_bb - lower_bb) / middle_bb
    is_in_squeeze = False
    if len(bb_width) > 200:
        if pd.notna(bb_width.iloc[-1]) and pd.notna(bb_width.rolling(200).min().iloc[-1]):
            squeeze_threshold = bb_width.rolling(200).min().iloc[-1] * 1.1
            is_in_squeeze = bb_width.iloc[-1] < squeeze_threshold

    if is_in_squeeze and st.session_state.strategy_squeeze_enabled:
        st.session_state.active_strategy_msg = "Squeeze: Stalking Breakout"
        if nifty_spot > upper_bb.iloc[-1] and bullish_confirmation:
            generate_signal("CE", "Squeeze Breakout")
        elif nifty_spot < lower_bb.iloc[-1] and bearish_confirmation:
            generate_signal("PE", "Squeeze Breakout")
        return

    # --- Morning Breakout Strategy ---
    if now_ist.time() <= datetime.strptime("10:30", "%H:%M").time():
        st.session_state.active_strategy_msg = "Morning: Breakout Strategy"
        if st.session_state.strategy_breakout_enabled and st.session_state.pdh is not None and st.session_state.pdl is not None:
            day_high, day_low = candles_1m['high'].max(), candles_1m['low'].min()
            breakout_levels = {"Day High": day_high, "PDH": st.session_state.pdh}
            breakdown_levels = {"Day Low": day_low, "PDL": st.session_state.pdl}
            
            for level_name, level_price in breakout_levels.items():
                if level_price and nifty_spot > level_price and rsi_1m > 60 and bullish_confirmation:
                    generate_signal("CE", f"{level_name} Breakout")
            for level_name, level_price in breakdown_levels.items():
                if level_price and nifty_spot < level_price and rsi_1m < 40 and bearish_confirmation:
                    generate_signal("PE", f"{level_name} Breakdown")
    # --- Mid-Day Strategies ---
    else: 
        st.session_state.active_strategy_msg = "Mid-Day: Multi-Strategy"
        
        if st.session_state.strategy_divergence_enabled:
            divergence = find_rsi_divergence(close_1m, rsi_1m_series)
            if divergence == "Bullish":
                generate_signal("CE", "RSI Divergence")
            elif divergence == "Bearish":
                generate_signal("PE", "RSI Divergence")
        
        if st.session_state.strategy_trendcont_enabled:
            if len(close_1m) > 50:
                ema9, ema20, ema50 = talib.EMA(close_1m, 9).iloc[-1], talib.EMA(close_1m, 20).iloc[-1], talib.EMA(close_1m, 50).iloc[-1]
                current_low, current_high = candles_1m['low'].iloc[-1], candles_1m['high'].iloc[-1]
                is_pullback_long = nifty_spot > ema50 and (current_low <= ema9 or current_low <= ema20) and nifty_spot > min(ema9, ema20)
                is_pullback_short = nifty_spot < ema50 and (current_high >= ema9 or current_high >= ema20) and nifty_spot < max(ema9, ema20)
                if is_pullback_long and rsi_1m > 50 and bullish_confirmation:
                    generate_signal("CE", "Trend Continuation")
                elif is_pullback_short and rsi_1m < 50 and bearish_confirmation:
                    generate_signal("PE", "Trend Continuation")

        if st.session_state.strategy_fibonacci_enabled:
            if len(candles_1m) > 1:
                day_high, day_low = candles_1m['high'].max(), candles_1m['low'].min()
                swing_range = day_high - day_low
                if swing_range > 0:
                    fib_zones = { "Shallow": (day_high - (swing_range * 0.382), day_high - (swing_range * 0.5)), "Deep": (day_high - (swing_range * 0.5), day_high - (swing_range * 0.618)) }
                    for zone_name, (top, bottom) in fib_zones.items():
                        if bottom <= nifty_spot <= top:
                            open_s, high_s, low_s, close_s = candles_1m['open'], candles_1m['high'], candles_1m['low'], candles_1m['close']
                            is_hammer, is_bull_engulf = talib.CDLHAMMER(open_s, high_s, low_s, close_s).iloc[-1] > 0, talib.CDLENGULFING(open_s, high_s, low_s, close_s).iloc[-1] > 0
                            is_shoot_star, is_bear_engulf = talib.CDLSHOOTINGSTAR(open_s, high_s, low_s, close_s).iloc[-1] > 0, talib.CDLENGULFING(open_s, high_s, low_s, close_s).iloc[-1] < 0
                            if (is_hammer or is_bull_engulf):
                                generate_signal("CE", f"Fib {zone_name} Reversal")
                            elif (is_shoot_star or is_bear_engulf):
                                generate_signal("PE", f"Fib {zone_name} Reversal")
    
    st.session_state.active_strategy_msg = "Hunting for signals..."

# --- Streamlit UI ---
st.set_page_config(layout="centered", page_title="Nifty Signal Bot")
st.title("Nifty Super Signal Bot 🚀")

now_ist = datetime.now(IST)
today_str = now_ist.strftime("%Y-%m-%d")

# --- Daily Reset ---
if st.session_state.last_run_day != today_str:
    st.session_state.update(
        last_run_day=today_str, signal_count=0, signal_log=[], 
        hoz_range=None, hoz_signal_given=False
    )

# --- Dynamic Refresh ---
# Bot is always active, so autorefresh is always on. 20 seconds.
refresh_interval = 20 * 1000 
st_autorefresh(interval=refresh_interval, key="mainrefresh")

# --- Vitals Header (Always active) ---
with st.container(border=True):
    v1, v2, v3 = st.columns(3)
    v1.metric("Nifty 50", f"{st.session_state.nifty_spot:.2f}")
    v2.metric("RSI (1-min)", f"{st.session_state.nifty_rsi:.2f}")
    v3.metric("ADX (1-min)", f"{st.session_state.nifty_adx:.2f}")

# --- Display Area ---
col1, col2 = st.columns(2)
col1.metric("Signals Today", st.session_state.signal_count)
col2.info(st.session_state.active_strategy_msg)

# --- Sidebar ---
st.sidebar.header("Daily Setup")
st.sidebar.metric("Previous Day High", f"{st.session_state.pdh or 'N/A'}")
st.sidebar.metric("Previous Day Low", f"{st.session_state.pdl or 'N/A'}")

st.sidebar.subheader("Contextual Filters")
bn_trend_icon = "▲" if st.session_state.banknifty_trend_up else "▼"
bn_trend_color = "green" if st.session_state.banknifty_trend_up else "red"
st.sidebar.markdown(f"BankNifty Trend: <span style='color:{bn_trend_color}; font-weight:bold;'>{bn_trend_icon}</span>", unsafe_allow_html=True)
obv_trend_icon = "▲" if st.session_state.is_obv_rising else "▼"
obv_trend_color = "green" if st.session_state.is_obv_rising else "red"
st.sidebar.markdown(f"Nifty OBV Trend: <span style='color:{obv_trend_color}; font-weight:bold;'>{obv_trend_icon}</span>", unsafe_allow_html=True)

st.sidebar.subheader("Recent FVG Zones")
st.sidebar.write("Bullish Gaps:", st.session_state.bull_fvgs)
st.sidebar.write("Bearish Gaps:", st.session_state.bear_fvgs)

# --- Strategy Toggles ---
st.sidebar.header("Strategy Toggles")
st.session_state.strategy_squeeze_enabled = st.sidebar.checkbox("Squeeze Breakout", value=st.session_state.strategy_squeeze_enabled)
st.session_state.strategy_breakout_enabled = st.sidebar.checkbox("Breakout (Morning)", value=st.session_state.strategy_breakout_enabled)
st.session_state.strategy_divergence_enabled = st.sidebar.checkbox("RSI Divergence", value=st.session_state.strategy_divergence_enabled)
st.session_state.strategy_trendcont_enabled = st.sidebar.checkbox("Trend Continuation", value=st.session_state.strategy_trendcont_enabled)
st.session_state.strategy_fibonacci_enabled = st.sidebar.checkbox("Fibonacci Reversal", value=st.session_state.strategy_fibonacci_enabled)
st.session_state.strategy_hero_enabled = st.sidebar.checkbox("Hero or Zero (Expiry @ 2PM+)", value=st.session_state.strategy_hero_enabled)

# --- Signal Log ---
if st.session_state.signal_log:
    st.subheader("Today's Signals")
    # Display the dataframe with the most recent signal at the top
    st.dataframe(st.session_state.signal_log, use_container_width=True)

# --- Main Analysis Loop (Always active) ---
try:
    spinner_message = f"Analyzing... Strategy: {st.session_state.active_strategy_msg}"
        
    with st.spinner(spinner_message):
        market_open, market_close = now_ist.replace(hour=9, minute=15), now_ist.replace(hour=15, minute=30)
        # Check if market is open
        if not (market_open <= now_ist <= market_close) or now_ist.strftime("%Y-%m-%d") in NSE_HOLIDAYS_2025:
            st.warning("Market is currently closed (or it's a holiday).")
        else:
            analyze_for_entry_signal() # Always analyze for a new signal

except Exception as e:
    st.error(f"A critical runtime error occurred: {e}")
    st.exception(e) # Show full traceback in Streamlit
