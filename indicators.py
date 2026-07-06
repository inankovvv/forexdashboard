"""
indicators.py — shared data fetching, indicator, and pattern-detection engine.

Covers:
  - MACD cross (12,26,9)
  - Bullish/Bearish Engulfing
  - EMA(9/21) cross with higher-timeframe trend + RSI zone + MACD momentum +
    Bollinger Band confluence filter ("confluence" signal, from the EMA/RSI/
    MACD/BB multi-indicator bot article)
  - Candlestick patterns: Doji, Hammer, Shooting Star, Bullish/Bearish Harami,
    Morning Star, Evening Star (from the candlestick recognition article)

Used by scanner.py (writes signals to SQLite) and app.py (Streamlit UI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

# ------------------- INSTRUMENTS -------------------
INSTRUMENTS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "EURCHF": "EURCHF=X",
    "EURAUD": "EURAUD=X",
    "EURNZD": "EURNZD=X",
    "GBPJPY": "GBPJPY=X",
    "GBPCHF": "GBPCHF=X",
    "GBPAUD": "GBPAUD=X",
    "GBPNZD": "GBPNZD=X",
    "AUDJPY": "AUDJPY=X",
    "AUDCHF": "AUDCHF=X",
    "NZDJPY": "NZDJPY=X",
    "NZDCHF": "NZDCHF=X",
    "WTI": "CL=F",
    "GER30": "^GDAXI",
    "NAS100": "^NDX",
    "US30": "^DJI",
    "XAU_USD": "GC=F",
}

# Timeframes we support, mapped to yfinance intervals.
# "4h" is built by resampling 1h candles (not native on yfinance).
TIMEFRAMES = {
    "15m": {"yf_interval": "15m", "resample": None, "max_days": 60},
    "30m": {"yf_interval": "30m", "resample": None, "max_days": 60},
    "1h":  {"yf_interval": "60m", "resample": None, "max_days": 730},
    "4h":  {"yf_interval": "60m", "resample": "4h", "max_days": 730},
    "1d":  {"yf_interval": "1d",  "resample": None, "max_days": 3650},
}

# For the confluence signal, we check trend on "one level up" from the
# timeframe being scanned (mirrors the 1m-signal / 5m-trend relationship
# in the source article, generalized across our 5 timeframes).
HIGHER_TF = {"15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d", "1d": "1d"}

MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
EMA_FAST, EMA_SLOW = 9, 21
RSI_PERIOD = 14
BB_PERIOD, BB_STD = 20, 2


# ------------------- DATA FETCHING -------------------

def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    return df.resample(rule).agg(agg).dropna()


def get_multi_data(instrument_keys: list[str], timeframe: str, lookback_days: int,
                    buffer_days: int = 20) -> dict[str, pd.DataFrame]:
    """Fetch OHLC data for multiple instruments in one batched request."""
    for k in instrument_keys:
        if k not in INSTRUMENTS:
            raise ValueError(f"Unknown instrument '{k}'. Choose from {list(INSTRUMENTS)}")
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from {list(TIMEFRAMES)}")

    cfg = TIMEFRAMES[timeframe]
    total_days = min(lookback_days + buffer_days, cfg["max_days"])
    period = f"{total_days}d"
    tickers = [INSTRUMENTS[k] for k in instrument_keys]

    raw = yf.download(
        tickers, period=period, interval=cfg["yf_interval"],
        progress=False, auto_adjust=False, group_by="ticker",
        threads=False,  # avoid yfinance's internal HTTP cache (SQLite-backed)
                        # hitting "unable to open database file" under concurrent writes
    )

    result: dict[str, pd.DataFrame] = {}
    for key in instrument_keys:
        ticker = INSTRUMENTS[key]
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                # group_by="ticker" puts the ticker symbol as the top column
                # level regardless of how many tickers were requested — always
                # select by ticker here rather than special-casing the count.
                df = raw[ticker].copy()
            else:
                df = raw.copy()
            df = df.dropna(how="all")
            if df.empty:
                result[key] = pd.DataFrame()
                continue
            if cfg["resample"]:
                df = _resample(df, cfg["resample"])
            result[key] = df
        except Exception as e:
            print(f"[WARN] no data for {key} ({ticker}): {e}")
            result[key] = pd.DataFrame()

    return result


def get_data(instrument: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    return get_multi_data([instrument], timeframe, lookback_days)[instrument]


# ------------------- INDICATORS -------------------

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line


def compute_bollinger(close: pd.Series, period=BB_PERIOD, std_mult=BB_STD):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return ma, ma + std_mult * std, ma - std_mult * std


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds EMA9/21, RSI, MACD, Bollinger columns to a raw OHLC DataFrame."""
    df = df.copy()
    close = df["Close"]

    df["ema_fast"] = close.ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=EMA_SLOW, adjust=False).mean()
    df["rsi"] = compute_rsi(close)

    macd, macd_signal, macd_hist = compute_macd(close)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd, macd_signal, macd_hist

    bb_ma, bb_upper, bb_lower = compute_bollinger(close)
    df["bb_ma"], df["bb_upper"], df["bb_lower"] = bb_ma, bb_upper, bb_lower

    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures candle-to-candle volatility."""
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Returns (%K, %D) Stochastic oscillator series."""
    low_min  = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    stoch_k  = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-10)
    return stoch_k, stoch_k.rolling(d_period).mean()


def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R oscillator (-100 to 0)."""
    high_max = df["High"].rolling(period).max()
    low_min  = df["Low"].rolling(period).min()
    return -100 * (high_max - df["Close"]) / (high_max - low_min + 1e-10)


def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    ma  = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-10)


# ------------------- SIGNAL SERIES (vectorized, whole history) -------------------

def detect_macd_cross_series(df: pd.DataFrame) -> pd.Series:
    hist, prev_hist = df["macd_hist"], df["macd_hist"].shift(1)
    result = pd.Series(index=df.index, dtype=object)
    result[(prev_hist < 0) & (hist > 0)] = "bullish"
    result[(prev_hist > 0) & (hist < 0)] = "bearish"
    return result


def detect_ema_cross_series(df: pd.DataFrame) -> pd.Series:
    fast, slow = df["ema_fast"], df["ema_slow"]
    prev_fast, prev_slow = fast.shift(1), slow.shift(1)
    result = pd.Series(index=df.index, dtype=object)
    result[(prev_fast <= prev_slow) & (fast > slow)] = "bullish"
    result[(prev_fast >= prev_slow) & (fast < slow)] = "bearish"
    return result


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds boolean pattern columns. Fully vectorized with numpy — no Python loop.
    Priority (highest first): engulfing > doji > hammer > shooting_star > harami > stars.
    Each candle receives at most one label (mutual exclusivity preserved).
    """
    df = df.copy()

    o  = df["Open"].values.astype(float)
    h  = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    c  = df["Close"].values.astype(float)
    n  = len(df)

    body       = np.abs(c - o)
    rng        = np.maximum(h - lo, 1e-6)
    lower_wick = np.minimum(o, c) - lo
    upper_wick = h - np.maximum(o, c)

    # Previous-candle values (shift +1) — NaN at index 0
    p_o = np.roll(o, 1); p_o[0] = np.nan
    p_c = np.roll(c, 1); p_c[0] = np.nan
    p_body = np.abs(p_c - p_o)

    # Two-candle-back values (shift +2) — NaN at indices 0-1
    o1 = np.roll(o, 2); o1[:2] = np.nan
    c1 = np.roll(c, 2); c1[:2] = np.nan

    # Middle-candle values for 3-candle patterns (shift +1)
    o2     = p_o
    c2     = p_c
    h_mid  = np.roll(h,  1); h_mid[0]  = np.nan
    l_mid  = np.roll(lo, 1); l_mid[0]  = np.nan
    mid_rng = np.maximum(h_mid - l_mid, 1e-6)
    min_mid = np.minimum(o2, c2)
    max_mid = np.maximum(o2, c2)

    # ---- Single-candle conditions ----
    cond_doji          = body / rng <= 0.1
    cond_hammer        = (lower_wick >= 2 * body) & (upper_wick <= body)
    cond_shooting_star = (upper_wick >= 2 * body) & (lower_wick <= body)

    # ---- Two-candle conditions (NaN comparisons yield False → safe at index 0) ----
    cond_bull_eng = (
        (p_c < p_o) & (c > o) &
        (o < p_c)   & (c > p_o) &
        (body > p_body)
    )
    cond_bear_eng = (
        (p_c > p_o) & (c < o) &
        (o > p_c)   & (c < p_o) &
        (body > p_body)
    )
    cond_bull_harami = (p_o > p_c) & (c > o) & (o > p_c) & (c < p_o)
    cond_bear_harami = (p_o < p_c) & (c < o) & (o < p_c) & (c > p_o)

    # ---- Three-candle conditions ----
    cond_morning_star = (
        (c1 < o1) &
        (np.abs(c2 - o2) < mid_rng * 0.3) &
        (c > o2) & (c > (o1 + c1) / 2) &
        (min_mid < c1) & (max_mid > c1)
    )
    cond_evening_star = (
        (c1 > o1) &
        (np.abs(c2 - o2) < mid_rng * 0.3) &
        (c < o2) & (c < (o1 + c1) / 2) &
        (max_mid > c1) & (min_mid < c1)
    )

    # ---- Priority chain ----
    # Assign lowest-priority first; higher-priority overwrites to enforce exclusivity.
    pattern = np.full(n, "", dtype=object)
    pattern[cond_evening_star]  = "evening_star"
    pattern[cond_morning_star]  = "morning_star"
    pattern[cond_bear_harami]   = "bearish_harami"
    pattern[cond_bull_harami]   = "bullish_harami"
    pattern[cond_shooting_star] = "shooting_star"
    pattern[cond_hammer]        = "hammer"
    pattern[cond_doji]          = "doji"
    pattern[cond_bear_eng]      = "bearish_engulfing"
    pattern[cond_bull_eng]      = "bullish_engulfing"

    for col in [
        "bullish_engulfing", "bearish_engulfing",
        "doji", "hammer", "shooting_star",
        "bullish_harami", "bearish_harami", "morning_star", "evening_star",
    ]:
        df[col] = pattern == col

    return df


PATTERN_COLUMNS = [
    "bullish_engulfing", "bearish_engulfing",
    "doji", "hammer", "shooting_star",
    "bullish_harami", "bearish_harami", "morning_star", "evening_star",
]

# Which patterns are inherently bullish vs bearish (for direction labeling)
BULLISH_PATTERNS = {"hammer", "bullish_engulfing", "bullish_harami", "morning_star"}
BEARISH_PATTERNS = {"shooting_star", "bearish_engulfing", "bearish_harami", "evening_star"}
# 'doji' has no inherent direction — it signals indecision, reported as "neutral"


def get_higher_tf_trend(higher_df: pd.DataFrame) -> str | None:
    """Returns 'UP' or 'DOWN' based on the last closed candle of a higher timeframe, using EMA20."""
    if higher_df is None or higher_df.empty or len(higher_df) < 25:
        return None
    close = higher_df["Close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    return "UP" if close.iloc[-1] > ema20.iloc[-1] else "DOWN"


def detect_confluence_signal(row: pd.Series, higher_trend: str | None) -> str | None:
    """A stricter confluence signal that requires clearer agreement across trend and momentum."""
    ema_cross = row.get("ema_cross")
    rsi, macd_hist = row["rsi"], row["macd_hist"]
    close, bb_upper, bb_lower = row["Close"], row["bb_upper"], row["bb_lower"]
    
    # Fetch middle band for structural support check
    bb_ma = row.get("bb_ma")

    if ema_cross != "bullish" and ema_cross != "bearish":
        return None

    if higher_trend is None:
        return None

    # Check for candlestick triggers (using your existing global sets)
    is_bullish_candle = any(row.get(p, False) for p in BULLISH_PATTERNS if p in row.index)
    is_bearish_candle = any(row.get(p, False) for p in BEARISH_PATTERNS if p in row.index)

    # Bullish: Trend UP, healthy RSI (50-70), MACD positive, Price above middle band (support), plus candlestick trigger
    if (ema_cross == "bullish" and higher_trend == "UP" 
        and 50 <= rsi <= 70 and macd_hist > 0 
        and bb_ma < close < bb_upper 
        and is_bullish_candle):
        return "bullish"

    # Bearish: Trend DOWN, healthy RSI (30-50), MACD negative, Price below middle band (resistance), plus candlestick trigger
    if (ema_cross == "bearish" and higher_trend == "DOWN" 
        and 30 <= rsi <= 50 and macd_hist < 0 
        and bb_lower < close < bb_ma 
        and is_bearish_candle):
        return "bearish"

    return None


def analyze(df: pd.DataFrame, higher_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full pipeline: indicators + MACD cross + EMA cross + candlestick patterns."""
    df = add_indicators(df)
    df["macd_cross"] = detect_macd_cross_series(df)
    df["ema_cross"] = detect_ema_cross_series(df)
    df = detect_candlestick_patterns(df)
    return df


# ------------------- PAIR ANALYSIS SNAPSHOT -------------------

def get_pair_analysis(df: pd.DataFrame) -> dict:
    """
    Returns a flat snapshot dict of all indicator values for the most recent
    closed candle. Used by the dashboard's single-instrument analysis panel.
    Returns {} when the DataFrame is too short to compute reliable values.
    """
    if df is None or df.empty or len(df) < 30:
        return {}

    close = df["Close"]
    last  = float(close.iloc[-1])
    prev  = float(close.iloc[-2])
    change     = last - prev
    change_pct = (change / prev * 100) if prev else 0.0

    ema9   = float(close.ewm(span=9,   adjust=False).mean().iloc[-1])
    ema21  = float(close.ewm(span=21,  adjust=False).mean().iloc[-1])
    ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

    rsi = float(compute_rsi(close).iloc[-1])

    macd_line, macd_sig, macd_hist_s = compute_macd(close)

    bb_ma, bb_upper, bb_lower = compute_bollinger(close)
    bb_u = float(bb_upper.iloc[-1])
    bb_l = float(bb_lower.iloc[-1])
    bb_m = float(bb_ma.iloc[-1])
    bb_pct_b     = (last - bb_l) / (bb_u - bb_l + 1e-10)
    bb_bandwidth = (bb_u - bb_l) / (bb_m + 1e-10)

    atr = float(compute_atr(df).iloc[-1])

    stoch_k, stoch_d = compute_stochastic(df)
    wr  = float(compute_williams_r(df).iloc[-1])
    cci = float(compute_cci(df).iloc[-1])

    volume = None
    if "Volume" in df.columns:
        v = float(df["Volume"].iloc[-1])
        if v > 0:
            volume = v

    return {
        "last": last, "change": change, "change_pct": change_pct,
        "high": float(df["High"].iloc[-1]), "low": float(df["Low"].iloc[-1]),
        "volume": volume,
        "ema9": ema9, "ema21": ema21, "ema50": ema50, "ema200": ema200,
        "rsi": rsi,
        "macd":        float(macd_line.iloc[-1]),
        "macd_signal": float(macd_sig.iloc[-1]),
        "macd_hist":   float(macd_hist_s.iloc[-1]),
        "bb_upper": bb_u, "bb_lower": bb_l, "bb_ma": bb_m,
        "bb_pct_b": bb_pct_b, "bb_bandwidth": bb_bandwidth,
        "atr": atr,
        "stoch_k": float(stoch_k.iloc[-1]), "stoch_d": float(stoch_d.iloc[-1]),
        "williams_r": wr, "cci": cci,
    }