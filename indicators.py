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

Used by scanner.py (writes signals to SQLite) and dashboard.py (Streamlit UI).
"""

from __future__ import annotations

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
    """Adds boolean columns for the manual candlestick patterns from the article."""
    df = df.copy()
    for col in [
        "doji", "hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing",
        "bullish_harami", "bearish_harami", "morning_star", "evening_star",
    ]:
        df[col] = False

    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    for i in range(len(df)):
        curr_o = o.iloc[i]
        curr_h = h.iloc[i]
        curr_l = l.iloc[i]
        curr_c = c.iloc[i]

        body = abs(curr_c - curr_o)
        rng = max(curr_h - curr_l, 1e-6)
        lower_wick = min(curr_o, curr_c) - curr_l
        upper_wick = curr_h - max(curr_o, curr_c)

        if body / rng <= 0.1:
            df.at[df.index[i], "doji"] = True
            continue

        if lower_wick >= 2 * body and upper_wick <= body:
            df.at[df.index[i], "hammer"] = True
            continue

        if upper_wick >= 2 * body and lower_wick <= body:
            df.at[df.index[i], "shooting_star"] = True
            continue

        if i >= 1:
            prev_o = o.iloc[i - 1]
            prev_c = c.iloc[i - 1]
            prev_body = abs(prev_c - prev_o)

            if (
                prev_c < prev_o
                and curr_c > curr_o
                and curr_o < prev_c
                and curr_c > prev_o
                and body > prev_body
            ):
                df.at[df.index[i], "bullish_engulfing"] = True
                continue

            if (
                prev_c > prev_o
                and curr_c < curr_o
                and curr_o > prev_c
                and curr_c < prev_o
                and body > prev_body
            ):
                df.at[df.index[i], "bearish_engulfing"] = True
                continue

            if prev_o > prev_c and curr_o < curr_c and curr_o > prev_c and curr_c < prev_o:
                df.at[df.index[i], "bullish_harami"] = True
                continue

            if prev_o < prev_c and curr_o > curr_c and curr_o < prev_c and curr_c > prev_o:
                df.at[df.index[i], "bearish_harami"] = True
                continue

        if i >= 2:
            o1, c1 = o.iloc[i - 2], c.iloc[i - 2]
            o2, c2 = o.iloc[i - 1], c.iloc[i - 1]
            prev_high = h.iloc[i - 1]
            prev_low = l.iloc[i - 1]

            if c1 < o1 and abs(c2 - o2) < (prev_high - prev_low) * 0.3 and curr_c > o2 and curr_c > (o1 + c1) / 2:
                if min(o2, c2) < c1 and max(o2, c2) > c1:
                    df.at[df.index[i], "morning_star"] = True
                    continue

            if c1 > o1 and abs(c2 - o2) < (prev_high - prev_low) * 0.3 and curr_c < o2 and curr_c < (o1 + c1) / 2:
                if max(o2, c2) > c1 and min(o2, c2) < c1:
                    df.at[df.index[i], "evening_star"] = True

    return df


PATTERN_COLUMNS = [
    "doji", "hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing",
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