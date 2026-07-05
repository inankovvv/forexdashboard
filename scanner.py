"""
scanner.py — runs the signal engine across instruments/timeframes and stores
results in SQLite (signals.db). Two modes:

  --mode backtest   Scan the last N days once, store everything found, exit.
  --mode live       Loop forever, checking for newly-closed candles, storing
                     new signals, and sending a Telegram alert for each one.

Usage:
  pip install yfinance pandas numpy requests
  python scanner.py --mode backtest --days 30
  python scanner.py --mode live --check-every 300
  python scanner.py --mode live --instruments EURUSD,GBPUSD,XAU_USD --timeframes 15m,1h
"""

from __future__ import annotations

import argparse
import time
import traceback
from datetime import datetime
from typing import Iterator

import pandas as pd
import requests

import db
from indicators import (
    INSTRUMENTS, TIMEFRAMES, HIGHER_TF, PATTERN_COLUMNS,
    BULLISH_PATTERNS, BEARISH_PATTERNS,
    get_multi_data, analyze, get_higher_tf_trend, detect_confluence_signal,
)

# ------------------- CONFIG -------------------
# Fill these in, or set as environment variables (recommended once on Replit:
# use Replit Secrets instead of hardcoding).
import os
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")
# -----------------------------------------------

SOFIA_TZ = "Europe/Sofia"


def to_sofia_time(ts) -> pd.Timestamp:
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert(SOFIA_TZ)


def format_candle_time(ts) -> str:
    return to_sofia_time(ts).isoformat()


def send_telegram_message(text: str) -> None:
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        return  # not configured yet, skip silently in backtest/testing
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        if resp.status_code != 200:
            print(f"[WARN] Telegram send failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[WARN] Telegram send error: {e}")


def process_instrument(key: str, tf: str, raw_df, higher_df, only_latest: bool,
                        lookback_days: int | None, alert: bool) -> int:
    """
    Analyzes one instrument's data, stores any signals found, optionally alerts.
    If only_latest=True, only the last closed candle is checked (live mode).
    If False, the whole lookback window is scanned (backtest mode).
    Returns count of NEW signals inserted.
    """
    if raw_df.empty or len(raw_df) < 40:
        return 0

    df = analyze(raw_df)
    higher_trend = get_higher_tf_trend(higher_df) if higher_df is not None else None

    if only_latest:
        if lookback_days is None or lookback_days <= 0:
            rows_to_check = df.iloc[[-1]]
        else:
            cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
            rows_to_check = df[df.index >= cutoff]
    else:
        cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
        rows_to_check = df[df.index >= cutoff]

    new_count = 0
    for ts, row in rows_to_check.iterrows():
        candle_time = format_candle_time(ts)
        price = round(float(row["Close"]), 5)
        found_signals = []  # list of (signal_type, direction)

        if row.get("macd_cross") in ("bullish", "bearish"):
            found_signals.append(("macd_cross", row["macd_cross"]))

        conf = detect_confluence_signal(row, higher_trend)
        if conf:
            found_signals.append(("confluence", conf))

        for pattern in PATTERN_COLUMNS:
            if bool(row.get(pattern)):
                if pattern in BULLISH_PATTERNS:
                    direction = "bullish"
                elif pattern in BEARISH_PATTERNS:
                    direction = "bearish"
                else:
                    direction = "neutral"
                found_signals.append((pattern, direction))

        for signal_type, direction in found_signals:
            inserted = db.insert_signal(key, tf, signal_type, direction, price, candle_time)
            if inserted:
                new_count += 1
                if alert:
                    arrow = "▲" if direction == "bullish" else ("▼" if direction == "bearish" else "◆")
                    text = (
                        f"{arrow} {direction.upper()} {signal_type.replace('_', ' ').title()}\n"
                        f"{key} — {tf}\n"
                        f"Price: {price}\n"
                        f"Candle: {ts.strftime('%Y-%m-%d %H:%M UTC')}"
                    )
                    print(f"[ALERT]\n{text}\n")
                    send_telegram_message(text)

    return new_count


import time as _time


def iter_scan_batches(timeframes: list[str], batch_size: int = 2) -> Iterator[list[str]]:
    """Yield timeframes in small batches so long scans stay responsive."""
    for i in range(0, len(timeframes), batch_size):
        yield timeframes[i:i + batch_size]


def run_scan(instrument_keys: list[str], timeframes: list[str], lookback_days: int,
             only_latest: bool, alert: bool, progress_callback=None) -> int:
    """
    Runs one full pass across all instruments x timeframes. Returns total new signals.
    progress_callback(tf, step, total_steps) is called after each timeframe finishes,
    if provided — used by the dashboard to show a progress bar.
    """
    total_new = 0
    total_steps = len(timeframes)
    for batch_idx, batch in enumerate(iter_scan_batches(timeframes, batch_size=2), start=1):
        for step_offset, tf in enumerate(batch, start=1):
            global_step = (batch_idx - 1) * 2 + step_offset
            needs_higher = tf in HIGHER_TF
            data = get_multi_data(instrument_keys, tf, lookback_days=max(lookback_days, 5))
            higher_data = {}
            if needs_higher:
                _time.sleep(1)
                higher_tf = HIGHER_TF[tf]
                higher_data = get_multi_data(instrument_keys, higher_tf, lookback_days=max(lookback_days, 10))

            for key in instrument_keys:
                raw_df = data.get(key)
                higher_df = higher_data.get(key) if higher_data else None
                try:
                    total_new += process_instrument(
                        key, tf, raw_df, higher_df, only_latest, lookback_days, alert
                    )
                except Exception:
                    print(f"[ERROR] processing {key} {tf}:")
                    traceback.print_exc()

            if progress_callback:
                progress_callback(tf, global_step, total_steps)

            _time.sleep(1)

    return total_new


def main():
    parser = argparse.ArgumentParser(description="Multi-instrument signal scanner.")
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    parser.add_argument("--days", type=int, default=30, help="Backtest lookback window in days")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,4h,1d")
    parser.add_argument("--instruments", type=str, default="all")
    parser.add_argument("--check-every", type=int, default=300, help="Seconds between checks (live mode)")
    args = parser.parse_args()

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    for tf in timeframes:
        if tf not in TIMEFRAMES:
            raise SystemExit(f"Unsupported timeframe '{tf}'. Choose from {list(TIMEFRAMES)}")

    if args.instruments.strip().lower() == "all":
        instrument_keys = list(INSTRUMENTS)
    else:
        instrument_keys = [i.strip() for i in args.instruments.split(",") if i.strip()]
        for k in instrument_keys:
            if k not in INSTRUMENTS:
                raise SystemExit(f"Unknown instrument '{k}'. Choose from {list(INSTRUMENTS)}")

    db.init_db()

    if args.mode == "backtest":
        print(f"Backtesting {len(instrument_keys)} instruments | timeframes={timeframes} | last {args.days} days")
        new_count = run_scan(instrument_keys, timeframes, args.days, only_latest=False, alert=False)
        print(f"\nDone. {new_count} new signals stored in {db.DB_PATH}.")
    else:
        print(f"Starting live scan | {len(instrument_keys)} instruments | timeframes={timeframes} | every {args.check_every}s")
        if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
            print("[WARN] TELEGRAM_BOT_TOKEN not set — alerts will be skipped. Set env vars or edit CONFIG.")
        while True:
            try:
                new_count = run_scan(instrument_keys, timeframes, lookback_days=5, only_latest=True, alert=True)
                if new_count:
                    print(f"[{datetime.utcnow().isoformat()}] {new_count} new signal(s) stored.")
            except Exception:
                print("[ERROR] in scan loop:")
                traceback.print_exc()
            time.sleep(args.check_every)


if __name__ == "__main__":
    main()