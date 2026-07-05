"""
app.py — Streamlit dashboard for the signal engine.

Usage:
  pip install streamlit yfinance pandas numpy requests plotly
  streamlit run app.py
"""

from datetime import datetime, timezone
from math import ceil
from urllib.parse import quote

import pandas as pd
import streamlit as st

import db
from indicators import INSTRUMENTS, TIMEFRAMES, PATTERN_COLUMNS
from scanner import run_scan

st.set_page_config(page_title="Signal Dashboard", layout="wide")
db.init_db()

PASSWORD_SECRET_KEY = "app_password"
PASSWORD = None
try:
    secrets = st.secrets
    if secrets and PASSWORD_SECRET_KEY in secrets:
        PASSWORD = secrets[PASSWORD_SECRET_KEY]
except Exception:
    PASSWORD = None

if "scan_running" not in st.session_state:
    st.session_state["scan_running"] = False

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if PASSWORD:
    if not st.session_state.get("authenticated"):
        login_placeholder = st.empty()
        with login_placeholder.container():
            st.title("🔒 Secure Dashboard")
            st.info("Enter the app password configured in Streamlit secrets to continue.")

            with st.form("login_form"):
                password_input = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Log in")

        if submit_button:
            if password_input == PASSWORD:
                st.session_state["authenticated"] = True
                login_placeholder.empty()
                st.success("Authentication successful. Loading dashboard...")
            else:
                st.error("Incorrect password. Try again.")

        if not st.session_state.get("authenticated"):
            st.stop()
else:
    st.warning(
        "No app password configured. Set `app_password` in Streamlit secrets to protect access."
    )

ALL_SIGNAL_TYPES = ["macd_cross", "confluence"] + PATTERN_COLUMNS
DEFAULT_SCAN_INSTRUMENTS = list(INSTRUMENTS)
DEFAULT_SCAN_TIMEFRAMES = list(TIMEFRAMES)
META_LAST_SCAN = "last_scan_time"
META_CLEARED_AT = "signals_cleared_at"

# Maps our instrument keys to TradingView's symbol format for the embedded
# widget. These are best-effort — TradingView's exact ticker for indices/
# commodities can vary by data feed, so there's a manual override in the UI
# if any of these don't load correctly for you.
TRADINGVIEW_SYMBOLS = {
    "EURUSD": "FX:EURUSD", "GBPUSD": "FX:GBPUSD", "USDJPY": "FX:USDJPY",
    "AUDUSD": "FX:AUDUSD", "NZDUSD": "FX:NZDUSD", "EURGBP": "FX:EURGBP",
    "EURJPY": "FX:EURJPY", "EURCHF": "FX:EURCHF", "EURAUD": "FX:EURAUD",
    "EURNZD": "FX:EURNZD", "GBPJPY": "FX:GBPJPY", "GBPCHF": "FX:GBPCHF",
    "GBPAUD": "FX:GBPAUD", "GBPNZD": "FX:GBPNZD", "AUDJPY": "FX:AUDJPY",
    "AUDCHF": "FX:AUDCHF", "NZDJPY": "FX:NZDJPY", "NZDCHF": "FX:NZDCHF",
    "WTI": "TVC:USOIL", "GER30": "TVC:DE30", "NAS100": "TVC:NDX",
    "US30": "TVC:DJI", "XAU_USD": "TVC:GOLD",
}


st.title("📊 Multi-Instrument Signal Dashboard")


def format_candle_time(value, timeframe=None):
    if not value:
        return "—"
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return str(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("Europe/Sofia")

    if timeframe in {"15m", "30m", "1h", "4h", "1d"}:
        delta = {
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "1d": pd.Timedelta(days=1),
        }[timeframe]
        open_ts = ts - delta
        return f"{open_ts.strftime('%Y-%m-%d %H:%M')} → {ts.strftime('%Y-%m-%d %H:%M')}"

    return ts.strftime("%Y-%m-%d %H:%M")


# ------------------- SIDEBAR: FILTERS + SCAN CONTROLS -------------------
with st.sidebar:
    st.header("Filters")
    selected_instruments = st.multiselect(
        "Instruments",
        options=list(INSTRUMENTS),
        default=DEFAULT_SCAN_INSTRUMENTS,
        help="All supported instruments are selected by default. Remove any instrument to narrow the scan.",
    )
    selected_timeframes = st.multiselect(
        "Timeframes",
        options=list(TIMEFRAMES),
        default=DEFAULT_SCAN_TIMEFRAMES,
        help="All supported timeframes are selected by default. Remove any timeframe to speed up the scan.",
    )
    selected_signal_types = st.multiselect(
        "Signal types", options=ALL_SIGNAL_TYPES,
        default=["confluence", "bullish_engulfing", "bearish_engulfing"],
        help="Doji/hammer/harami/stars/MACD cross tend to be frequent — add them if you want the noise. "
             "See the FAQ tab for what each one means.",
    )
    row_limit = st.slider("Max rows shown", 50, 2000, 300, step=50)
    st.caption("Tip: lower the row limit or remove instruments/timeframes if the UI feels slow.")

    st.divider()
    st.header("Run a scan")

    scan_running = st.session_state.get("scan_running", False)

    last_scan_iso = db.get_meta(META_LAST_SCAN)
    cleared_at_iso = db.get_meta(META_CLEARED_AT)
    last_scan = None
    cleared_at = None

    if last_scan_iso:
        try:
            last_scan = datetime.fromisoformat(last_scan_iso)
        except ValueError:
            last_scan = None

    if cleared_at_iso:
        try:
            cleared_at = datetime.fromisoformat(cleared_at_iso)
        except ValueError:
            cleared_at = None

    if last_scan:
        st.markdown(f"**Last latest-candles scan:** {last_scan.strftime('%Y-%m-%d %H:%M UTC')}")
    else:
        st.markdown("**Last latest-candles scan:** _never run yet_")

    if cleared_at:
        st.markdown(
            f"**Database cleared:** {cleared_at.strftime('%Y-%m-%d %H:%M UTC')} — the next latest-candles scan will start fresh."
        )

    st.info(
        "Latest-candles scans are incremental when possible: repeated clicks scan only new data since the last successful check."
    )

    def safe_run_scan(*args, **kwargs):
        try:
            return run_scan(*args, **kwargs)
        except Exception as exc:
            st.error(f"Scan failed: {exc}")
            return 0
        finally:
            st.session_state["scan_running"] = False

    if st.button(
        "⚡ Check latest candles now",
        width='stretch',
        disabled=scan_running,
        help="Check the latest recent candles for each selected instrument/timeframe and store any new signals.",
    ):
        st.session_state["scan_running"] = True

        last_scan_iso = db.get_meta(META_LAST_SCAN)
        cleared_at_iso = db.get_meta(META_CLEARED_AT)
        now = datetime.now(timezone.utc)
        last_scan = None
        cleared_at = None

        if last_scan_iso:
            try:
                last_scan = datetime.fromisoformat(last_scan_iso)
            except ValueError:
                last_scan = None

        if cleared_at_iso:
            try:
                cleared_at = datetime.fromisoformat(cleared_at_iso)
            except ValueError:
                cleared_at = None

        if last_scan and cleared_at and cleared_at > last_scan:
            last_scan = None
            st.info("Database was cleared since the last scan. Running a fresh latest-candles scan.")

        if last_scan:
            elapsed_days = ceil((now - last_scan).total_seconds() / 86400)
            lookback_days = min(max(1, elapsed_days + 1), 30)
            st.info(
                f"Rescanning from last check ({last_scan.strftime('%Y-%m-%d %H:%M UTC')}) "
                f"with a {lookback_days}-day window."
            )
        else:
            lookback_days = 1

        with st.spinner("Scanning latest candles..."):
            progress_bar = st.progress(0, text="Starting...")

            def _update_progress2(tf, step, total_steps):
                progress_bar.progress(step / total_steps, text=f"Finished {tf} ({step}/{total_steps})")

            new_count = safe_run_scan(
                selected_instruments or list(INSTRUMENTS),
                selected_timeframes or list(TIMEFRAMES),
                lookback_days=lookback_days,
                only_latest=True,
                alert=False,
                progress_callback=_update_progress2,
            )

        db.set_meta(META_LAST_SCAN, now.isoformat())

        if new_count:
            st.success(f"Check complete — {new_count} new signals stored.")
        else:
            st.info("Scan complete — no new signals were found.")

    st.caption("Backtest controls")
    st.info("Backtests scan the selected history and store any new signals. Use this when you want to rebuild signal history from the selected window.")
    scan_days = st.number_input("Backtest lookback (days)", min_value=1, max_value=90, value=10)

    if st.session_state.get("confirm_backtest", False):
        st.warning("This will scan the full selected history and store all new signals. Please confirm to proceed.")
        if st.button("Confirm full backtest scan now", disabled=scan_running, key="confirm_backtest_confirm"):
            st.session_state["scan_running"] = True
            st.session_state["confirm_backtest"] = False
            with st.spinner("Running backtest scan..."):
                progress_bar = st.progress(0, text="Starting...")

                def _update_progress(tf, step, total_steps):
                    progress_bar.progress(step / total_steps, text=f"Finished {tf} ({step}/{total_steps})")

                new_count = safe_run_scan(
                    selected_instruments or list(INSTRUMENTS),
                    selected_timeframes or list(TIMEFRAMES),
                    lookback_days=scan_days,
                    only_latest=False,
                    alert=False,
                    progress_callback=_update_progress,
                )

            if new_count:
                st.success(f"Backtest complete — {new_count} new signals stored.")
            else:
                st.info("Backtest complete — no new signals were found.")
        if st.button("Cancel backtest", key="cancel_backtest"):
            st.session_state["confirm_backtest"] = False
    else:
        if st.button(
            "🔍 Run backtest scan now",
            width='stretch',
            disabled=scan_running,
            help="Scan the full selected history for the chosen lookback period and store all new signals.",
        ):
            st.session_state["confirm_backtest"] = True

    st.divider()
    st.caption("Background scans: run the live scanner in another terminal to update every 5 minutes even when the dashboard is closed.")
    st.caption("Database")
    st.info("Clearing stored signals removes all rows and resets the scan history metadata. Use only if you want to restart from scratch.")

    if st.session_state.get("confirm_clear", False):
        st.warning("This will delete all stored signals from the database. Please confirm to proceed.")
        if st.button("Confirm clear stored signals", key="confirm_clear_confirm"):
            deleted = db.clear_signals()
            st.session_state.pop("signals_df", None)
            st.session_state["confirm_clear"] = False
            st.success(f"Cleared {deleted} stored signal(s).")
        if st.button("Cancel clear", key="cancel_clear"):
            st.session_state["confirm_clear"] = False
    else:
        if st.button("🗑️ Clear all stored signals", width='stretch', disabled=scan_running):
            st.session_state["confirm_clear"] = True

# ------------------- TABS -------------------
tab_dashboard, tab_faq = st.tabs(["📊 Dashboard", "📖 FAQ / Help"])

with tab_dashboard:
    # ------------------- METRICS -------------------
    rows = db.fetch_signals(
        instruments=selected_instruments or None,
        timeframes=selected_timeframes or None,
        signal_types=selected_signal_types or None,
        limit=row_limit,
    )
    signals_df = pd.DataFrame([dict(r) for r in rows])

    col1, col2, col3 = st.columns(3)
    col1.metric("Signals shown", len(signals_df))
    if not signals_df.empty:
        col2.metric("Instruments covered", signals_df["instrument"].nunique())
        latest_row = signals_df.sort_values("candle_time", ascending=False).iloc[0]
        col3.metric(
            "Most recent",
            format_candle_time(latest_row["candle_time"], latest_row["timeframe"]),
        )
    else:
        col2.metric("Instruments covered", 0)
        col3.metric("Most recent", "—")

    st.divider()

    # ------------------- SIGNALS TABLE -------------------
    st.subheader("Signals")
    if signals_df.empty:
        st.info("No signals yet — run a scan from the sidebar to populate this table.")
    else:
        display_df = signals_df.copy()
        display_df["candle_time"] = display_df.apply(
            lambda row: format_candle_time(row["candle_time"], row["timeframe"]),
            axis=1,
        )
        display_df = display_df[["instrument", "timeframe", "signal_type", "direction", "price", "candle_time"]]
        display_df.columns = ["Instrument", "Timeframe", "Signal", "Direction", "Price", "Candle Open → Close"]

        def highlight_direction(val):
            if val == "bullish":
                return "color: #2ecc71; font-weight: bold"
            if val == "bearish":
                return "color: #e74c3c; font-weight: bold"
            return "color: #95a5a6"

        st.dataframe(
            display_df.style.map(highlight_direction, subset=["Direction"]),
            width='stretch',
            hide_index=True,
            height=450,
        )

        csv = signals_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download as CSV", csv, "signals.csv", "text/csv")

    st.divider()

    # ------------------- TRADINGVIEW CHART -------------------
    st.subheader("Chart view")
    st.caption(
        "Use TradingView's own symbol search (top-left of the chart) and interval "
        "selector (top toolbar) to switch instruments/timeframes."
    )

    default_symbol = TRADINGVIEW_SYMBOLS["EURUSD"]
    tradingview_url = (
        "https://www.tradingview.com/widgetembed/?frameElementId=tradingview_chart"
        "&widgetType=widget"
        f"&symbol={quote(default_symbol, safe='')}"
        "&interval=60"
        "&theme=dark"
        "&style=1"
        "&locale=en"
        f"&timezone={quote('Europe/Sofia', safe='')}"
        "&toolbarbg=%23f1f3f6"
        "&allow_symbol_change=true"
        "&hide_side_toolbar=false"
    )
    st.iframe(tradingview_url, height=620)
    st.caption(
        "If an index/commodity symbol doesn't load by default (data feeds vary by provider), "
        "use TradingView's own symbol search inside the chart — try prefixes like OANDA:, "
        "CAPITALCOM:, or FX_IDC: for that instrument."
    )


with tab_faq:
    st.subheader("What each signal means")

    with st.expander("📊 MACD Cross"):
        st.markdown("""
The MACD (Moving Average Convergence Divergence) line crosses its signal line.

- **Bullish**: MACD histogram flips from negative to positive — momentum turning up.
- **Bearish**: MACD histogram flips from positive to negative — momentum turning down.

Settings used: 12/26/9 (the standard, most widely used combination).

**Caveat**: MACD crosses fire often and are prone to false signals in choppy/sideways
markets — this is a raw momentum signal, not a standalone trade trigger.
        """)

    with st.expander("🕯️ Bullish / Bearish Engulfing"):
        st.markdown("""
A two-candle reversal pattern where the second candle's body completely
"engulfs" the body of the candle before it.

- **Bullish engulfing**: a red (down) candle followed by a larger green (up)
  candle whose body fully covers the red candle's body. Suggests buyers just
  overwhelmed sellers.
- **Bearish engulfing**: the mirror image — a green candle followed by a
  larger red candle that fully covers it. Suggests sellers just took control.

Generally considered more reliable at the end of a clear trend (i.e. a
bullish engulfing after a downtrend) than in the middle of a sideways range.
        """)

    with st.expander("🎯 Confluence (multi-filter signal)"):
        st.markdown("""
The strictest signal in the dashboard — it only fires when **all 5** of these
agree at once:

1. **EMA9/21 cross** — fast/slow moving average crossover signals a trend shift
2. **Higher-timeframe trend** — the timeframe one level up agrees with the direction
   (e.g. a 15m signal needs the 1h trend to agree)
3. **RSI in a healthy zone** — 40–70 for bullish, 30–60 for bearish (momentum
   present but not already exhausted)
4. **MACD histogram** — confirms the same direction
5. **Not at a Bollinger Band extreme** — price isn't already stretched into a
   likely reversal zone

**Why it matters**: any single indicator alone throws off a lot of false
signals. Confluence means several independent signals are pointing the same
way at the same time — a rarer, generally higher-conviction setup. You'll see
far fewer confluence signals than MACD crosses or engulfing bars — that's expected.

**Caveat**: confluence reduces false signals, it doesn't eliminate them. It's
not a guarantee, just better odds than any one filter alone.
        """)

    with st.expander("🕯️ Candlestick Patterns (Doji, Hammer, Shooting Star, Harami, Stars)"):
        st.markdown("""
- **Doji** — open and close are almost identical, tiny real body. Signals
  indecision, not a directional bet by itself. *(Direction: neutral)*
- **Hammer** — small body near the top of the range, long lower wick (2x+ the
  body). Suggests sellers pushed price down but buyers rejected it. Bullish
  reversal signal, especially after a downtrend.
- **Shooting Star** — the mirror of the hammer: small body near the bottom,
  long upper wick. Bearish reversal signal, especially after an uptrend.
- **Bullish/Bearish Harami** — a large candle followed by a small candle whose
  body sits entirely *inside* the prior candle's body. Signals momentum
  pausing/potentially reversing.
- **Morning Star** — 3-candle bullish reversal: big red candle → small
  indecisive candle → big green candle closing well into the first candle's body.
- **Evening Star** — the bearish mirror of the Morning Star.

**Caveat**: single-candle patterns (doji, hammer, shooting star) are common
and noisy — they fire on a large fraction of candles. Best used as
*confirmation* alongside another signal, not traded alone.
        """)

    st.divider()
    st.subheader("Other things worth knowing")

    with st.expander("Why do timeframes matter, and what's special about 4h?"):
        st.markdown("""
- 15m/30m/1h/1d all come directly from Yahoo Finance.
- **4h isn't a real Yahoo interval** — it's built by resampling 1h candles into
  4-hour blocks. This is accurate for *closed* candles, but the current
  still-forming 4h candle may look slightly different until all its
  underlying 1h candles have closed.
        """)

    with st.expander("What does Direction (bullish/bearish/neutral) mean?"):
        st.markdown("""
- **Bullish** — the signal suggests upward price movement.
- **Bearish** — the signal suggests downward price movement.
- **Neutral** — currently only applies to Doji, which signals indecision
  rather than a direction.
        """)

    with st.expander("How should I actually trade these signals? (general framework)"):
        st.markdown("""
This dashboard flags *setups* — it doesn't manage risk for you. A common,
disciplined framework people use:

1. **Entry**: on the close of the candle where the signal fires (not mid-candle
   — indicators can still shift before the candle closes).
2. **Stop loss**: below the recent swing low (bullish) or above the recent
   swing high (bearish).
3. **Position size**: risk a small, fixed % of account per trade (1–2% is
   common) rather than a fixed lot size, so a losing streak doesn't compound.
4. **Target**: decide your risk:reward ratio (e.g. 1:2) *before* entering, not
   after.
5. **Context**: avoid trading through major news releases (check Forex
   Factory's calendar), and be aware of session — forex signals on lower
   timeframes tend to be more reliable during the London/NY session overlap.

**Before trading anything live**: use the backtest scan to see how often a
given signal type actually fired historically for your instrument/timeframe,
and what happened afterward — don't assume a signal works just because the
logic sounds reasonable.

*This is general educational information, not personalized financial advice.*
        """)

    with st.expander("Why are there so many signals when I include all patterns?"):
        st.markdown("""
Doji, hammer, and harami patterns are single/two-candle shapes that occur
naturally and often — especially on lower timeframes (15m/30m) with 23
instruments running in parallel. This is expected, not a bug. The sidebar
defaults to hiding the noisiest patterns (doji, hammer, shooting star, harami,
stars) — add them back in if you want to see everything.
        """)