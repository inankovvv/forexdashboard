"""
db.py — SQLite storage for detected signals. Simple, file-based, and
portable to Replit as-is (no external database needed for a personal tool).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager

DB_PATH = "signals.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    signal_type TEXT NOT NULL,      -- 'macd_cross','confluence','doji','hammer', etc.
    direction TEXT NOT NULL,        -- 'bullish','bearish','neutral'
    price REAL NOT NULL,
    candle_time TEXT NOT NULL,      -- ISO timestamp of the candle
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(instrument, timeframe, signal_type, candle_time)
);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);
CREATE INDEX IF NOT EXISTS idx_signals_instrument ON signals(instrument);
"""


@contextmanager
def get_conn(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def clear_signals(db_path: str = DB_PATH) -> int:
    """Delete all stored signals and recreate the table schema."""
    with get_conn(db_path) as conn:
        deleted = conn.execute("DELETE FROM signals").rowcount
        conn.commit()
        return deleted


def insert_signal(instrument: str, timeframe: str, signal_type: str, direction: str,
                   price: float, candle_time: str, db_path: str = DB_PATH) -> bool:
    """Returns True if a new row was inserted, False if it was a duplicate (already exists)."""
    with get_conn(db_path) as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO signals
               (instrument, timeframe, signal_type, direction, price, candle_time)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (instrument, timeframe, signal_type, direction, price, candle_time),
        )
        conn.commit()
        return cur.rowcount > 0


def fetch_signals(db_path: str = DB_PATH, instruments=None, timeframes=None,
                   signal_types=None, limit: int = 500):
    """Returns matching signals as a list of sqlite3.Row, most recent first."""
    query = "SELECT * FROM signals WHERE 1=1"
    params = []

    if instruments:
        query += f" AND instrument IN ({','.join('?' * len(instruments))})"
        params += list(instruments)
    if timeframes:
        query += f" AND timeframe IN ({','.join('?' * len(timeframes))})"
        params += list(timeframes)
    if signal_types:
        query += f" AND signal_type IN ({','.join('?' * len(signal_types))})"
        params += list(signal_types)

    query += " ORDER BY candle_time DESC LIMIT ?"
    params.append(limit)

    with get_conn(db_path) as conn:
        return conn.execute(query, params).fetchall()
