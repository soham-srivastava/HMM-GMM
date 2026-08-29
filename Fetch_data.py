"""Data loading with a local cache (yfinance shapes vary between versions)."""

import os

import numpy as np
import pandas as pd

CACHE_DIR = "data_cache"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _normalise(data, ticker):
    """
    yfinance's column layout depends on version and on group_by: flat,
    (ticker, field), or (field, ticker).  Pick whichever level actually holds
    the OHLCV names instead of assuming a position.
    """
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            data = data[ticker]
        else:
            field_level = next(
                (lvl for lvl in range(data.columns.nlevels)
                 if set(OHLCV) & set(data.columns.get_level_values(lvl))),
                data.columns.nlevels - 1,
            )
            data = data.copy()
            data.columns = data.columns.get_level_values(field_level)
    missing = [c for c in OHLCV if c not in data.columns]
    if missing:
        raise RuntimeError(f"missing columns {missing} for {ticker}; "
                           f"got {list(data.columns)[:10]}")
    data = data[OHLCV].copy()
    data = data.loc[:, ~data.columns.duplicated()]
    data.index = pd.to_datetime(data.index)
    return data.sort_index()


def get_data(ticker, start_date, end_date, use_cache=True, auto_adjust=True):
    """Download OHLCV and attach simple returns. Cached to data_cache/."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{ticker}_{start_date}_{end_date}_{int(auto_adjust)}.csv".replace("/", "-")
    path = os.path.join(CACHE_DIR, key)

    if use_cache and os.path.exists(path):
        data = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    else:
        import yfinance as yf
        raw = yf.download(ticker, start=start_date, end=end_date,
                          auto_adjust=auto_adjust, progress=False,
                          group_by="ticker")
        if raw is None or raw.empty:
            raise RuntimeError(f"no data returned for {ticker}")
        data = _normalise(raw, ticker)
        data.index.name = "Date"
        data.to_csv(path)

    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
    data["returns"] = data["Close"].pct_change()
    return data


def get_data_from_csv(file_path, date_col=None):
    """
    Read OHLCV from a CSV.  Column names and the date column vary between
    vendors (date/Date/timestamp, lower- or title-case OHLCV), so normalise
    rather than assuming this project's own export format.
    """
    data = pd.read_csv(file_path)
    if date_col is None:
        date_col = next((c for c in data.columns
                         if c.lower() in ("date", "datetime", "timestamp", "time")),
                        data.columns[0])
    try:
        parsed = pd.to_datetime(data[date_col], errors="coerce")
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)   # keep local wall time
    except (TypeError, ValueError):
        parsed = pd.to_datetime(data[date_col], utc=True,
                                errors="coerce").dt.tz_localize(None)
    data[date_col] = parsed
    data = data.dropna(subset=[date_col]).set_index(date_col).sort_index()
    data.index.name = "Date"

    canon = {c.lower(): c for c in OHLCV}
    data = data.rename(columns={c: canon[c.lower()] for c in data.columns
                                if c.lower() in canon})
    missing = [c for c in OHLCV if c not in data.columns]
    if missing:
        raise RuntimeError(f"{file_path}: missing columns {missing}")

    data = data[OHLCV].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
    data["returns"] = data["Close"].pct_change()
    return data
