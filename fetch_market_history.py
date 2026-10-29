"""Download 5 years of daily Colombo Stock Exchange prices.

The CSE website chart API only returns about one year. Yahoo Finance keeps
the longer daily history. CSE symbols such as COMB.N0000 are not Yahoo
tickers. Yahoo uses a hyphen and a .CM suffix: COMB-N0000.CM.

Run from this folder:

    python fetch_market_history.py

Output:
    data/market/<SYMBOL>.csv     one file per company
    data/cse_prices_5y.csv        all five companies in one table
"""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path("data/market")
COMBINED_PATH = Path("data/cse_prices_5y.csv")

# Five liquid names already used by the charts and the share list.
COMPANIES = [
    {"symbol": "COMB.N0000", "name": "Commercial Bank"},
    {"symbol": "SAMP.N0000", "name": "Sampath Bank"},
    {"symbol": "HNB.N0000", "name": "Hatton National Bank"},
    {"symbol": "JKH.N0000", "name": "John Keells Holdings"},
    {"symbol": "DIAL.N0000", "name": "Dialog Axiata"},
]

COLUMNS = [
    "date",
    "symbol",
    "company",
    "yahoo",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def to_yahoo(symbol: str) -> str:
    """COMB.N0000 -> COMB-N0000.CM"""
    if symbol.upper().endswith(".CM"):
        return symbol
    return symbol.replace(".", "-") + ".CM"


def fetch_chart(yahoo: str) -> dict:
    query = urllib.parse.urlencode({"range": "5y", "interval": "1d", "events": "div,splits"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo)}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        payload = json.loads(response.read().decode("utf-8"))
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(chart["error"])
    result = (chart.get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise RuntimeError(f"No price history returned for {yahoo}")
    return result


def rows_from_chart(company: dict, result: dict) -> list[dict]:
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose") or []
    rows = []
    for index, stamp in enumerate(timestamps):
        close = quote["close"][index] if index < len(quote["close"]) else None
        if close is None:
            continue
        when = datetime.fromtimestamp(stamp, tz=timezone.utc).date().isoformat()
        volume = quote["volume"][index] if index < len(quote["volume"]) else None
        rows.append(
            {
                "date": when,
                "symbol": company["symbol"],
                "company": company["name"],
                "yahoo": to_yahoo(company["symbol"]),
                "open": _num(quote["open"][index]),
                "high": _num(quote["high"][index]),
                "low": _num(quote["low"][index]),
                "close": _num(close),
                "adj_close": _num(adj[index]) if index < len(adj) else _num(close),
                "volume": "" if volume is None else int(volume),
            }
        )
    return rows


def _num(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def trim_stale_tail(rows: list[dict]) -> list[dict]:
    """Yahoo repeats the last CSE close with zero volume after it stops updating."""
    last_trade = 0
    for index, row in enumerate(rows):
        volume = row["volume"]
        if volume not in ("", "0") and int(volume) > 0:
            last_trade = index
    return rows[: last_trade + 1]


def load_history_frame(symbol: str):
    """Load a saved company file as a price table. Returns None if it is missing."""
    import pandas as pd

    path = OUT_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    combined: list[dict] = []
    for company in COMPANIES:
        yahoo = to_yahoo(company["symbol"])
        print(f"Downloading {company['name']} ({company['symbol']} -> {yahoo})...")
        result = fetch_chart(yahoo)
        rows = rows_from_chart(company, result)
        rows = trim_stale_tail(rows)
        if len(rows) < 200:
            raise RuntimeError(f"{yahoo} returned only {len(rows)} rows, which is too short for 5 years.")
        closes = [float(row["close"]) for row in rows]
        unique = len({round(price, 2) for price in closes})
        if unique < 30:
            raise RuntimeError(f"{yahoo} prices barely move ({unique} distinct closes). Not saving that series.")
        out_path = OUT_DIR / f"{company['symbol']}.csv"
        write_csv(out_path, rows)
        combined.extend(rows)
        print(
            f"  {len(rows)} sessions, {rows[0]['date']} to {rows[-1]['date']}, "
            f"close {closes[0]:.2f} -> {closes[-1]:.2f} LKR, saved {out_path}"
        )
        time.sleep(0.4)

    write_csv(COMBINED_PATH, combined)
    print(f"\nSaved {len(combined)} rows to {COMBINED_PATH}")


if __name__ == "__main__":
    main()
