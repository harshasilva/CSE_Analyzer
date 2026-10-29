"""Download the daily price list published by the Colombo Stock Exchange website.

The public chart API is https://www.cse.lk/api/companyChartDataByStock.
It needs the numeric stock id from tradeSummary, plus Origin and Referer headers.
Period 5 is the longest range the site will return: about one year of daily bars.
A from/to date range is rejected.

Run from this folder:

    python fetch_cse_history.py
"""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = "https://www.cse.lk/api/"
OUT_DIR = Path("data/cse_site")
COMBINED_PATH = Path("data/cse_site_prices_1y.csv")
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://www.cse.lk",
    "Referer": "https://www.cse.lk/",
    "Content-Type": "application/x-www-form-urlencoded",
}

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
    "high",
    "low",
    "close",
    "volume",
    "turnover",
]


def post(path: str, fields: dict) -> dict | list:
    body = urllib.parse.urlencode(fields).encode()
    request = urllib.request.Request(BASE + path, data=body, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.loads(response.read().decode("utf-8"))


def stock_ids() -> dict[str, str]:
    payload = post("tradeSummary", {})
    wanted = {company["symbol"] for company in COMPANIES}
    found = {}
    for item in payload.get("reqTradeSummery", []):
        symbol = item.get("symbol")
        if symbol in wanted and item.get("id") is not None:
            found[symbol] = str(item["id"])
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"CSE trade summary did not include {', '.join(sorted(missing))}")
    return found


def daily_rows(company: dict, stock_id: str) -> list[dict]:
    payload = post("companyChartDataByStock", {"stockId": stock_id, "period": "5"})
    points = payload.get("chartData") or []
    rows = []
    for point in points:
        stamp = point.get("t")
        close = point.get("p")
        if stamp is None or close is None:
            continue
        when = datetime.fromtimestamp(stamp / 1000, tz=ZoneInfo("Asia/Colombo")).date().isoformat()
        rows.append(
            {
                "date": when,
                "symbol": company["symbol"],
                "company": company["name"],
                "high": point.get("h") if point.get("h") is not None else "",
                "low": point.get("l") if point.get("l") is not None else "",
                "close": close,
                "volume": point.get("q") if point.get("q") is not None else "",
                "turnover": point.get("s") if point.get("s") is not None else "",
            }
        )
    rows.sort(key=lambda row: row["date"])
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ids = stock_ids()
    combined: list[dict] = []
    for company in COMPANIES:
        symbol = company["symbol"]
        print(f"Downloading {company['name']} ({symbol}, CSE id {ids[symbol]})...")
        rows = daily_rows(company, ids[symbol])
        if len(rows) < 100:
            raise RuntimeError(f"{symbol} returned only {len(rows)} daily rows.")
        write_csv(OUT_DIR / f"{symbol}.csv", rows)
        combined.extend(rows)
        print(f"  {len(rows)} sessions, {rows[0]['date']} to {rows[-1]['date']}, close {rows[-1]['close']}")
        time.sleep(0.4)
    write_csv(COMBINED_PATH, combined)
    print(f"\nSaved {len(combined)} official CSE rows to {COMBINED_PATH}")


if __name__ == "__main__":
    main()
