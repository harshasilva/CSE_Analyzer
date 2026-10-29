"""Download the CSE market report for every listed stock.

Today's full board is the live market report. Each symbol also gets the
longest daily chart the CSE website returns (period 5, about one year).

Run from this folder:

    python fetch_market_all.py

Output:
    data/cse_market_all_today.csv   today's price, change, volume, and turnover
    data/cse_api/charts/<SYMBOL>.csv
    data/cse_market_all.csv         one year of daily bars for every symbol
    data/cse_market_all_archive.csv every official close seen so far
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import requests

from fetch_cse_api import (
    BASE,
    HEADERS,
    OUT,
    PAUSE,
    as_list,
    chart_rows,
    flat_row,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parent
TODAY_PATH = ROOT / "data" / "cse_market_all_today.csv"
HISTORY_PATH = ROOT / "data" / "cse_market_all.csv"
ARCHIVE_PATH = ROOT / "data" / "cse_market_all_archive.csv"
CHART_DIR = OUT / "charts"
FRESH_SECONDS = 18 * 60 * 60
CHART_COLUMNS = [
    "date",
    "symbol",
    "company",
    "open",
    "high",
    "low",
    "close",
    "change",
    "change_pct",
    "volume",
    "turnover",
]
TODAY_COLUMNS = [
    "symbol",
    "name",
    "id",
    "lastTradedPrice",
    "previousClose",
    "change",
    "percentageChange",
    "open",
    "high",
    "low",
    "closingPrice",
    "sharevolume",
    "turnover",
    "tradevolume",
]

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def post(path: str, fields: dict | None = None) -> object:
    delay = 0.6
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = SESSION.post(BASE + path, data=fields or {}, timeout=45)
            response.raise_for_status()
            if not response.content:
                return None
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                raise
            if attempt == 3:
                break
            time.sleep(delay)
            delay *= 2
    if last_error is not None:
        raise last_error
    return None


def listed_stocks() -> list[dict]:
    board = as_list(post("tradeSummary"), "reqTradeSummery")
    rows = [flat_row(item) for item in board]
    write_csv(TODAY_PATH, rows, TODAY_COLUMNS)
    stocks = []
    seen: set[str] = set()
    for item in board:
        symbol = item.get("symbol")
        stock_id = item.get("id")
        if not symbol or stock_id is None or symbol in seen:
            continue
        seen.add(symbol)
        stocks.append(
            {
                "symbol": str(symbol),
                "id": str(stock_id),
                "name": str(item.get("name") or ""),
            }
        )
    stocks.sort(key=lambda row: row["symbol"])
    print(f"Today's market report: {len(rows)} rows, {len(stocks)} symbols -> {TODAY_PATH}", flush=True)
    return stocks


def chart_is_fresh(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 80:
        return False
    return (time.time() - path.stat().st_mtime) < FRESH_SECONDS


def chart_last_date(path: Path) -> str:
    last = ""
    if not path.exists():
        return ""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text and not text.startswith("date,"):
                last = text.split(",")[0]
    return last


def download_charts(stocks: list[dict], needed_date: str | None = None) -> dict:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    skipped = 0
    failed = []
    for index, stock in enumerate(stocks, start=1):
        symbol = stock["symbol"]
        path = CHART_DIR / f"{symbol}.csv"
        if needed_date:
            already = chart_last_date(path) >= needed_date
        else:
            already = chart_is_fresh(path)
        if already:
            skipped += 1
            print(f"  {index}/{len(stocks)} {symbol} already saved", flush=True)
            continue
        try:
            payload = post("companyChartDataByStock", {"stockId": stock["id"], "period": "5"})
        except requests.RequestException as exc:
            failed.append({"symbol": symbol, "error": str(exc)})
            print(f"  {index}/{len(stocks)} {symbol} failed: {exc}", flush=True)
            time.sleep(PAUSE)
            continue
        points = payload.get("chartData") if isinstance(payload, dict) else []
        full, _site = chart_rows(symbol, stock["name"], points or [])
        write_csv(path, full, CHART_COLUMNS)
        if full:
            saved.append(symbol)
            print(
                f"  {index}/{len(stocks)} {symbol}: {len(full)} sessions, {full[0]['date']} to {full[-1]['date']}",
                flush=True,
            )
        else:
            failed.append({"symbol": symbol, "error": "no chart rows"})
            print(f"  {index}/{len(stocks)} {symbol}: no chart rows", flush=True)
        time.sleep(PAUSE)
    return {"downloaded": len(saved), "skipped": skipped, "failed": failed}


def remember_official_closes(rows: list[dict]) -> int:
    """Merge these daily bars into the archive.

    The CSE chart download only covers about one year. The archive keeps
    older sessions when a later download no longer includes them.
    """
    existing: list[dict] = []
    if ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size > 0:
        with ARCHIVE_PATH.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    keyed: dict[tuple[str, str], dict] = {}
    for row in existing:
        symbol = str(row.get("symbol") or "")
        day = str(row.get("date") or "")
        if symbol and day:
            keyed[(symbol, day)] = row
    for row in rows:
        symbol = str(row.get("symbol") or "")
        day = str(row.get("date") or "")
        if not symbol or not day:
            continue
        keyed[(symbol, day)] = {column: row.get(column, "") for column in CHART_COLUMNS}
    merged = [keyed[key] for key in sorted(keyed)]
    write_csv(ARCHIVE_PATH, merged, CHART_COLUMNS)
    print(f"Official close archive: {len(merged)} rows -> {ARCHIVE_PATH}", flush=True)
    return len(merged)


def seed_archive_if_missing() -> None:
    if ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size > 0:
        return
    if not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0:
        return
    with HISTORY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows:
        remember_official_closes(rows)


def combine_charts(stocks: list[dict]) -> dict:
    names = {stock["symbol"]: stock["name"] for stock in stocks}
    paths = sorted(CHART_DIR.glob("*.csv"))
    rows = []
    coverage = []
    for path in paths:
        symbol_rows = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not row.get("date") or not row.get("symbol"):
                    continue
                board_name = names.get(row["symbol"])
                if board_name:
                    row["company"] = board_name
                symbol_rows.append(row)
        if not symbol_rows:
            continue
        symbol_rows.sort(key=lambda row: row["date"])
        rows.extend(symbol_rows)
        coverage.append(
            {
                "symbol": path.stem,
                "company": symbol_rows[0].get("company") or names.get(path.stem, ""),
                "sessions": len(symbol_rows),
                "from": symbol_rows[0]["date"],
                "to": symbol_rows[-1]["date"],
                "close": symbol_rows[-1].get("close") or "",
            }
        )
    rows.sort(key=lambda row: (row["symbol"], row["date"]))
    write_csv(HISTORY_PATH, rows, CHART_COLUMNS)
    archive_rows = remember_official_closes(rows)
    write_csv(
        ROOT / "data" / "cse_market_all_symbols.csv",
        coverage,
        ["symbol", "company", "sessions", "from", "to", "close"],
    )
    print(f"Combined history: {len(rows)} rows, {len(coverage)} symbols -> {HISTORY_PATH}", flush=True)
    return {"rows": len(rows), "symbols": len(coverage), "archive_rows": archive_rows}


def max_saved_session() -> str:
    path = ROOT / "data" / "cse_market_all_symbols.csv"
    latest = ""
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                day = str(row.get("to") or "")
                if day > latest:
                    latest = day
    if latest:
        return latest
    return chart_last_date(HISTORY_PATH)


def latest_traded_session() -> str:
    from fetch_cse_api import stamp_to_date

    if not TODAY_PATH.exists() or TODAY_PATH.stat().st_size == 0:
        return ""
    latest = ""
    with TODAY_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            day = stamp_to_date(row.get("lastTradedTime"))
            if day and day > latest:
                latest = day
    return latest


def refresh_prices() -> str:
    """Save today's full board. Download daily charts only when a newer session exists."""
    seed_archive_if_missing()
    stocks = listed_stocks()
    session = latest_traded_session()
    saved = max_saved_session()
    if session and saved >= session:
        return (
            f"Daily charts already include {saved}. "
            f"Live board saved for {len(stocks)} symbols."
        )
    print(f"Charts stop at {saved or 'nothing'}. Downloading through {session or 'the latest bar'}.", flush=True)
    charts = download_charts(stocks, needed_date=session or None)
    history = combine_charts(stocks)
    return (
        f"Charts updated: {history['rows']} rows, {history['symbols']} symbols, "
        f"downloaded {charts['downloaded']}, skipped {charts['skipped']}, "
        f"failed {len(charts['failed'])}."
    )


def main() -> None:
    seed_archive_if_missing()
    stocks = listed_stocks()
    charts = download_charts(stocks)
    history = combine_charts(stocks)
    manifest = {
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": BASE + "companyChartDataByStock",
        "period": "5",
        "listed_symbols": len(stocks),
        "charts_downloaded": charts["downloaded"],
        "charts_skipped_fresh": charts["skipped"],
        "charts_failed": charts["failed"],
        "history_rows": history["rows"],
        "history_symbols": history["symbols"],
        "today": str(TODAY_PATH),
        "history": str(HISTORY_PATH),
        "archive": str(ARCHIVE_PATH),
        "archive_rows": history.get("archive_rows", 0),
        "note": (
            "Period 5 is the longest daily range the CSE chart API returns, about one year. "
            "Each download is merged into the close archive, so older sessions remain after that window moves on."
        ),
    }
    write_json(OUT / "market_all_manifest.json", manifest)
    print(f"Failed symbols: {len(charts['failed'])}", flush=True)
    if history["symbols"] < 30:
        raise SystemExit("Too few charts were saved to train on.")


if __name__ == "__main__":
    main()
