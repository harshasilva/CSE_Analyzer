"""Download old prices, today's board, and company details from the CSE website API.

The public API is https://www.cse.lk/api/. It does not take a start and end date.
The longest share chart is period 5, about one year of daily bars. Today's board,
trades, and company pages are separate calls.

Followed shares (charts, profiles, and financial reports):
    COMB, SAMP, HNB, JKH, DIAL

Every listed symbol is saved from the live board and from companyInfoSummery.

    python fetch_cse_api.py
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

BASE = "https://www.cse.lk/api/"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "cse_api"
HISTORY = OUT / "history"
SITE_DIR = ROOT / "data" / "cse_site"
SITE_COMBINED = ROOT / "data" / "cse_site_prices_1y.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://www.cse.lk",
    "Referer": "https://www.cse.lk/",
    "Accept": "application/json, text/plain, */*",
}
PAUSE = 0.25

FOLLOWED = [
    {"symbol": "COMB.N0000", "name": "Commercial Bank"},
    {"symbol": "SAMP.N0000", "name": "Sampath Bank"},
    {"symbol": "HNB.N0000", "name": "Hatton National Bank"},
    {"symbol": "JKH.N0000", "name": "John Keells Holdings"},
    {"symbol": "DIAL.N0000", "name": "Dialog Axiata"},
]
SITE_COLUMNS = [
    "date",
    "symbol",
    "company",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
]


def post(path: str, fields: dict | None = None) -> object:
    response = requests.post(
        BASE + path,
        data=fields or {},
        headers=HEADERS,
        timeout=45,
    )
    response.raise_for_status()
    if not response.content:
        return None
    return response.json()


def stamp_to_date(value) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number > 10_000_000_000:
        number = number / 1000
    if number < 1_000_000_000:
        return str(value)
    # Chart bars are stamped at 18:30 UTC, which is midnight on the Colombo session date.
    return datetime.fromtimestamp(number, tz=ZoneInfo("Asia/Colombo")).date().isoformat()


def write_csv(path: Path, rows: list[dict], first: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for key in first or []:
        if key not in seen:
            seen.add(key)
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def today_colombo() -> str:
    return datetime.now(ZoneInfo("Asia/Colombo")).date().isoformat()


def append_history(path: Path, rows: list[dict], keys: list[str], extra: dict | None = None) -> None:
    """Keep one row per key. A later download for the same key replaces the older row."""
    if not rows:
        return
    stamped = []
    for row in rows:
        item = {key: value for key, value in row.items()}
        if extra:
            item.update(extra)
        stamped.append(item)
    existing: list[dict] = []
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    chosen: dict[tuple, dict] = {}
    for row in existing + stamped:
        chosen[tuple(str(row.get(key, "")) for key in keys)] = row
    write_csv(path, list(chosen.values()), list(keys))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def cell(value) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def flat_row(item: dict) -> dict:
    return {key: cell(value) for key, value in item.items()}


def as_list(payload: object, *keys: str) -> list:
    if isinstance(payload, list):
        rows = []
        for item in payload:
            if isinstance(item, list):
                rows.extend(piece for piece in item if isinstance(piece, dict))
            elif isinstance(item, dict):
                rows.append(item)
        return rows
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
    return []


def chart_rows(symbol: str, company: str, points: list) -> tuple[list[dict], list[dict]]:
    full = []
    site = []
    for point in points:
        if not isinstance(point, dict) or point.get("t") is None or point.get("p") is None:
            continue
        when = stamp_to_date(point.get("t"))
        full.append(
            {
                "date": when,
                "symbol": symbol,
                "company": company,
                "open": point.get("o") if point.get("o") is not None else "",
                "high": point.get("h") if point.get("h") is not None else "",
                "low": point.get("l") if point.get("l") is not None else "",
                "close": point.get("p"),
                "change": point.get("c") if point.get("c") is not None else "",
                "change_pct": point.get("pc") if point.get("pc") is not None else "",
                "volume": point.get("q") if point.get("q") is not None else "",
                "turnover": point.get("s") if point.get("s") is not None else "",
            }
        )
        site.append(
            {
                "date": when,
                "symbol": symbol,
                "company": company,
                "high": point.get("h") if point.get("h") is not None else "",
                "low": point.get("l") if point.get("l") is not None else "",
                "close": point.get("p"),
                "volume": point.get("q") if point.get("q") is not None else "",
                "turnover": point.get("s") if point.get("s") is not None else "",
            }
        )
    full.sort(key=lambda row: row["date"])
    site.sort(key=lambda row: row["date"])
    return full, site


def index_rows(points: list, name: str) -> list[dict]:
    rows = []
    for point in points:
        if not isinstance(point, dict):
            continue
        rows.append(
            {
                "date": stamp_to_date(point.get("d")),
                "index": name,
                "value": point.get("v") if point.get("v") is not None else "",
                "change_pct": point.get("pc") if point.get("pc") is not None else "",
            }
        )
    rows.sort(key=lambda row: row["date"])
    return rows


def quote_row(symbol: str, payload: dict) -> dict:
    info = payload.get("reqSymbolInfo") or {}
    beta = payload.get("reqSymbolBetaInfo") or {}
    row = {"symbol": symbol}
    if isinstance(info, dict):
        row.update(flat_row(info))
    if isinstance(beta, dict):
        for key, value in beta.items():
            row[f"beta_{key}"] = cell(value)
    return row


def profile_row(symbol: str, payload: dict) -> dict:
    summary = (payload.get("reqComSumInfo") or [{}])[0] or {}
    business = (payload.get("infoCompanyBusinessSummary") or [{}])[0] or {}
    directors = []
    for person in (payload.get("topPosts") or []) + (payload.get("infoCompanyDirector") or []):
        if not isinstance(person, dict):
            continue
        name = " ".join(
            part.strip()
            for part in (person.get("firstName") or "", person.get("lastName") or "")
            if part and part.strip()
        )
        role = person.get("designationOther") or ""
        directors.append(f"{name} ({role})".strip() if role else name)
    return {
        "symbol": summary.get("symbol") or symbol,
        "name": summary.get("name") or "",
        "sector": summary.get("sector") or "",
        "board": summary.get("boardType") or "",
        "established": summary.get("established") or "",
        "address": summary.get("registerOffice") or summary.get("registeredOffice") or "",
        "phone": summary.get("tel1") or "",
        "fax": summary.get("fax") or "",
        "email": summary.get("email1") or "",
        "website": summary.get("web") or "",
        "auditors": summary.get("auditors") or "",
        "secretaries": summary.get("secretaries") or "",
        "business": (business.get("body") or "").strip(),
        "directors": "; ".join(item for item in directors if item),
    }


def report_rows(symbol: str, payload: dict) -> list[dict]:
    rows = []
    for kind, key in (
        ("annual", "infoAnnualData"),
        ("quarterly", "infoQuarterlyData"),
        ("other", "infoOtherData"),
    ):
        for item in payload.get(key) or []:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "report_type": kind,
                    "title": item.get("fileText") or "",
                    "date": stamp_to_date(item.get("manualDate") or item.get("uploadedDate")),
                    "file_path": item.get("path") or "",
                }
            )
    return rows


def save_market() -> dict:
    print("Live board, market totals, and today's trades...")
    board = as_list(post("tradeSummary"), "reqTradeSummery")
    write_csv(
        OUT / "live_board.csv",
        [flat_row(item) for item in board],
        ["symbol", "name", "id", "lastTradedPrice", "lastTrade", "previousClose", "change", "percentageChange"],
    )
    market = {
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "marketStatus": post("marketStatus"),
        "marketSummery": post("marketSummery"),
        "aspiData": post("aspiData"),
        "snpData": post("snpData"),
        "topGainers": post("topGainers"),
        "topLooses": post("topLooses"),
        "mostActiveTrades": post("mostActiveTrades"),
    }
    write_json(OUT / "market_now.json", market)
    sectors = as_list(post("allSectors"))
    write_csv(OUT / "sectors.csv", [flat_row(item) for item in sectors], ["name", "symbol", "indexValue", "change", "percentage"])
    daily = as_list(post("dailyMarketSummery"))
    for row in daily:
        if "tradeDate" in row:
            row["tradeDate"] = stamp_to_date(row.get("tradeDate"))
    write_csv(OUT / "daily_market.csv", [flat_row(item) for item in daily], ["tradeDate", "asi", "spp", "marketTurnover", "marketTrades"])
    trades = as_list(post("detailedTrades"), "reqDetailTrades")
    write_csv(
        OUT / "today_trades.csv",
        [flat_row(item) for item in trades],
        ["symbol", "name", "price", "qty", "trades", "change", "changePercentage"],
    )
    print(f"  {len(board)} listed symbols, {len(trades)} trade rows, {len(daily)} market days")
    as_of = today_colombo()
    append_history(HISTORY / "daily_market.csv", [flat_row(item) for item in daily], ["tradeDate"])
    append_history(
        HISTORY / "sectors.csv",
        [flat_row(item) for item in sectors],
        ["as_of", "name"],
        {"as_of": as_of},
    )
    status = market.get("marketStatus") if isinstance(market.get("marketStatus"), dict) else {}
    summary = market.get("marketSummery") if isinstance(market.get("marketSummery"), dict) else {}
    aspi = market.get("aspiData") if isinstance(market.get("aspiData"), dict) else {}
    snp = market.get("snpData") if isinstance(market.get("snpData"), dict) else {}
    append_history(
        HISTORY / "market_snapshots.csv",
        [
            {
                "as_of": as_of,
                "status": status.get("status") or "",
                "trades": summary.get("trades") if summary else "",
                "share_volume": summary.get("shareVolume") if summary else "",
                "turnover": summary.get("tradeVolume") if summary else "",
                "aspi": aspi.get("value") if aspi else "",
                "aspi_change": aspi.get("change") if aspi else "",
                "aspi_change_pct": aspi.get("percentage") if aspi else "",
                "aspi_high": aspi.get("highValue") if aspi else "",
                "aspi_low": aspi.get("lowValue") if aspi else "",
                "snp": snp.get("value") if snp else "",
                "snp_change": snp.get("change") if snp else "",
                "snp_change_pct": snp.get("percentage") if snp else "",
            }
        ],
        ["as_of"],
    )
    return {"board": board, "trades": len(trades), "market_days": len(daily)}


def save_index_history() -> dict:
    print("Index history (longest range the API returns)...")
    saved = {}
    for chart_id, name, filename in (
        ("1", "ASPI", "aspi_history.csv"),
        ("2", "S&P SL20", "snp_history.csv"),
    ):
        try:
            payload = post("chartData", {"chartId": chart_id, "period": "5"})
        except requests.RequestException as exc:
            print(f"  {name} chart failed: {exc}")
            continue
        points = payload if isinstance(payload, list) else []
        rows = index_rows(points, name)
        if not rows:
            print(f"  {name} returned no points")
            continue
        write_csv(OUT / filename, rows, ["date", "index", "value", "change_pct"])
        saved[name] = {"rows": len(rows), "from": rows[0]["date"], "to": rows[-1]["date"]}
        print(f"  {name}: {len(rows)} points, {rows[0]['date']} to {rows[-1]['date']}")
        time.sleep(PAUSE)
    return saved


def save_share_history(board: list[dict]) -> dict:
    print("One-year daily prices for the five followed shares...")
    ids = {}
    for item in board:
        symbol = item.get("symbol")
        if symbol and item.get("id") is not None:
            ids[symbol] = str(item["id"])
    combined = []
    summary = {}
    for company in FOLLOWED:
        symbol = company["symbol"]
        stock_id = ids.get(symbol)
        if not stock_id:
            print(f"  {symbol} was not on the live board")
            continue
        payload = post("companyChartDataByStock", {"stockId": stock_id, "period": "5"})
        points = payload.get("chartData") if isinstance(payload, dict) else []
        full, site = chart_rows(symbol, company["name"], points or [])
        write_csv(
            OUT / "charts" / f"{symbol}.csv",
            full,
            ["date", "symbol", "company", "open", "high", "low", "close", "change", "change_pct", "volume", "turnover"],
        )
        write_csv(SITE_DIR / f"{symbol}.csv", site, SITE_COLUMNS)
        combined.extend(site)
        if full:
            summary[symbol] = {"rows": len(full), "from": full[0]["date"], "to": full[-1]["date"], "close": full[-1]["close"]}
            print(f"  {symbol}: {len(full)} sessions, {full[0]['date']} to {full[-1]['date']}, close {full[-1]['close']}")
        time.sleep(PAUSE)
    write_csv(SITE_COMBINED, combined, SITE_COLUMNS)
    return summary


def save_company_details(board: list[dict]) -> dict:
    symbols = []
    seen = set()
    for item in board:
        symbol = item.get("symbol")
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    print(f"Company quote details for {len(symbols)} symbols...")
    quotes = []
    for index, symbol in enumerate(symbols, start=1):
        try:
            payload = post("companyInfoSummery", {"symbol": symbol})
        except requests.RequestException as exc:
            print(f"  {symbol} quote failed: {exc}")
            time.sleep(PAUSE)
            continue
        if isinstance(payload, dict):
            quotes.append(quote_row(symbol, payload))
        if index % 40 == 0 or index == len(symbols):
            print(f"  quotes {index}/{len(symbols)}")
        time.sleep(PAUSE)
    write_csv(
        OUT / "company_quotes.csv",
        quotes,
        ["symbol", "name", "id", "isin", "lastTradedPrice", "previousClose", "change", "changePercentage", "marketCap", "hiTrade", "lowTrade"],
    )
    append_history(HISTORY / "quote_snapshots.csv", quotes, ["as_of", "symbol"], {"as_of": today_colombo()})

    print("Profiles and financial reports for the five followed shares...")
    profiles = []
    reports = []
    for company in FOLLOWED:
        symbol = company["symbol"]
        try:
            profile = post("companyProfile", {"symbol": symbol})
            financials = post("financials", {"symbol": symbol})
        except requests.RequestException as exc:
            print(f"  {symbol} details failed: {exc}")
            continue
        if isinstance(profile, dict):
            write_json(OUT / "profiles" / f"{symbol}.json", profile)
            profiles.append(profile_row(symbol, profile))
        if isinstance(financials, dict):
            statements = financials.get("reqFinancial")
            write_json(
                OUT / "financials" / f"{symbol}.json",
                {
                    "infoAnnualData": financials.get("infoAnnualData") or [],
                    "infoQuarterlyData": financials.get("infoQuarterlyData") or [],
                    "infoOtherData": financials.get("infoOtherData") or [],
                    "statement_lines": len(statements) if isinstance(statements, list) else 0,
                },
            )
            reports.extend(report_rows(symbol, financials))
        print(f"  {symbol}: profile and {sum(1 for row in reports if row['symbol'] == symbol)} reports")
        time.sleep(PAUSE)
    write_csv(OUT / "company_profiles.csv", profiles)
    write_csv(OUT / "financial_reports.csv", reports, ["symbol", "report_type", "title", "date", "file_path"])
    return {"quotes": len(quotes), "profiles": len(profiles), "reports": len(reports)}


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def refresh_tracked() -> str:
    """Save the live market, index history, and the five followed shares.

    The full symbol list stays in company_quotes.csv. This only replaces the
    five shares the model follows, then stores today's snapshot.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    save_market()
    save_index_history()
    quotes = []
    profiles = []
    reports = []
    for company in FOLLOWED:
        symbol = company["symbol"]
        try:
            payload = post("companyInfoSummery", {"symbol": symbol})
            profile = post("companyProfile", {"symbol": symbol})
            financials = post("financials", {"symbol": symbol})
        except requests.RequestException as exc:
            print(f"  {symbol} refresh failed: {exc}")
            time.sleep(PAUSE)
            continue
        if isinstance(payload, dict):
            quotes.append(quote_row(symbol, payload))
        if isinstance(profile, dict):
            write_json(OUT / "profiles" / f"{symbol}.json", profile)
            profiles.append(profile_row(symbol, profile))
        if isinstance(financials, dict):
            statements = financials.get("reqFinancial")
            write_json(
                OUT / "financials" / f"{symbol}.json",
                {
                    "infoAnnualData": financials.get("infoAnnualData") or [],
                    "infoQuarterlyData": financials.get("infoQuarterlyData") or [],
                    "infoOtherData": financials.get("infoOtherData") or [],
                    "statement_lines": len(statements) if isinstance(statements, list) else 0,
                },
            )
            reports.extend(report_rows(symbol, financials))
        print(f"  {symbol}: quote, profile, and reports refreshed")
        time.sleep(PAUSE)

    if quotes:
        current = {row.get("symbol"): row for row in _read_csv_rows(OUT / "company_quotes.csv")}
        for row in quotes:
            current[row["symbol"]] = {key: "" if value is None else value for key, value in row.items()}
        merged = list(current.values())
        write_csv(
            OUT / "company_quotes.csv",
            merged,
            ["symbol", "name", "id", "isin", "lastTradedPrice", "previousClose", "change", "changePercentage", "marketCap", "hiTrade", "lowTrade"],
        )
        append_history(HISTORY / "quote_snapshots.csv", merged, ["as_of", "symbol"], {"as_of": today_colombo()})
    if profiles:
        current_profiles = {row.get("symbol"): row for row in _read_csv_rows(OUT / "company_profiles.csv")}
        for row in profiles:
            current_profiles[row["symbol"]] = row
        write_csv(OUT / "company_profiles.csv", list(current_profiles.values()))
    if reports:
        kept = [row for row in _read_csv_rows(OUT / "financial_reports.csv") if row.get("symbol") not in {item["symbol"] for item in reports}]
        write_csv(
            OUT / "financial_reports.csv",
            kept + reports,
            ["symbol", "report_type", "title", "date", "file_path"],
        )
    return f"Saved the CSE market snapshot and refreshed {len(quotes)} followed shares."


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    market = save_market()
    indexes = save_index_history()
    history = save_share_history(market["board"])
    details = save_company_details(market["board"])
    manifest = {
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": BASE,
        "listed_symbols": len(market["board"]),
        "today_trades": market["trades"],
        "daily_market_rows": market["market_days"],
        "indexes": indexes,
        "share_history": history,
        "company_quotes": details["quotes"],
        "profiles": details["profiles"],
        "financial_reports": details["reports"],
        "note": "Share charts use period 5, the longest range this API returns (about one year). Prices older than that remain in data/cse_prices_5y.csv.",
    }
    write_json(OUT / "manifest.json", manifest)
    print(f"\nSaved the CSE API download to {OUT}")


if __name__ == "__main__":
    main()
