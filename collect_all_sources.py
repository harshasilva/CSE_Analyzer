"""
Collect CSE announcements, Sri Lanka business news, and price snapshots
into a single combined CSV/JSON dataset.

Sources:
- CSE announcements API
- CSE tradeSummary (live price snapshots)
- Ada Derana Biz
- Daily FT Business
- EconomyNext
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Iterable
import requests
from bs4 import BeautifulSoup


OUTPUT_DIR = "cse_output"
COMBINED_JSON = os.path.join(OUTPUT_DIR, "cse_combined_sources.json")
COMBINED_CSV = os.path.join(OUTPUT_DIR, "cse_combined_sources.csv")

# Date range for CSE announcements
START_YEAR = 2021
END_YEAR = date.today().year

NEWS_SOURCES = [
    {
        "name": "Adaderana Biz",
        "url": "https://bizenglish.adaderana.lk/",
        "tags": ["h2", "h3", "h4"],
    },
    {
        "name": "Daily FT Business",
        "url": "https://www.ft.lk/business/34",
        "tags": ["h3", "h4"],
    },
    {
        "name": "EconomyNext",
        "url": "https://economynext.com/",
        "tags": ["h3", "h4"],
    },
]

COMPANY_SYMBOLS = {
    "commercial bank": "COMB.N0000",
    "hatton national bank": "HNB.N0000",
    "sampath bank": "SAMP.N0000",
    "seylan bank": "SEYB.N0000",
    "bank of ceylon": "BOC.N0000",
    "dfcc bank": "DFCC.N0000",
    "nations trust bank": "NTB.N0000",
    "pan asia bank": "PABC.N0000",
    "john keells holdings": "JKH.N0000",
    "dialog axiata": "DIAL.N0000",
}

SYMBOL_TO_COMPANY = {symbol: name.title() for name, symbol in COMPANY_SYMBOLS.items()}

CSE_BASE_URL = "https://www.cse.lk/api/"
CSE_ENDPOINTS = [
    ("approvedAnnouncement", "approvedAnnouncements", "Approved Announcements"),
    ("getFinancialAnnouncement", "financialAnnouncements", "Financial Announcements"),
    ("circularAnnouncement", "circularAnnouncements", "Circular Announcements"),
    ("directiveAnnouncement", "directiveAnnouncements", "Directive Announcements"),
    ("getNonComplianceAnnouncements", "nonComplianceAnnouncements", "Non-Compliance Announcements"),
    ("getNewListingsRelatedNoticesAnnouncements", "newListingRelatedAnnouncements", "New Listings Announcements"),
    ("getBuyInBoardAnnouncements", "buyInBoardAnnouncements", "Buy-In Board Announcements"),
    ("getCOVIDAnnouncements", "covidAnnouncements", "COVID Announcements"),
]

DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d %B %Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%b %d, %Y",
    "%B %d, %Y",
]

DATE_FIELDS = [
    "date",
    "announcementDate",
    "publishedDate",
    "createdDate",
    "updatedDate",
    "dateTime",
    "published_date",
    "created_at",
    "timestamp",
    "effectiveDate",
    "submittedDate",
    "noticeDate",
]


@dataclass
class CombinedRow:
    source_type: str
    source: str
    title: str | None = None
    url: str | None = None
    published_date: str | None = None
    company: str | None = None
    category: str | None = None
    symbol: str | None = None
    close: float | None = None
    return_20d: float | None = None
    return_60d: float | None = None
    text: str | None = None


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    s = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s[: len(fmt)], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def request_with_backoff(method: str, url: str, max_attempts: int = 3, **kwargs) -> requests.Response | None:
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if attempt == max_attempts:
                print(f"[!] Request failed after {attempt} attempts: {url} ({exc})")
                return None
            print(f"[!] Request failed (attempt {attempt}/{max_attempts}): {url} ({exc})")
            time.sleep(delay)
            delay *= 2


def extract_date_from_text(text: str) -> date | None:
    if not text:
        return None
    pattern = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}")
    match = pattern.search(text)
    if match:
        return parse_date(match.group(0))
    return None


def fetch_news_items() -> list[CombinedRow]:
    rows: list[CombinedRow] = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for source in NEWS_SOURCES:
        resp = request_with_backoff("GET", source["url"], headers=headers, timeout=20)
        if resp is None:
            print(f"[!] News fetch failed for {source['name']}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        seen = set()
        for tag in source["tags"]:
            for h in soup.find_all(tag):
                a = h.find("a") or h.parent if h.parent and h.parent.name == "a" else None
                title = h.get_text(strip=True)
                if not title or len(title) < 10:
                    continue
                url = None
                if a and a.get("href"):
                    url = a["href"]
                    if url and url.startswith("/"):
                        url = source["url"].rstrip("/") + url
                key = (title, url)
                if key in seen:
                    continue
                seen.add(key)

                parent_text = h.parent.get_text(" ", strip=True) if h.parent else ""
                published = extract_date_from_text(parent_text)
                published_str = published.isoformat() if published else None
                rows.append(
                    CombinedRow(
                        source_type="news",
                        source=source["name"],
                        title=title,
                        url=url,
                        published_date=published_str,
                        text=parent_text,
                    )
                )
    return rows


def fetch_cse_announcements(start_year: int, end_year: int) -> list[CombinedRow]:
    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.cse.lk/",
        "Origin": "https://www.cse.lk",
    }

    rows: list[CombinedRow] = []

    for name, key, label in CSE_ENDPOINTS:
        url = CSE_BASE_URL + name
        resp = request_with_backoff("POST", url, headers=headers, timeout=30)
        if resp is None:
            print(f"[!] CSE endpoint failed ({label})")
            continue

        try:
            data = resp.json()
        except Exception as exc:
            print(f"[!] CSE JSON parse failed ({label}): {exc}")
            continue

        items = data.get(key, []) if isinstance(data, dict) else []
        if not items:
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list) and value:
                        items = value
                        break

        for item in items or []:
            parsed_date = None
            for field in DATE_FIELDS:
                if field in item and item[field]:
                    parsed_date = parse_date(str(item[field]))
                    if parsed_date:
                        break
            if not parsed_date:
                continue
            if not (start_date <= parsed_date <= end_date):
                continue

            company = str(item.get("company", "")) or None
            title = str(item.get("subject") or item.get("announcementType") or label)
            text = str(item.get("description") or item.get("remarks") or "")

            rows.append(
                CombinedRow(
                    source_type="announcement",
                    source="CSE",
                    title=title,
                    published_date=parsed_date.isoformat(),
                    company=company,
                    category=label,
                    text=text,
                )
            )

    return rows


def fetch_price_snapshots(symbols: Iterable[str]) -> list[CombinedRow]:
    rows: list[CombinedRow] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.cse.lk/",
        "Origin": "https://www.cse.lk",
    }

    for symbol in symbols:
        resp = request_with_backoff(
            "POST",
            f"{CSE_BASE_URL}tradeSummary",
            data={"symbol": symbol},
            headers=headers,
            timeout=20,
        )
        if resp is None:
            print(f"[!] CSE tradeSummary failed for {symbol}")
            continue

        try:
            data = resp.json()
        except Exception as exc:
            print(f"[!] CSE tradeSummary JSON parse failed for {symbol}: {exc}")
            continue

        price = None
        for item in data.get("reqTradeSummery", []) or []:
            if item.get("symbol") == symbol and item.get("price"):
                try:
                    price = float(item["price"])
                except (TypeError, ValueError):
                    price = None
                break

        if price is None:
            print(f"[!] No live price for {symbol}")
            continue

        rows.append(
            CombinedRow(
                source_type="price_snapshot",
                source="CSE tradeSummary",
                title=f"{symbol} live price",
                published_date=date.today().isoformat(),
                company=SYMBOL_TO_COMPANY.get(symbol),
                symbol=symbol,
                close=price,
            )
        )

    return rows


def write_outputs(rows: list[CombinedRow]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(COMBINED_JSON, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)

    fieldnames = list(asdict(rows[0]).keys()) if rows else [f.name for f in CombinedRow.__dataclass_fields__.values()]
    with open(COMBINED_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"[OK] Wrote {len(rows)} rows")
    print(f"[OK] JSON -> {COMBINED_JSON}")
    print(f"[OK] CSV  -> {COMBINED_CSV}")


def main() -> None:
    print("Collecting CSE announcements, Sri Lanka business news, and price snapshots...")
    news_rows = fetch_news_items()
    announcement_rows = fetch_cse_announcements(START_YEAR, END_YEAR)
    price_rows = fetch_price_snapshots(sorted(set(COMPANY_SYMBOLS.values())))

    all_rows = news_rows + announcement_rows + price_rows
    write_outputs(all_rows)


if __name__ == "__main__":
    main()
