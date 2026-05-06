"""
CSE (Colombo Stock Exchange) — Last 6 Years News Scraper
=========================================================
Fetches all announcement types from the CSE API for the
last 6 years (2021–2026) and saves:

  cse_output/
    ├── cse_news_ALL_2021_2026.json   ← full structured data
    ├── cse_news_ALL_2021_2026.csv    ← all years combined (spreadsheet)
  ├── cse_news_2021.csv
  ├── cse_news_2022.csv
  ├── cse_news_2023.csv
  ├── cse_news_2024.csv
    ├── cse_news_2025.csv
    └── cse_news_2026.csv

Usage:
    pip install requests
    python cse_news_scraper.py

Note: The CSE API returns its full dataset in one call per
endpoint — there is no pagination by date. This script
fetches each endpoint once, then filters by year locally.
"""

import requests
import json
import csv
import time
import os
from datetime import datetime, date

# ─────────────────────────────────────────────────────────
# CONFIG  ← edit here if needed
# ─────────────────────────────────────────────────────────

BASE_URL = "https://www.cse.lk/api/"

# Range of years to collect
START_YEAR = 2021
END_YEAR   = 2026          # inclusive
YEARS      = list(range(START_YEAR, END_YEAR + 1))

OUTPUT_DIR = "cse_output"

REQUEST_DELAY = 1.5        # seconds between API calls — be polite

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json, text/plain, */*",
    "Referer":    "https://www.cse.lk/",
    "Origin":     "https://www.cse.lk",
}

# All announcement endpoints → (endpoint name, response JSON key, human label)
ENDPOINTS = [
    ("approvedAnnouncement",                        "approvedAnnouncements",            "Approved Announcements"),
    ("getFinancialAnnouncement",                    "financialAnnouncements",            "Financial Announcements"),
    ("circularAnnouncement",                        "circularAnnouncements",             "Circular Announcements"),
    ("directiveAnnouncement",                       "directiveAnnouncements",            "Directive Announcements"),
    ("getNonComplianceAnnouncements",               "nonComplianceAnnouncements",        "Non-Compliance Announcements"),
    ("getNewListingsRelatedNoticesAnnouncements",   "newListingRelatedAnnouncements",    "New Listings Announcements"),
    ("getBuyInBoardAnnouncements",                  "buyInBoardAnnouncements",           "Buy-In Board Announcements"),
    ("getCOVIDAnnouncements",                       "covidAnnouncements",                "COVID Announcements"),
]

# CSV column order — any extra fields are appended after these
CSV_PRIORITY_COLS = [
    "_year", "_category", "_parsed_date",
    "company", "subject", "description", "heading", "title",
    "announcementType", "type",
    "date", "announcementDate", "publishedDate", "createdDate",
]

# ─────────────────────────────────────────────────────────
# DATE PARSING
# ─────────────────────────────────────────────────────────

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
    "%d-%b-%Y",
    "%B %d, %Y",
]

DATE_FIELDS = [
    "date", "announcementDate", "publishedDate", "createdDate",
    "updatedDate", "dateTime", "published_date", "created_at",
    "timestamp", "effectiveDate", "submittedDate", "noticeDate",
]

def parse_date(value):
    """Try multiple formats; return a date object or None."""
    if not value:
        return None
    s = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s[:19], fmt).date()
        except (ValueError, TypeError):
            try:
                return datetime.strptime(s, fmt).date()
            except (ValueError, TypeError):
                continue
    # Fallback: grab first 4-digit year token
    for token in s.split():
        token = token.strip(".,;:-/")
        if token.isdigit() and len(token) == 4:
            try:
                return date(int(token), 1, 1)
            except ValueError:
                pass
    return None

def extract_date(item):
    """Find the best date field in an item; return (date_obj, field_name)."""
    for field in DATE_FIELDS:
        if field in item and item[field]:
            d = parse_date(item[field])
            if d:
                return d, field
    return None, None

# ─────────────────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────────────────

def fetch_endpoint(name, key, label):
    """POST to one CSE API endpoint. Returns full list of raw items."""
    url = BASE_URL + name
    print(f"  -> {label}")

    try:
        resp = requests.post(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"     X  Connection error — check your internet.")
        return []
    except requests.exceptions.Timeout:
        print(f"     X  Request timed out.")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"     X  HTTP error: {e}")
        return []
    except Exception as e:
        print(f"     X  Unexpected error: {e}")
        return []

    try:
        data = resp.json()
    except Exception:
        print(f"     X  Could not parse JSON response.")
        return []

    # Try the expected key first, then scan all top-level lists
    items = data.get(key) if isinstance(data, dict) else None
    if not items:
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list) and v:
                    items = v
                    break

    if not items:
        print(f"     !  Empty response.")
        return []

    print(f"     OK {len(items):,} total items received.")
    return items

# ─────────────────────────────────────────────────────────
# FILTER & TAG
# ─────────────────────────────────────────────────────────

def tag_and_filter(items, years, label):
    """
    Tag each item with metadata fields.
    Returns dict: { year: [tagged_items] }
    """
    year_set = set(years)
    buckets  = {y: [] for y in years}
    no_date  = 0

    for item in items:
        d, field = extract_date(item)
        if d is None:
            no_date += 1
            continue
        if d.year in year_set:
            tagged = dict(item)
            tagged["_year"]        = d.year
            tagged["_parsed_date"] = str(d)
            tagged["_date_field"]  = field
            tagged["_category"]    = label
            buckets[d.year].append(tagged)

    if no_date:
        print(f"     !  {no_date} items had no parseable date — skipped.")

    for y in years:
        print(f"        {y}: {len(buckets[y]):>5} items")

    return buckets

# ─────────────────────────────────────────────────────────
# SAVE HELPERS
# ─────────────────────────────────────────────────────────

def ordered_fieldnames(items):
    """Priority columns first, then any extras found in items."""
    seen   = set()
    fields = []
    for col in CSV_PRIORITY_COLS:
        if col not in seen:
            fields.append(col)
            seen.add(col)
    for item in items:
        for k in item:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    return fields

def write_csv(filepath, items):
    if not items:
        print(f"     !  Nothing to write -> {filepath}")
        return
    fields = ordered_fieldnames(items)
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
    print(f"     OK {len(items):,} rows -> {os.path.basename(filepath)}")

def write_json(filepath, payload):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"     OK JSON -> {os.path.basename(filepath)}")

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"  CSE News Scraper  —  {START_YEAR} to {END_YEAR}")
    print("=" * 60)
    print(f"  Endpoints : {len(ENDPOINTS)}")
    print(f"  Years     : {', '.join(str(y) for y in YEARS)}")
    print(f"  Output    : {OUTPUT_DIR}/")
    print("=" * 60)

    # json_store[year][category_label] = [items]
    json_store   = {y: {} for y in YEARS}
    flat_by_year = {y: [] for y in YEARS}
    flat_all     = []

    # ── Fetch & sort each endpoint ────────────────────────
    for name, key, label in ENDPOINTS:
        print(f"\n[{label}]")
        raw = fetch_endpoint(name, key, label)
        time.sleep(REQUEST_DELAY)

        if not raw:
            for y in YEARS:
                json_store[y][label] = []
            continue

        buckets = tag_and_filter(raw, YEARS, label)

        for y in YEARS:
            json_store[y][label] = buckets[y]
            flat_by_year[y].extend(buckets[y])
            flat_all.extend(buckets[y])

    # ── Sort newest → oldest ──────────────────────────────
    sort_key = lambda x: x.get("_parsed_date", "")
    for y in YEARS:
        flat_by_year[y].sort(key=sort_key, reverse=True)
    flat_all.sort(key=sort_key, reverse=True)

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    grand_total = 0
    for y in YEARS:
        count = len(flat_by_year[y])
        grand_total += count
        print(f"  {y}  ->  {count:,} announcements")
    print(f"  {'─' * 30}")
    print(f"  TOTAL  ->  {grand_total:,} announcements ({START_YEAR}–{END_YEAR})")
    print("=" * 60)

    print("\n  Category breakdown:")
    header = f"  {'Category':<42}" + "".join(f"  {y}" for y in YEARS) + "   Total"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for name, key, label in ENDPOINTS:
        row = f"  {label:<42}"
        cat_total = 0
        for y in YEARS:
            c = len(json_store[y].get(label, []))
            cat_total += c
            row += f"  {c:>4}"
        row += f"   {cat_total:>5}"
        print(row)

    # ── Save files ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SAVING FILES")
    print("=" * 60)

    # 1. Combined JSON
    write_json(
        os.path.join(OUTPUT_DIR, f"cse_news_ALL_{START_YEAR}_{END_YEAR}.json"),
        {
            "meta": {
                "scraped_at":  datetime.now().isoformat(),
                "start_year":  START_YEAR,
                "end_year":    END_YEAR,
                "total_items": grand_total,
                "source":      BASE_URL,
            },
            "years": {
                str(y): {
                    "total":      len(flat_by_year[y]),
                    "categories": json_store[y],
                }
                for y in YEARS
            },
        },
    )

    # 2. Combined CSV (all years)
    write_csv(
        os.path.join(OUTPUT_DIR, f"cse_news_ALL_{START_YEAR}_{END_YEAR}.csv"),
        flat_all,
    )

    # 3. Per-year CSVs
    for y in YEARS:
        write_csv(
            os.path.join(OUTPUT_DIR, f"cse_news_{y}.csv"),
            flat_by_year[y],
        )

    print(f"\n  All files saved in: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)
    print("\n  Output files:")
    print(f"    cse_news_ALL_{START_YEAR}_{END_YEAR}.json  (full structured data)")
    print(f"    cse_news_ALL_{START_YEAR}_{END_YEAR}.csv   (all years combined)")
    for y in YEARS:
        print(f"    cse_news_{y}.csv")
    print("\n  Done!")


if __name__ == "__main__":
    main()
