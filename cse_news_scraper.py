"""
CSE (Colombo Stock Exchange) Historical News Scraper
=====================================================
Fetches all announcement types from the CSE API,
filters for the last year (2026), and saves to:
    - cse_news_2026.json   (full raw data)
    - cse_news_2026.csv    (flat spreadsheet)

Usage:
    pip install requests
    python cse_news_scraper.py

Author: Generated for CSE data collection
"""

import requests
import json
import csv
import time
import os
from datetime import datetime, date

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_URL = "https://www.cse.lk/api/"

# Date range to filter (inclusive)
TARGET_START = date(2026, 1, 1)
TARGET_END = date.today()

OUTPUT_DIR = "cse_output"
JSON_FILE  = f"{OUTPUT_DIR}/cse_news_2026.json"
CSV_FILE   = f"{OUTPUT_DIR}/cse_news_2026.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cse.lk/",
    "Origin": "https://www.cse.lk",
}

# Delay between requests (seconds) — be respectful to the server
REQUEST_DELAY = 1.5

# All announcement endpoints with their response key and label
ENDPOINTS = [
    {
        "name":     "approvedAnnouncement",
        "label":    "Approved Announcements",
        "key":      "approvedAnnouncements",
    },
    {
        "name":     "getFinancialAnnouncement",
        "label":    "Financial Announcements",
        "key":      "financialAnnouncements",
    },
    {
        "name":     "circularAnnouncement",
        "label":    "Circular Announcements",
        "key":      "circularAnnouncements",
    },
    {
        "name":     "directiveAnnouncement",
        "label":    "Directive Announcements",
        "key":      "directiveAnnouncements",
    },
    {
        "name":     "getNonComplianceAnnouncements",
        "label":    "Non-Compliance Announcements",
        "key":      "nonComplianceAnnouncements",
    },
    {
        "name":     "getNewListingsRelatedNoticesAnnouncements",
        "label":    "New Listings Announcements",
        "key":      "newListingRelatedAnnouncements",
    },
    {
        "name":     "getBuyInBoardAnnouncements",
        "label":    "Buy-In Board Announcements",
        "key":      "buyInBoardAnnouncements",
    },
    {
        "name":     "getCOVIDAnnouncements",
        "label":    "COVID Announcements",
        "key":      "covidAnnouncements",
    },
]

# ─────────────────────────────────────────────
# DATE PARSING
# ─────────────────────────────────────────────

# Common date formats used in CSE API responses
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
]

def parse_date(value):
    """Try multiple date formats and return a date object or None."""
    if not value:
        return None
    value = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value[:len(fmt)], fmt).date()
        except (ValueError, TypeError):
            continue
    # Try just extracting a 4-digit year
    for part in value.split():
        if part.isdigit() and len(part) == 4:
            try:
                return date(int(part), 1, 1)
            except ValueError:
                pass
    return None

def get_date_from_item(item):
    """Try common date field names to find a date in an announcement item."""
    date_fields = [
        "date", "announcementDate", "publishedDate", "createdDate",
        "updatedDate", "dateTime", "published_date", "created_at",
        "timestamp", "effectiveDate", "submittedDate",
    ]
    for field in date_fields:
        if field in item and item[field]:
            parsed = parse_date(item[field])
            if parsed:
                return parsed, field
    return None, None

# ─────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────

def fetch_endpoint(endpoint_info):
    """POST to a CSE API endpoint and return the list of items."""
    url = BASE_URL + endpoint_info["name"]
    label = endpoint_info["label"]
    key = endpoint_info["key"]

    print(f"  → Fetching: {label} ...")

    try:
        resp = requests.post(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"     ✗ Connection error — check your internet or VPN.")
        return []
    except requests.exceptions.Timeout:
        print(f"     ✗ Request timed out.")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"     ✗ HTTP error: {e}")
        return []
    except Exception as e:
        print(f"     ✗ Unexpected error: {e}")
        return []

    try:
        data = resp.json()
    except Exception:
        print(f"     ✗ Could not parse JSON response.")
        return []

    # Try the expected key first
    items = data.get(key, [])

    # If not found, scan all values for a list
    if not items:
        for v in data.values():
            if isinstance(v, list) and len(v) > 0:
                items = v
                break

    if not items:
        print(f"     ⚠ No items found (empty response).")
        return []

    print(f"     ✓ Got {len(items)} total items.")
    return items

# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────

def filter_by_date_range(items, start_date, end_date, category_label):
    """Filter announcement items to only those within the target date range."""
    filtered = []
    no_date_count = 0

    for item in items:
        parsed_date, date_field = get_date_from_item(item)

        if parsed_date is None:
            no_date_count += 1
            # Include items with no date if you want — comment out the continue to keep them
            continue

        if start_date <= parsed_date <= end_date:
            item["_parsed_date"]  = str(parsed_date)
            item["_date_field"]   = date_field
            item["_category"]     = category_label
            filtered.append(item)

    if no_date_count:
        print(f"     ℹ {no_date_count} items had no parseable date and were skipped.")

    print(f"     ✓ {len(filtered)} items from {start_date} to {end_date}.")
    return filtered

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_json(all_data):
    """Save all results to a structured JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  ✓ JSON saved → {JSON_FILE}")

def save_csv(all_items):
    """Save flat list of all items to CSV."""
    if not all_items:
        print("  ⚠ No items to write to CSV.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all unique keys across items
    all_keys = []
    seen = set()
    priority = ["_category", "_parsed_date", "company", "subject", "description",
                "announcementType", "type", "date", "announcementDate", "publishedDate"]
    for key in priority:
        if key not in seen:
            all_keys.append(key)
            seen.add(key)
    for item in all_items:
        for k in item.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for item in all_items:
            writer.writerow(item)

    print(f"  ✓ CSV saved  → {CSV_FILE}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print(f"  CSE News Scraper — Fetching {TARGET_START} to {TARGET_END} announcements")
    print("=" * 55)
    print(f"  Base URL : {BASE_URL}")
    print(f"  Target   : {TARGET_START} to {TARGET_END}")
    print(f"  Output   : {OUTPUT_DIR}/")
    print("=" * 55)

    all_results  = {}   # structured by category (for JSON)
    all_flat     = []   # flat list (for CSV)
    total_found  = 0

    for ep in ENDPOINTS:
        print(f"\n[{ep['label']}]")

        raw_items = fetch_endpoint(ep)
        time.sleep(REQUEST_DELAY)

        if not raw_items:
            all_results[ep["label"]] = []
            continue

        year_items = filter_by_date_range(raw_items, TARGET_START, TARGET_END, ep["label"])

        all_results[ep["label"]] = year_items
        all_flat.extend(year_items)
        total_found += len(year_items)

    # Sort flat list by date
    all_flat.sort(key=lambda x: x.get("_parsed_date", ""), reverse=True)

    print("\n" + "=" * 55)
    print(f"  TOTAL announcements from {TARGET_START} to {TARGET_END}: {total_found}")
    print("=" * 55)

    # Summary by category
    print("\n  Summary by category:")
    for label, items in all_results.items():
        print(f"    {len(items):>5}  {label}")

    # Save outputs
    print("\n  Saving files...")
    save_json({
        "meta": {
            "scraped_at":   datetime.now().isoformat(),
            "target_start": TARGET_START,
            "target_end":   TARGET_END,
            "total_items":  total_found,
            "source":       BASE_URL,
        },
        "data": all_results,
    })
    save_csv(all_flat)

    print("\n  Done! Open the cse_output/ folder for your files.")
    print("=" * 55)


if __name__ == "__main__":
    main()
