"""Download about five years of dated headlines for the CSE price study.

Three streams, from October 2021 through today:
- world: rates, oil, recession, and stock-market headlines (one week at a time)
- sri_lanka: Sri Lanka economy headlines (one month at a time)
- market: Colombo Stock Exchange and the five followed companies

The source is Google News search. A full run takes roughly fifteen minutes.
Progress is saved after every window. Run the same command again to continue.

    python fetch_news_history.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from news_reactions import COLOMBO, clean_title

ROOT = Path(__file__).resolve().parent
ARCHIVE_PATH = ROOT / "data" / "news_archive.csv"
CHECKPOINT_PATH = ROOT / "data" / "news_archive_checkpoint.json"
LOG_PATH = ROOT / "cse_output" / "news_archive.log"

SEARCH_URL = "https://news.google.com/rss/search"
MIN_GAP_SECONDS = 1.2
FULL_PAGE = 95
START = date(2021, 10, 1)

QUERIES = {
    "sri_lanka": '"Sri Lanka" (economy OR rupee OR inflation OR IMF OR election OR stocks OR bank OR export)',
    "world": '("Federal Reserve" OR "oil price" OR "stock market" OR recession OR "interest rate")',
    "market": (
        '"Colombo Stock Exchange" OR "Sri Lanka stocks" OR "All Share Price Index" '
        'OR "John Keells" OR "Dialog Axiata" OR "Sampath Bank" OR "Hatton National Bank" '
        'OR "Commercial Bank of Ceylon"'
    ),
}
MARKET_TERMS = (
    "sri lanka",
    "colombo",
    "stock exchange",
    "keells",
    "dialog axiata",
    "sampath",
    "hatton national bank",
    "commercial bank",
    "all share",
)
COLUMNS = ["seen_at", "date", "region", "title", "url", "source", "country"]


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        if cursor.month == 12:
            following = date(cursor.year + 1, 1, 1)
        else:
            following = date(cursor.year, cursor.month + 1, 1)
        windows.append((max(cursor, start), min(following - timedelta(days=1), end)))
        cursor = following
    return windows


def week_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=6), end)
        windows.append((cursor, stop))
        cursor = stop + timedelta(days=1)
    return windows


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        return {"completed": [], "split": [], "saved": 0}
    try:
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"completed": [], "split": [], "saved": 0}
    data.setdefault("completed", [])
    data.setdefault("split", [])
    data.setdefault("saved", 0)
    return data


def save_checkpoint(checkpoint: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    temporary.replace(CHECKPOINT_PATH)


def window_key(region: str, start: date, end: date) -> str:
    return f"{region}|{start.isoformat()}|{end.isoformat()}"


def halves(start: date, end: date) -> tuple[tuple[date, date], tuple[date, date]] | None:
    midpoint = start + timedelta(days=(end - start).days // 2)
    if midpoint <= start or midpoint >= end:
        return None
    return (start, midpoint), (midpoint + timedelta(days=1), end)


def _seen_from_pubdate(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=COLOMBO)
    return parsed.astimezone(COLOMBO)


def _split_source(title: str) -> tuple[str, str]:
    if " - " not in title:
        return title, "Google News"
    headline, source = title.rsplit(" - ", 1)
    headline = headline.strip()
    if len(headline) < 12:
        return title, "Google News"
    return headline, source.strip()


class Client:
    def __init__(self) -> None:
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0"

    def articles(self, region: str, start: date, end: date) -> list[dict]:
        locale = ("en-US", "US", "US:en") if region == "world" else ("en-LK", "LK", "LK:en")
        # `before` is the next day so the window's last date is included.
        query = f"{QUERIES[region]} after:{start.isoformat()} before:{(end + timedelta(days=1)).isoformat()}"
        delay = 15.0
        for attempt in range(1, 5):
            wait = MIN_GAP_SECONDS - (time.time() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.time()
            try:
                response = self.session.get(
                    SEARCH_URL,
                    params={"q": query, "hl": locale[0], "gl": locale[1], "ceid": locale[2]},
                    timeout=40,
                )
            except requests.RequestException as exc:
                log(f"Request error (attempt {attempt}): {exc}")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if response.status_code == 429:
                log(f"News search asked for a slower pace. Waiting {int(delay)} seconds.")
                time.sleep(delay)
                delay = min(delay * 2, 90)
                continue
            if response.status_code != 200:
                log(f"News search HTTP {response.status_code} (attempt {attempt}).")
                time.sleep(delay)
                continue
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError:
                log(f"News search did not return a feed (attempt {attempt}).")
                time.sleep(delay)
                continue
            return list(root.findall(".//item"))
        raise RuntimeError(f"News search failed for {region} {start.isoformat()} to {end.isoformat()}")


def append_articles(region: str, items: list[ET.Element]) -> int:
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not ARCHIVE_PATH.exists()
    written = 0
    with ARCHIVE_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for item in items:
            raw_title = clean_title(item.findtext("title") or "")
            url = clean_title(item.findtext("link") or item.findtext("guid") or "")
            seen = _seen_from_pubdate(item.findtext("pubDate") or "")
            if seen is None or len(raw_title) < 12 or len(url) < 8:
                continue
            title, source = _split_source(raw_title)
            if region == "market" and not any(term in title.lower() for term in MARKET_TERMS):
                continue
            writer.writerow(
                {
                    "seen_at": seen.isoformat(),
                    "date": seen.date().isoformat(),
                    "region": region,
                    "title": title[:500],
                    "url": url,
                    "source": source[:120],
                    "country": "",
                }
            )
            written += 1
    return written


def fetch_window(client: Client, checkpoint: dict, region: str, start: date, end: date, depth: int) -> None:
    key = window_key(region, start, end)
    if key in checkpoint["completed"]:
        return
    if key in checkpoint["split"]:
        parts = halves(start, end)
        if parts is None:
            return
        fetch_window(client, checkpoint, region, parts[0][0], parts[0][1], depth + 1)
        fetch_window(client, checkpoint, region, parts[1][0], parts[1][1], depth + 1)
        return
    items = client.articles(region, start, end)
    span = (end - start).days
    truncated = len(items) >= FULL_PAGE
    if truncated and region != "world" and span > 10 and depth < 3:
        parts = halves(start, end)
        if parts is not None:
            checkpoint["split"].append(key)
            save_checkpoint(checkpoint)
            log(f"{region} {start} to {end} returned {len(items)} headlines. Splitting the window.")
            fetch_window(client, checkpoint, region, parts[0][0], parts[0][1], depth + 1)
            fetch_window(client, checkpoint, region, parts[1][0], parts[1][1], depth + 1)
            return
    written = append_articles(region, items)
    checkpoint["completed"].append(key)
    checkpoint["saved"] = int(checkpoint.get("saved", 0)) + written
    save_checkpoint(checkpoint)
    note = " capped" if truncated else ""
    log(f"{region} {start} to {end}: saved {written}{note}. Archive rows this run: {checkpoint['saved']}.")


def dedupe_archive() -> None:
    if not ARCHIVE_PATH.exists():
        return
    import pandas as pd

    frame = pd.read_csv(ARCHIVE_PATH)
    before = len(frame)
    frame = frame.drop_duplicates(["region", "url"], keep="first")
    frame.to_csv(ARCHIVE_PATH, index=False)
    log(f"Deduped archive from {before} rows to {len(frame)}.")


def jobs(end: date) -> list[tuple[str, date, date]]:
    planned = []
    for start, stop in month_windows(START, end):
        planned.append(("sri_lanka", start, stop))
        planned.append(("market", start, stop))
    for start, stop in week_windows(START, end):
        planned.append(("world", start, stop))
    order = {"sri_lanka": 0, "market": 1, "world": 2}
    return sorted(planned, key=lambda item: (item[1], order[item[0]]))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    end = date.today()
    planned = jobs(end)
    checkpoint = load_checkpoint()
    client = Client()
    log(f"Google News archive from {START.isoformat()} to {end.isoformat()}. {len(planned)} windows.")
    for region, start, stop in planned:
        try:
            fetch_window(client, checkpoint, region, start, stop, 0)
        except RuntimeError as exc:
            log(str(exc))
            log("Stopped. Run python fetch_news_history.py again to continue.")
            raise SystemExit(1) from exc
    dedupe_archive()
    log(f"ARCHIVE DONE. Checkpoint count {checkpoint['saved']}. File: {ARCHIVE_PATH}")


if __name__ == "__main__":
    main()
