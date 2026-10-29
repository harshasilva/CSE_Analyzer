"""Download the latest Sri Lanka business headlines from public news APIs.

EconomyNext and Lanka Business Online both expose a WordPress REST API.
This script asks those APIs for business, markets, economy, banking, and
finance stories, then saves a CSV.

    python fetch_business_news.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import requests

from news_reactions import clean_title, parse_seen

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "cse_output" / "sri_lanka_business_news.csv"
LATEST_PATH = ROOT / "data" / "news_latest.csv"

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")

# EconomyNext category ids from /wp-json/wp/v2/categories
ECONOMYNEXT_CATEGORIES = (
    (2689, "Business"),
    (4, "Markets"),
    (3, "Banking"),
    (2, "Finance"),
    (12, "Economy"),
)

COLUMNS = ["seen_at", "date", "region", "title", "url", "source", "country", "category", "excerpt"]
HOURLY_COLUMNS = ["seen_at", "date", "region", "title", "url", "source", "country"]


def _plain(value: str) -> str:
    text = TAG.sub(" ", value or "")
    text = html.unescape(text)
    return SPACE.sub(" ", text).strip()


def _title(payload: dict) -> str:
    rendered = payload.get("title") or {}
    if isinstance(rendered, dict):
        rendered = rendered.get("rendered") or ""
    return clean_title(_plain(str(rendered)))


def _excerpt(payload: dict) -> str:
    rendered = payload.get("excerpt") or {}
    if isinstance(rendered, dict):
        rendered = rendered.get("rendered") or ""
    return _plain(str(rendered))[:500]


def _get_json(url: str) -> list | None:
    response = requests.get(url, headers=HEADERS, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return None
    return payload


def _row(source: str, category: str, item: dict) -> dict | None:
    title = _title(item)
    url = (item.get("link") or "").strip()
    if len(title) < 12 or not url:
        return None
    seen = parse_seen(item.get("date"))
    if seen is None:
        return None
    return {
        "seen_at": seen.isoformat(),
        "date": seen.date().isoformat(),
        "region": "sri_lanka",
        "title": title[:500],
        "url": url,
        "source": source,
        "country": "LK",
        "category": category,
        "excerpt": _excerpt(item),
    }


def fetch_articles() -> tuple[list[dict], list[str]]:
    """Return the latest Sri Lanka business articles and a short note per source."""
    rows: list[dict] = []
    notes: list[str] = []
    seen_urls: set[str] = set()

    for category_id, category in ECONOMYNEXT_CATEGORIES:
        url = (
            "https://economynext.com/wp-json/wp/v2/posts"
            f"?categories={category_id}&per_page=25&orderby=date&order=desc"
            "&_fields=date,link,title,excerpt"
        )
        try:
            payload = _get_json(url)
        except (requests.RequestException, ValueError) as exc:
            notes.append(f"EconomyNext {category} was not available ({exc.__class__.__name__}).")
            continue
        if payload is None:
            notes.append(f"EconomyNext {category} did not return a story list.")
            continue
        added = 0
        for item in payload:
            if not isinstance(item, dict):
                continue
            row = _row("EconomyNext", category, item)
            if row is None or row["url"] in seen_urls:
                continue
            seen_urls.add(row["url"])
            rows.append(row)
            added += 1
        notes.append(f"EconomyNext {category}: {added} headlines.")

    lbo_url = (
        "https://www.lankabusinessonline.com/wp-json/wp/v2/posts"
        "?per_page=25&orderby=date&order=desc&_fields=date,link,title,excerpt"
    )
    try:
        payload = _get_json(lbo_url)
    except (requests.RequestException, ValueError) as exc:
        notes.append(f"Lanka Business Online was not available ({exc.__class__.__name__}).")
        payload = None
    if payload is None and not any(note.startswith("Lanka Business Online was not") for note in notes):
        notes.append("Lanka Business Online did not return a story list.")
    elif payload is not None:
        added = 0
        for item in payload:
            if not isinstance(item, dict):
                continue
            row = _row("Lanka Business Online", "Business", item)
            if row is None or row["url"] in seen_urls:
                continue
            seen_urls.add(row["url"])
            rows.append(row)
            added += 1
        notes.append(f"Lanka Business Online: {added} headlines.")

    rows.sort(key=lambda item: item["seen_at"], reverse=True)
    return rows, notes


def fetch_rows() -> tuple[list[dict], list[str]]:
    """Save the business CSV and return rows in the hourly headline shape."""
    articles, notes = fetch_articles()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(articles, columns=COLUMNS).to_csv(OUT_PATH, index=False)
    notes.append(f"Saved {len(articles)} Sri Lanka business headlines to {OUT_PATH.name}.")
    hourly = [{key: article[key] for key in HOURLY_COLUMNS} for article in articles]
    return hourly, notes


def merge_latest(rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LATEST_PATH.exists():
        previous = pd.read_csv(LATEST_PATH)
        combined = pd.concat([previous, frame], ignore_index=True)
    else:
        combined = frame
    has_url = combined["url"].fillna("").astype(str).str.len() > 8
    by_url = combined["region"].astype(str) + "|" + combined["url"].fillna("").astype(str)
    by_title = (
        combined["region"].astype(str)
        + "|"
        + combined["title"].fillna("").astype(str).str.lower()
        + "|"
        + combined["date"].astype(str)
    )
    combined["dedupe"] = by_url.where(has_url, by_title)
    combined = combined.drop_duplicates("dedupe", keep="last").drop(columns=["dedupe"])
    combined.to_csv(LATEST_PATH, index=False)


def main() -> None:
    articles, notes = fetch_articles()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(articles, columns=COLUMNS).to_csv(OUT_PATH, index=False)
    hourly = [{key: article[key] for key in HOURLY_COLUMNS} for article in articles]
    merge_latest(hourly)
    for note in notes:
        print(note)
    print(f"Saved {len(articles)} headlines to {OUT_PATH}")
    print()
    for article in articles[:20]:
        print(f"{article['date']}  [{article['source']} / {article['category']}]  {article['title']}")


if __name__ == "__main__":
    main()
