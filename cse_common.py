"""Shared helpers for CSE dates, live prices, and news files."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTHS.update({name[:3]: month for name, month in list(_MONTHS.items())})

_TEXT_DATE = re.compile(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})")
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_YMD_DATE = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})")
_DMY_DATE = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})")
_YEAR_ONLY = re.compile(r"^\d{4}$")
_NEWS_YEAR = re.compile(r"cse_news_(\d{4})\.csv$")


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_cse_date(value) -> date | None:
    """Parse CSE date text without turning a bare year into 1 January."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "nat"}:
        return None
    if _YEAR_ONLY.fullmatch(text):
        return None

    iso_match = _ISO_DATE.search(text)
    if iso_match:
        return _safe_date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))

    text_match = _TEXT_DATE.search(text)
    if text_match:
        month = _MONTHS.get(text_match.group(2).lower())
        if month:
            parsed = _safe_date(int(text_match.group(3)), month, int(text_match.group(1)))
            if parsed:
                return parsed

    ymd_match = _YMD_DATE.match(text)
    if ymd_match:
        return _safe_date(int(ymd_match.group(1)), int(ymd_match.group(2)), int(ymd_match.group(3)))

    dmy_match = _DMY_DATE.match(text)
    if dmy_match:
        return _safe_date(int(dmy_match.group(3)), int(dmy_match.group(2)), int(dmy_match.group(1)))

    return None


def _as_price(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    return price


def extract_trade_price(item: dict) -> float | None:
    """Read a CSE tradeSummary row. The live field is lastTrade, not price."""
    if not isinstance(item, dict):
        return None
    for key in ("lastTrade", "lastTradedPrice", "price"):
        price = _as_price(item.get(key))
        if price is not None:
            return price
    return _as_price(item.get("previousClose"))


def select_news_csv_names(names: list[str]) -> list[str]:
    """Use one combined news file when it exists so yearly copies are not loaded again."""
    csv_names = [name for name in names if name.startswith("cse_news") and name.endswith(".csv")]
    combined = sorted(name for name in csv_names if "_ALL_" in name)
    if combined:
        return [combined[-1]]
    yearly = sorted(name for name in csv_names if _NEWS_YEAR.fullmatch(name))
    return yearly or sorted(csv_names)


def select_news_files(directory: Path) -> list[Path]:
    files = [path for path in directory.glob("cse_news_*.csv") if path.is_file()]
    chosen = select_news_csv_names([path.name for path in files])
    by_name = {path.name: path for path in files}
    return [by_name[name] for name in chosen if name in by_name]
