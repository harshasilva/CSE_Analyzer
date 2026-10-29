"""Fetch world, Sri Lanka, and stock-market news, then score the five CSE stocks.

Run once an hour. The score comes from the model trained on about five years
of older headlines and the price move that followed each one.

    python hourly_news.py
    python hourly_news.py --install

--install registers a Windows task named "CSE Hourly News".
This is a research forecast, not a trading instruction.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from news_reactions import (
    COLOMBO,
    MODEL_PATH,
    REGION_LABELS,
    clean_title,
    latest_closes,
    load_prices,
    parse_seen,
    relevant_to_prices,
    score_headlines,
)

LATEST_PATH = ROOT / "data" / "news_latest.csv"
HEADLINE_PATH = ROOT / "cse_output" / "hourly_headlines.csv"
FORECAST_PATH = ROOT / "cse_output" / "hourly_news_forecast.csv"
REPORT_PATH = ROOT / "cse_output" / "hourly_news_report.txt"
LOG_PATH = ROOT / "cse_output" / "hourly_news.log"
TASK_NAME = "CSE Hourly News"

FEEDS = (
    ("world", "BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    (
        "world",
        "Google markets",
        "https://news.google.com/rss/search?q=%22stock+market%22+OR+%22federal+reserve%22+OR+%22oil+price%22+when:1d&hl=en-US&gl=US&ceid=US:en",
    ),
    (
        "sri_lanka",
        "Google Sri Lanka",
        "https://news.google.com/rss/search?q=%22Sri+Lanka%22+when:1d&hl=en-LK&gl=LK&ceid=LK:en",
    ),
    (
        "sri_lanka",
        "Google Sri Lanka economy",
        "https://news.google.com/rss/search?q=%22Sri+Lanka%22+(economy+OR+rupee+OR+inflation+OR+IMF+OR+stocks+OR+bank)+when:1d&hl=en-LK&gl=LK&ceid=LK:en",
    ),
    ("sri_lanka", "EconomyNext", "https://economynext.com/feed/"),
    ("sri_lanka", "Ada Derana", "https://www.adaderana.lk/rss.php"),
    (
        "market",
        "Google CSE",
        "https://news.google.com/rss/search?q=%22Colombo+Stock+Exchange%22+OR+ASPI+OR+%22Sri+Lanka+stocks%22+when:2d&hl=en-LK&gl=LK&ceid=LK:en",
    ),
)

CSE_ENDPOINTS = (
    ("approvedAnnouncement", "approvedAnnouncements", "CSE"),
    ("getFinancialAnnouncement", "financialAnnouncements", "CSE financial"),
    ("circularAnnouncement", "circularAnnouncements", "CSE circular"),
    ("directiveAnnouncement", "directiveAnnouncements", "CSE directive"),
    ("getNonComplianceAnnouncements", "nonComplianceAnnouncements", "CSE compliance"),
    ("getNewListingsRelatedNoticesAnnouncements", "newListingRelatedAnnouncements", "CSE listings"),
    ("getBuyInBoardAnnouncements", "buyInBoardAnnouncements", "CSE buy-in"),
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://www.cse.lk",
    "Referer": "https://www.cse.lk/",
}
ATOM = "{http://www.w3.org/2005/Atom}"


def log(message: str) -> None:
    stamp = datetime.now(COLOMBO).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _text(node: ET.Element | None, name: str) -> str:
    if node is None:
        return ""
    child = node.find(name)
    if child is None or child.text is None:
        return ""
    return clean_title(child.text)


def parse_feed(region: str, source: str, payload: str) -> list[dict]:
    root = ET.fromstring(payload)
    rows = []
    for item in root.findall(".//item"):
        title = _text(item, "title")
        outlet = None
        if " - " in title:
            headline, maybe_outlet = title.rsplit(" - ", 1)
            if len(headline) >= 12:
                title = headline.strip()
                outlet = maybe_outlet.strip()
        if len(title) < 12:
            continue
        link = _text(item, "link") or _text(item, "guid")
        seen = _parse_pub_date(_text(item, "pubDate"))
        source_node = item.find("source")
        feed_source = outlet or source
        if source_node is not None and source_node.text:
            feed_source = clean_title(source_node.text)
        rows.append(_row(region, feed_source, title, link, seen))
    for entry in root.findall(f".//{ATOM}entry"):
        title = _text(entry, f"{ATOM}title")
        if len(title) < 12:
            continue
        link = ""
        for link_node in entry.findall(f"{ATOM}link"):
            link = link_node.attrib.get("href") or link
            if link_node.attrib.get("rel", "alternate") == "alternate" and link:
                break
        seen = _parse_pub_date(_text(entry, f"{ATOM}published") or _text(entry, f"{ATOM}updated"))
        rows.append(_row(region, source, title, link, seen))
    return rows


def _parse_pub_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        parsed = None
    if parsed is None:
        stamp = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(stamp):
            return None
        parsed = stamp.to_pydatetime()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed.astimezone(COLOMBO)


def _row(region: str, source: str, title: str, url: str, seen: datetime | None) -> dict:
    if seen is None:
        seen = datetime.now(COLOMBO)
    return {
        "seen_at": seen.isoformat(),
        "date": seen.date().isoformat(),
        "region": region,
        "title": title[:500],
        "url": url,
        "source": source,
        "country": "",
    }


def fetch_feeds() -> tuple[list[dict], list[str]]:
    rows = []
    notes = []
    for region, source, url in FEEDS:
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
            response.raise_for_status()
            parsed = parse_feed(region, source, response.text)
        except (requests.RequestException, ET.ParseError) as exc:
            notes.append(f"{source} was not available ({exc.__class__.__name__}).")
            continue
        parsed.sort(key=lambda item: item["seen_at"], reverse=True)
        rows.extend(parsed[:12])
        notes.append(f"{source}: {min(len(parsed), 12)} headlines.")
    return rows, notes


def fetch_cse_announcements(since: datetime) -> tuple[list[dict], str]:
    rows = []
    try:
        for endpoint, key, source in CSE_ENDPOINTS:
            response = requests.post(f"https://www.cse.lk/api/{endpoint}", headers=HEADERS, timeout=30)
            response.raise_for_status()
            payload = response.json()
            items = payload.get(key) or []
            if not items:
                for value in payload.values():
                    if isinstance(value, list) and value:
                        items = value
                        break
            for item in items:
                if not isinstance(item, dict):
                    continue
                seen = parse_seen(item.get("createdDate") or item.get("date"))
                if seen is None or seen < since:
                    continue
                parts = [
                    item.get("company") or item.get("companyName") or "",
                    item.get("announcementCategory") or "",
                    item.get("remarks") or "",
                    item.get("title") or item.get("subject") or "",
                ]
                title = clean_title(" ".join(str(part) for part in parts if part))
                if len(title) < 12:
                    continue
                rows.append(_row("market", source, title, f"cse-{item.get('id', title[:40])}", seen))
    except requests.RequestException as exc:
        return rows, f"CSE announcements were not available ({exc.__class__.__name__})."
    return rows, f"CSE announcements in the last two days: {len(rows)}."


def _dedupe_key(frame: pd.DataFrame) -> pd.Series:
    has_url = frame["url"].fillna("").astype(str).str.len() > 8
    by_url = frame["region"].astype(str) + "|" + frame["url"].fillna("").astype(str)
    by_title = frame["region"].astype(str) + "|" + frame["title"].str.lower() + "|" + frame["date"].astype(str)
    return by_url.where(has_url, by_title)


def remember_headlines(headlines: pd.DataFrame) -> None:
    if headlines.empty:
        return
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LATEST_PATH.exists():
        previous = pd.read_csv(LATEST_PATH)
        combined = pd.concat([previous, headlines], ignore_index=True)
    else:
        combined = headlines.copy()
    combined["dedupe"] = _dedupe_key(combined)
    combined = combined.drop_duplicates("dedupe", keep="last").drop(columns=["dedupe"])
    combined.to_csv(LATEST_PATH, index=False)


def recent_only(headlines: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if headlines.empty:
        return headlines
    seen = pd.to_datetime(headlines["seen_at"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(now - timedelta(hours=36))
    chosen = headlines.loc[seen >= cutoff].copy()
    chosen["_seen"] = seen.loc[chosen.index]
    chosen = chosen.sort_values("_seen", ascending=False).drop_duplicates("title").head(40)
    return chosen.drop(columns=["_seen"])


def _percent(value: float) -> str:
    return f"{value * 100:+.2f}%"


def write_report(
    now: datetime,
    notes: list[str],
    headlines: pd.DataFrame,
    forecast: pd.DataFrame,
    metrics: dict | None,
) -> str:
    lines = [
        "Hourly news and CSE price reaction",
        now.strftime("%a %d %b %Y, %H:%M Colombo"),
        "",
        "Research only. Not a trading instruction.",
        "",
    ]
    lines.extend(notes)
    lines.append("")
    if forecast.empty:
        lines.append("No stock forecast this hour.")
    else:
        lines.append("Expected move from the last close, using the average price change after similar old headlines:")
        for row in forecast.sort_values("company").itertuples(index=False):
            if pd.isna(row.close) or pd.isna(row.predicted_close):
                lines.append(f"  {row.company}: next session {_percent(row.pred_1d)}.")
                continue
            live = ""
            if pd.notna(getattr(row, "live_price", None)):
                live = f", live {row.live_price:.2f}"
            lines.append(
                f"  {row.company}: next session {_percent(row.pred_1d)}, "
                f"about {row.predicted_close:.2f} LKR from {row.close:.2f} on {pd.Timestamp(row.date).date().isoformat()}{live}. "
                f"Five sessions {_percent(row.pred_5d)}. Headlines used: {int(row.headline_count)}."
            )
    if not headlines.empty:
        lines.extend(["", "Headlines used this hour:"])
        ordered = headlines.assign(abs_move=headlines["historical_1d"].abs().fillna(0)).sort_values("abs_move", ascending=False)
        for row in ordered.head(12).itertuples(index=False):
            if pd.notna(row.historical_1d):
                history = (
                    f"after similar old news the next session averaged {_percent(row.historical_1d)} "
                    f"({int(row.historical_n)} rows, {row.historical_label})"
                )
            else:
                history = "no similar group in the training years"
            lines.append(f"  [{REGION_LABELS.get(row.region, row.region)}] {row.tone}. {history}")
            lines.append(f"    {row.source}: {row.title}")
    if metrics:
        lines.extend(
            [
                "",
                (
                    f"The model was trained on reactions from {metrics['train_start']} to {metrics['train_end']} "
                    f"and tested on {metrics['test_start']} to {metrics['test_end']}."
                ),
                (
                    f"On those later sessions it got the direction right {metrics['session_direction']:.0%} of the time. "
                    f"Always guessing the common side scored {metrics['majority_direction']:.0%}. "
                    f"Average miss {metrics['session_mae']:.2f} percentage points, versus {metrics['zero_mae']:.2f} for a flat call."
                ),
            ]
        )
    else:
        lines.extend(["", "The reaction model is not trained yet. Run python train_news_reaction.py after the five-year download finishes."])
    text = "\n".join(lines).strip() + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    return text


def python_executable() -> str:
    """Prefer a real python.exe. The WindowsApps alias often fails inside Task Scheduler."""
    current = Path(sys.executable)
    if "WindowsApps" not in str(current) and current.exists():
        return str(current)
    local = Path.home() / "AppData" / "Local" / "Programs" / "Python"
    found = sorted(local.glob("Python*/python.exe")) if local.exists() else []
    if found:
        return str(found[-1])
    return str(current)


def install_task() -> int:
    executable = python_executable()
    script = str(ROOT / "hourly_news.py")
    command = (
        "$action = New-ScheduledTaskAction "
        f"-Execute '{executable}' -Argument '{script}' -WorkingDirectory '{ROOT}'; "
        "$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date '00:05') "
        "-RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 9999); "
        "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew; "
        f"Register-ScheduledTask -TaskName '{TASK_NAME}' -Action $action -Trigger $trigger "
        "-Settings $settings -Force | Out-Null; "
        f"Write-Output 'Registered {TASK_NAME} with {executable}'"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
    )
    message = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0:
        log(f"Could not register the hourly task. {message}")
        return 1
    log(message or f"Windows will run {script} every hour.")
    return 0


def live_prices() -> dict[str, float]:
    try:
        from hourly_alert import fetch_live_quotes

        quotes = fetch_live_quotes()
    except Exception:
        return {}
    return {symbol: float(quote["price"]) for symbol, quote in quotes.items() if quote.get("price")}


def run() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    now = datetime.now(COLOMBO)
    try:
        from hourly_alert import refresh_history_if_stale

        notes_refresh = refresh_history_if_stale(now)
    except Exception as exc:
        notes_refresh = f"Daily prices were not refreshed ({exc.__class__.__name__})."
    feed_rows, notes = fetch_feeds()
    from fetch_business_news import fetch_rows

    biz_rows, biz_notes = fetch_rows()
    feed_rows.extend(biz_rows)
    notes.extend(biz_notes)
    notes.insert(0, notes_refresh)
    cse_rows, cse_note = fetch_cse_announcements(now - timedelta(days=2))
    notes.append(cse_note)
    headlines = pd.DataFrame(feed_rows + cse_rows)
    if not headlines.empty:
        headlines["dedupe"] = _dedupe_key(headlines)
        headlines = headlines.drop_duplicates("dedupe").drop(columns=["dedupe"])
        remember_headlines(headlines)
    current = recent_only(headlines, now) if not headlines.empty else headlines
    scored = current[current["title"].map(relevant_to_prices)].copy() if not current.empty else current
    log(
        f"Collected {0 if headlines.empty else len(headlines)} headlines, "
        f"{0 if current.empty else len(current)} in the last 36 hours, "
        f"{0 if scored.empty else len(scored)} used for the price score."
    )
    notes.append(
        f"Price score used {0 if scored.empty else len(scored)} of "
        f"{0 if current.empty else len(current)} recent headlines. "
        "Sports and other unrelated items were saved but not scored."
    )
    metrics = None
    headline_scores = pd.DataFrame()
    forecast = pd.DataFrame()
    if not MODEL_PATH.exists():
        notes.append("Reaction model not found yet.")
    elif scored.empty:
        notes.append("No economy or market headlines in the last 36 hours.")
    else:
        bundle = joblib.load(MODEL_PATH)
        metrics = bundle.get("metrics")
        headline_scores, forecast = score_headlines(scored, bundle)
        prices = load_prices()
        forecast = forecast.merge(latest_closes(prices), on=["symbol", "company"], how="left")
        live = live_prices()
        forecast["live_price"] = forecast["symbol"].map(live)
        forecast["predicted_close"] = (forecast["close"] * (1 + forecast["pred_1d"])).round(2)
        forecast["close"] = forecast["close"].round(2)
        FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(FORECAST_PATH, index=False)
        headline_scores.to_csv(HEADLINE_PATH, index=False)
    for note in notes:
        log(note)
    report = write_report(now, notes, headline_scores, forecast, metrics)
    print(report)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly world, Sri Lanka, and stock-market news forecast")
    parser.add_argument("--install", action="store_true", help="Register a Windows task that runs this every hour")
    args = parser.parse_args()
    if args.install:
        raise SystemExit(install_task())
    raise SystemExit(run())


if __name__ == "__main__":
    main()
