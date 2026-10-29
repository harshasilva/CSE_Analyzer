"""Refresh the CSE API, save world news, retrain, and predict every listed stock.

Run once an hour:

    python hourly_market.py
    python hourly_market.py --install

--install registers a Windows task named "CSE Hourly Market".
The forecast is research output, not a trading instruction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hourly_news import fetch_cse_announcements, fetch_feeds, python_executable
from news_reactions import COLOMBO, relevant_to_prices

OUT_DIR = ROOT / "cse_output"
NEWS_ARCHIVE = ROOT / "data" / "news_archive.csv"
NEWS_LATEST = ROOT / "data" / "news_latest.csv"
HEADLINE_PATH = OUT_DIR / "hourly_market_headlines.csv"
REPORT_PATH = OUT_DIR / "hourly_market_report.txt"
STATUS_PATH = OUT_DIR / "hourly_market_status.json"
LOG_PATH = OUT_DIR / "hourly_market.log"
LOCK_PATH = OUT_DIR / "hourly_market.lock"
FORECAST_PATH = OUT_DIR / "cse_market_all_forecast.csv"
MODEL_REPORT_PATH = OUT_DIR / "cse_market_all_report.txt"
TASK_NAME = "CSE Hourly Market"
ARCHIVE_COLUMNS = ["seen_at", "date", "region", "title", "url", "source", "country"]
HIGHLIGHT_MIN_VOLUME = 1000
HIGHLIGHT_MIN_PRICE = 5


def log(message: str) -> None:
    stamp = datetime.now(COLOMBO).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {message}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _dedupe_key(frame: pd.DataFrame) -> pd.Series:
    url = frame["url"].fillna("").astype(str)
    by_url = frame["region"].astype(str) + "|" + url
    by_title = (
        frame["region"].astype(str)
        + "|"
        + frame["title"].fillna("").str.lower()
        + "|"
        + frame["date"].astype(str)
    )
    return by_url.where(url.str.len() > 8, by_title)


def append_news(headlines: pd.DataFrame) -> int:
    """Add new headlines to the archive the full-market model trains on."""
    if headlines.empty:
        return 0
    incoming = headlines.reindex(columns=ARCHIVE_COLUMNS)
    if NEWS_ARCHIVE.exists() and NEWS_ARCHIVE.stat().st_size:
        previous = pd.read_csv(NEWS_ARCHIVE)
        for column in ARCHIVE_COLUMNS:
            if column not in previous.columns:
                previous[column] = ""
        previous = previous.reindex(columns=ARCHIVE_COLUMNS)
        before = len(previous)
        combined = pd.concat([previous, incoming], ignore_index=True)
    else:
        before = 0
        combined = incoming.copy()
    combined["dedupe"] = _dedupe_key(combined)
    combined = combined.drop_duplicates("dedupe", keep="last").drop(columns=["dedupe"])
    NEWS_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(NEWS_ARCHIVE, index=False)
    if NEWS_LATEST.exists() and NEWS_LATEST.stat().st_size:
        latest = pd.read_csv(NEWS_LATEST)
        for column in ARCHIVE_COLUMNS:
            if column not in latest.columns:
                latest[column] = ""
        latest = pd.concat([latest.reindex(columns=ARCHIVE_COLUMNS), incoming], ignore_index=True)
    else:
        latest = incoming.copy()
    latest["dedupe"] = _dedupe_key(latest)
    latest = latest.drop_duplicates("dedupe", keep="last").drop(columns=["dedupe"])
    latest.to_csv(NEWS_LATEST, index=False)
    return max(len(combined) - before, 0)


def collect_news(now: datetime) -> tuple[pd.DataFrame, list[str], int]:
    feed_rows, notes = fetch_feeds()
    cse_rows, cse_note = fetch_cse_announcements(now - timedelta(days=2))
    notes.append(cse_note)
    headlines = pd.DataFrame(feed_rows + cse_rows)
    if headlines.empty:
        return headlines, notes, 0
    for column in ARCHIVE_COLUMNS:
        if column not in headlines.columns:
            headlines[column] = ""
    headlines = headlines.reindex(columns=ARCHIVE_COLUMNS)
    keep = headlines["title"].map(lambda title: relevant_to_prices(str(title)))
    # CSE announcement rows are already market items. Keep them even if the word list misses.
    cse_mask = headlines["source"].fillna("").astype(str).str.startswith("CSE")
    chosen = headlines[keep | cse_mask].copy()
    notes.append(
        f"Saved {len(chosen)} economy and market headlines. "
        f"Left out {len(headlines) - len(chosen)} unrelated items."
    )
    if not chosen.empty:
        chosen["dedupe"] = _dedupe_key(chosen)
        chosen = chosen.drop_duplicates("dedupe").drop(columns=["dedupe"])
        HEADLINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        chosen.to_csv(HEADLINE_PATH, index=False)
    added = append_news(chosen)
    notes.append(f"New headlines added to the training archive: {added}.")
    return chosen, notes, added


def api_snapshot() -> str:
    from fetch_cse_api import save_market

    market = save_market()
    return (
        f"CSE API saved {len(market['board'])} listed symbols, "
        f"{market['trades']} trade rows, {market['market_days']} market days."
    )


def market_status_lines() -> list[str]:
    from hourly_alert import market_lines

    return market_lines()


def write_hourly_report(now: datetime, notes: list[str]) -> None:
    lines = [
        "CSE hourly market update",
        now.strftime("%a %d %b %Y, %H:%M Colombo"),
        "",
        "Research only. Not a trading instruction.",
        "",
    ]
    lines.extend(market_status_lines())
    lines.append("")
    lines.extend(notes)
    if FORECAST_PATH.exists():
        forecast = pd.read_csv(FORECAST_PATH)
        forecast = forecast.dropna(subset=["predicted_change_pct"])
        shown = forecast[
            (forecast["volume"] >= HIGHLIGHT_MIN_VOLUME) & (forecast["last_close"] >= HIGHLIGHT_MIN_PRICE)
        ].sort_values("predicted_change_pct", ascending=False)
        gains = shown[shown["predicted_change_pct"] > 0]
        declines = shown[shown["predicted_change_pct"] < 0].sort_values("predicted_change_pct")
        horizon = ""
        if "horizon_sessions" in forecast.columns and forecast["horizon_sessions"].notna().any():
            horizon = f" over the next {int(forecast['horizon_sessions'].iloc[0])} sessions"
        lines.extend(
            [
                "",
                f"Forecast for {len(forecast)} symbols{horizon}.",
                f"Median predicted change: {forecast['predicted_change_pct'].median():+.2f}%.",
                f"Called higher: {int((forecast['predicted_change_pct'] > 0).sum())}.",
                "",
                "Largest predicted gains:",
            ]
        )
        for row in gains.head(8).itertuples(index=False):
            lines.append(
                f"  {row.symbol} {row.company}: {row.last_close} -> {row.predicted_change_pct:+.2f}% "
                f"(up probability {row.up_probability:.0%})"
            )
        lines.extend(["", "Largest predicted declines:"])
        if declines.empty:
            lines.append("  None. Every published call is higher.")
        for row in declines.head(8).itertuples(index=False):
            lines.append(
                f"  {row.symbol} {row.company}: {row.last_close} -> {row.predicted_change_pct:+.2f}% "
                f"(up probability {row.up_probability:.0%})"
            )
        lines.extend(["", f"Full forecast: {FORECAST_PATH}", f"Model report: {MODEL_REPORT_PATH}"])
    text = "\n".join(lines).strip() + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")
    STATUS_PATH.write_text(
        json.dumps(
            {
                "updated_at": now.isoformat(timespec="seconds"),
                "notes": notes,
                "forecast": str(FORECAST_PATH),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def install_task() -> int:
    executable = python_executable()
    script = str(ROOT / "hourly_market.py")
    command = (
        "$action = New-ScheduledTaskAction "
        f"-Execute '{executable}' -Argument '{script}' -WorkingDirectory '{ROOT}'; "
        "$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date '00:10') "
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


def run() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    now = datetime.now(COLOMBO)
    if LOCK_PATH.exists() and time.time() - LOCK_PATH.stat().st_mtime < 50 * 60:
        log("A run is already in progress. Skipping.")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(now.isoformat(), encoding="utf-8")
    notes: list[str] = []
    try:
        try:
            notes.append(api_snapshot())
        except Exception as exc:
            notes.append(f"CSE API snapshot failed ({exc.__class__.__name__}: {exc}).")
            log(notes[-1])
        try:
            from fetch_market_all import refresh_prices

            notes.append(refresh_prices())
        except Exception as exc:
            notes.append(f"Price refresh failed ({exc.__class__.__name__}: {exc}).")
            log(notes[-1])
        _headlines, news_notes, _added = collect_news(now)
        notes.extend(news_notes)
        for note in notes:
            log(note)
        log("Training the full-market model and writing the next-session forecast.")
        from train_market_all import main as train_main

        train_main()
        write_hourly_report(now, notes)
        log(f"Hourly forecast saved to {FORECAST_PATH}")
        if REPORT_PATH.exists():
            print(REPORT_PATH.read_text(encoding="utf-8"))
        return 0
    except Exception:
        log(traceback.format_exc().strip())
        return 1
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly CSE API, world news, train, and predict")
    parser.add_argument("--install", action="store_true", help="Register a Windows task that runs this every hour")
    args = parser.parse_args()
    if args.install:
        raise SystemExit(install_task())
    raise SystemExit(run())


if __name__ == "__main__":
    main()
