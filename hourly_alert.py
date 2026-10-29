"""Predict the five CSE stocks from historical prices and news, then send Telegram.

Runs every hour. It refreshes the official CSE daily prices, reads the model
trained on about five years of prices and headlines, adds the live trade
price and the latest CSE and market news, and sends one Telegram message.

GOOD means the model expects the next close to be higher.
BAD means it expects the next close to be lower.
UNCLEAR means the chance of a higher close is near the middle.

Buy limit is the highest price this outlook treats as a buy.
Sell limit is the lowest price this outlook treats as a sell.

This is a research alert, not a trading instruction.

    python hourly_alert.py
    python hourly_alert.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import requests
import sklearn._loss._loss as _sklearn_cy_loss

# The saved model names a Cython loss module "_loss". Python only finds it
# after this alias, including under the scikit-learn build that created it.
sys.modules.setdefault("_loss", _sklearn_cy_loss)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

os.chdir(ROOT)

from cse_common import extract_trade_price
from fetch_cse_history import COMBINED_PATH, COMPANIES, post
from train_news_price import FEATURES, MODEL_PATH, build_panel
from telegram_alert import send_message as send_telegram

COLOMBO = ZoneInfo("Asia/Colombo")
OUT_DIR = ROOT / "cse_output"
CONFIG_PATH = OUT_DIR / "hourly_config.json"
OUTLOOK_PATH = OUT_DIR / "hourly_outlook.csv"
MESSAGE_PATH = OUT_DIR / "hourly_message.txt"
LOG_PATH = OUT_DIR / "hourly_alert.log"
HISTORY_STAMP_PATH = OUT_DIR / "history_refresh.json"
LOCK_PATH = OUT_DIR / "hourly_alert.lock"
LIVE_URL = "https://www.cse.lk/api/tradeSummary"

DEFAULT_CONFIG = {
    "weekdays_only": False,
    "start_hour": 0,
    "end_hour": 24,
    "send_always": True,
}


def log(message: str) -> None:
    stamp = datetime.now(COLOMBO).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {message}"
    print(line)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_hourly_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    if not CONFIG_PATH.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return config
    if isinstance(saved, dict):
        config.update(saved)
    return config


def inside_send_window(now: datetime, config: dict, force: bool) -> bool:
    if force or config.get("send_always"):
        return True
    if config.get("weekdays_only", True) and now.weekday() >= 5:
        return False
    start = int(config.get("start_hour", 8))
    end = int(config.get("end_hour", 18))
    return start <= now.hour < end


def previous_weekday(day: datetime) -> datetime:
    cursor = day - timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    return cursor


def expected_session_date(now: datetime) -> datetime.date:
    """Last completed weekday. Today's bar is often missing until the next morning."""
    return previous_weekday(now).date()


def latest_saved_date() -> datetime.date | None:
    if not COMBINED_PATH.exists():
        return None
    frame = pd.read_csv(COMBINED_PATH, usecols=["date"])
    if frame.empty:
        return None
    parsed = pd.to_datetime(frame["date"], errors="coerce").max()
    if pd.isna(parsed):
        return None
    return parsed.date()


def refresh_history_if_stale(now: datetime) -> str:
    latest = latest_saved_date()
    needed = expected_session_date(now)
    latest_text = latest.isoformat() if latest else None
    if latest is not None and latest >= needed:
        return f"Daily prices already include {latest_text}."

    stamp = {}
    if HISTORY_STAMP_PATH.exists():
        try:
            stamp = json.loads(HISTORY_STAMP_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stamp = {}
    if stamp.get("checked_on") == now.date().isoformat() and stamp.get("latest") == latest_text:
        return f"Daily prices still stop at {latest_text}. The CSE chart has no newer bar yet."

    from fetch_cse_history import main as download_history

    log(f"Daily prices stop at {latest_text}. Downloading CSE history through {needed.isoformat()}.")
    download_history()
    refreshed = latest_saved_date()
    refreshed_text = refreshed.isoformat() if refreshed else None
    HISTORY_STAMP_PATH.write_text(
        json.dumps({"checked_on": now.date().isoformat(), "latest": refreshed_text}, indent=2),
        encoding="utf-8",
    )
    return f"Daily prices refreshed through {refreshed_text}."


def build_feature_panel(prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Same feature table the trainer saves, including the CSE index and report counts."""
    del prices
    panel, _undated, _stats = build_panel()
    panel[FEATURES] = panel[FEATURES].replace([float("inf"), float("-inf")], pd.NA)
    return panel


def predict_latest(panel: pd.DataFrame) -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing saved model: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    features = list(bundle.get("features") or FEATURES)
    missing = [name for name in features if name not in panel.columns]
    if missing:
        raise RuntimeError(f"Price table is missing model features: {', '.join(missing)}")
    usable = panel.dropna(subset=features).sort_values("date")
    latest = usable.groupby("symbol", as_index=False).tail(1)
    if latest.empty:
        raise RuntimeError("No complete price rows to predict.")
    change = bundle["regressor"].predict(latest[features])
    probability = bundle["classifier"].predict_proba(latest[features])[:, 1]
    names = {company["symbol"]: company["name"] for company in COMPANIES}
    forecast = pd.DataFrame(
        {
            "symbol": latest["symbol"].values,
            "company": latest["symbol"].map(names).fillna(latest["symbol"]).values,
            "as_of": latest["date"].dt.date.astype(str).values,
            "last_close": latest["close"].astype(float).values,
            "predicted_change_pct": change * 100,
            "up_probability": probability,
        }
    )
    return forecast


def fetch_live_quotes() -> dict[str, dict]:
    try:
        payload = post("tradeSummary", {})
    except Exception:
        response = requests.post(
            LIVE_URL,
            json={},
            headers={
                "User-Agent": "Mozilla/5.0",
                "Origin": "https://www.cse.lk",
                "Referer": "https://www.cse.lk/",
                "X-Requested-With": "XMLHttpRequest",
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    quotes = {}
    for item in payload.get("reqTradeSummery", []):
        symbol = str(item.get("symbol") or "")
        price = extract_trade_price(item)
        if not symbol or price is None:
            continue
        change = item.get("percentageChange")
        try:
            change_pct = float(change) if change not in (None, "") else None
        except (TypeError, ValueError):
            change_pct = None
        quotes[symbol] = {"price": price, "change_pct": change_pct}
    return quotes


def outlook_for(change_pct: float, probability: float) -> tuple[str, str]:
    if probability >= 0.55 and change_pct > 0:
        return "GOOD", "BUY"
    if probability <= 0.45 and change_pct < 0:
        return "BAD", "SELL"
    return "UNCLEAR", "WAIT"


def price_limits(price: float, change_pct: float, outlook: str) -> tuple[float, float]:
    """Return the buy limit and the sell limit in LKR."""
    target = price * (1 + change_pct / 100)
    if outlook == "GOOD":
        buy = price
        sell = target if target > price else price * 1.01
    elif outlook == "BAD":
        sell = price
        buy = target if target < price else price * 0.99
    else:
        buy = price * 0.99
        sell = price * 1.01
    return round(buy, 2), round(sell, 2)


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def saved_quotes() -> dict[str, dict]:
    path = ROOT / "data" / "cse_api" / "company_quotes.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "symbol" not in frame.columns:
        return {}
    details = {}
    for row in frame.drop_duplicates("symbol", keep="last").itertuples(index=False):
        details[str(row.symbol)] = {
            "beta": _number(getattr(row, "beta_triASIBetaValue", None)),
            "foreign_pct": _number(getattr(row, "foreignPercentage", None)),
            "high_52w": _number(getattr(row, "p12HiPrice", None)),
            "low_52w": _number(getattr(row, "p12LowPrice", None)),
        }
    return details


def market_lines() -> list[str]:
    path = ROOT / "data" / "cse_api" / "market_now.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    lines = []
    status = payload.get("marketStatus") or {}
    if isinstance(status, dict) and status.get("status"):
        lines.append(f"CSE: {status['status']}")
    aspi = payload.get("aspiData") or {}
    snp = payload.get("snpData") or {}
    if isinstance(aspi, dict) and _number(aspi.get("value")) is not None:
        change = _number(aspi.get("percentage"))
        change_text = f" ({change:+.2f}%)" if change is not None else ""
        lines.append(f"ASPI {float(aspi['value']):,.2f}{change_text}")
    if isinstance(snp, dict) and _number(snp.get("value")) is not None:
        change = _number(snp.get("percentage"))
        change_text = f" ({change:+.2f}%)" if change is not None else ""
        lines.append(f"S&P SL20 {float(snp['value']):,.2f}{change_text}")
    return lines


def build_rows(forecast: pd.DataFrame, quotes: dict[str, dict]) -> list[dict]:
    saved = saved_quotes()
    rows = []
    for item in forecast.itertuples(index=False):
        live = quotes.get(item.symbol, {})
        price = float(live.get("price") or item.last_close)
        detail = saved.get(item.symbol, {})
        high_52 = detail.get("high_52w")
        off_high = round((price / high_52 - 1) * 100, 1) if high_52 else None
        price_source = "live" if live.get("price") else "last close"
        change_pct = float(item.predicted_change_pct)
        probability = float(item.up_probability)
        outlook, action = outlook_for(change_pct, probability)
        buy_limit, sell_limit = price_limits(price, change_pct, outlook)
        rows.append(
            {
                "symbol": item.symbol,
                "company": item.company,
                "as_of": item.as_of,
                "price": round(price, 2),
                "price_source": price_source,
                "today_change_pct": live.get("change_pct"),
                "predicted_change_pct": round(change_pct, 3),
                "predicted_next": round(price * (1 + change_pct / 100), 2),
                "up_probability": round(probability, 3),
                "outlook": outlook,
                "action": action,
                "buy_limit": buy_limit,
                "sell_limit": sell_limit,
                "api_beta": None if detail.get("beta") is None else round(detail["beta"], 2),
                "foreign_pct": None if detail.get("foreign_pct") is None else round(detail["foreign_pct"], 2),
                "off_52w_high_pct": off_high,
            }
        )
    order = {"GOOD": 0, "BAD": 1, "UNCLEAR": 2}
    rows.sort(key=lambda row: (order[row["outlook"]], -row["up_probability"], row["company"]))
    return rows


def news_brief(now: datetime) -> list[str]:
    """Live CSE announcements and the move that followed similar old headlines."""
    import hourly_news
    from news_reactions import MODEL_PATH as NEWS_MODEL
    from news_reactions import REGION_LABELS, relevant_to_prices, score_headlines

    feed_rows, notes = hourly_news.fetch_feeds()
    cse_rows, cse_note = hourly_news.fetch_cse_announcements(now - timedelta(days=2))
    notes.append(cse_note)
    headlines = pd.DataFrame(feed_rows + cse_rows)
    if headlines.empty:
        return ["Live news", *notes, "No headlines this hour."]
    headlines["dedupe"] = hourly_news._dedupe_key(headlines)
    headlines = headlines.drop_duplicates("dedupe").drop(columns=["dedupe"])
    hourly_news.remember_headlines(headlines)
    current = hourly_news.recent_only(headlines, now)
    scored = current[current["title"].map(relevant_to_prices)].copy() if not current.empty else current
    lines = [
        "Live news and CSE market",
        f"Headlines this hour: {len(current)}. Used for the news score: {len(scored)}.",
        cse_note,
    ]
    if not NEWS_MODEL.exists() or scored.empty:
        if not NEWS_MODEL.exists():
            lines.append("The news-reaction model is not trained yet.")
        return lines
    bundle = joblib.load(NEWS_MODEL)
    headline_scores, forecast = score_headlines(scored, bundle)
    if not forecast.empty:
        from news_reactions import latest_closes
        from news_reactions import load_prices as load_reaction_prices

        closes = latest_closes(load_reaction_prices()).drop(columns=["company"])
        forecast = forecast.merge(closes, on="symbol", how="left")
        forecast["live_price"] = pd.NA
        forecast["predicted_close"] = (forecast["close"] * (1 + forecast["pred_1d"])).round(2)
        forecast["close"] = forecast["close"].round(2)
        lines.append("After similar old headlines, the next session averaged:")
        for row in forecast.sort_values("company").itertuples(index=False):
            lines.append(
                f"  {row.company}: {row.pred_1d * 100:+.2f}% "
                f"({int(row.headline_count)} headlines)"
            )
    if not headline_scores.empty:
        lines.append("Latest headlines:")
        ordered = headline_scores.assign(abs_move=headline_scores["historical_1d"].abs().fillna(0))
        ordered = ordered.sort_values("abs_move", ascending=False)
        for row in ordered.head(6).itertuples(index=False):
            region = REGION_LABELS.get(row.region, row.region)
            title = str(row.title)
            if len(title) > 140:
                title = title[:137].rstrip() + "..."
            if pd.notna(row.historical_1d):
                history = f"old similar news {row.historical_1d * 100:+.2f}%"
            else:
                history = "no similar old group"
            lines.append(f"  [{region}] {row.tone}, {history}")
            lines.append(f"    {row.source}: {title}")
        forecast.to_csv(hourly_news.FORECAST_PATH, index=False)
        headline_scores.to_csv(hourly_news.HEADLINE_PATH, index=False)
    hourly_news.write_report(now, notes, headline_scores, forecast, bundle.get("metrics"))
    return lines


def format_message(rows: list[dict], now: datetime, news_lines: list[str] | None = None) -> str:
    lines = [
        "CSE hourly outlook",
        now.strftime("%a %d %b %Y, %H:%M Colombo"),
        "",
        "Research only. Not a trading instruction.",
        "",
    ]
    lines.extend(market_lines())
    if lines[-1] != "":
        lines.append("")
    if not rows:
        lines.append("No stock outlook this hour.")
        return "\n".join(lines)

    labels = {"GOOD": "GOOD future", "BAD": "BAD future", "UNCLEAR": "UNCLEAR future"}
    for row in rows:
        today = row["today_change_pct"]
        today_text = f", today {today:+.2f}%" if today is not None else ""
        source = "Live" if row["price_source"] == "live" else "Last close"
        lines.extend(
            [
                f"{labels[row['outlook']]} — {row['action']}",
                f"{row['company']} ({row['symbol']})",
                f"{source} {row['price']:.2f} LKR{today_text}",
                (
                    f"Next close about {row['predicted_next']:.2f} "
                    f"({row['predicted_change_pct']:+.2f}%, up chance {row['up_probability']:.0%})"
                ),
                f"Buy limit: {row['buy_limit']:.2f} LKR",
                f"Sell limit: {row['sell_limit']:.2f} LKR",
            ]
        )
        quote_bits = []
        if row.get("api_beta") is not None:
            quote_bits.append(f"beta {row['api_beta']:.2f}")
        if row.get("foreign_pct") is not None:
            quote_bits.append(f"foreign holding {row['foreign_pct']:.1f}%")
        if row.get("off_52w_high_pct") is not None:
            quote_bits.append(f"{row['off_52w_high_pct']:+.1f}% vs 52-week high")
        if quote_bits:
            lines.append("CSE quote: " + ", ".join(quote_bits))
        lines.append("")
    lines.append("Buy limit = highest price to buy. Sell limit = lowest price to sell.")
    lines.append(f"Price model uses daily closes through {rows[0]['as_of']}.")
    if news_lines:
        lines.extend(["", *news_lines])
    text = "\n".join(lines).strip()
    if len(text) > 4000:
        text = text[:3990].rstrip() + "\n..."
    return text


def save_outputs(rows: list[dict], message: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTLOOK_PATH, index=False)
    MESSAGE_PATH.write_text(message + "\n", encoding="utf-8")


def run(force: bool) -> int:
    now = datetime.now(COLOMBO)
    config = load_hourly_config()
    if not inside_send_window(now, config, force):
        log("Outside the send window. No message sent.")
        return 0
    if LOCK_PATH.exists() and time.time() - LOCK_PATH.stat().st_mtime < 50 * 60:
        log("A run is already in progress. Skipping.")
        return 0
    LOCK_PATH.write_text(now.isoformat(), encoding="utf-8")
    try:
        note = refresh_history_if_stale(now)
        log(note)
        try:
            from fetch_cse_api import refresh_tracked

            log(refresh_tracked())
        except Exception as exc:
            log(f"CSE detail refresh was skipped. {exc}")
        forecast = predict_latest(build_feature_panel())
        try:
            quotes = fetch_live_quotes()
            log(f"Live prices loaded for {len(quotes)} symbols.")
        except Exception as exc:
            quotes = {}
            log(f"Live prices unavailable. Using last closes. {exc}")
        rows = build_rows(forecast, quotes)
        try:
            news_lines = news_brief(now)
            log(news_lines[0] if news_lines else "No news section.")
        except Exception as exc:
            news_lines = [f"Live news was not added ({exc.__class__.__name__})."]
            log(news_lines[0])
        message = format_message(rows, now, news_lines)
        save_outputs(rows, message)
        log("Outlook saved.")
        print(message)
        ok, detail = send_telegram(message)
        log(detail if ok else f"Telegram not sent. {detail}")
        return 0
    except Exception:
        log(traceback.format_exc().strip())
        return 1
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly CSE outlook Telegram alert")
    parser.add_argument("--force", action="store_true", help="Send even outside 08:00-18:00 Colombo on weekdays")
    args = parser.parse_args()
    raise SystemExit(run(force=args.force))


if __name__ == "__main__":
    main()
