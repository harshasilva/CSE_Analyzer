"""Predict the next CSE session's price change from five years of prices and news.

The price history is the Yahoo Finance daily file, with official CSE closes
used wherever the exchange file has that session. News is the CSE announcement
file plus the saved world, Sri Lanka, and stock-market headline archive.

A headline is attached to a stock when it names that company. Other headlines
are market-wide and apply to every stock. The model only sees news from the
day itself and the four sessions before it, then predicts the following close.

The CSE chart adds the official ASPI return where that date exists, how far
the close sits from its trailing one-year high and low, and how many financial
reports were published in the last five sessions. A quote field that the API
only returns for today, such as beta or foreign holding, is not copied onto
older rows.

Run from this folder:

    python train_news_price.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_absolute_error

from news_sentiment import extract_date, score_text

PRICE_PATH = Path("data/cse_prices_5y.csv")
SITE_PRICE_PATH = Path("data/cse_site_prices_1y.csv")
ASPI_PATH = Path("data/cse_api/aspi_history.csv")
FINANCIAL_DIR = Path("data/cse_api/financials")
ANNOUNCEMENTS_PATH = Path("cse_output/cse_news_ALL_2021_2026.csv")
WORLD_NEWS_PATH = Path("cse_output/cse_combined_sources.csv")
ARCHIVE_PATH = Path("data/news_archive.csv")
MODEL_PATH = Path("models/cse_news_price_model.joblib")
REPORT_PATH = Path("cse_output/cse_news_price_report.txt")
FORECAST_PATH = Path("cse_output/cse_news_price_forecast.csv")

# A one-day move larger than this is a split or a bad print, not a normal return.
MAX_ABS_RETURN = 0.35
TEST_FRACTION = 0.20

COMPANIES = {
    "COMB.N0000": ("commercial bank", "combank"),
    "SAMP.N0000": ("sampath bank", "sampath"),
    "HNB.N0000": ("hatton national", "hnb"),
    "JKH.N0000": ("john keells", "keells"),
    "DIAL.N0000": ("dialog axiata", "dialog"),
}

PRICE_FEATURES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_10",
    "vol_20",
    "volume_ratio",
    "dist_ma_10",
    "dist_ma_20",
    "range_pct",
    "dist_high_252",
    "dist_low_252",
    "rsi_14",
    "mkt_ret_1",
]
CONTEXT_FEATURES = [
    "index_ret_1",
    "index_ret_5",
    "report_count_5d",
]
NEWS_FEATURES = [
    "ann_count_5d",
    "ann_sent_5d",
    "listings_5d",
    "buyin_5d",
    "compliance_5d",
    "world_count_5d",
    "world_sent_5d",
    "lanka_count_5d",
    "lanka_sent_5d",
    "tape_count_5d",
    "tape_sent_5d",
    "company_count_5d",
    "company_sent_5d",
]
SYMBOL_FEATURES = [f"sym_{symbol.split('.')[0]}" for symbol in COMPANIES]
FEATURES = PRICE_FEATURES + NEWS_FEATURES + SYMBOL_FEATURES + CONTEXT_FEATURES


def _mentions(text: str, aliases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(alias in lowered for alias in aliases)


def _category_flags(category: str) -> dict[str, int]:
    text = category.lower()
    return {
        "is_listing": int("listing" in text),
        "is_buyin": int("buy-in" in text or "buy in" in text),
        "is_compliance": int("non-compliance" in text or "non compliance" in text),
    }


def load_prices() -> pd.DataFrame:
    """Yahoo holds the five-year history. Official CSE closes replace it where they exist."""
    frames = []
    if PRICE_PATH.exists():
        yahoo = pd.read_csv(PRICE_PATH)
        yahoo["priority"] = 0
        frames.append(yahoo)
    if SITE_PRICE_PATH.exists():
        site = pd.read_csv(SITE_PRICE_PATH)
        site["priority"] = 1
        if "open" not in site.columns:
            site["open"] = site["close"]
        frames.append(site)
    if not frames:
        raise FileNotFoundError(f"Missing price history: {PRICE_PATH}")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "symbol", "close"])
    frame = frame[frame["symbol"].isin(COMPANIES)]
    frame = frame.sort_values(["symbol", "date", "priority"])
    frame = frame.drop_duplicates(["symbol", "date"], keep="last")
    frame["open"] = frame["open"].fillna(frame["close"])
    frame["high"] = frame["high"].fillna(frame["close"])
    frame["low"] = frame["low"].fillna(frame["close"])
    frame["volume"] = frame["volume"].fillna(0)
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def _event(when, text: str, *, world: int = 0, lanka: int = 0, tape: int = 0, **flags) -> dict:
    named = [symbol for symbol, aliases in COMPANIES.items() if _mentions(text, aliases)]
    return {
        "date": pd.Timestamp(when),
        "sentiment": score_text(text),
        "is_world": world,
        "is_lanka": lanka,
        "is_tape": tape,
        "symbols": named,
        "is_listing": flags.get("is_listing", 0),
        "is_buyin": flags.get("is_buyin", 0),
        "is_compliance": flags.get("is_compliance", 0),
    }


def load_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return dated news events, and world headlines that have no usable date."""
    events: list[dict] = []
    undated: list[dict] = []

    if ANNOUNCEMENTS_PATH.exists():
        announcements = pd.read_csv(ANNOUNCEMENTS_PATH)
        for row in announcements.itertuples(index=False):
            category = str(getattr(row, "announcementCategory", "") or getattr(row, "_category", "") or "")
            remarks = str(getattr(row, "remarks", "") or "")
            company = str(getattr(row, "company", "") or "")
            text = " ".join(part for part in (company, category, remarks) if part and part != "nan")
            when = extract_date(getattr(row, "createdDate", None)) or extract_date(getattr(row, "_parsed_date", None))
            if when is None or not text.strip():
                continue
            flags = _category_flags(f"{category} {getattr(row, '_category', '')}")
            events.append(_event(when, text, **flags))

    if WORLD_NEWS_PATH.exists():
        world = pd.read_csv(WORLD_NEWS_PATH)
        world = world[world["source_type"] == "news"]
        for row in world.itertuples(index=False):
            title = str(getattr(row, "title", "") or "")
            body = str(getattr(row, "text", "") or "")
            text = f"{title} {body}".strip()
            if len(text) < 20:
                continue
            when = extract_date(getattr(row, "published_date", None)) or extract_date(text)
            named = [symbol for symbol, aliases in COMPANIES.items() if _mentions(text, aliases)]
            item = {
                "date": None if when is None else pd.Timestamp(when),
                "sentiment": score_text(text),
                "title": title,
                "source": str(getattr(row, "source", "") or ""),
                "symbols": named,
            }
            if when is None:
                undated.append(item)
                continue
            events.append(_event(when, text, world=1))

    if ARCHIVE_PATH.exists():
        archive = pd.read_csv(ARCHIVE_PATH)
        for row in archive.itertuples(index=False):
            title = str(getattr(row, "title", "") or "")
            if len(title) < 12:
                continue
            when = extract_date(getattr(row, "date", None)) or extract_date(getattr(row, "seen_at", None))
            if when is None:
                continue
            region = str(getattr(row, "region", "") or "")
            if region == "world":
                events.append(_event(when, title, world=1))
            elif region == "sri_lanka":
                events.append(_event(when, title, lanka=1))
            elif region == "market":
                events.append(_event(when, title, tape=1))

    columns = ["date", "sentiment", "is_world", "is_lanka", "is_tape", "symbols", "is_listing", "is_buyin", "is_compliance"]
    event_frame = pd.DataFrame(events, columns=columns) if events else pd.DataFrame(columns=columns)
    undated_frame = pd.DataFrame(undated)
    return event_frame, undated_frame


def _rolling_news(daily: pd.DataFrame, trade_dates: pd.DatetimeIndex, prefix: str) -> pd.DataFrame:
    aligned = daily.reindex(trade_dates).fillna(0.0)
    count_5d = aligned["count"].rolling(5, min_periods=1).sum()
    sent_sum_5d = aligned["sent_sum"].rolling(5, min_periods=1).sum()
    out = pd.DataFrame(index=trade_dates)
    out[f"{prefix}_count_5d"] = count_5d
    out[f"{prefix}_sent_5d"] = np.where(count_5d > 0, sent_sum_5d / count_5d, 0.0)
    return out


def news_feature_frames(events: pd.DataFrame, trade_dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Market features for every stock, plus company features keyed by symbol and date."""
    empty_market = pd.DataFrame(index=trade_dates)
    for column in NEWS_FEATURES:
        if column.startswith("company_"):
            continue
        empty_market[column] = 0.0
    empty_company = pd.DataFrame(columns=["symbol", "date", "company_count_5d", "company_sent_5d"])
    if events.empty:
        return empty_market, empty_company

    unnamed = events["symbols"].map(lambda symbols: len(symbols) == 0)
    announcements = events[(events["is_world"] == 0) & (events["is_lanka"] == 0) & (events["is_tape"] == 0)]
    world = events[(events["is_world"] == 1) & unnamed]
    lanka = events[(events["is_lanka"] == 1) & unnamed]
    tape = events[(events["is_tape"] == 1) & unnamed]

    def _bucket(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["count", "sent_sum"])
        grouped = frame.groupby("date").agg(count=("sentiment", "size"), sent_sum=("sentiment", "sum"))
        return grouped

    market = _rolling_news(_bucket(announcements), trade_dates, "ann")
    market = market.rename(columns={"ann_count_5d": "ann_count_5d", "ann_sent_5d": "ann_sent_5d"})
    extra = announcements.groupby("date")[["is_listing", "is_buyin", "is_compliance"]].sum() if not announcements.empty else pd.DataFrame()
    extra = extra.reindex(trade_dates).fillna(0.0)
    market["listings_5d"] = extra["is_listing"].rolling(5, min_periods=1).sum() if "is_listing" in extra else 0.0
    market["buyin_5d"] = extra["is_buyin"].rolling(5, min_periods=1).sum() if "is_buyin" in extra else 0.0
    market["compliance_5d"] = extra["is_compliance"].rolling(5, min_periods=1).sum() if "is_compliance" in extra else 0.0
    market = market.join(_rolling_news(_bucket(world), trade_dates, "world"))
    market = market.join(_rolling_news(_bucket(lanka), trade_dates, "lanka"))
    market = market.join(_rolling_news(_bucket(tape), trade_dates, "tape"))

    company_rows = []
    named = events[events["symbols"].map(lambda symbols: len(symbols) > 0)]
    for symbol in COMPANIES:
        matched = named[named["symbols"].map(lambda symbols, symbol=symbol: symbol in symbols)]
        rolled = _rolling_news(_bucket(matched), trade_dates, "company")
        rolled = rolled.reset_index(names="date")
        rolled["symbol"] = symbol
        company_rows.append(rolled)
    company = pd.concat(company_rows, ignore_index=True)
    return market, company


def add_price_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frames = []
    dropped_spikes = 0
    for symbol, frame in prices.groupby("symbol", sort=False):
        frame = frame.sort_values("date").copy()
        frame["ret_1"] = frame["close"].pct_change()
        spike = frame["ret_1"].abs().gt(MAX_ABS_RETURN)
        dropped_spikes += int(spike.sum())
        frame.loc[spike, "ret_1"] = np.nan
        frame["ret_5"] = frame["close"].pct_change(5)
        frame["ret_10"] = frame["close"].pct_change(10)
        frame["ret_20"] = frame["close"].pct_change(20)
        frame.loc[frame["ret_5"].abs().gt(MAX_ABS_RETURN), "ret_5"] = np.nan
        frame.loc[frame["ret_10"].abs().gt(MAX_ABS_RETURN), "ret_10"] = np.nan
        frame.loc[frame["ret_20"].abs().gt(MAX_ABS_RETURN), "ret_20"] = np.nan
        frame["vol_10"] = frame["ret_1"].rolling(10).std()
        frame["vol_20"] = frame["ret_1"].rolling(20).std()
        average_volume = frame["volume"].rolling(20).mean()
        frame["volume_ratio"] = frame["volume"] / average_volume.replace(0, np.nan)
        ma_10 = frame["close"].rolling(10).mean()
        ma_20 = frame["close"].rolling(20).mean()
        frame["dist_ma_10"] = frame["close"] / ma_10.replace(0, np.nan) - 1
        frame["dist_ma_20"] = frame["close"] / ma_20.replace(0, np.nan) - 1
        frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)
        frame = add_trailing_range(frame)
        delta = frame["close"].diff()
        gains = delta.clip(lower=0).rolling(14).mean()
        losses = (-delta.clip(upper=0)).rolling(14).mean()
        strength = gains / losses.replace(0, np.nan)
        frame["rsi_14"] = 100 - (100 / (1 + strength))
        frame["target_ret"] = frame["close"].shift(-1) / frame["close"] - 1
        frame.loc[frame["target_ret"].abs().gt(MAX_ABS_RETURN), "target_ret"] = np.nan
        frame["symbol_name"] = frame["company"]
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    market_sum = panel.groupby("date")["ret_1"].transform(lambda series: series.sum(min_count=1))
    market_count = panel.groupby("date")["ret_1"].transform("count")
    others = market_count - panel["ret_1"].notna().astype(int)
    panel["mkt_ret_1"] = (market_sum - panel["ret_1"].fillna(0)) / others.replace(0, np.nan)
    for symbol in COMPANIES:
        panel[f"sym_{symbol.split('.')[0]}"] = (panel["symbol"] == symbol).astype(int)
    return panel, dropped_spikes


def add_trailing_range(frame: pd.DataFrame) -> pd.DataFrame:
    """Distance from the trailing high and low, using only closes already known."""
    high = frame["close"].rolling(252, min_periods=20).max()
    low = frame["close"].rolling(252, min_periods=20).min()
    frame["dist_high_252"] = frame["close"] / high.replace(0, np.nan) - 1
    frame["dist_low_252"] = frame["close"] / low.replace(0, np.nan) - 1
    return frame


def _known_date(value) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return pd.Timestamp(parsed).tz_localize(None).normalize()
    if number > 10_000_000_000:
        number = number / 1000
    if number < 1_000_000_000:
        return None
    when = datetime.fromtimestamp(number, tz=ZoneInfo("Asia/Colombo")).date()
    return pd.Timestamp(when)


def _report_events() -> pd.DataFrame:
    rows = []
    if not FINANCIAL_DIR.exists():
        return pd.DataFrame(columns=["symbol", "date"])
    for path in FINANCIAL_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("infoAnnualData", "infoQuarterlyData", "infoOtherData"):
            for item in payload.get(key) or []:
                if not isinstance(item, dict):
                    continue
                when = _known_date(item.get("uploadedDate") or item.get("authorizedDate") or item.get("manualDate"))
                if when is not None:
                    rows.append({"symbol": path.stem, "date": when})
    if not rows:
        return pd.DataFrame(columns=["symbol", "date"])
    return pd.DataFrame(rows)


def _report_counts(panel: pd.DataFrame) -> tuple[pd.Series, int]:
    counts = pd.Series(0.0, index=panel.index)
    events = _report_events()
    if events.empty:
        return counts, 0
    trade_dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    used = 0
    for symbol, group in events.groupby("symbol"):
        known = []
        first_session = trade_dates[0]
        for when in group["date"]:
            stamp = pd.Timestamp(when)
            if stamp < first_session:
                continue
            position = int(trade_dates.searchsorted(stamp, side="left"))
            if position < len(trade_dates):
                known.append(trade_dates[position])
                used += 1
        if not known:
            continue
        daily = pd.Series(1.0, index=pd.DatetimeIndex(known))
        daily = daily.groupby(level=0).sum().reindex(trade_dates).fillna(0.0)
        rolled = daily.rolling(5, min_periods=1).sum()
        mask = panel["symbol"].eq(symbol)
        counts.loc[mask] = panel.loc[mask, "date"].map(rolled).fillna(0.0).to_numpy()
    return counts, used


def _aspi_returns() -> pd.DataFrame:
    empty = pd.DataFrame(columns=["date", "aspi_ret_1", "aspi_ret_5"])
    if not ASPI_PATH.exists():
        return empty
    aspi = pd.read_csv(ASPI_PATH)
    if "date" not in aspi.columns or "value" not in aspi.columns:
        return empty
    aspi["date"] = pd.to_datetime(aspi["date"], errors="coerce").dt.normalize()
    aspi["value"] = pd.to_numeric(aspi["value"], errors="coerce")
    aspi = aspi.dropna(subset=["date", "value"]).sort_values("date")
    aspi = aspi.drop_duplicates("date", keep="last")
    aspi["aspi_ret_1"] = aspi["value"].pct_change()
    aspi["aspi_ret_5"] = aspi["value"].pct_change(5)
    for column in ("aspi_ret_1", "aspi_ret_5"):
        aspi.loc[aspi[column].abs().gt(MAX_ABS_RETURN), column] = np.nan
    return aspi[["date", "aspi_ret_1", "aspi_ret_5"]]


def attach_api_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add CSE index returns and report counts that were known on that session."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    aspi = _aspi_returns()
    if aspi.empty:
        panel["aspi_ret_1"] = np.nan
        panel["aspi_ret_5"] = np.nan
        aspi_sessions = 0
    else:
        panel = panel.merge(aspi, on="date", how="left")
        aspi_sessions = int(panel["aspi_ret_1"].notna().groupby(panel["date"]).any().sum()) if "date" in panel else 0
    panel["index_ret_1"] = panel["aspi_ret_1"].fillna(panel["mkt_ret_1"])
    daily = panel.groupby("date")["ret_1"].mean().sort_index()
    compound = (1 + daily.fillna(0)).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1
    panel["index_ret_5"] = panel["aspi_ret_5"].fillna(panel["date"].map(compound))
    report_counts, report_events = _report_counts(panel)
    panel["report_count_5d"] = report_counts
    for column in CONTEXT_FEATURES:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    stats = {
        "aspi_sessions": aspi_sessions,
        "report_events": report_events,
    }
    return panel, stats


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    prices = load_prices()
    events, undated = load_events()
    panel, dropped_spikes = add_price_features(prices)
    trade_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    market, company = news_feature_frames(events, trade_dates)
    panel = panel.merge(market, left_on="date", right_index=True, how="left")
    panel = panel.merge(company, on=["symbol", "date"], how="left")
    panel[NEWS_FEATURES] = panel[NEWS_FEATURES].fillna(0.0)
    panel, api_stats = attach_api_features(panel)
    stats = {
        "price_rows": int(len(prices)),
        "price_start": prices["date"].min().date().isoformat(),
        "price_end": prices["date"].max().date().isoformat(),
        "events": int(len(events)),
        "world_events": int((events["is_world"] == 1).sum()) if not events.empty else 0,
        "lanka_events": int((events["is_lanka"] == 1).sum()) if not events.empty else 0,
        "tape_events": int((events["is_tape"] == 1).sum()) if not events.empty else 0,
        "announcements": int(((events["is_world"] == 0) & (events["is_lanka"] == 0) & (events["is_tape"] == 0)).sum()) if not events.empty else 0,
        "company_events": int(events["symbols"].map(len).gt(0).sum()) if not events.empty else 0,
        "undated_world": int(len(undated)),
        "dropped_spikes": dropped_spikes,
        "aspi_sessions": api_stats["aspi_sessions"],
        "report_events": api_stats["report_events"],
    }
    return panel, undated, stats


def _fit_models(train: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingClassifier]:
    features = train[FEATURES]
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    regressor.fit(features, train["target_ret"])
    classifier.fit(features, (train["target_ret"] > 0).astype(int))
    return regressor, classifier


def _direction_label(probability: float) -> str:
    if probability >= 0.55:
        return "up"
    if probability <= 0.45:
        return "down"
    return "unclear"


def _latest_rows(panel: pd.DataFrame) -> pd.DataFrame:
    usable = panel.dropna(subset=PRICE_FEATURES).sort_values("date")
    return usable.groupby("symbol", as_index=False).tail(1)


def write_forecast(panel: pd.DataFrame, regressor, classifier) -> pd.DataFrame:
    latest = _latest_rows(panel)
    predicted = regressor.predict(latest[FEATURES])
    probability = classifier.predict_proba(latest[FEATURES])[:, 1]
    forecast = pd.DataFrame(
        {
            "symbol": latest["symbol"].values,
            "company": latest["symbol_name"].values,
            "as_of": latest["date"].dt.date.astype(str).values,
            "last_close": latest["close"].round(2).values,
            "predicted_change_pct": np.round(predicted * 100, 3),
            "predicted_next_close": np.round(latest["close"].values * (1 + predicted), 2),
            "direction": [_direction_label(value) for value in probability],
            "up_probability": np.round(probability, 3),
            "ann_sent_5d": latest["ann_sent_5d"].round(3).values,
            "world_sent_5d": latest["world_sent_5d"].round(3).values,
            "company_sent_5d": latest["company_sent_5d"].round(3).values,
        }
    )
    FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(FORECAST_PATH, index=False)
    return forecast


def evaluate(train: pd.DataFrame, test: pd.DataFrame, regressor, classifier) -> dict:
    predicted = regressor.predict(test[FEATURES])
    probability = classifier.predict_proba(test[FEATURES])[:, 1]
    direction = (probability >= 0.5).astype(int)
    actual_direction = (test["target_ret"] > 0).astype(int)
    momentum = (test["ret_1"] > 0).astype(int)
    majority = int(actual_direction.mean() >= 0.5)
    importances = permutation_importance(
        classifier,
        test[FEATURES],
        actual_direction,
        n_repeats=5,
        random_state=42,
        scoring="accuracy",
    )
    order = np.argsort(importances.importances_mean)[::-1]
    ranked = [(FEATURES[index], float(importances.importances_mean[index])) for index in order[:8]]
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": train["date"].min().date().isoformat(),
        "train_end": train["date"].max().date().isoformat(),
        "test_start": test["date"].min().date().isoformat(),
        "test_end": test["date"].max().date().isoformat(),
        "mae": float(mean_absolute_error(test["target_ret"], predicted) * 100),
        "baseline_mae": float(mean_absolute_error(test["target_ret"], np.zeros(len(test))) * 100),
        "direction_accuracy": float(accuracy_score(actual_direction, direction)),
        "momentum_accuracy": float(accuracy_score(actual_direction, momentum)),
        "majority_accuracy": float(accuracy_score(actual_direction, np.full(len(test), majority))),
        "importances": ranked,
    }


def write_report(stats: dict, metrics: dict, forecast: pd.DataFrame, undated: pd.DataFrame) -> None:
    lines = [
        "CSE next-session price model",
        "",
        "This model replaces the old sentiment classifier and the single-stock random forest.",
        "It predicts the percent change from one close to the next close.",
        "The test period is the later part of the price history. Those days were not used for training.",
        "",
        f"Prices: {stats['price_rows']} rows, {stats['price_start']} to {stats['price_end']}",
        f"Dated announcements used: {stats['announcements']}",
        f"Dated world headlines used: {stats['world_events']}",
        f"Dated Sri Lanka headlines used: {stats['lanka_events']}",
        f"Dated stock-market headlines used: {stats['tape_events']}",
        f"Headlines that name one of the five stocks: {stats['company_events']}",
        f"World headlines with no date, kept out of training: {stats['undated_world']}",
        f"Sessions with an official ASPI return: {stats['aspi_sessions']}",
        f"Financial reports placed on a later session: {stats['report_events']}",
        "Where the ASPI chart has no row, the index return uses the other stocks' move that day.",
        "Beta, foreign holding, and today's market ratios are saved with each download. They are not copied onto older sessions.",
        f"Split or bad-print days removed: {stats['dropped_spikes']}",
        "",
        f"Train: {metrics['train_rows']} rows, {metrics['train_start']} to {metrics['train_end']}",
        f"Test:  {metrics['test_rows']} rows, {metrics['test_start']} to {metrics['test_end']}",
        "",
        f"Average miss on the test period: {metrics['mae']:.3f} percentage points",
        f"Average miss if the model always says 0: {metrics['baseline_mae']:.3f} percentage points",
        f"Direction accuracy: {metrics['direction_accuracy']:.1%}",
        f"Direction accuracy of 'same way as yesterday': {metrics['momentum_accuracy']:.1%}",
        f"Direction accuracy of always guessing the common side: {metrics['majority_accuracy']:.1%}",
        "",
        (
            "The size of the predicted move beat a no-change forecast."
            if metrics["mae"] + 0.01 < metrics["baseline_mae"]
            else "The size of the predicted move was less accurate than assuming the price does not change."
        ),
        (
            "The direction call beat always guessing the more common side."
            if metrics["direction_accuracy"] >= metrics["majority_accuracy"] + 0.02
            else "The direction call did not clearly beat always guessing the more common side."
        ),
        "A one-day move is mostly noise. Judge the model by the test numbers above, not by the latest call.",
        "",
        "Features that mattered most on the test period:",
    ]
    for name, value in metrics["importances"]:
        lines.append(f"  {name}: {value:.4f}")
    lines.extend(
        [
            "",
            "Next session after the last saved close.",
            "",
        ]
    )
    for row in forecast.itertuples(index=False):
        lines.append(
            f"{row.company}: last close {row.last_close:.2f} on {row.as_of}, "
            f"next session {row.direction} ({row.predicted_change_pct:+.3f}%, "
            f"up probability {row.up_probability:.0%})"
        )
    if not undated.empty:
        lines.extend(["", "Latest undated business headlines and their tone:"])
        sample = undated.assign(abs_score=undated["sentiment"].abs()).sort_values("abs_score", ascending=False).head(8)
        for row in sample.itertuples(index=False):
            tone = "positive" if row.sentiment > 0.2 else "negative" if row.sentiment < -0.2 else "neutral"
            lines.append(f"  [{tone}] {row.source}: {row.title}")
    lines.extend(
        [
            "",
            "A next-day stock move is mostly noise. Use the test accuracy above, not the headline forecast, to judge the model.",
            "This is a research model, not a trading instruction.",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    panel, undated, stats = build_panel()
    trainable = panel.dropna(subset=FEATURES + ["target_ret"]).copy()
    if len(trainable) < 200:
        raise RuntimeError(f"Only {len(trainable)} usable rows. That is not enough to train.")
    dates = np.sort(trainable["date"].unique())
    cutoff = dates[int(len(dates) * (1 - TEST_FRACTION))]
    train = trainable[trainable["date"] < cutoff]
    test = trainable[trainable["date"] >= cutoff]
    regressor, classifier = _fit_models(train)
    metrics = evaluate(train, test, regressor, classifier)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "regressor": regressor,
            "classifier": classifier,
            "features": FEATURES,
            "companies": COMPANIES,
            "trained_through": metrics["train_end"],
        },
        MODEL_PATH,
    )
    forecast = write_forecast(panel, regressor, classifier)
    write_report(stats, metrics, forecast, undated)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Test direction accuracy: {metrics['direction_accuracy']:.1%}")
    print(f"Test average miss: {metrics['mae']:.3f} percentage points")
    print(f"Forecast written to {FORECAST_PATH}")
    print(f"Report written to {REPORT_PATH}")
    print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
