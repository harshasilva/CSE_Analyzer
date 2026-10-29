"""Predict the next five CSE sessions for liquid listed stocks.

Prices are the official daily charts in data/cse_market_all.csv, plus any
older closes kept in data/cse_market_all_archive.csv. A headline is used
only when it names a listed company, and its tone comes from FinBERT.
The model sees information available at the close, then predicts the close
five sessions later.

A percent is published only when the size model and the up/down model agree
and the chance of a rise is at least 60% or at most 40%. Other stocks stay
unclear, with no future price. Training and the forecast both skip a session
that closed below 5 or traded fewer than 1000 shares.

Run after fetch_market_all.py:

    python train_market_all.py
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_absolute_error

from news_sentiment import extract_date
from train_news_price import (
    CONTEXT_FEATURES,
    MAX_ABS_RETURN,
    NEWS_FEATURES,
    PRICE_FEATURES,
    TEST_FRACTION,
    _category_flags,
    _rolling_news,
    add_trailing_range,
    attach_api_features,
)

ROOT = Path(__file__).resolve().parent
PRICE_PATH = ROOT / "data" / "cse_market_all.csv"
ARCHIVE_PATH = ROOT / "data" / "cse_market_all_archive.csv"
TODAY_PATH = ROOT / "data" / "cse_market_all_today.csv"
ANNOUNCEMENTS_PATH = ROOT / "cse_output" / "cse_news_ALL_2021_2026.csv"
WORLD_NEWS_PATH = ROOT / "cse_output" / "cse_combined_sources.csv"
ARCHIVE_NEWS_PATH = ROOT / "data" / "news_archive.csv"
SCORE_CACHE = ROOT / "data" / "finbert_company_scores.csv"
MODEL_PATH = ROOT / "models" / "cse_market_all_model.joblib"
REPORT_PATH = ROOT / "cse_output" / "cse_market_all_report.txt"
FORECAST_PATH = ROOT / "cse_output" / "cse_market_all_forecast.csv"

HORIZON = 5
HORIZON_MAX_ABS = 0.50
MIN_PRICE = 5.0
MIN_VOLUME = 1000.0
CALL_UP = 0.60
CALL_DOWN = 0.40
FINBERT_MODEL = "ProsusAI/finbert"
SCORER_USED = "FinBERT"
COMPANY_FEATURES = ["company_count_5d", "company_sent_5d"]
FEATURES = PRICE_FEATURES + COMPANY_FEATURES + CONTEXT_FEATURES
COUNT_FEATURES = [
    "ann_count_5d",
    "listings_5d",
    "buyin_5d",
    "compliance_5d",
    "world_count_5d",
    "lanka_count_5d",
    "tape_count_5d",
    "company_count_5d",
]
MIN_SESSIONS = 40
NAME_DROP = {"plc", "limited", "ltd", "company", "the", "of"}


def normalize_name(name: str) -> str:
    text = str(name or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(word for word in text.split() if word not in NAME_DROP)


def _read_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_prices() -> tuple[pd.DataFrame, int, int]:
    frames = []
    archive_rows = 0
    if ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size > 0:
        archived = _read_prices(ARCHIVE_PATH)
        archive_rows = int(len(archived))
        frames.append(archived)
    if PRICE_PATH.exists() and PRICE_PATH.stat().st_size > 0:
        frames.append(_read_prices(PRICE_PATH))
    if not frames:
        raise FileNotFoundError(f"Missing {PRICE_PATH}. Run python fetch_market_all.py first.")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["company"] = frame["company"].fillna("").astype(str)
    if TODAY_PATH.exists():
        board = pd.read_csv(TODAY_PATH, usecols=["symbol", "name"])
        official = board.drop_duplicates("symbol").set_index("symbol")["name"].astype(str)
        frame["company"] = frame["symbol"].map(official).fillna(frame["company"])
    frame = frame.dropna(subset=["date", "symbol", "close"])
    frame = frame[frame["close"] > 0]
    frame = frame.sort_values(["symbol", "date"])
    # The one-year file is read after the archive, so a fresh official bar wins.
    frame = frame.drop_duplicates(["symbol", "date"], keep="last")
    counts = frame.groupby("symbol").size()
    keep = counts[counts >= MIN_SESSIONS].index
    excluded = int((counts < MIN_SESSIONS).sum())
    frame = frame[frame["symbol"].isin(keep)].copy()
    frame["open"] = frame["open"].fillna(frame["close"])
    frame["high"] = frame["high"].fillna(frame["close"])
    frame["low"] = frame["low"].fillna(frame["close"])
    frame["volume"] = frame["volume"].fillna(0)
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True), excluded, archive_rows


def company_aliases(prices: pd.DataFrame) -> list[tuple[str, str]]:
    """Longer names are matched first so 'acl cables' is not lost inside a shorter key."""
    aliases = []
    seen: set[str] = set()
    for symbol, company in prices.groupby("symbol")["company"].first().items():
        key = normalize_name(company)
        if len(key) < 4 or key in seen:
            continue
        seen.add(key)
        aliases.append((key, str(symbol)))
    aliases.sort(key=lambda item: len(item[0]), reverse=True)
    return aliases


def match_symbols(text: str, aliases: list[tuple[str, str]]) -> list[str]:
    key = normalize_name(text)
    if len(key) < 4:
        return []
    found = []
    for alias, symbol in aliases:
        if alias in key and symbol not in found:
            found.append(symbol)
    return found


def linked_symbols(text: str, aliases: list[tuple[str, str]], known: set[str], explicit: str = "") -> list[str]:
    found = match_symbols(text, aliases)
    token = str(explicit or "").strip()
    if token.lower() == "nan":
        token = ""
    if token in known and token not in found:
        found.append(token)
    return found


def _event(when, text: str, symbols: list[str], sentiment: float, *, world: int = 0, lanka: int = 0, tape: int = 0, **flags) -> dict:
    return {
        "date": pd.Timestamp(when),
        "sentiment": sentiment,
        "is_world": world,
        "is_lanka": lanka,
        "is_tape": tape,
        "symbols": symbols,
        "is_listing": flags.get("is_listing", 0),
        "is_buyin": flags.get("is_buyin", 0),
        "is_compliance": flags.get("is_compliance", 0),
    }


def _text_key(text: str) -> str:
    return hashlib.sha1(" ".join(str(text).lower().split()).encode("utf-8")).hexdigest()


def _load_score_cache() -> dict[str, float]:
    if not SCORE_CACHE.exists() or SCORE_CACHE.stat().st_size == 0:
        return {}
    frame = pd.read_csv(SCORE_CACHE)
    if frame.empty or "key" not in frame.columns or "score" not in frame.columns:
        return {}
    if "scorer" not in frame.columns:
        return {}
    frame = frame[frame["scorer"].astype(str).eq(FINBERT_MODEL)]
    return dict(zip(frame["key"].astype(str), pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)))


def _save_score_cache(cache: dict[str, float]) -> None:
    frame = pd.DataFrame(
        {"key": list(cache), "score": [cache[key] for key in cache], "scorer": FINBERT_MODEL}
    )
    SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SCORE_CACHE, index=False)


def _prepare_finbert() -> None:
    """Load FinBERT on this PC's mismatched torch and torchvision install."""
    import enum
    import importlib.machinery
    import sys
    import types

    def stub(name: str):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        module.__path__ = []
        sys.modules[name] = module
        return module

    if "torchvision" not in sys.modules:
        vision = stub("torchvision")
        vision_io = stub("torchvision.io")
        transforms = stub("torchvision.transforms")
        functional = stub("torchvision.transforms.functional")

        class ImageReadMode(enum.IntEnum):
            UNCHANGED = 0
            GRAY = 1
            GRAY_ALPHA = 2
            RGB = 3
            RGB_ALPHA = 4

        class InterpolationMode(enum.Enum):
            NEAREST = "nearest"
            NEAREST_EXACT = "nearest-exact"
            BILINEAR = "bilinear"
            BICUBIC = "bicubic"
            BOX = "box"
            HAMMING = "hamming"
            LANCZOS = "lanczos"

        vision_io.ImageReadMode = ImageReadMode
        vision_io.decode_image = lambda *_args, **_kwargs: None
        transforms.InterpolationMode = InterpolationMode
        functional.pil_to_tensor = lambda *_args, **_kwargs: None
        transforms.functional = functional
        vision.io = vision_io
        vision.transforms = transforms

    import torch
    import transformers.utils.import_utils as import_utils

    import_utils.get_torch_version = lambda: torch.__version__.split("+")[0]
    import_utils.is_torch_greater_or_equal.cache_clear()
    import_utils.is_torch_less_or_equal.cache_clear()


def _signed_sentiment(label: str, confidence: float) -> float:
    name = label.lower()
    if "positive" in name:
        return confidence
    if "negative" in name:
        return -confidence
    return 0.0


def score_company_texts(texts: list[str]) -> list[float]:
    """Score company headlines with FinBERT. Repeated headlines use the saved cache."""
    global SCORER_USED
    if not texts:
        return []
    keys = [_text_key(text) for text in texts]
    cache = _load_score_cache()
    missing: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, text in zip(keys, texts):
        if key not in cache and key not in seen:
            seen.add(key)
            missing.append((key, text))
    if missing:
        print(f"Scoring {len(missing)} company headlines with FinBERT...", flush=True)
        _prepare_finbert()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        model.eval()
        labels = {int(index): str(name) for index, name in model.config.id2label.items()}
        SCORER_USED = f"FinBERT ({FINBERT_MODEL})"
        batch_size = 32
        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            encoded = tokenizer(
                [text for _key, text in chunk],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                probabilities = torch.softmax(model(**encoded).logits, dim=-1)
            for row, (key, _text) in zip(probabilities, chunk):
                choice = int(torch.argmax(row).item())
                cache[key] = _signed_sentiment(labels.get(choice, ""), float(row[choice].item()))
            _save_score_cache(cache)
            print(f"  FinBERT {min(start + batch_size, len(missing))}/{len(missing)}", flush=True)
    else:
        SCORER_USED = f"FinBERT ({FINBERT_MODEL})"
        print(f"Using cached FinBERT scores for {len(set(keys))} company headlines.", flush=True)
    return [float(cache[key]) for key in keys]


def load_events(aliases: list[tuple[str, str]], known: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Keep a headline only when it names a listed company."""
    print("Reading headlines and keeping those that name a listed company...", flush=True)
    pending: list[dict] = []
    undated_items: list[dict] = []
    ignored = 0

    def consider(when, text: str, symbols: list[str], source: str = "", title: str = "", **flags) -> None:
        nonlocal ignored
        cleaned = " ".join(str(text or "").split())
        if len(cleaned) < 12:
            return
        if not symbols:
            ignored += 1
            return
        item = {
            "when": when,
            "text": cleaned,
            "symbols": symbols,
            "source": source,
            "title": title or cleaned[:180],
            "flags": flags,
        }
        if when is None:
            undated_items.append(item)
            return
        pending.append(item)

    if ANNOUNCEMENTS_PATH.exists():
        announcements = pd.read_csv(ANNOUNCEMENTS_PATH)
        for row in announcements.itertuples(index=False):
            category = str(getattr(row, "announcementCategory", "") or getattr(row, "_category", "") or "")
            remarks = str(getattr(row, "remarks", "") or "")
            company = str(getattr(row, "company", "") or getattr(row, "companyName", "") or "")
            text = " ".join(part for part in (company, category, remarks) if part and part != "nan")
            when = extract_date(getattr(row, "createdDate", None)) or extract_date(getattr(row, "_parsed_date", None))
            flags = _category_flags(f"{category} {getattr(row, '_category', '')}")
            consider(when, text, linked_symbols(text, aliases, known), source="CSE announcement", title=company or category, **flags)

    if WORLD_NEWS_PATH.exists():
        world = pd.read_csv(WORLD_NEWS_PATH)
        world = world[world["source_type"] == "news"]
        for row in world.itertuples(index=False):
            title = str(getattr(row, "title", "") or "")
            body = str(getattr(row, "text", "") or "")
            text = f"{title} {body}".strip()
            when = extract_date(getattr(row, "published_date", None)) or extract_date(text)
            explicit = str(getattr(row, "symbol", "") or "")
            company = str(getattr(row, "company", "") or "")
            consider(
                when,
                f"{company} {text}",
                linked_symbols(f"{company} {text}", aliases, known, explicit),
                source=str(getattr(row, "source", "") or ""),
                title=title,
                world=1,
            )

    if ARCHIVE_NEWS_PATH.exists():
        archive = pd.read_csv(ARCHIVE_NEWS_PATH)
        for row in archive.itertuples(index=False):
            title = str(getattr(row, "title", "") or "")
            when = extract_date(getattr(row, "date", None)) or extract_date(getattr(row, "seen_at", None))
            region = str(getattr(row, "region", "") or "")
            flags = {"world": int(region == "world"), "lanka": int(region == "sri_lanka"), "tape": int(region == "market")}
            consider(
                when,
                title,
                linked_symbols(title, aliases, known),
                source=str(getattr(row, "source", "") or ""),
                title=title,
                **flags,
            )

    scored = score_company_texts([item["text"] for item in pending] + [item["text"] for item in undated_items])
    pending_scores = scored[: len(pending)]
    undated_scores = scored[len(pending) :]
    events: list[dict] = []
    for item, sentiment in zip(pending, pending_scores):
        flags = item["flags"]
        events.append(
            _event(
                item["when"],
                item["text"],
                item["symbols"],
                sentiment,
                world=flags.get("world", 0),
                lanka=flags.get("lanka", 0),
                tape=flags.get("tape", 0),
                is_listing=flags.get("is_listing", 0),
                is_buyin=flags.get("is_buyin", 0),
                is_compliance=flags.get("is_compliance", 0),
            )
        )
    undated_rows = [
        {"sentiment": sentiment, "title": item["title"], "source": item["source"]}
        for item, sentiment in zip(undated_items, undated_scores)
    ]
    columns = ["date", "sentiment", "is_world", "is_lanka", "is_tape", "symbols", "is_listing", "is_buyin", "is_compliance"]
    event_frame = pd.DataFrame(events, columns=columns) if events else pd.DataFrame(columns=columns)
    info = {
        "company_headlines": int(len(pending)),
        "ignored_headlines": int(ignored),
        "undated_company": int(len(undated_items)),
    }
    return event_frame, pd.DataFrame(undated_rows), info


def news_feature_frames(events: pd.DataFrame, trade_dates: pd.DatetimeIndex, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    empty_market = pd.DataFrame(index=trade_dates)
    for column in NEWS_FEATURES:
        if column.startswith("company_"):
            continue
        empty_market[column] = 0.0
    empty_company = pd.DataFrame(columns=["symbol", "date", "company_count_5d", "company_sent_5d"])
    if events.empty:
        return empty_market, empty_company

    unnamed = events["symbols"].map(lambda names: len(names) == 0)
    announcements = events[(events["is_world"] == 0) & (events["is_lanka"] == 0) & (events["is_tape"] == 0)]
    world = events[(events["is_world"] == 1) & unnamed]
    lanka = events[(events["is_lanka"] == 1) & unnamed]
    tape = events[(events["is_tape"] == 1) & unnamed]

    def _bucket(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["count", "sent_sum"])
        return frame.groupby("date").agg(count=("sentiment", "size"), sent_sum=("sentiment", "sum"))

    market = _rolling_news(_bucket(announcements), trade_dates, "ann")
    extra = announcements.groupby("date")[["is_listing", "is_buyin", "is_compliance"]].sum() if not announcements.empty else pd.DataFrame()
    extra = extra.reindex(trade_dates).fillna(0.0)
    market["listings_5d"] = extra["is_listing"].rolling(5, min_periods=1).sum() if "is_listing" in extra else 0.0
    market["buyin_5d"] = extra["is_buyin"].rolling(5, min_periods=1).sum() if "is_buyin" in extra else 0.0
    market["compliance_5d"] = extra["is_compliance"].rolling(5, min_periods=1).sum() if "is_compliance" in extra else 0.0
    market = market.join(_rolling_news(_bucket(world), trade_dates, "world"))
    market = market.join(_rolling_news(_bucket(lanka), trade_dates, "lanka"))
    market = market.join(_rolling_news(_bucket(tape), trade_dates, "tape"))

    company_rows = []
    named = events[events["symbols"].map(lambda names: len(names) > 0)]
    mentioned = set()
    if not named.empty:
        for names in named["symbols"]:
            mentioned.update(names)
    for symbol in symbols:
        if symbol not in mentioned:
            continue
        matched = named[named["symbols"].map(lambda names, symbol=symbol: symbol in names)]
        rolled = _rolling_news(_bucket(matched), trade_dates, "company")
        rolled = rolled.reset_index(names="date")
        rolled["symbol"] = symbol
        company_rows.append(rolled)
    company = pd.concat(company_rows, ignore_index=True) if company_rows else empty_company
    return market, company


def add_price_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frames = []
    dropped_spikes = 0
    for _symbol, frame in prices.groupby("symbol", sort=False):
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
        frame["target_ret"] = frame["close"].shift(-HORIZON) / frame["close"] - 1
        frame.loc[frame["target_ret"].abs().gt(HORIZON_MAX_ABS), "target_ret"] = np.nan
        frame["symbol_name"] = frame["company"]
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    market_sum = panel.groupby("date")["ret_1"].transform(lambda series: series.sum(min_count=1))
    market_count = panel.groupby("date")["ret_1"].transform("count")
    others = market_count - panel["ret_1"].notna().astype(int)
    panel["mkt_ret_1"] = (market_sum - panel["ret_1"].fillna(0)) / others.replace(0, np.nan)
    return panel, dropped_spikes


def stabilize_news_counts(panel: pd.DataFrame) -> pd.DataFrame:
    """Compare headline volume with the previous 60 sessions.

    The saved archive gets denser over time. A raw count would treat that
    collection change as a market shock. A ratio stays near 1 when coverage
    is simply higher, and rises only when the latest week is busier than
    the recent pace.
    """
    market_counts = [column for column in COUNT_FEATURES if column != "company_count_5d" and column in panel.columns]
    for column in market_counts:
        daily = panel.groupby("date")[column].first().sort_index()
        baseline = daily.rolling(60, min_periods=20).median().shift(1)
        ratio = daily / baseline.where(baseline > 0)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 4.0)
        panel[column] = panel["date"].map(ratio)
    if "company_count_5d" in panel.columns:
        panel["company_count_5d"] = np.log1p(panel["company_count_5d"].clip(lower=0))
    return panel


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    prices, excluded, archive_rows = load_prices()
    aliases = company_aliases(prices)
    known = set(prices["symbol"].astype(str).unique())
    events, undated, news_info = load_events(aliases, known)
    panel, dropped_spikes = add_price_features(prices)
    trade_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    market, company = news_feature_frames(events, trade_dates, list(prices["symbol"].unique()))
    panel = panel.merge(market, left_on="date", right_index=True, how="left")
    panel = panel.merge(company, on=["symbol", "date"], how="left")
    panel[NEWS_FEATURES] = panel[NEWS_FEATURES].fillna(0.0)
    panel, api_stats = attach_api_features(panel)
    panel = stabilize_news_counts(panel)
    panel["liquid"] = (panel["close"] >= MIN_PRICE) & (panel["volume"].fillna(0) >= MIN_VOLUME)
    stats = {
        "price_rows": int(len(prices)),
        "symbols": int(prices["symbol"].nunique()),
        "excluded_symbols": excluded,
        "price_start": prices["date"].min().date().isoformat(),
        "price_end": prices["date"].max().date().isoformat(),
        "archive_rows": archive_rows,
        "liquid_rows": int(panel["liquid"].sum()),
        "liquid_symbols": int(panel.loc[panel["liquid"], "symbol"].nunique()),
        "thin_rows": int((~panel["liquid"]).sum()),
        "company_headlines": news_info["company_headlines"],
        "ignored_headlines": news_info["ignored_headlines"],
        "undated_company": news_info["undated_company"],
        "company_events": int(events["symbols"].map(len).gt(0).sum()) if not events.empty else 0,
        "dropped_spikes": dropped_spikes,
        "aspi_sessions": api_stats["aspi_sessions"],
        "report_events": api_stats["report_events"],
    }
    return panel, undated, stats


def _fit_models(train: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingClassifier]:
    features = train[FEATURES]
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=1.0,
        random_state=42,
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=1.0,
        random_state=42,
    )
    regressor.fit(features, train["target_ret"])
    classifier.fit(features, (train["target_ret"] > 0).astype(int))
    return regressor, classifier


def _latest_rows(panel: pd.DataFrame) -> pd.DataFrame:
    usable = panel.dropna(subset=PRICE_FEATURES).sort_values("date")
    return usable.groupby("symbol", as_index=False).tail(1)


def _call_direction(change: float, probability: float) -> str:
    if change > 0 and probability >= CALL_UP:
        return "up"
    if change < 0 and probability <= CALL_DOWN:
        return "down"
    return "unclear"


def _published_mask(predicted: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return ((predicted > 0) & (probability >= CALL_UP)) | ((predicted < 0) & (probability <= CALL_DOWN))


def write_forecast(panel: pd.DataFrame, regressor, classifier, band_pct: float) -> pd.DataFrame:
    latest = _latest_rows(panel)
    session = latest["date"].max()
    latest = latest[latest["date"].eq(session) & latest["liquid"]].copy()
    predicted = regressor.predict(latest[FEATURES])
    probability = classifier.predict_proba(latest[FEATURES])[:, 1]
    directions = [_call_direction(float(change), float(chance)) for change, chance in zip(predicted, probability)]
    published = np.array([direction != "unclear" for direction in directions])
    change_pct = np.round(predicted * 100, 3)
    low_pct = np.round(change_pct - band_pct, 3)
    high_pct = np.round(change_pct + band_pct, 3)
    close = latest["close"].to_numpy()
    next_close = np.round(close * (1 + predicted), 4)
    close_low = np.round(close * (1 + (change_pct - band_pct) / 100), 4)
    close_high = np.round(close * (1 + (change_pct + band_pct) / 100), 4)
    forecast = pd.DataFrame(
        {
            "symbol": latest["symbol"].values,
            "company": latest["symbol_name"].values,
            "as_of": latest["date"].dt.date.astype(str).values,
            "last_close": np.round(close, 4),
            "volume": latest["volume"].round(0).values,
            "horizon_sessions": HORIZON,
            "direction": directions,
            "up_probability": np.round(probability, 3),
            "predicted_change_pct": np.where(published, change_pct, np.nan),
            "predicted_change_low_pct": np.where(published, low_pct, np.nan),
            "predicted_change_high_pct": np.where(published, high_pct, np.nan),
            "predicted_close": np.where(published, next_close, np.nan),
            "predicted_close_low": np.where(published, close_low, np.nan),
            "predicted_close_high": np.where(published, close_high, np.nan),
            "company_sent_5d": latest["company_sent_5d"].round(3).values,
        }
    )
    up = forecast[forecast["direction"].eq("up")].sort_values(["predicted_change_pct", "symbol"], ascending=[False, True])
    down = forecast[forecast["direction"].eq("down")].sort_values(["predicted_change_pct", "symbol"], ascending=[True, True])
    unclear = forecast[forecast["direction"].eq("unclear")].sort_values("symbol")
    forecast = pd.concat([up, down, unclear], ignore_index=True)
    FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(FORECAST_PATH, index=False)
    return forecast


def _subset_scores(actual: np.ndarray, predicted: np.ndarray, probability: np.ndarray) -> dict:
    if len(actual) == 0:
        return {
            "rows": 0,
            "mae": float("nan"),
            "baseline_mae": float("nan"),
            "direction_accuracy": float("nan"),
            "majority_accuracy": float("nan"),
        }
    actual_direction = (actual > 0).astype(int)
    direction = (probability >= 0.5).astype(int)
    majority = int(actual_direction.mean() >= 0.5)
    return {
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted) * 100),
        "baseline_mae": float(mean_absolute_error(actual, np.zeros(len(actual))) * 100),
        "direction_accuracy": float(accuracy_score(actual_direction, direction)),
        "majority_accuracy": float(accuracy_score(actual_direction, np.full(len(actual), majority))),
    }


def evaluate(train: pd.DataFrame, test: pd.DataFrame, regressor, classifier) -> dict:
    predicted = regressor.predict(test[FEATURES])
    probability = classifier.predict_proba(test[FEATURES])[:, 1]
    actual = test["target_ret"].to_numpy()
    actual_direction = (actual > 0).astype(int)
    momentum = (test["ret_1"].to_numpy() > 0).astype(int)
    full = _subset_scores(actual, predicted, probability)
    mask = _published_mask(predicted, probability)
    published = _subset_scores(actual[mask], predicted[mask], probability[mask])
    band_from_published = published["rows"] >= 30 and np.isfinite(published["mae"])
    band_pct = float(published["mae"] if band_from_published else full["mae"])
    sample = test
    if len(test) > 4000:
        sample = test.sample(4000, random_state=42)
    importances = permutation_importance(
        classifier,
        sample[FEATURES],
        (sample["target_ret"] > 0).astype(int),
        n_repeats=3,
        random_state=42,
        scoring="accuracy",
    )
    order = np.argsort(importances.importances_mean)[::-1]
    ranked = [(FEATURES[index], float(importances.importances_mean[index])) for index in order[:8]]
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_symbols": int(train["symbol"].nunique()),
        "test_symbols": int(test["symbol"].nunique()),
        "train_start": train["date"].min().date().isoformat(),
        "train_end": train["date"].max().date().isoformat(),
        "test_start": test["date"].min().date().isoformat(),
        "test_end": test["date"].max().date().isoformat(),
        "mae": full["mae"],
        "baseline_mae": full["baseline_mae"],
        "direction_accuracy": full["direction_accuracy"],
        "momentum_accuracy": float(accuracy_score(actual_direction, momentum)),
        "majority_accuracy": full["majority_accuracy"],
        "published_rows": published["rows"],
        "published_mae": published["mae"],
        "published_baseline_mae": published["baseline_mae"],
        "published_direction_accuracy": published["direction_accuracy"],
        "published_majority_accuracy": published["majority_accuracy"],
        "band_pct": band_pct,
        "band_from_published": band_from_published,
        "importances": ranked,
    }


def _fmt_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def _fmt_rate(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.1%}"


def _score_lines(label: str, mae: float, baseline: float, direction: float, majority: float) -> list[str]:
    lines = [
        f"{label}",
        f"Average miss: {_fmt_pct(mae)} percentage points",
        f"Average miss if the model always says 0: {_fmt_pct(baseline)} percentage points",
        f"Direction accuracy: {_fmt_rate(direction)}",
        f"Direction accuracy of always guessing the common side: {_fmt_rate(majority)}",
    ]
    if not np.isfinite(mae):
        lines.append("There were no rows in this group.")
        return lines
    if mae + 0.01 < baseline:
        lines.append("The size of the predicted move beat a no-change forecast.")
    else:
        lines.append("The size of the predicted move missed by at least as much as a no-change forecast.")
    if direction >= majority + 0.02:
        lines.append("The direction call beat always guessing the more common side.")
    else:
        lines.append("The direction call stayed within two points of always guessing the more common side, or trailed it.")
    return lines


def _call_lines(forecast: pd.DataFrame, direction: str) -> list[str]:
    calls = forecast[forecast["direction"].eq(direction)]
    if direction == "down":
        calls = calls.sort_values("predicted_change_pct", ascending=True)
    else:
        calls = calls.sort_values("predicted_change_pct", ascending=False)
    if calls.empty:
        return ["  None. No liquid stock cleared the agreement filter."]
    lines = []
    for row in calls.head(12).itertuples(index=False):
        lines.append(
            f"  {row.symbol} {row.company}: close {row.last_close:.2f} on {row.as_of}, "
            f"next {HORIZON} sessions {row.predicted_change_pct:+.3f}% "
            f"(range {row.predicted_change_low_pct:+.3f}% to {row.predicted_change_high_pct:+.3f}%, "
            f"up probability {row.up_probability:.0%})"
        )
    return lines


def write_report(stats: dict, metrics: dict, forecast: pd.DataFrame, undated: pd.DataFrame) -> None:
    published = int(forecast["direction"].ne("unclear").sum()) if not forecast.empty else 0
    band_source = "published test calls" if metrics["band_from_published"] else "all liquid test rows"
    lines = [
        "CSE full-market five-session model",
        "",
        f"Each row predicts the percent change from this close to the close {HORIZON} sessions later.",
        f"Training and the forecast use only sessions that closed at {MIN_PRICE:.0f} or above and traded at least {MIN_VOLUME:.0f} shares.",
        f"A percent is published only when that percent and the up/down model agree, and the chance of a rise is at least {CALL_UP:.0%} or at most {CALL_DOWN:.0%}.",
        "Other liquid stocks stay unclear, with no future price.",
        f"A published percent is shown with a band of ±{metrics['band_pct']:.3f} points, the average miss on {band_source}.",
        "The last four sessions have no completed five-session outcome yet, so they are not in the test score.",
        f"Headlines count only when they name a listed company. Tone comes from {SCORER_USED}.",
        "Official closes are kept in data/cse_market_all_archive.csv. A CSE chart download is still about one year; older days remain in the archive.",
        "The test period is the later part of the liquid history. Those days were not used for training.",
        "",
        f"Prices: {stats['price_rows']} rows, {stats['symbols']} symbols, {stats['price_start']} to {stats['price_end']}",
        f"Archive rows on disk: {stats['archive_rows']}",
        f"Symbols with fewer than {MIN_SESSIONS} sessions, left out: {stats['excluded_symbols']}",
        f"Liquid sessions: {stats['liquid_rows']} rows, {stats['liquid_symbols']} symbols",
        f"Thin sessions left out of training and the forecast: {stats['thin_rows']}",
        f"Company headlines scored: {stats['company_headlines']}",
        f"Headlines with no listed company, left out: {stats['ignored_headlines']}",
        f"Undated company headlines, kept out of training: {stats['undated_company']}",
        f"Split or bad-print days removed: {stats['dropped_spikes']}",
        "",
        f"Train: {metrics['train_rows']} rows, {metrics['train_symbols']} symbols, {metrics['train_start']} to {metrics['train_end']}",
        f"Test:  {metrics['test_rows']} rows, {metrics['test_symbols']} symbols, {metrics['test_start']} to {metrics['test_end']}",
        f"Published calls inside the test: {metrics['published_rows']} of {metrics['test_rows']}",
        "",
    ]
    lines.extend(
        _score_lines(
            "All liquid test rows:",
            metrics["mae"],
            metrics["baseline_mae"],
            metrics["direction_accuracy"],
            metrics["majority_accuracy"],
        )
    )
    lines.append(f"Direction accuracy of 'same way as yesterday': {metrics['momentum_accuracy']:.1%}")
    lines.append("")
    lines.extend(
        _score_lines(
            "Published calls only, unclear days left out:",
            metrics["published_mae"],
            metrics["published_baseline_mae"],
            metrics["published_direction_accuracy"],
            metrics["published_majority_accuracy"],
        )
    )
    lines.extend(
        [
            "",
            "Judge a published percent by the published-call numbers above.",
            "",
            "Features that mattered most on the test sample:",
        ]
    )
    for name, value in metrics["importances"]:
        lines.append(f"  {name}: {value:.4f}")
    lines.extend(["", f"Strongest up calls for the next {HORIZON} sessions:", ""])
    lines.extend(_call_lines(forecast, "up"))
    lines.extend(["", f"Strongest down calls for the next {HORIZON} sessions:", ""])
    lines.extend(_call_lines(forecast, "down"))
    lines.extend(
        [
            "",
            f"Full forecast for {len(forecast)} liquid symbols ({published} with a published percent): {FORECAST_PATH}",
            "This is a research model, not a trading instruction.",
        ]
    )
    if not undated.empty and "sentiment" in undated.columns:
        lines.extend(["", "Undated company headlines and their FinBERT tone:"])
        sample = undated.assign(abs_score=undated["sentiment"].abs()).sort_values("abs_score", ascending=False).head(8)
        for row in sample.itertuples(index=False):
            tone = "positive" if row.sentiment > 0.2 else "negative" if row.sentiment < -0.2 else "neutral"
            lines.append(f"  [{tone}] {row.source}: {row.title}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("Building the full-market panel...", flush=True)
    panel, undated, stats = build_panel()
    trainable = panel.loc[panel["liquid"]].dropna(subset=FEATURES + ["target_ret"]).copy()
    if len(trainable) < 500:
        raise RuntimeError(f"Only {len(trainable)} usable liquid rows. That is not enough to train.")
    dates = np.sort(trainable["date"].unique())
    cutoff = dates[int(len(dates) * (1 - TEST_FRACTION))]
    train = trainable[trainable["date"] < cutoff]
    test = trainable[trainable["date"] >= cutoff]
    print(f"Fitting on {len(train)} liquid rows, testing on {len(test)} liquid rows...", flush=True)
    regressor, classifier = _fit_models(train)
    metrics = evaluate(train, test, regressor, classifier)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "regressor": regressor,
            "classifier": classifier,
            "features": FEATURES,
            "horizon_sessions": HORIZON,
            "min_price": MIN_PRICE,
            "min_volume": MIN_VOLUME,
            "call_up": CALL_UP,
            "call_down": CALL_DOWN,
            "band_pct": metrics["band_pct"],
            "sentiment": FINBERT_MODEL,
            "count_transform": "company_log1p_market_ratio_unused",
            "count_features": COUNT_FEATURES,
            "symbols": int(stats["symbols"]),
            "trained_through": metrics["train_end"],
            "tested_through": metrics["test_end"],
        },
        MODEL_PATH,
    )
    forecast = write_forecast(panel, regressor, classifier, metrics["band_pct"])
    write_report(stats, metrics, forecast, undated)
    published = int(forecast["direction"].ne("unclear").sum())
    print(f"Saved model to {MODEL_PATH}")
    print(f"Symbols: {stats['symbols']}  rows: {stats['price_rows']}  {stats['price_start']} to {stats['price_end']}")
    print(f"Company headlines: {stats['company_headlines']}  ignored headlines: {stats['ignored_headlines']}")
    print(f"Test direction accuracy: {metrics['direction_accuracy']:.1%}")
    print(f"Test average miss: {metrics['mae']:.3f} percentage points")
    print(f"No-change miss: {metrics['baseline_mae']:.3f} percentage points")
    print(
        f"Published test calls: {metrics['published_rows']}  "
        f"miss {_fmt_pct(metrics['published_mae'])} vs no-change {_fmt_pct(metrics['published_baseline_mae'])}  "
        f"direction {_fmt_rate(metrics['published_direction_accuracy'])} vs common side {_fmt_rate(metrics['published_majority_accuracy'])}"
    )
    print(f"Forecast written to {FORECAST_PATH} ({published} published of {len(forecast)} liquid symbols)")
    print(f"Report written to {REPORT_PATH}")
    print("Strongest up calls:")
    print(forecast[forecast["direction"].eq("up")].head(8).to_string(index=False))
    print("Strongest down calls:")
    print(forecast[forecast["direction"].eq("down")].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
