"""Learn how the five CSE stocks moved after each news headline.

Headlines are world news, Sri Lanka news, or stock-market news. Each headline
is joined to the next Colombo session that could still react to it. A story
timestamped before 14:30 Colombo is matched to that session. A later story is
matched to the following session.

The five stocks are Commercial Bank, Sampath Bank, Hatton National Bank,
John Keells Holdings, and Dialog Axiata.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from news_sentiment import label_from_score, score_text

ROOT = Path(__file__).resolve().parent
YAHOO_PRICES = ROOT / "data" / "cse_prices_5y.csv"
SITE_PRICES = ROOT / "data" / "cse_site_prices_1y.csv"
ARCHIVE_PATH = ROOT / "data" / "news_archive.csv"
ANNOUNCEMENTS_PATH = ROOT / "cse_output" / "cse_news_ALL_2021_2026.csv"
MODEL_PATH = ROOT / "models" / "news_reaction_model.joblib"
REACTIONS_PATH = ROOT / "data" / "news_reactions.csv"
PATTERNS_PATH = ROOT / "cse_output" / "news_reaction_patterns.csv"
REPORT_PATH = ROOT / "cse_output" / "news_reaction_report.txt"
FORECAST_PATH = ROOT / "cse_output" / "news_reaction_forecast.csv"

COLOMBO = ZoneInfo("Asia/Colombo")
MARKET_CLOSE_MINUTES = 14 * 60 + 30
MAX_ABS_RETURN = 0.35

# Short aliases use a whole word. Longer aliases use a phrase.
COMPANIES = (
    ("COMB.N0000", "Commercial Bank", ("commercial bank", "combank")),
    ("SAMP.N0000", "Sampath Bank", ("sampath bank", "sampath")),
    ("HNB.N0000", "Hatton National Bank", ("hatton national", "hnb")),
    ("JKH.N0000", "John Keells Holdings", ("john keells", "keells")),
    ("DIAL.N0000", "Dialog Axiata", ("dialog axiata", "dialog")),
)
SYMBOLS = tuple(item[0] for item in COMPANIES)
COMPANY_NAMES = {symbol: name for symbol, name, _aliases in COMPANIES}
SYMBOL_CODES = {symbol: symbol.split(".")[0] for symbol in SYMBOLS}

REGIONS = ("world", "sri_lanka", "market")
REGION_LABELS = {
    "world": "World",
    "sri_lanka": "Sri Lanka",
    "market": "Stock market",
}

TOPICS = {
    "rates": ("interest rate", "policy rate", "federal reserve", "central bank", "monetary", "rate hike", "rate cut"),
    "inflation": ("inflation", "cost of living"),
    "imf": ("imf", "international monetary fund"),
    "rupee": ("rupee", "exchange rate", "us dollar", "forex"),
    "oil": ("oil price", "crude oil", "fuel price", "petrol", "diesel"),
    "politics": ("election", "president", "parliament", "government"),
    "earnings": ("profit", "profits", "earnings", "dividend", "revenue"),
    "crisis": ("crisis", "default", "bankruptcy", "recession", "crash"),
    "conflict": ("war", "invasion", "sanction", "sanctions", "attack"),
}

FEATURES = (
    ["sentiment", "mentions_company"]
    + [f"region_{region}" for region in REGIONS]
    + [f"topic_{topic}" for topic in TOPICS]
    + [f"sym_{SYMBOL_CODES[symbol]}" for symbol in SYMBOLS]
)

_TAG = re.compile(r"<[^>]+>")
_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$", re.IGNORECASE)


def clean_title(value) -> str:
    text = _TAG.sub(" ", str(value or ""))
    text = text.replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"')
    return re.sub(r"\s+", " ", text).strip()


def _contains(text: str, phrase: str) -> bool:
    if " " in phrase or len(phrase) > 4:
        return phrase in text
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def symbols_mentioned(text: str) -> list[str]:
    lowered = clean_title(text).lower()
    found = []
    for symbol, _name, aliases in COMPANIES:
        if any(_contains(lowered, alias) for alias in aliases):
            found.append(symbol)
    return found


def topic_flags(text: str) -> dict[str, int]:
    lowered = clean_title(text).lower()
    return {f"topic_{name}": int(any(_contains(lowered, phrase) for phrase in phrases)) for name, phrases in TOPICS.items()}


def topic_names(flags: dict[str, int]) -> list[str]:
    return [name for name in TOPICS if flags.get(f"topic_{name}")]


PRICE_TERMS = (
    "bank",
    "bond",
    "budget",
    "central bank",
    "colombo",
    "debt",
    "dividend",
    "dollar",
    "economy",
    "economic",
    "election",
    "exchange rate",
    "export",
    "federal",
    "forex",
    "fuel",
    "gdp",
    "imf",
    "import",
    "inflation",
    "interest",
    "investor",
    "market",
    "monetary",
    "oil",
    "petrol",
    "profit",
    "recession",
    "reserve",
    "rupee",
    "sanction",
    "share",
    "stock",
    "tariff",
    "tax",
    "trade",
    "war",
)


def relevant_to_prices(title: str) -> bool:
    """True when a headline is about markets, the economy, or politics that can move prices."""
    lowered = clean_title(title).lower()
    if symbols_mentioned(lowered):
        return True
    if any(topic_flags(lowered).values()):
        return True
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in PRICE_TERMS)


def parse_seen(value) -> datetime | None:
    """Read a timestamp and return it in Colombo time."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat", "null"}:
        return None
    if re.fullmatch(r"\d{8}T\d{6}Z", text):
        parsed = datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(COLOMBO)
    # CSE announcement times have no zone. They are Colombo local.
    # Strings that already carry Z or an offset are converted from that zone.
    if _OFFSET.search(text):
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime().astimezone(COLOMBO)
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    local = parsed.to_pydatetime()
    if local.tzinfo is None:
        local = local.replace(tzinfo=COLOMBO)
    return local.astimezone(COLOMBO)


def load_prices() -> pd.DataFrame:
    """Official CSE closes where they exist, Yahoo Finance closes before that."""
    frames = []
    if YAHOO_PRICES.exists():
        frames.append(_read_prices(YAHOO_PRICES, 0))
    if SITE_PRICES.exists():
        frames.append(_read_prices(SITE_PRICES, 1))
    if not frames:
        raise FileNotFoundError("No CSE price file found in data/.")
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["symbol", "date", "priority"])
    prices = prices.drop_duplicates(["symbol", "date"], keep="last")
    return prices.sort_values(["symbol", "date"]).reset_index(drop=True)


def _read_prices(path: Path, priority: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["priority"] = priority
    keep = ["date", "symbol", "company", "close", "priority"]
    frame = frame.dropna(subset=["date", "symbol", "close"])
    frame = frame[frame["symbol"].isin(SYMBOLS)]
    return frame[keep]


def _empty_headlines() -> pd.DataFrame:
    columns = ["seen_at", "date", "region", "title", "url", "source", "country"]
    return pd.DataFrame(columns=columns)


def load_archive() -> pd.DataFrame:
    if not ARCHIVE_PATH.exists():
        return _empty_headlines()
    frame = pd.read_csv(ARCHIVE_PATH)
    for column in ("seen_at", "date", "region", "title", "url", "source", "country"):
        if column not in frame.columns:
            frame[column] = ""
    return frame[["seen_at", "date", "region", "title", "url", "source", "country"]]


def load_announcements() -> pd.DataFrame:
    """CSE company announcements already saved for 2021-2026."""
    if not ANNOUNCEMENTS_PATH.exists():
        return _empty_headlines()
    frame = pd.read_csv(ANNOUNCEMENTS_PATH)
    rows = []
    for record in frame.itertuples(index=False):
        seen = parse_seen(getattr(record, "createdDate", None))
        if seen is None:
            continue
        parts = [
            getattr(record, "company", ""),
            getattr(record, "companyName", ""),
            getattr(record, "announcementCategory", ""),
            getattr(record, "_category", ""),
            getattr(record, "remarks", ""),
            getattr(record, "title", ""),
            getattr(record, "subject", ""),
        ]
        title = clean_title(" ".join(str(part) for part in parts if str(part) not in {"", "nan", "None"}))
        if len(title) < 12:
            continue
        rows.append(
            {
                "seen_at": seen.isoformat(),
                "date": seen.date().isoformat(),
                "region": "market",
                "title": title[:500],
                "url": f"cse-announcement-{getattr(record, 'id', '')}",
                "source": "CSE",
                "country": "Sri Lanka",
            }
        )
    if not rows:
        return _empty_headlines()
    return pd.DataFrame(rows)


def load_headlines() -> pd.DataFrame:
    """Dated world, Sri Lanka, market, and CSE announcement headlines."""
    frames = [frame for frame in (load_archive(), load_announcements()) if not frame.empty]
    if not frames:
        return _empty_headlines()
    headlines = pd.concat(frames, ignore_index=True)
    headlines["title"] = headlines["title"].map(clean_title)
    headlines["region"] = headlines["region"].astype(str)
    headlines = headlines[headlines["region"].isin(REGIONS)]
    headlines = headlines[headlines["title"].str.len() >= 12]
    headlines["url"] = headlines["url"].fillna("").astype(str)
    headlines["rank"] = headlines["region"].map({"market": 0, "sri_lanka": 1, "world": 2}).fillna(3)
    has_url = headlines["url"].str.len() > 8
    headlines["dedupe_key"] = np.where(
        has_url,
        headlines["region"] + "|" + headlines["url"],
        headlines["region"] + "|" + headlines["title"].str.lower() + "|" + headlines["date"].astype(str),
    )
    headlines = headlines.sort_values("rank").drop_duplicates("dedupe_key", keep="first")
    headlines["title_key"] = headlines["title"].str.lower()
    headlines = headlines.drop_duplicates(["region", "title_key", "date"], keep="first")
    headlines = headlines.drop(columns=["rank", "dedupe_key", "title_key"])
    return headlines.reset_index(drop=True)


def explode_headlines(headlines: pd.DataFrame) -> pd.DataFrame:
    """One row per headline and stock that headline can apply to."""
    rows = []
    for record in headlines.itertuples(index=False):
        title = clean_title(record.title)
        named = symbols_mentioned(title)
        targets = named or list(SYMBOLS)
        sentiment = score_text(title)
        flags = topic_flags(title)
        seen = getattr(record, "seen_at", "")
        for symbol in targets:
            row = {
                "seen_at": seen,
                "date": getattr(record, "date", ""),
                "region": record.region,
                "title": title,
                "url": getattr(record, "url", ""),
                "source": getattr(record, "source", ""),
                "country": getattr(record, "country", ""),
                "symbol": symbol,
                "company": COMPANY_NAMES[symbol],
                "sentiment": sentiment,
                "tone": label_from_score(sentiment),
                "mentions_company": int(symbol in named),
                "topics": "|".join(topic_names(flags)),
            }
            for region in REGIONS:
                row[f"region_{region}"] = int(record.region == region)
            row.update(flags)
            for code_symbol, code in SYMBOL_CODES.items():
                row[f"sym_{code}"] = int(symbol == code_symbol)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _price_books(prices: pd.DataFrame) -> dict[str, pd.Series]:
    books = {}
    for symbol, part in prices.groupby("symbol", sort=False):
        part = part.sort_values("date").drop_duplicates("date", keep="last")
        closes = part.set_index("date")["close"].astype(float).sort_index()
        closes.index = pd.DatetimeIndex(closes.index).tz_localize(None)
        books[symbol] = closes
    return books


def attach_returns(rows: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Add the price change after each headline. Rows with no following close are dropped."""
    if rows.empty:
        return rows
    books = _price_books(prices)
    pieces = []
    for symbol, closes in books.items():
        part = rows[rows["symbol"] == symbol].copy()
        if part.empty:
            continue
        local = pd.to_datetime(part["seen_at"], utc=True, errors="coerce").dt.tz_convert(COLOMBO)
        valid_time = local.notna()
        part = part.loc[valid_time].copy()
        local = local.loc[valid_time]
        if part.empty:
            continue
        minutes = local.dt.hour * 60 + local.dt.minute
        before_close = (minutes < MARKET_CLOSE_MINUTES).to_numpy()
        days = local.dt.tz_localize(None).dt.normalize().to_numpy(dtype="datetime64[ns]")
        sessions = pd.DatetimeIndex(closes.index)
        pos = np.where(
            before_close,
            sessions.searchsorted(days, side="left"),
            sessions.searchsorted(days, side="right"),
        )
        usable = (pos > 0) & (pos < len(sessions))
        if not usable.any():
            continue
        part = part.iloc[np.flatnonzero(usable)].copy()
        pos = pos[usable]
        ref_idx = pos - 1
        close_values = closes.to_numpy(dtype=float)
        part["ref_date"] = sessions[ref_idx]
        part["react_date"] = sessions[pos]
        part["close_ref"] = close_values[ref_idx]
        part["close_1d"] = close_values[pos]
        part["return_1d"] = part["close_1d"] / part["close_ref"] - 1
        five_idx = ref_idx + 5
        has_five = five_idx < len(sessions)
        close_5 = np.full(len(part), np.nan)
        close_5[has_five] = close_values[five_idx[has_five]]
        react_5 = np.full(len(part), np.datetime64("NaT"), dtype="datetime64[ns]")
        react_5[has_five] = sessions.to_numpy()[five_idx[has_five]]
        part["close_5d"] = close_5
        part["react_5_date"] = react_5
        part["return_5d"] = part["close_5d"] / part["close_ref"] - 1
        part = part[(part["close_ref"] > 0) & (part["return_1d"].abs() <= MAX_ABS_RETURN)]
        pieces.append(part)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def build_reactions(headlines: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return attach_returns(explode_headlines(headlines), prices)


def latest_closes(prices: pd.DataFrame) -> pd.DataFrame:
    latest = prices.sort_values("date").groupby("symbol", as_index=False).tail(1)
    return latest[["symbol", "company", "date", "close"]].reset_index(drop=True)


def pattern_table(reactions: pd.DataFrame) -> pd.DataFrame:
    """Average price change after each kind of headline, from the training rows."""
    frames = []

    def _aggregate(frame: pd.DataFrame, scope: str, topic: str = "", symbol: str = "") -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        grouped = frame.groupby(["region", "tone"], as_index=False).agg(
            n=("return_1d", "size"),
            mean_1d=("return_1d", "mean"),
            mean_5d=("return_5d", "mean"),
        )
        grouped["scope"] = scope
        grouped["topic"] = topic
        grouped["symbol"] = symbol
        return grouped

    frames.append(_aggregate(reactions, "region_tone"))
    tone_only = reactions.groupby("tone", as_index=False).agg(
        n=("return_1d", "size"),
        mean_1d=("return_1d", "mean"),
        mean_5d=("return_5d", "mean"),
    )
    tone_only["scope"] = "tone"
    tone_only["region"] = ""
    tone_only["topic"] = ""
    tone_only["symbol"] = ""
    frames.append(tone_only)
    for topic in TOPICS:
        flagged = reactions[reactions[f"topic_{topic}"] == 1]
        frames.append(_aggregate(flagged, "topic", topic=topic))
    named = reactions[reactions["mentions_company"] == 1]
    if not named.empty:
        company = named.groupby(["symbol", "tone"], as_index=False).agg(
            n=("return_1d", "size"),
            mean_1d=("return_1d", "mean"),
            mean_5d=("return_5d", "mean"),
        )
        company["scope"] = "company"
        company["region"] = ""
        company["topic"] = ""
        frames.append(company)
    table = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    return table.sort_values(["scope", "region", "topic", "symbol", "tone"]).reset_index(drop=True)


def lookup_historical(patterns: pd.DataFrame, region: str, tone: str, topics: list[str]) -> dict | None:
    """The busiest matching average from the training years."""
    if patterns is None or patterns.empty:
        return None
    topic_rows = patterns[
        (patterns["scope"] == "topic")
        & (patterns["region"] == region)
        & (patterns["tone"] == tone)
        & (patterns["topic"].isin(topics))
        & (patterns["n"] >= 30)
    ]
    if not topic_rows.empty:
        row = topic_rows.sort_values("n", ascending=False).iloc[0]
        return row.to_dict()
    general = patterns[
        (patterns["scope"] == "region_tone")
        & (patterns["region"] == region)
        & (patterns["tone"] == tone)
        & (patterns["n"] >= 20)
    ]
    if not general.empty:
        return general.iloc[0].to_dict()
    tone_rows = patterns[(patterns["scope"] == "tone") & (patterns["tone"] == tone)]
    if not tone_rows.empty:
        return tone_rows.iloc[0].to_dict()
    return None


def score_headlines(headlines: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict the next-session and five-session move for each current headline."""
    if headlines.empty:
        return pd.DataFrame(), pd.DataFrame()
    prepared = headlines.copy().reset_index(drop=True)
    prepared["headline_id"] = np.arange(len(prepared))
    rows = _attach_headline_ids(prepared)
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    features = rows[bundle["features"]]
    rows["pred_1d"] = bundle["regressor_1d"].predict(features)
    rows["pred_5d"] = bundle["regressor_5d"].predict(features)
    probability = bundle["classifier_1d"].predict_proba(features)
    up_index = list(bundle["classifier_1d"].classes_).index(1)
    rows["up_probability"] = probability[:, up_index]
    patterns = bundle.get("patterns")
    if not isinstance(patterns, pd.DataFrame):
        patterns = pd.DataFrame(patterns or [])

    headline_rows = []
    for headline_id, part in rows.groupby("headline_id"):
        source_row = prepared.loc[prepared["headline_id"] == headline_id].iloc[0]
        topics = [name for name in str(part["topics"].iloc[0]).split("|") if name]
        historical = lookup_historical(patterns, source_row["region"], part["tone"].iloc[0], topics)
        headline_rows.append(
            {
                "headline_id": int(headline_id),
                "seen_at": source_row.get("seen_at", ""),
                "region": source_row["region"],
                "title": source_row["title"],
                "source": source_row.get("source", ""),
                "url": source_row.get("url", ""),
                "tone": part["tone"].iloc[0],
                "sentiment": float(part["sentiment"].iloc[0]),
                "topics": "|".join(topics),
                "stocks": ", ".join(part["company"].tolist()),
                "pred_1d": float(part["pred_1d"].mean()),
                "pred_5d": float(part["pred_5d"].mean()),
                "up_probability": float(part["up_probability"].mean()),
                "historical_1d": None if historical is None else float(historical["mean_1d"]),
                "historical_5d": None if historical is None else float(historical["mean_5d"]),
                "historical_n": 0 if historical is None else int(historical["n"]),
                "historical_label": _historical_label(historical),
            }
        )
    headline_frame = pd.DataFrame(headline_rows)
    by_id = headline_frame.set_index("headline_id")
    rows["quote_1d"] = rows["headline_id"].map(by_id["historical_1d"]).fillna(rows["pred_1d"])
    rows["quote_5d"] = rows["headline_id"].map(by_id["historical_5d"]).fillna(rows["pred_5d"])

    stock_rows = []
    for symbol, part in rows.groupby("symbol"):
        stock_rows.append(
            {
                "symbol": symbol,
                "company": COMPANY_NAMES[symbol],
                "headline_count": int(part["headline_id"].nunique()),
                "pred_1d": float(part["quote_1d"].mean()),
                "pred_5d": float(part["quote_5d"].mean()),
                "up_probability": float(part["up_probability"].mean()),
            }
        )
    stock_frame = pd.DataFrame(stock_rows)
    return headline_frame, stock_frame


def _attach_headline_ids(prepared: pd.DataFrame) -> pd.DataFrame:
    """Explode headlines and keep the caller's headline id on every stock row."""
    frames = []
    for record in prepared.itertuples(index=False):
        one = pd.DataFrame(
            [
                {
                    "seen_at": record.seen_at,
                    "date": getattr(record, "date", ""),
                    "region": record.region,
                    "title": record.title,
                    "url": getattr(record, "url", ""),
                    "source": getattr(record, "source", ""),
                    "country": getattr(record, "country", ""),
                }
            ]
        )
        exploded = explode_headlines(one)
        if exploded.empty:
            continue
        exploded["headline_id"] = int(record.headline_id)
        frames.append(exploded)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _historical_label(historical: dict | None) -> str:
    if not historical:
        return "no similar training headlines"
    topic = str(historical.get("topic") or "")
    region = REGION_LABELS.get(str(historical.get("region") or ""), "Any region")
    tone = str(historical.get("tone") or "")
    if topic:
        return f"{region} {tone} {topic} news"
    if historical.get("region"):
        return f"{region} {tone} news"
    return f"{tone} news"
