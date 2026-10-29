from __future__ import annotations

from pathlib import Path
import json
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, dash_table, no_update
from plotly.subplots import make_subplots

from cse_common import extract_trade_price, select_news_csv_names
from fetch_market_history import COMPANIES as MARKET_COMPANIES

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "cse_output"
MARKET_DIR = ROOT / "data" / "market"
BOOK_PATH = DATA_DIR / "paper_book.json"
NEWS_PREFIX = "cse_news"
PREDICTION_FILE = "cse_prediction_results.csv"
COLOMBO = ZoneInfo("Asia/Colombo")
PAPER_CASH = 1_000_000.0
MARKET = [{"symbol": row["symbol"], "name": row["name"]} for row in MARKET_COMPANIES]
_BOOK_LOCK = threading.Lock()
_QUOTE_CACHE: dict = {"at": 0.0, "data": {}}
_HISTORY_CACHE: dict[str, tuple] = {}
_SITE_CACHE: dict = {"stamp": None, "frame": None}
SITE_PRICES = ROOT / "data" / "cse_site_prices_1y.csv"
PRICE_COLUMNS = ["live_price", "price", "close", "last_price"]
NEWS_SHOW_COLUMNS = [
    "_parsed_date",
    "_year",
    "company",
    "_category",
    "announcementCategory",
    "subject",
    "title",
    "remarks",
    "description",
]
_DATA_CACHE: dict = {"stamp": None, "data": {}}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, low_memory=False)


def _read_json(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except Exception:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return pd.json_normalize(payload)
        except Exception:
            return pd.DataFrame()


def _data_stamp() -> tuple:
    if not DATA_DIR.exists():
        return tuple()
    files = sorted(DATA_DIR.glob("*.csv")) + sorted(DATA_DIR.glob("*.json"))
    return tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in files)


def load_all_data() -> Dict[str, pd.DataFrame]:
    stamp = _data_stamp()
    if stamp == _DATA_CACHE["stamp"] and _DATA_CACHE["data"]:
        return _DATA_CACHE["data"]

    data: Dict[str, pd.DataFrame] = {}
    if DATA_DIR.exists():
        for csv_path in sorted(DATA_DIR.glob("*.csv")):
            data[csv_path.name] = _read_csv(csv_path)
        for json_path in sorted(DATA_DIR.glob("*.json")):
            data[json_path.name] = _read_json(json_path)
    _DATA_CACHE["stamp"] = stamp
    _DATA_CACHE["data"] = data
    return data


def build_news_df(data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    news_names = select_news_csv_names(
        [name for name in data_map if name.startswith(NEWS_PREFIX) and name.endswith(".csv")]
    )
    frames = []
    for name in news_names:
        df = data_map.get(name)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["_source_file"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    news = pd.concat(frames, ignore_index=True, sort=False)
    if "announcementId" in news.columns and news["announcementId"].notna().any():
        news = news.drop_duplicates(subset=["announcementId"], keep="first")
    elif "id" in news.columns and news["id"].notna().any():
        news = news.drop_duplicates(subset=["id"], keep="first")
    return news.reset_index(drop=True)


def present_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    columns = [col for col in NEWS_SHOW_COLUMNS if col in df.columns]
    return df[columns].copy() if columns else df


def frame_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%d")
    clean = clean.astype(object).where(pd.notna(clean), None)
    return clean.to_dict("records")


def build_predictions_df(data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data_map.get(PREDICTION_FILE)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()


def ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ["_parsed_date", "createdDate", "announcementDate", "publishedDate", "date"]:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if parsed.notna().any():
            df["_year"] = parsed.dt.year
            return df
    if "_year" in df.columns:
        df["_year"] = pd.to_numeric(df["_year"], errors="coerce")
    return df


def text_search_mask(df: pd.DataFrame, keyword: str, columns: Iterable[str]) -> pd.Series:
    if not keyword:
        return pd.Series(True, index=df.index)
    keyword = str(keyword).strip()
    if not keyword:
        return pd.Series(True, index=df.index)

    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            mask = mask | df[col].fillna("").astype(str).str.contains(keyword, case=False, na=False)
    return mask


def filter_news(
    df: pd.DataFrame,
    year_value: int | None,
    keyword: str | None,
    company_value: str | None,
) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    if df.empty:
        return df, notes

    df = ensure_year_column(df)
    if year_value and "_year" in df.columns:
        df = df[df["_year"] == year_value]
    elif year_value:
        notes.append("Year filter ignored (no _year column).")

    if company_value and "company" in df.columns:
        df = df[df["company"].fillna("").astype(str).str.contains(company_value, case=False, na=False)]
    elif company_value:
        notes.append("Company filter ignored (no company column).")

    text_cols = ["company", "subject", "description", "heading", "title", "remarks"]
    if keyword:
        mask = text_search_mask(df, keyword, text_cols)
        df = df[mask]

    return df, notes


def add_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "combined_score" not in df.columns:
        return df
    df = df.copy()
    df["combined_score"] = pd.to_numeric(df["combined_score"], errors="coerce")

    def score_to_risk(score: float | None) -> str:
        if score is None or pd.isna(score):
            return "unknown"
        if score >= 8:
            return "low"
        if score >= 4:
            return "medium"
        return "high"

    df["risk_level"] = df["combined_score"].apply(score_to_risk)
    return df


def filter_predictions(
    df: pd.DataFrame,
    name_value: str | None,
    risk_value: str | None,
    score_min: float | None,
    score_max: float | None,
    price_min: float | None,
    price_max: float | None,
) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    if df.empty:
        return df, notes

    df = add_risk_level(df)

    if name_value and "company" in df.columns:
        df = df[df["company"].fillna("").astype(str).str.contains(name_value, case=False, na=False)]
    elif name_value:
        notes.append("Name filter ignored (no company column).")

    if risk_value and "risk_level" in df.columns:
        df = df[df["risk_level"] == risk_value]
    elif risk_value:
        notes.append("Risk filter ignored (no combined_score column).")

    if "combined_score" in df.columns and (score_min is not None or score_max is not None):
        df = _filter_numeric(df, "combined_score", score_min, score_max)
    elif score_min is not None or score_max is not None:
        notes.append("Score filter ignored (no combined_score column).")

    price_col = next((col for col in PRICE_COLUMNS if col in df.columns), None)
    if price_col and (price_min is not None or price_max is not None):
        df = _filter_numeric(df, price_col, price_min, price_max)
    elif price_min is not None or price_max is not None:
        notes.append("Price filter ignored (no price column).")

    return df, notes


def _filter_numeric(df: pd.DataFrame, column: str, low: float | None, high: float | None) -> pd.DataFrame:
    values = pd.to_numeric(df[column], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if low is not None:
        mask &= values >= low
    if high is not None:
        mask &= values <= high
    return df.loc[mask]


def to_table_columns(df: pd.DataFrame) -> List[dict]:
    return [{"name": col, "id": col} for col in df.columns]


def file_options(data_map: Dict[str, pd.DataFrame]) -> List[dict]:
    return [{"label": name, "value": name} for name in sorted(data_map.keys())]


def prepare_file_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.head(500).copy()


def filter_file_df(df: pd.DataFrame, company_value: str | None) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    if df.empty:
        return df, notes
    if company_value and "company" in df.columns:
        df = df[df["company"].fillna("").astype(str).str.contains(company_value, case=False, na=False)]
    elif company_value:
        notes.append("Company filter ignored (no company column).")
    return df, notes


def _chart_palette(theme: str | None) -> tuple[str, str, str]:
    if theme == "dark":
        return "#e7ebf5", "#9aa6c2", "#2c3450"
    return "#1E1B39", "#4a5a52", "#e6e9f2"


def build_company_figure(
    pred_df: pd.DataFrame, company_value: str | None, theme: str | None = None
) -> Tuple[go.Figure, List[dict], str | None]:
    ink, muted, grid = _chart_palette(theme)
    if pred_df.empty or "company" not in pred_df.columns:
        fig = go.Figure()
        fig.update_layout(
            height=300,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                {
                    "text": "No prediction data available",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 14, "color": muted},
                }
            ],
        )
        return fig, [], None

    companies = sorted({str(c) for c in pred_df["company"].dropna().unique()})
    options = [{"label": company, "value": company} for company in companies]
    if company_value not in companies:
        company_value = companies[0] if companies else None

    row = pred_df[pred_df["company"] == company_value].head(1)
    if row.empty:
        return build_company_figure(pred_df.iloc[:0], company_value, theme)

    row = row.iloc[0]
    metrics: List[dict] = []
    for label, key in [
        ("Combined", "combined_score"),
        ("Historical", "hist_score"),
        ("Live", "live_score"),
        ("Hist mentions", "hist_mentions"),
        ("Live mentions", "live_mentions"),
        ("Positive %", "pct_positive"),
        ("Negative %", "pct_negative"),
        ("Neutral %", "pct_neutral"),
    ]:
        if key in pred_df.columns:
            value = pd.to_numeric(row.get(key), errors="coerce")
            if pd.notna(value):
                metrics.append({"metric": label, "value": float(value)})

    fig = go.Figure()
    if metrics:
        fig.add_bar(
            x=[m["metric"] for m in metrics],
            y=[m["value"] for m in metrics],
            marker_color="#003DA5",
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    fig.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Poppins", "color": ink},
        yaxis_title="",
        xaxis_title="",
        xaxis_tickangle=-20,
        yaxis={"gridcolor": grid},
    )
    return fig, options, company_value


def build_insight_message(pred_df: pd.DataFrame) -> str:
    if pred_df.empty or "signal" not in pred_df.columns or "company" not in pred_df.columns:
        return ""

    strong = pred_df[pred_df["signal"] == "*** STRONG BUY ***"]
    if strong.empty:
        return "No strong signals right now. Keep monitoring upcoming announcements and score changes."

    names = strong["company"].dropna().astype(str).unique().tolist()
    names = names[:6]
    joined = ", ".join(names)
    return (
        "Informational only: the current signals highlight these shares as strong candidates to watch based on the "
        f"engine metrics: {joined}."
    )


def fetch_live_quotes() -> dict[str, dict]:
    """Latest CSE last-trade snapshot, reused for a few seconds across callbacks."""
    now = time.time()
    if _QUOTE_CACHE["data"] and now - float(_QUOTE_CACHE["at"]) < 8:
        return _QUOTE_CACHE["data"]
    try:
        response = requests.post(
            "https://www.cse.lk/api/tradeSummary",
            json={},
            headers={
                "User-Agent": "Mozilla/5.0",
                "Origin": "https://www.cse.lk",
                "Referer": "https://www.cse.lk/",
                "X-Requested-With": "XMLHttpRequest",
            },
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return _QUOTE_CACHE["data"]

    quotes: dict[str, dict] = {}
    for item in payload.get("reqTradeSummery", []):
        symbol = str(item.get("symbol") or "")
        price = extract_trade_price(item)
        if not symbol or price is None:
            continue
        raw_change = item.get("percentageChange")
        try:
            change_pct = float(raw_change) if raw_change not in (None, "") else None
        except (TypeError, ValueError):
            change_pct = None
        quotes[symbol] = {"price": price, "change_pct": change_pct}
    _QUOTE_CACHE["at"] = now
    _QUOTE_CACHE["data"] = quotes
    return quotes


def _as_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Official CSE bars have high, low, and close. Open is the previous close."""
    out = frame.copy()
    out["Close"] = pd.to_numeric(out["close"] if "close" in out.columns else out["Close"], errors="coerce")
    high_col = "high" if "high" in out.columns else "High"
    low_col = "low" if "low" in out.columns else "Low"
    vol_col = "volume" if "volume" in out.columns else "Volume" if "Volume" in out.columns else None
    out["High"] = pd.to_numeric(out[high_col], errors="coerce")
    out["Low"] = pd.to_numeric(out[low_col], errors="coerce")
    if "open" in out.columns or "Open" in out.columns:
        out["Open"] = pd.to_numeric(out["open"] if "open" in out.columns else out["Open"], errors="coerce")
    else:
        out["Open"] = out["Close"].shift(1).fillna(out["Close"])
    out["Volume"] = pd.to_numeric(out[vol_col], errors="coerce") if vol_col else 0
    out["High"] = out[["High", "Open", "Close"]].max(axis=1)
    out["Low"] = out[["Low", "Open", "Close"]].min(axis=1)
    # A few CSE bars have a wild low or high that flattens the whole chart.
    wild_low = out["Low"] < out["Close"] * 0.90
    wild_high = out["High"] > out["Close"] * 1.10
    out.loc[wild_low, "Low"] = out.loc[wild_low, ["Open", "Close"]].min(axis=1)
    out.loc[wild_high, "High"] = out.loc[wild_high, ["Open", "Close"]].max(axis=1)
    out = out.dropna(subset=["Close"]).sort_index()
    return out[["Open", "High", "Low", "Close", "Volume"]]


def _site_history() -> pd.DataFrame:
    if not SITE_PRICES.exists():
        return pd.DataFrame()
    stamp = SITE_PRICES.stat().st_mtime_ns
    if _SITE_CACHE["stamp"] == stamp and _SITE_CACHE["frame"] is not None:
        return _SITE_CACHE["frame"]
    frame = pd.read_csv(SITE_PRICES, parse_dates=["date"])
    _SITE_CACHE["stamp"] = stamp
    _SITE_CACHE["frame"] = frame
    return frame


def history_for(symbol: str | None) -> pd.DataFrame:
    """Prefer the official CSE daily file. It is current; the Yahoo files stop early."""
    if not symbol:
        return pd.DataFrame()
    site = _site_history()
    if not site.empty and "symbol" in site.columns:
        part = site.loc[site["symbol"].astype(str) == symbol]
        if not part.empty:
            return _as_ohlc(part.set_index("date"))

    path = MARKET_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    stamp = path.stat().st_mtime_ns
    cached = _HISTORY_CACHE.get(symbol)
    if cached and cached[0] == stamp:
        return cached[1]
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    ohlc = _as_ohlc(frame)
    _HISTORY_CACHE[symbol] = (stamp, ohlc)
    return ohlc


def with_live_bar(frame: pd.DataFrame, price: float | None) -> pd.DataFrame:
    """Move today's candle to the latest CSE last trade when it differs from the saved close."""
    if frame is None or frame.empty or price is None:
        return frame if frame is not None else pd.DataFrame()
    out = frame.copy()
    today = pd.Timestamp(datetime.now(COLOMBO).date())
    last_day = pd.Timestamp(out.index[-1]).normalize()
    price = float(price)
    prev = float(out["Close"].iloc[-1])
    if last_day == today or abs(price - prev) < 0.005:
        if last_day == today and abs(price - prev) >= 0.005:
            idx = out.index[-1]
            out.loc[idx, "Close"] = price
            out.loc[idx, "High"] = max(float(out.loc[idx, "High"]), price)
            out.loc[idx, "Low"] = min(float(out.loc[idx, "Low"]), price)
        return out
    if last_day > today or (today - last_day).days > 6:
        return out
    out.loc[today, ["Open", "High", "Low", "Close", "Volume"]] = [
        prev,
        max(prev, price),
        min(prev, price),
        price,
        0,
    ]
    return out


def tradable_price(symbol: str, quotes: dict[str, dict]) -> tuple[float | None, str]:
    quote = quotes.get(symbol) or {}
    if quote.get("price"):
        return float(quote["price"]), "live"
    frame = history_for(symbol)
    if not frame.empty and pd.notna(frame["Close"].iloc[-1]):
        return float(frame["Close"].iloc[-1]), "last saved close"
    return None, ""


def _fmt_px(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.2f}"


def _fmt_chg(value) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "No session change", "chg-flat"
    value = float(value)
    if value > 0:
        return f"+{value:.2f}%", "chg-up"
    if value < 0:
        return f"{value:.2f}%", "chg-down"
    return "0.00%", "chg-flat"


def empty_book() -> dict:
    return {"cash": PAPER_CASH, "fills": []}


def load_book() -> dict:
    if not BOOK_PATH.exists():
        return empty_book()
    try:
        payload = json.loads(BOOK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_book()
    if not isinstance(payload, dict):
        return empty_book()
    payload.setdefault("cash", PAPER_CASH)
    payload.setdefault("fills", [])
    return payload


def save_book(book: dict) -> None:
    BOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOOK_PATH.write_text(json.dumps(book, indent=2), encoding="utf-8")


def replay_positions(book: dict) -> dict[str, dict]:
    state: dict[str, dict] = {}
    for fill in book.get("fills") or []:
        symbol = str(fill.get("symbol") or "")
        if not symbol:
            continue
        slot = state.setdefault(symbol, {"qty": 0, "cost": 0.0})
        try:
            qty = int(fill.get("qty") or 0)
            price = float(fill.get("price") or 0)
        except (TypeError, ValueError):
            continue
        if qty <= 0 or price <= 0:
            continue
        if str(fill.get("side")).upper() == "BUY":
            slot["qty"] += qty
            slot["cost"] += qty * price
            continue
        if slot["qty"] <= 0:
            continue
        sell_qty = min(qty, slot["qty"])
        avg = slot["cost"] / slot["qty"]
        slot["qty"] -= sell_qty
        slot["cost"] -= avg * sell_qty
        if slot["qty"] <= 0:
            slot["qty"] = 0
            slot["cost"] = 0.0
    return state


def place_order(symbol: str, side: str, qty, price: float | None, source: str) -> str:
    try:
        shares = int(qty)
    except (TypeError, ValueError):
        return "Enter a whole number of shares."
    if shares <= 0:
        return "Quantity has to be at least 1."
    if not symbol or price is None or price <= 0:
        return "No price is available for that share yet."
    side = side.upper()
    notion = round(shares * float(price), 2)
    with _BOOK_LOCK:
        book = load_book()
        held = replay_positions(book)
        if side == "BUY":
            if notion > float(book["cash"]) + 1e-6:
                return (
                    f"Not enough paper cash. This buy needs LKR {notion:,.2f}. "
                    f"Cash is LKR {float(book['cash']):,.2f}."
                )
            book["cash"] = round(float(book["cash"]) - notion, 2)
        elif side == "SELL":
            owned = int(held.get(symbol, {}).get("qty", 0))
            if shares > owned:
                return f"You hold {owned:,} shares of {symbol}."
            book["cash"] = round(float(book["cash"]) + notion, 2)
        else:
            return "Unknown order."
        book["fills"].append(
            {
                "id": uuid.uuid4().hex[:12],
                "time": datetime.now(COLOMBO).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "side": side,
                "qty": shares,
                "price": round(float(price), 2),
                "value": notion,
            }
        )
        save_book(book)
    verb = "Bought" if side == "BUY" else "Sold"
    return f"{verb} {shares:,} {symbol} at LKR {float(price):,.2f} ({source})."


def reset_book() -> str:
    with _BOOK_LOCK:
        save_book(empty_book())
    return "Paper book reset. Cash is LKR 1,000,000 and no shares are held."


def held_text(symbol: str, positions: dict[str, dict], quote: dict) -> str:
    slot = positions.get(symbol) or {}
    qty = int(slot.get("qty") or 0)
    if qty <= 0:
        return "No shares held"
    avg = float(slot["cost"]) / qty
    last = quote.get("price")
    if last:
        pnl = (float(last) - avg) * qty
        return f"Held {qty:,} · avg {_fmt_px(avg)} · P&L {pnl:+,.2f}"
    return f"Held {qty:,} · avg {_fmt_px(avg)}"


def book_summary(book: dict, quotes: dict[str, dict], positions: dict[str, dict]) -> tuple[float, float, float]:
    cash = float(book.get("cash") or 0)
    market_value = 0.0
    pnl = 0.0
    for symbol, slot in positions.items():
        qty = int(slot.get("qty") or 0)
        if qty <= 0:
            continue
        last = (quotes.get(symbol) or {}).get("price")
        if last is None:
            frame = history_for(symbol)
            if frame.empty:
                continue
            last = float(frame["Close"].iloc[-1])
        avg = float(slot["cost"]) / qty
        market_value += float(last) * qty
        pnl += (float(last) - avg) * qty
    return cash, cash + market_value, pnl


def model_by_symbol() -> dict[str, str]:
    path = DATA_DIR / "hourly_outlook.csv"
    if not path.exists():
        return {}
    try:
        outlook = pd.read_csv(path)
    except Exception:
        return {}
    if "symbol" not in outlook.columns or "action" not in outlook.columns:
        return {}
    return {str(row.symbol): str(row.action) for row in outlook.itertuples(index=False)}


def badge_class(action: str) -> str:
    action = (action or "").upper()
    if action == "BUY":
        return "badge badge-buy"
    if action == "SELL":
        return "badge badge-sell"
    if action == "WAIT":
        return "badge badge-wait"
    return "badge"


def _empty_figure(height: int, message: str, color: str = "#4a5a52") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 13, "color": color},
            }
        ],
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def mini_figure(frame: pd.DataFrame, session_up: bool | None = None) -> go.Figure:
    if frame is None or frame.empty or "Close" not in frame.columns:
        return _empty_figure(132, "No prices")
    window = frame.tail(90)
    close = pd.to_numeric(window["Close"], errors="coerce").dropna()
    if close.empty:
        return _empty_figure(132, "No prices")
    if session_up is None:
        rising = float(close.iloc[-1]) >= float(close.iloc[0])
    else:
        rising = session_up
    color = "#15CF74" if rising else "#E23B32"
    fill = "rgba(21,207,116,0.22)" if rising else "rgba(226,59,50,0.22)"
    baseline = float(close.min())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=[baseline] * len(close),
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close,
            mode="lines",
            line={"color": color, "width": 2},
            fill="tonexty",
            fillcolor=fill,
            hovertemplate="%{x|%d %b %Y}<br>LKR %{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=132,
        margin={"l": 0, "r": 0, "t": 4, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="x",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def price_figure(
    symbol: str,
    frame: pd.DataFrame,
    window_days: int | None,
    change_pct: float | None,
    revision: str | None = None,
    theme: str | None = None,
) -> go.Figure:
    ink, muted, grid = _chart_palette(theme)
    if frame is None or frame.empty or "Close" not in frame.columns:
        return _empty_figure(460, "No saved price history for this share", muted)
    opened = pd.to_numeric(frame["Open"], errors="coerce")
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce") if "Volume" in frame.columns else pd.Series(0, index=frame.index)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=opened,
            high=high,
            low=low,
            close=close,
            increasing_line_color="#15CF74",
            decreasing_line_color="#E23B32",
            increasing_fillcolor="#15CF74",
            decreasing_fillcolor="#E23B32",
            name="Price",
            hovertemplate="LKR %{close:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=frame.index,
            y=volume,
            marker_color=["rgba(21,207,116,0.45)" if flag else "rgba(226,59,50,0.4)" for flag in (close >= opened).fillna(False)],
            name="Volume",
            hovertemplate="Volume %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    if change_pct not in (None, 0) and pd.notna(close.iloc[-1]):
        prev = float(close.iloc[-1]) / (1 + float(change_pct) / 100)
        fig.add_hline(
            y=prev,
            line_dash="dot",
            line_color="#9aa3b8",
            row=1,
            col=1,
            annotation_text="Prev close",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color=muted,
        )
    fig.update_layout(
        height=420,
        margin={"l": 8, "r": 12, "t": 12, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Poppins", "color": ink, "size": 12},
        showlegend=False,
        uirevision=revision or symbol or "market",
        hovermode="x unified",
        yaxis={"title": "LKR", "gridcolor": grid},
        yaxis2={"showticklabels": False, "gridcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(showgrid=False, rangeslider_visible=False, color=ink)
    if len(frame.index):
        end = pd.Timestamp(frame.index[-1])
        span = int(window_days or 186)
        start = end - pd.Timedelta(days=span)
        fig.update_xaxes(
            range=[start, end + pd.Timedelta(days=2)],
            autorange=False,
        )
        visible = frame.loc[frame.index >= start]
        if visible.empty:
            visible = frame
        low = float(visible["Low"].min())
        high = float(visible["High"].max())
        pad = max((high - low) * 0.12, 0.4)
        fig.update_yaxes(range=[low - pad, high + pad], autorange=False, row=1, col=1)
    return fig


def position_table(positions: dict[str, dict], quotes: dict[str, dict], names: dict[str, str]) -> tuple[list, list]:
    columns = [
        {"name": "Share", "id": "name"},
        {"name": "Symbol", "id": "symbol"},
        {"name": "Shares", "id": "qty", "type": "numeric", "format": {"specifier": ","}},
        {"name": "Avg", "id": "avg", "type": "numeric", "format": {"specifier": ",.2f"}},
        {"name": "Last", "id": "last", "type": "numeric", "format": {"specifier": ",.2f"}},
        {"name": "Value", "id": "value", "type": "numeric", "format": {"specifier": ",.2f"}},
        {"name": "P&L", "id": "pnl", "type": "numeric", "format": {"specifier": "+,.2f"}},
    ]
    rows = []
    for symbol, slot in positions.items():
        qty = int(slot.get("qty") or 0)
        if qty <= 0:
            continue
        avg = float(slot["cost"]) / qty
        last = (quotes.get(symbol) or {}).get("price")
        if last is None:
            frame = history_for(symbol)
            last = None if frame.empty else float(frame["Close"].iloc[-1])
        value = None if last is None else round(float(last) * qty, 2)
        pnl = None if last is None else round((float(last) - avg) * qty, 2)
        rows.append(
            {
                "name": names.get(symbol, symbol),
                "symbol": symbol,
                "qty": qty,
                "avg": round(avg, 2),
                "last": None if last is None else round(float(last), 2),
                "value": value,
                "pnl": pnl,
            }
        )
    rows.sort(key=lambda row: row["name"])
    return rows, columns


def fill_table(book: dict) -> tuple[list, list]:
    columns = [
        {"name": "Time", "id": "time"},
        {"name": "Symbol", "id": "symbol"},
        {"name": "Side", "id": "side"},
        {"name": "Shares", "id": "qty", "type": "numeric", "format": {"specifier": ","}},
        {"name": "Price", "id": "price", "type": "numeric", "format": {"specifier": ",.2f"}},
        {"name": "Value", "id": "value", "type": "numeric", "format": {"specifier": ",.2f"}},
    ]
    rows = []
    for fill in reversed(book.get("fills") or []):
        rows.append(
            {
                "time": fill.get("time"),
                "symbol": fill.get("symbol"),
                "side": fill.get("side"),
                "qty": fill.get("qty"),
                "price": fill.get("price"),
                "value": fill.get("value"),
            }
        )
        if len(rows) >= 40:
            break
    return rows, columns


def stock_card(company: dict) -> html.Div:
    symbol = company["symbol"]
    selected = symbol == MARKET[0]["symbol"]
    return html.Div(
        id={"type": "stock-card", "symbol": symbol},
        className="stock-card is-selected" if selected else "stock-card",
        children=[
            html.Button(
                id={"type": "select-stock", "symbol": symbol},
                n_clicks=0,
                type="button",
                className="stock-select",
                children=[
                    html.Div(
                        className="stock-top",
                        children=[
                            html.Div(
                                children=[
                                    html.Div(company["name"], className="stock-name"),
                                    html.Div(symbol, className="stock-symbol"),
                                ]
                            ),
                            html.Span(id={"type": "model-badge", "symbol": symbol}, className="badge"),
                        ],
                    ),
                    html.Div("—", id={"type": "live-price", "symbol": symbol}, className="px"),
                    html.Div("Waiting for a price", id={"type": "live-change", "symbol": symbol}, className="chg-flat"),
                ],
            ),
            dcc.Graph(
                id={"type": "mini-chart", "symbol": symbol},
                figure=mini_figure(pd.DataFrame()),
                config={"displayModeBar": False},
                style={"height": "132px"},
            ),
            html.Div("No shares held", id={"type": "held", "symbol": symbol}, className="held"),
            html.Div(
                className="card-trade",
                children=[
                    dcc.Input(
                        id={"type": "card-qty", "symbol": symbol},
                        type="number",
                        min=1,
                        step=1,
                        value=100,
                        placeholder="Qty",
                    ),
                    html.Button(
                        "Buy",
                        id={"type": "card-buy", "symbol": symbol},
                        n_clicks=0,
                        type="button",
                        className="btn btn-buy btn-small",
                    ),
                    html.Button(
                        "Sell",
                        id={"type": "card-sell", "symbol": symbol},
                        n_clicks=0,
                        type="button",
                        className="btn btn-sell btn-small",
                    ),
                ],
            ),
        ],
    )


APP_STYLE = """
:root {
    --paper: #f5f7fb;
    --ink: #1e1b39;
    --muted: #464646;
    --accent: #003da5;
    --accent-strong: #002f80;
    --navy: #1d2443;
    --up: #15cf74;
    --down: #e23b32;
    --surface: #ffffff;
    --line: #e4e7ee;
    --shadow: 0 8px 24px rgba(29, 36, 67, 0.08);
    --shadow-soft: 0 4px 16px rgba(29, 36, 67, 0.06);
    --topbar: #ffffff;
    --field: #ffffff;
    --pill: #eef2f8;
    --insight-bg: #eef3fb;
    --insight-line: #d5e0f2;
    --row-alt: #f5f7fb;
    --row-hover: #e8eef8;
    --note: #8a2d2d;
    color-scheme: light;
}

html[data-theme="dark"] {
    --paper: #0e1220;
    --ink: #e8edf7;
    --muted: #a7b0c6;
    --accent: #6ea2ff;
    --accent-strong: #9ec0ff;
    --navy: #243056;
    --surface: #171d30;
    --line: #2c3550;
    --shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    --shadow-soft: 0 4px 16px rgba(0, 0, 0, 0.28);
    --topbar: #12182a;
    --field: #101522;
    --pill: #1c2744;
    --insight-bg: #1a2744;
    --insight-line: #31456e;
    --row-alt: #12182a;
    --row-hover: #243056;
    --note: #ffb4ae;
    --Dash-Stroke-Strong: #3a4668;
    --Dash-Stroke-Weak: rgba(255, 255, 255, 0.08);
    --Dash-Fill-Interactive-Strong: #6ea2ff;
    --Dash-Fill-Interactive-Weak: rgba(110, 162, 255, 0.16);
    --Dash-Fill-Inverse-Strong: #171d30;
    --Dash-Text-Primary: #e8edf7;
    --Dash-Text-Strong: #e8edf7;
    --Dash-Text-Weak: #a7b0c6;
    --Dash-Text-Disabled: #7d879d;
    --Dash-Fill-Primary-Hover: rgba(255, 255, 255, 0.06);
    --Dash-Fill-Primary-Active: rgba(255, 255, 255, 0.1);
    --Dash-Fill-Disabled: rgba(255, 255, 255, 0.08);
    color-scheme: dark;
}

* { box-sizing: border-box; }

body {
    margin: 0;
    font-family: "Poppins", "Segoe UI", sans-serif;
    background: var(--paper);
    color: var(--ink);
}

.app-shell {
    min-height: 100vh;
    padding: 0 22px 60px;
}

.topbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin: 0 -22px 18px;
    padding: 16px 28px;
    background: var(--topbar);
    border-bottom: 1px solid var(--line);
}

.brand { display: flex; align-items: center; gap: 12px; }

.brand-logo {
    width: 54px;
    height: 54px;
    display: block;
    border-radius: 14px;
}

.brand-title { margin: 0; font-weight: 700; font-size: 22px; letter-spacing: -0.3px; line-height: 1.15; }
.brand-cse { color: #003da5; }
.brand-analyser { color: #0f9f5b; }
html[data-theme="dark"] .brand-cse { color: #7eb0ff; }
html[data-theme="dark"] .brand-analyser { color: #3ee08a; }
.brand-sub { margin: 0; color: var(--muted); font-size: 12px; }

.controls {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}

.btn {
    border: none;
    background: var(--accent);
    color: white;
    padding: 10px 18px;
    border-radius: 999px;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
}

.btn:hover { background: var(--accent-strong); }

.btn-buy { background: var(--up); color: #08301c; box-shadow: none; }
.btn-sell { background: var(--down); color: #ffffff; box-shadow: none; }
.btn-buy:hover { background: #12b868; box-shadow: none; }
.btn-sell:hover { background: #c92f28; box-shadow: none; }
.btn-ghost {
    background: var(--surface);
    color: var(--accent);
    border: 1px solid var(--line);
    box-shadow: none;
    margin-top: 10px;
}

.theme-toggle {
    display: inline-flex;
    align-items: center;
    border: none;
    border-radius: 999px;
    padding: 10px 16px;
    font-weight: 700;
    font-family: inherit;
    font-size: 14px;
    cursor: pointer;
    background: linear-gradient(135deg, #1d2443 0%, #003da5 100%);
    color: #ffffff;
    box-shadow: 0 6px 16px rgba(0, 61, 165, 0.28);
}

.theme-toggle:hover { filter: brightness(1.08); }

html[data-theme="dark"] .theme-toggle {
    background: linear-gradient(135deg, #ffe08a 0%, #fff7df 55%, #ffffff 100%);
    color: #1d2443;
    box-shadow: 0 6px 16px rgba(255, 214, 102, 0.25);
}

.theme-when-light,
.theme-when-dark {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.theme-when-dark { display: none; }
html[data-theme="dark"] .theme-when-light { display: none; }
html[data-theme="dark"] .theme-when-dark { display: inline-flex; }

.theme-icon { font-size: 15px; line-height: 1; }
.theme-icon-moon { color: #f6c453; }
.theme-icon-sun { color: #e08a00; }
.btn-small { padding: 8px 12px; flex: 1; border-radius: 8px; }

.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    background: var(--pill);
    color: var(--accent);
    font-weight: 600;
    font-size: 13px;
    padding: 8px 12px;
}

.pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--up);
    box-shadow: 0 0 0 0 rgba(21, 207, 116, 0.6);
    animation: pulse 1.8s ease-out infinite;
}

.quote-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 230px), 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

.stock-card {
    background: var(--navy);
    border: 1px solid var(--navy);
    border-radius: 10px;
    padding: 8px 8px 12px;
    box-shadow: none;
    min-width: 0;
    color: #ffffff;
}

.stock-card.is-selected {
    box-shadow: 0 0 0 2px var(--accent);
}

.stock-select {
    width: 100%;
    border: none;
    background: transparent;
    text-align: left;
    cursor: pointer;
    padding: 6px 8px 0;
    font-family: inherit;
    color: inherit;
}

.stock-top {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    align-items: flex-start;
}

.stock-name { font-weight: 600; font-size: 15px; color: #ffffff; }
.stock-symbol { color: rgba(255, 255, 255, 0.62); font-size: 12px; margin-top: 2px; }

.px {
    font-family: "Poppins", "Segoe UI", sans-serif;
    font-weight: 700;
    font-size: 28px;
    margin-top: 8px;
    letter-spacing: -0.4px;
    color: #ffffff;
}

.chg-up, .chg-down, .chg-flat { font-weight: 600; font-size: 13px; margin-top: 2px; }
.chg-up { color: var(--up); }
.chg-down { color: #ff6b63; }
.chg-flat { color: rgba(255, 255, 255, 0.7); }

.held {
    color: rgba(255, 255, 255, 0.72);
    font-size: 12px;
    padding: 0 8px 8px;
    min-height: 18px;
}

.card-trade {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 0 6px;
}

.card-trade input {
    width: 72px;
    border-radius: 8px;
    border: none;
    padding: 8px 10px;
    font-family: inherit;
    background: #ffffff;
    color: #1e1b39;
}

.badge {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.4px;
}
.badge:empty { display: none; }
.badge-buy { background: rgba(21, 207, 116, 0.18); color: var(--up); }
.badge-sell { background: rgba(226, 59, 50, 0.2); color: #ff8d86; }
.badge-wait { background: rgba(255, 255, 255, 0.12); color: #f6c453; }

.market-layout {
    display: grid;
    grid-template-columns: minmax(0, 1.7fr) minmax(280px, 0.78fr);
    gap: 16px;
    margin-bottom: 16px;
}

.ticket {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 16px;
    box-shadow: var(--shadow-soft);
}

.ticket .chart-subtitle { margin-bottom: 12px; }

.cash-label {
    font-weight: 600;
    margin-bottom: 8px;
}

.position-line {
    color: var(--muted);
    margin-bottom: 12px;
    min-height: 18px;
}

.ticket input {
    width: 100%;
    border-radius: 12px;
    border: 1px solid var(--line);
    padding: 10px 12px;
    font-family: inherit;
    margin-bottom: 12px;
    background: var(--field);
    color: var(--ink);
}

.ticket-actions { display: flex; gap: 8px; }
.ticket-actions .btn { flex: 1; }

.trade-status {
    margin-top: 12px;
    font-weight: 600;
    min-height: 20px;
}

.section-label {
    margin: 4px 0 10px;
    font-size: 18px;
    font-weight: 700;
    color: var(--ink);
}

.range-pills {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}

.range-pills { margin: 0 0 8px; }

.range-pills label {
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 6px 12px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    color: var(--muted);
    background: var(--surface);
}

.range-pills label.selected,
.range-pills label[aria-selected="true"] {
    background: var(--accent);
    color: #ffffff;
    border-color: var(--accent);
}

.range-pills .dash-options-list-option-wrapper,
.range-input {
    position: absolute;
    width: 0;
    height: 0;
    overflow: hidden;
    opacity: 0;
    pointer-events: none;
}

.chart-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px 16px 6px;
    box-shadow: var(--shadow-soft);
    margin-bottom: 16px;
}

.insight-banner {
    background: var(--insight-bg);
    border: 1px solid var(--insight-line);
    border-radius: 10px;
    padding: 14px 16px;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 14px;
}

.chart-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}

.chart-title {
    margin: 0;
    font-size: 18px;
}

.chart-subtitle {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 13px;
}

.tabs {
  margin-top: 18px;
}

.tab {
    background: var(--surface);
    border-radius: 10px 10px 0 0;
    border: 1px solid var(--line);
    padding: 12px 16px;
    font-weight: 600;
    color: var(--muted);
}

.tab--selected {
    background: var(--surface);
    border-bottom: 3px solid var(--accent);
    color: var(--accent);
}

.panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 18px 16px 24px;
}

.filter-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin: 12px 0 10px;
}

.filter-row input,
.filter-row .Select-control,
.filter-row .Select-menu-outer,
.filter-row .Select-placeholder,
.filter-row .Select-input,
.filter-row .Select-value-label {
    font-family: "Poppins", "Segoe UI", sans-serif;
}

.filter-row input,
.filter-row .Select-control,
.filter-row .dash-dropdown {
    border-radius: 12px !important;
    border: 1px solid var(--line) !important;
    background: var(--field) !important;
    color: var(--ink);
    min-height: 38px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
}

.filter-row input:focus,
.filter-row .Select-control:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0, 61, 165, 0.16);
}

.dash-table-container {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--line);
}

.dash-spreadsheet td,
.dash-spreadsheet th {
    border-color: var(--line) !important;
    font-family: "Poppins", "Segoe UI", sans-serif !important;
}

.dash-spreadsheet td {
    background-color: var(--surface) !important;
    color: var(--ink);
}

.dash-spreadsheet th {
    background-color: var(--navy) !important;
    color: #ffffff !important;
}

.dash-spreadsheet tr:nth-child(even) td {
    background-color: var(--row-alt) !important;
}

.dash-spreadsheet tr:hover td {
    background-color: var(--row-hover) !important;
}

.dash-dropdown-content,
.dash-dropdown-search-container,
.dash-dropdown-options {
    background: var(--field);
    color: var(--ink);
}

.note { color: var(--note); margin-bottom: 6px; }

html[data-theme="dark"] .previous-next-container,
html[data-theme="dark"] .page-number,
html[data-theme="dark"] .current-page,
html[data-theme="dark"] .dash-spreadsheet-menu {
    color: var(--ink);
}

html[data-theme="dark"] .previous-next-container button,
html[data-theme="dark"] .dash-spreadsheet-container input {
    background: var(--field);
    color: var(--ink);
    border-color: var(--line);
}
.count { margin-bottom: 10px; font-weight: 600; color: var(--muted); }

.count::before {
    content: "\25CF";
    color: var(--accent);
    margin-right: 8px;
}

@media (max-width: 980px) {
    .market-layout { grid-template-columns: 1fr; }
}

@media (max-width: 720px) {
    .topbar { padding: 16px; margin-left: -12px; margin-right: -12px; }
    .filter-row { gap: 8px; }
    .tab { padding: 10px 12px; }
    .panel { padding: 16px 12px; }
    .chart-card { padding: 12px; }
    .quote-grid { grid-template-columns: 1fr; }
}

@keyframes slideIn {
  from { transform: translateY(12px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(21, 207, 116, 0.55); }
  70% { box-shadow: 0 0 0 8px rgba(21, 207, 116, 0); }
  100% { box-shadow: 0 0 0 0 rgba(21, 207, 116, 0); }
}
"""

app = Dash(__name__)
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>CSE Analyser</title>
        <script>
        try {
            var savedTheme = JSON.parse(localStorage.getItem("theme-mode"));
            if (savedTheme && savedTheme.data) savedTheme = savedTheme.data;
            if (savedTheme === "dark") document.documentElement.setAttribute("data-theme", "dark");
        } catch (e) {}
        </script>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
        """ + APP_STYLE + """
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="topbar",
            children=[
                html.Div(
                    className="brand",
                    children=[
                        html.Img(src="/assets/logo.svg", className="brand-logo", alt="CSE Analyser"),
                        html.Div(
                            children=[
                                html.Div(
                                    className="brand-title",
                                    children=[
                                        html.Span("CSE", className="brand-cse"),
                                        " ",
                                        html.Span("Analyser", className="brand-analyser"),
                                    ],
                                ),
                                html.Div("Live share prices. Paper trades stay on this computer.", className="brand-sub"),
                            ]
                        ),
                    ],
                ),
                html.Div(
                    className="controls",
                    children=[
                        html.Div(className="live-pill", children=[html.Span(className="pulse"), "Live"]),
                        html.Button("Refresh", id="refresh-btn", n_clicks=0, className="btn"),
                        html.Button(
                            id="theme-toggle",
                            n_clicks=0,
                            type="button",
                            className="theme-toggle",
                            title="Switch to dark mode",
                            children=[
                                html.Span(
                                    className="theme-when-light",
                                    children=[html.Span("☾", className="theme-icon theme-icon-moon"), "Dark"],
                                ),
                                html.Span(
                                    className="theme-when-dark",
                                    children=[html.Span("☀", className="theme-icon theme-icon-sun"), "Light"],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        dcc.Interval(id="refresh-interval", interval=10_000, n_intervals=0),
        dcc.Store(id="theme-mode", data="light", storage_type="local"),
        dcc.Store(id="selected-symbol", data=MARKET[0]["symbol"]),
        dcc.Store(id="book-version", data=0),
        dcc.Tabs(
            className="tabs",
            children=[
                dcc.Tab(
                    label="Live shares",
                    className="tab",
                    selected_className="tab--selected",
                    children=[
                        html.Div(
                            className="panel",
                            children=[
                                html.Div("Each share", className="section-label"),
                                html.Div(className="quote-grid", children=[stock_card(company) for company in MARKET]),
                                html.Div(
                                    className="market-layout",
                                    children=[
                                        html.Div(
                                            className="chart-card",
                                            children=[
                                                html.Div(
                                                    className="chart-header",
                                                    children=[
                                                        html.Div(
                                                            children=[
                                                                html.H3(id="market-title", className="chart-title", children=MARKET[0]["name"]),
                                                                html.P(
                                                                    id="market-subtitle",
                                                                    className="chart-subtitle",
                                                                    children="Daily price bars. The latest point follows the live last trade.",
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                                dcc.RadioItems(
                                                    id="chart-range",
                                                    options=[
                                                        {"label": "1M", "value": 31},
                                                        {"label": "3M", "value": 93},
                                                        {"label": "6M", "value": 186},
                                                        {"label": "1Y", "value": 380},
                                                    ],
                                                    value=186,
                                                    className="range-pills",
                                                    inputClassName="range-input",
                                                    labelClassName="range-label",
                                                    inline=True,
                                                ),
                                                dcc.Graph(
                                                    id="market-chart",
                                                    config={"displayModeBar": False},
                                                    style={"height": "440px"},
                                                    figure=price_figure(
                                                        MARKET[0]["symbol"],
                                                        history_for(MARKET[0]["symbol"]),
                                                        186,
                                                        None,
                                                        f"{MARKET[0]['symbol']}:186",
                                                    ),
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="ticket",
                                            children=[
                                                html.H3("Paper order", className="chart-title"),
                                                html.P(
                                                    "Buy and sell the selected share at the latest last trade. "
                                                    "This book stays on this computer and is not sent to a broker.",
                                                    className="chart-subtitle",
                                                ),
                                                html.Div(id="cash-label", className="cash-label", children="Cash LKR 1,000,000.00"),
                                                html.Div(id="position-line", className="position-line", children="No shares held"),
                                                dcc.Input(
                                                    id="order-qty",
                                                    type="number",
                                                    min=1,
                                                    step=1,
                                                    value=100,
                                                    placeholder="Shares",
                                                ),
                                                html.Div(
                                                    className="ticket-actions",
                                                    children=[
                                                        html.Button("Buy", id="buy-btn", n_clicks=0, type="button", className="btn btn-buy"),
                                                        html.Button("Sell", id="sell-btn", n_clicks=0, type="button", className="btn btn-sell"),
                                                    ],
                                                ),
                                                html.Div(id="trade-status", className="trade-status"),
                                                html.Button(
                                                    "Reset paper book",
                                                    id="reset-book",
                                                    n_clicks=0,
                                                    type="button",
                                                    className="btn btn-ghost",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div("Open positions", className="section-label"),
                                dash_table.DataTable(
                                    id="positions-table",
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "minWidth": "90px", "padding": "8px"},
                                    style_header={
                                        "backgroundColor": "#1d2443",
                                        "color": "#ffffff",
                                        "fontWeight": "600",
                                        "border": "1px solid #1d2443",
                                    },
                                    style_data_conditional=[
                                        {
                                            "if": {"filter_query": "{pnl} > 0", "column_id": "pnl"},
                                            "color": "#0f9f5b",
                                            "fontWeight": "600",
                                        },
                                        {
                                            "if": {"filter_query": "{pnl} < 0", "column_id": "pnl"},
                                            "color": "#c24141",
                                            "fontWeight": "600",
                                        },
                                    ],
                                ),
                                html.Div("Recent fills", className="section-label"),
                                dash_table.DataTable(
                                    id="fills-table",
                                    page_size=8,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "minWidth": "90px", "padding": "8px"},
                                    style_header={
                                        "backgroundColor": "#1d2443",
                                        "color": "#ffffff",
                                        "fontWeight": "600",
                                        "border": "1px solid #1d2443",
                                    },
                                    style_data_conditional=[
                                        {
                                            "if": {"filter_query": '{side} = "BUY"', "column_id": "side"},
                                            "color": "#0f9f5b",
                                            "fontWeight": "600",
                                        },
                                        {
                                            "if": {"filter_query": '{side} = "SELL"', "column_id": "side"},
                                            "color": "#c24141",
                                            "fontWeight": "600",
                                        },
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="News",
                    className="tab",
                    selected_className="tab--selected",
                    children=[
                        html.Div(
                            className="panel",
                            children=[
                                html.Div(
                                    className="filter-row",
                                    children=[
                                        dcc.Dropdown(id="news-year", placeholder="Year", style={"width": "160px"}),
                                        dcc.Input(id="news-company", type="text", placeholder="Company"),
                                        dcc.Input(id="news-keyword", type="text", placeholder="Keyword"),
                                    ],
                                ),
                                html.Div(id="news-filter-note", className="note"),
                                html.Div(id="news-count", className="count"),
                                dash_table.DataTable(
                                    id="news-table",
                                    page_size=20,
                                    filter_action="native",
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "minWidth": "120px", "maxWidth": "300px"},
                                    style_header={
                                        "backgroundColor": "#1d2443",
                                        "color": "#ffffff",
                                        "fontWeight": "600",
                                        "border": "1px solid #1d2443",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Predictions",
                    className="tab",
                    selected_className="tab--selected",
                    children=[
                        html.Div(
                            className="panel",
                            children=[
                                html.Div(id="pred-insight", className="insight-banner"),
                                html.Div(
                                    className="chart-card",
                                    children=[
                                        html.Div(
                                            className="chart-header",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.H3("Share snapshot", className="chart-title"),
                                                        html.P(
                                                            "Scores, mentions, and sentiment mix for a selected share.",
                                                            className="chart-subtitle",
                                                        ),
                                                    ]
                                                ),
                                                dcc.Dropdown(
                                                    id="pred-company-graph",
                                                    placeholder="Select share",
                                                    style={"width": "260px"},
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(
                                            id="pred-graph",
                                            config={"displayModeBar": False},
                                            style={"height": "320px"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="filter-row",
                                    children=[
                                        dcc.Input(id="pred-name", type="text", placeholder="Company"),
                                        dcc.Dropdown(
                                            id="pred-risk",
                                            placeholder="Risk",
                                            options=[
                                                {"label": "Low", "value": "low"},
                                                {"label": "Medium", "value": "medium"},
                                                {"label": "High", "value": "high"},
                                            ],
                                            style={"width": "160px"},
                                        ),
                                        dcc.Input(id="pred-score-min", type="number", placeholder="Min score"),
                                        dcc.Input(id="pred-score-max", type="number", placeholder="Max score"),
                                        dcc.Input(id="pred-price-min", type="number", placeholder="Min price"),
                                        dcc.Input(id="pred-price-max", type="number", placeholder="Max price"),
                                    ],
                                ),
                                html.Div(id="pred-filter-note", className="note"),
                                html.Div(id="pred-count", className="count"),
                                dash_table.DataTable(
                                    id="pred-table",
                                    page_size=20,
                                    filter_action="native",
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "minWidth": "120px", "maxWidth": "300px"},
                                    style_header={
                                        "backgroundColor": "#1d2443",
                                        "color": "#ffffff",
                                        "fontWeight": "600",
                                        "border": "1px solid #1d2443",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Files",
                    className="tab",
                    selected_className="tab--selected",
                    children=[
                        html.Div(
                            className="panel",
                            children=[
                                html.Div(
                                    className="filter-row",
                                    children=[
                                        dcc.Dropdown(id="file-select", placeholder="Select file", style={"width": "320px"}),
                                        dcc.Input(id="file-company", type="text", placeholder="Company"),
                                    ],
                                ),
                                html.Div(id="file-filter-note", className="note"),
                                html.Div(id="file-count", className="count"),
                                dash_table.DataTable(
                                    id="file-table",
                                    page_size=20,
                                    filter_action="native",
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "minWidth": "120px", "maxWidth": "300px"},
                                    style_header={
                                        "backgroundColor": "#1d2443",
                                        "color": "#ffffff",
                                        "fontWeight": "600",
                                        "border": "1px solid #1d2443",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        ),
    ],
)


@app.callback(
    Output("news-year", "options"),
    Output("news-table", "data"),
    Output("news-table", "columns"),
    Output("news-count", "children"),
    Output("news-filter-note", "children"),
    Output("pred-table", "data"),
    Output("pred-table", "columns"),
    Output("pred-count", "children"),
    Output("pred-filter-note", "children"),
    Output("pred-insight", "children"),
    Output("pred-company-graph", "options"),
    Output("pred-company-graph", "value"),
    Output("pred-graph", "figure"),
    Output("file-select", "options"),
    Output("file-select", "value"),
    Output("file-table", "data"),
    Output("file-table", "columns"),
    Output("file-filter-note", "children"),
    Output("file-count", "children"),
    Input("refresh-btn", "n_clicks"),
    Input("refresh-interval", "n_intervals"),
    Input("pred-company-graph", "value"),
    Input("theme-mode", "data"),
    State("news-year", "value"),
    State("news-company", "value"),
    State("news-keyword", "value"),
    State("pred-name", "value"),
    State("pred-risk", "value"),
    State("pred-score-min", "value"),
    State("pred-score-max", "value"),
    State("pred-price-min", "value"),
    State("pred-price-max", "value"),
    State("file-select", "value"),
    State("file-company", "value"),
)
def refresh_all(
    _refresh_clicks: int,
    _refresh_ticks: int,
    pred_company_graph: str | None,
    theme: str | None,
    news_year: int | None,
    news_company: str | None,
    news_keyword: str | None,
    pred_name: str | None,
    pred_risk: str | None,
    pred_score_min: float | None,
    pred_score_max: float | None,
    pred_price_min: float | None,
    pred_price_max: float | None,
    file_value: str | None,
    file_company: str | None,
):
    data_map = load_all_data()

    news_all = ensure_year_column(build_news_df(data_map))
    news_year_options = []
    if not news_all.empty and "_year" in news_all.columns:
        years = sorted(int(year) for year in news_all["_year"].dropna().unique())
        news_year_options = [{"label": str(year), "value": year} for year in years]
    news_df, news_notes = filter_news(news_all, news_year, news_keyword, news_company)
    news_view = present_news(news_df)

    pred_df = build_predictions_df(data_map)
    pred_df, pred_notes = filter_predictions(
        pred_df,
        pred_name,
        pred_risk,
        pred_score_min,
        pred_score_max,
        pred_price_min,
        pred_price_max,
    )

    pred_insight = build_insight_message(pred_df)

    pred_figure, pred_company_opts, pred_company_value = build_company_figure(
        pred_df,
        pred_company_graph,
        theme,
    )

    file_opts = file_options(data_map)
    if not file_value and file_opts:
        file_value = file_opts[0]["value"]
    file_df = data_map.get(file_value, pd.DataFrame()) if file_value else pd.DataFrame()
    file_df, file_notes = filter_file_df(file_df, file_company)
    file_total = len(file_df)
    file_view = prepare_file_table(file_df)

    news_count = f"Rows: {len(news_df):,}" if not news_df.empty else "No news rows"
    pred_count = f"Rows: {len(pred_df):,}" if not pred_df.empty else "No prediction rows"
    if file_total == 0:
        file_count = "No file rows"
    elif file_total > len(file_view):
        file_count = f"Rows: {file_total:,} (showing {len(file_view):,})"
    else:
        file_count = f"Rows: {file_total:,}"

    return (
        news_year_options,
        frame_records(news_view),
        to_table_columns(news_view),
        news_count,
        " ".join(news_notes),
        frame_records(pred_df),
        to_table_columns(pred_df),
        pred_count,
        " ".join(pred_notes),
        pred_insight,
        pred_company_opts,
        pred_company_value,
        pred_figure,
        file_opts,
        file_value,
        frame_records(file_view),
        to_table_columns(file_view),
        " ".join(file_notes),
        file_count,
    )


@app.callback(
    Output("selected-symbol", "data"),
    Input({"type": "select-stock", "symbol": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_stock(_clicks):
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or not triggered.get("symbol"):
        return no_update
    return triggered["symbol"]


@app.callback(
    Output("trade-status", "children"),
    Output("book-version", "data"),
    Input("buy-btn", "n_clicks"),
    Input("sell-btn", "n_clicks"),
    Input("reset-book", "n_clicks"),
    Input({"type": "card-buy", "symbol": ALL}, "n_clicks"),
    Input({"type": "card-sell", "symbol": ALL}, "n_clicks"),
    State("selected-symbol", "data"),
    State("order-qty", "value"),
    State({"type": "card-qty", "symbol": ALL}, "value"),
    State({"type": "card-qty", "symbol": ALL}, "id"),
    State("book-version", "data"),
    prevent_initial_call=True,
)
def trade(_buy, _sell, _reset, _card_buys, _card_sells, selected, qty, card_qtys, card_qty_ids, version):
    triggered = ctx.triggered_id
    version = int(version or 0)
    if triggered == "reset-book":
        return reset_book(), version + 1
    if triggered == "buy-btn":
        symbol, side, amount = selected, "BUY", qty
    elif triggered == "sell-btn":
        symbol, side, amount = selected, "SELL", qty
    elif isinstance(triggered, dict) and triggered.get("type") in {"card-buy", "card-sell"}:
        symbol = triggered["symbol"]
        side = "BUY" if triggered["type"] == "card-buy" else "SELL"
        amount = next(
            (value for value, ident in zip(card_qtys or [], card_qty_ids or []) if ident.get("symbol") == symbol),
            None,
        )
    else:
        return no_update, no_update
    quotes = fetch_live_quotes()
    price, source = tradable_price(symbol, quotes)
    message = place_order(symbol, side, amount, price, source)
    if message.startswith("Bought") or message.startswith("Sold"):
        return message, version + 1
    return message, no_update


@app.callback(
    Output({"type": "live-price", "symbol": ALL}, "children"),
    Output({"type": "live-change", "symbol": ALL}, "children"),
    Output({"type": "live-change", "symbol": ALL}, "className"),
    Output({"type": "mini-chart", "symbol": ALL}, "figure"),
    Output({"type": "held", "symbol": ALL}, "children"),
    Output({"type": "stock-card", "symbol": ALL}, "className"),
    Output({"type": "model-badge", "symbol": ALL}, "children"),
    Output({"type": "model-badge", "symbol": ALL}, "className"),
    Output("market-chart", "figure"),
    Output("market-title", "children"),
    Output("market-subtitle", "children"),
    Output("cash-label", "children"),
    Output("position-line", "children"),
    Output("positions-table", "data"),
    Output("positions-table", "columns"),
    Output("fills-table", "data"),
    Output("fills-table", "columns"),
    Input("refresh-interval", "n_intervals"),
    Input("refresh-btn", "n_clicks"),
    Input("selected-symbol", "data"),
    Input("book-version", "data"),
    Input("chart-range", "value"),
    Input("theme-mode", "data"),
    State({"type": "live-price", "symbol": ALL}, "id"),
)
def update_market(_ticks, _clicks, selected, _version, chart_range, theme, price_ids):
    symbols = [item["symbol"] for item in price_ids]
    if selected not in symbols and symbols:
        selected = symbols[0]
    quotes = fetch_live_quotes()
    book = load_book()
    positions = replay_positions(book)
    models = model_by_symbol()
    names = {row["symbol"]: row["name"] for row in MARKET}

    prices, changes, change_cls = [], [], []
    figures, helds, cards, badges, badge_cls = [], [], [], [], []
    for symbol in symbols:
        quote = quotes.get(symbol) or {}
        price = quote.get("price")
        change_pct = quote.get("change_pct")
        text, css = _fmt_chg(change_pct)
        prices.append(_fmt_px(price) if price is not None else "—")
        changes.append(text if price is not None else "Feed unavailable")
        change_cls.append(css if price is not None else "chg-flat")
        session_up = None if change_pct is None else float(change_pct) >= 0
        figures.append(mini_figure(with_live_bar(history_for(symbol), price), session_up))
        helds.append(held_text(symbol, positions, quote))
        cards.append("stock-card is-selected" if symbol == selected else "stock-card")
        action = models.get(symbol, "")
        badges.append(action)
        badge_cls.append(badge_class(action))

    name = names.get(selected, selected or "")
    quote = quotes.get(selected) or {}
    price = quote.get("price")
    change_text, _css = _fmt_chg(quote.get("change_pct"))
    window_days = int(chart_range or 186)
    chart = price_figure(
        selected or "",
        with_live_bar(history_for(selected), price),
        window_days,
        quote.get("change_pct") if price is not None else None,
        f"{selected}:{window_days}:{theme or 'light'}",
        theme,
    )
    asof = datetime.now(COLOMBO).strftime("%H:%M:%S")
    if price is None:
        subtitle = f"{selected} · waiting for a live price · {asof}"
    else:
        subtitle = f"{selected} · LKR {_fmt_px(price)} · {change_text} · updated {asof}"
    cash, equity, pnl = book_summary(book, quotes, positions)
    cash_label = f"Cash LKR {cash:,.2f}  ·  Equity LKR {equity:,.2f}  ·  Open P&L {pnl:+,.2f}"
    position_rows, position_cols = position_table(positions, quotes, names)
    fills, fill_cols = fill_table(book)
    return (
        prices,
        changes,
        change_cls,
        figures,
        helds,
        cards,
        badges,
        badge_cls,
        chart,
        name,
        subtitle,
        cash_label,
        held_text(selected, positions, quote) if selected else "",
        position_rows,
        position_cols,
        fills,
        fill_cols,
    )


app.clientside_callback(
    """
    function (_clicks, theme) {
        return theme === "dark" ? "light" : "dark";
    }
    """,
    Output("theme-mode", "data"),
    Input("theme-toggle", "n_clicks"),
    State("theme-mode", "data"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function (theme) {
        var mode = theme === "dark" ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", mode);
        return mode === "dark" ? "Switch to light mode" : "Switch to dark mode";
    }
    """,
    Output("theme-toggle", "title"),
    Input("theme-mode", "data"),
)


if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=False)
