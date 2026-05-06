from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf


DEFAULT_SYMBOL = "ACL.N0000"
DEFAULT_COMPANY_NAME = "Commercial Bank"
OUTPUT_DIR = Path("cse_output")
NEWS_CSV = OUTPUT_DIR / "cse_news_ALL_2021_2026.csv"

COMPANY_SYMBOLS = {
    "commercial bank": "COMB.N0000",
    "hatton national bank": "HNB.N0000",
    "sampath bank": "SAMP.N0000",
    "seylan bank": "SEYB.N0000",
    "bank of ceylon": "BOC.N0000",
    "dfcc bank": "DFCC.N0000",
    "nations trust bank": "NTB.N0000",
    "pan asia bank": "PABC.N0000",
    "john keells holdings": "JKH.N0000",
    "dialog axiata": "DIAL.N0000",
}


def resolve_symbol(company_name: str) -> str | None:
    if not company_name:
        return None
    key = company_name.lower().strip()
    return COMPANY_SYMBOLS.get(key)


def fetch_live_price(symbol: str):
    print(f"Fetching live price for {symbol} from CSE API...")
    try:
        url = "https://www.cse.lk/api/tradeSummary"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.post(url, data={"symbol": symbol}, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        for item in data.get("reqTradeSummery", []):
            if item.get("symbol") == symbol and item.get("price"):
                price = float(item["price"])
                print(f"[OK] Live price: {price:.2f} LKR")
                return price
    except Exception as exc:
        print(f"[!] Live price lookup failed: {exc}")
    return None


def fetch_price_history(symbol: str) -> pd.DataFrame:
    print(f"Fetching 1-year price history for {symbol}...")
    try:
        df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                close_level = df.columns.get_level_values(0)
                if "Close" in close_level:
                    df = df.xs("Close", axis=1, level=0).to_frame(name="Close")
                else:
                    df.columns = df.columns.get_level_values(-1)
            df = df.dropna().copy()
            df.index = pd.to_datetime(df.index)
            print(f"[OK] Retrieved {len(df):,} price rows from Yahoo Finance.")
            return df
    except Exception as exc:
        print(f"[!] Price history fetch failed: {exc}")

    print("[!] Falling back to synthetic history anchored to the live price.")
    live_price = fetch_live_price(symbol) or 100.0
    dates = pd.date_range(end=datetime.today(), periods=252, freq="B")
    np.random.seed(42)
    random_walk = np.random.randn(252).cumsum()
    prices = random_walk - random_walk[-1] + live_price
    return pd.DataFrame({"Close": prices}, index=dates)


def load_company_events(company_name: str) -> pd.DataFrame:
    if not NEWS_CSV.exists():
        print(f"[!] News file not found: {NEWS_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(NEWS_CSV, encoding="utf-8-sig", low_memory=False)
    text_cols = ["company", "subject", "description", "heading", "title", "remarks"]
    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        if col in df.columns:
            mask = mask | df[col].fillna("").astype(str).str.contains(company_name, case=False, na=False)
    events = df.loc[mask].copy()

    date_source = None
    for col in ["_parsed_date", "createdDate", "announcementDate", "publishedDate", "date"]:
        if col in events.columns and events[col].notna().any():
            date_source = col
            break

    if date_source is None:
        print("[!] No parseable event date column found.")
        return pd.DataFrame()

    events["event_date"] = pd.to_datetime(events[date_source], errors="coerce")
    events = events.dropna(subset=["event_date"]).sort_values("event_date").copy()
    events["event_text"] = events[[c for c in text_cols if c in events.columns]].fillna("").agg(" ".join, axis=1)
    return events


def classify_event(row: pd.Series):
    text = str(row.get("event_text", "")).lower()
    category = str(row.get("_category", "")).lower()

    if "non-compliance" in category or "non-compliance" in text:
        return "negative", "crimson", -1
    if "financial" in category or any(word in text for word in ["profit", "results", "dividend", "revenue", "earnings"]):
        return "positive", "green", 1
    if "new listings" in category or any(word in text for word in ["debenture", "bond", "ipo", "allotment", "listing"]):
        return "positive", "darkgreen", 1
    return "neutral", "gray", 0


def build_forecast_note(price_df: pd.DataFrame, events: pd.DataFrame) -> str:
    close_series = price_df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze("columns")
    latest = float(close_series.iloc[-1])
    recent_20 = close_series.tail(20)
    recent_60 = close_series.tail(60)
    short_return = (recent_20.iloc[-1] / recent_20.iloc[0] - 1) * 100 if len(recent_20) > 1 else 0.0
    medium_return = (recent_60.iloc[-1] / recent_60.iloc[0] - 1) * 100 if len(recent_60) > 1 else 0.0

    score = 0
    if short_return > 3:
        score += 1
    elif short_return < -3:
        score -= 1
    if medium_return > 5:
        score += 1
    elif medium_return < -5:
        score -= 1

    if not events.empty:
        recent_events = events[events["event_date"] >= (price_df.index.max() - pd.Timedelta(days=120))]
        event_score = sum(classify_event(row)[2] for _, row in recent_events.iterrows()) if not recent_events.empty else 0
        if event_score > 0:
            score += 1
        elif event_score < 0:
            score -= 1

    if score >= 2:
        outlook = "Bullish bias"
        idea = "price can keep improving if the next announcements stay positive"
    elif score == 1:
        outlook = "Mildly bullish"
        idea = "chart is improving, but upside depends on fresh earnings or corporate news"
    elif score == 0:
        outlook = "Neutral"
        idea = "expect range-bound movement unless a strong announcement breaks the pattern"
    else:
        outlook = "Cautious"
        idea = "weak price momentum plus event risk suggests a possible pullback"

    return (
        f"Latest close: {latest:.2f} LKR | 20D return: {short_return:+.2f}% | 60D return: {medium_return:+.2f}%\n"
        f"Forecast view: {outlook}. {idea}."
    )


def main() -> None:
    print("\nCSE chart analysis\n")
    company_name = input(f"Enter company name (default: {DEFAULT_COMPANY_NAME}): ").strip()
    if not company_name:
        company_name = DEFAULT_COMPANY_NAME
    symbol = input(f"Enter symbol (leave blank to auto-detect, default: {DEFAULT_SYMBOL}): ").strip()
    if not symbol:
        symbol = resolve_symbol(company_name) or DEFAULT_SYMBOL

    print(f"\n{company_name} chart analysis for {symbol}\n")

    price_df = fetch_price_history(symbol)
    events = load_company_events(company_name)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(price_df.index, price_df["Close"], color="#1f4e79", linewidth=2.2, label="Close")

    if not events.empty:
        events = events[(events["event_date"] >= price_df.index.min()) & (events["event_date"] <= price_df.index.max())].copy()
        for _, row in events.iterrows():
            sentiment, color, direction = classify_event(row)
            event_date = row["event_date"]
            nearest_idx = price_df.index.get_indexer([event_date], method="nearest")[0]
            price_at_event = float(price_df.iloc[nearest_idx]["Close"])
            ax.scatter(event_date, price_at_event, color=color, s=55, zorder=5, marker="^" if direction >= 0 else "v")

        top_events = events.tail(6)
        for i, (_, row) in enumerate(top_events.iterrows()):
            event_date = row["event_date"]
            nearest_idx = price_df.index.get_indexer([event_date], method="nearest")[0]
            price_at_event = float(price_df.iloc[nearest_idx]["Close"])
            label = str(row.get("remarks", row.get("title", "Event")))[:58]
            ax.annotate(
                f"{event_date:%Y-%m-%d}\n{label}",
                xy=(event_date, price_at_event),
                xytext=(8, 12 + (i % 3) * 16),
                textcoords="offset points",
                fontsize=8,
                color="#333333",
                arrowprops={"arrowstyle": "->", "color": "#999999", "lw": 0.6},
            )

    ax.set_title(f"{company_name} price chart with announcement markers", fontsize=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (LKR)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    print(build_forecast_note(price_df, events))
    if not events.empty:
        print(f"Announcement markers on chart: {len(events)}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()