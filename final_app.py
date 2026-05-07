from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from transformers import pipeline

OUTPUT_DIR = Path("cse_output")
COMPACT_DIR = OUTPUT_DIR / "compact"
NEWS_GLOB = "cse_news_*.csv"

DEFAULT_NEWS_ROWS = 300
DEFAULT_OTHER_ROWS = 1500
DEFAULT_HEADLINES = 10
DEFAULT_COMPANY = "Commercial Bank"
DEFAULT_SYMBOL = "COMB.N0000"

TEXT_COLUMNS = [
    "title",
    "remarks",
    "subject",
    "description",
    "announcementCategory",
    "category",
    "announcementType",
    "type",
]

DATE_COLUMNS = [
    "createdDate",
    "publishedDate",
    "announcementDate",
    "date",
    "_parsed_date",
]


def find_news_files() -> list[Path]:
    files = sorted(OUTPUT_DIR.glob(NEWS_GLOB))
    return [path for path in files if path.is_file()]


def pick_date_column(df: pd.DataFrame) -> str | None:
    for col in DATE_COLUMNS:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def build_news_text(df: pd.DataFrame) -> pd.Series:
    available = [col for col in TEXT_COLUMNS if col in df.columns]
    if not available:
        return pd.Series([""] * len(df), index=df.index)
    text_df = df[available].fillna("").astype(str)
    text_df = text_df.replace("nan", "")
    return text_df.bfill(axis=1).iloc[:, 0].str.strip()


def load_news_sources(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df["_source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["news_text"] = build_news_text(df)
    df = df[df["news_text"].str.len() > 0]
    date_col = pick_date_column(df)
    if date_col:
        df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values("_parsed_date", ascending=False)
    df = df.drop_duplicates(subset=["news_text"], keep="first")
    return df


def reduce_dataframe(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if df.empty:
        return df
    date_col = pick_date_column(df)
    if date_col:
        df = df.copy()
        df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values("_parsed_date", ascending=False)
    return df.head(max_rows).reset_index(drop=True)


def write_compact_outputs(news_df: pd.DataFrame, max_news_rows: int, max_other_rows: int) -> None:
    COMPACT_DIR.mkdir(parents=True, exist_ok=True)
    if not news_df.empty:
        compact_news = news_df.copy()
        keep_cols = [col for col in ["news_text", "company", "_parsed_date"] if col in compact_news.columns]
        if keep_cols:
            compact_news = compact_news[keep_cols]
        compact_news = reduce_dataframe(compact_news, max_news_rows)
        compact_news.to_csv(COMPACT_DIR / "cse_news_compact.csv", index=False)
        compact_news.to_json(COMPACT_DIR / "cse_news_compact.json", orient="records", indent=2)

    for path in OUTPUT_DIR.glob("*.csv"):
        if path.name == "cse_news_compact.csv":
            continue
        df = pd.read_csv(path, low_memory=False)
        df = reduce_dataframe(df, max_other_rows)
        out_name = path.stem + "_compact.csv"
        df.to_csv(COMPACT_DIR / out_name, index=False)


def fetch_price_history(symbol: str) -> pd.DataFrame:
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
        return df

    dates = pd.date_range(end=datetime.today(), periods=252, freq="B")
    rng = np.random.default_rng(42)
    random_walk = rng.standard_normal(252).cumsum()
    prices = random_walk - random_walk[-1] + 100.0
    return pd.DataFrame({"Close": prices}, index=dates)


def load_company_events(news_df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame()
    df = news_df.copy()
    text_col = "news_text"
    if text_col not in df.columns:
        return pd.DataFrame()
    mask = df[text_col].str.contains(company_name, case=False, na=False)
    events = df.loc[mask].copy()
    if events.empty:
        return events
    if "_parsed_date" not in events.columns:
        date_col = pick_date_column(events)
        if date_col:
            events["_parsed_date"] = pd.to_datetime(events[date_col], errors="coerce")
    events = events.dropna(subset=["_parsed_date"]).sort_values("_parsed_date")
    events = events.rename(columns={"_parsed_date": "event_date"})
    return events


def build_forecast_note(price_df: pd.DataFrame, events: pd.DataFrame) -> str:
    close_series = price_df["Close"].squeeze("columns")
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
        if not recent_events.empty:
            score += 1

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


def render_chart(company_name: str, symbol: str, news_df: pd.DataFrame) -> None:
    price_df = fetch_price_history(symbol)
    events = load_company_events(news_df, company_name)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(price_df.index, price_df["Close"], color="#1f4e79", linewidth=2.2, label="Close")

    if not events.empty:
        events = events[(events["event_date"] >= price_df.index.min()) & (events["event_date"] <= price_df.index.max())]
        for _, row in events.iterrows():
            event_date = row["event_date"]
            nearest_idx = price_df.index.get_indexer([event_date], method="nearest")[0]
            price_at_event = float(price_df.iloc[nearest_idx]["Close"])
            ax.scatter(event_date, price_at_event, color="#64748b", s=45, zorder=5)

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


def run_dashboard(args: argparse.Namespace) -> None:
    print("Loading AI model (finbert). This may take a minute the first time.")
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    news_files = find_news_files()
    if not news_files:
        raise FileNotFoundError("No cse_news_*.csv files found in cse_output.")

    news_df = normalize_news(load_news_sources(news_files))
    if news_df.empty:
        raise ValueError("No news rows with usable text were found.")

    if not args.skip_compact:
        write_compact_outputs(news_df, args.max_news_rows, args.max_other_rows)

    print("\n" + "=" * 60)
    print("CSE AI NEWS SENTIMENT + FUTURE CHART")
    print("=" * 60)

    headline_df = news_df.head(args.headlines)
    for _, row in headline_df.iterrows():
        text = str(row["news_text"]).strip()
        if not text:
            continue
        result = sentiment_pipeline(text)[0]
        label = result["label"].upper()
        score = round(float(result["score"]), 2)
        date_value = row.get("_parsed_date")
        date_str = date_value.strftime("%Y-%m-%d") if pd.notna(date_value) else ""
        date_display = f" | Date: {date_str}" if date_str else ""
        print(f"\n[{label}] (Confidence: {score}){date_display}")
        print(f"News: {text}")

    if not args.skip_chart:
        print("\nGenerating future chart analysis...")
        render_chart(args.company, args.symbol, news_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSE final analyzer output")
    parser.add_argument("--max-news-rows", type=int, default=DEFAULT_NEWS_ROWS)
    parser.add_argument("--max-other-rows", type=int, default=DEFAULT_OTHER_ROWS)
    parser.add_argument("--headlines", type=int, default=DEFAULT_HEADLINES)
    parser.add_argument("--company", type=str, default=DEFAULT_COMPANY)
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--skip-chart", action="store_true")
    parser.add_argument("--skip-compact", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_dashboard(parse_args())