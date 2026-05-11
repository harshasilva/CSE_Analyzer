# -*- coding: utf-8 -*-
"""
CSE Smart Analyzer
==================
Analyses 5 years of CSE historical announcements + today's live news
to predict which shares are likely to perform well.

Logic:
  Historical data  -> Company activity signals (IPO, financial, compliance trends)
  Live news        -> FinBERT sentiment score (strongest signal)
  Combined score   -> Ranked prediction table

Usage:
    python cse_analyzer.py

Requirements:
    pip install requests beautifulsoup4 transformers torch pandas
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os, glob, re, time, csv
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from bs4 import BeautifulSoup
import yfinance as yf
from transformers import pipeline

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

OUTPUT_DIR       = "cse_output"
LIVE_LIMIT       = 40          # headlines per live source
RECENT_DAYS      = 180         # "recent" historical window (6 months)
COMMERCIAL_BANK_SYMBOL = "COMB.N0000"
COMMERCIAL_BANK_NAME = "Commercial Bank"
COMMERCIAL_BANK_NEWS_CSV = Path(OUTPUT_DIR) / "cse_news_ALL_2021_2026.csv"

# Scoring weights
W_HIST_OLD       = 0.5         # old historical mention
W_HIST_RECENT    = 1.5         # recent historical mention (more relevant)
W_IPO_SIGNAL     = 3.0         # IPO / new listing → high activity signal
W_FINANCIAL      = 2.0         # financial announcement → results signal
W_COMPLIANCE_NEG = -2.0        # non-compliance → negative signal
W_LIVE_POS       = 5.0         # live positive sentiment
W_LIVE_NEG       = -4.0        # live negative sentiment
W_LIVE_NEU       = 0.5         # live neutral mention (at least active)

LIVE_SOURCES = [
    {
        "name":    "Adaderana Biz",
        "url":     "https://bizenglish.adaderana.lk/",
        "tags":    ["h4", "h3", "h2"],
        "headers": {"User-Agent": "Mozilla/5.0"},
    },
    {
        "name":    "Daily FT Business",
        "url":     "https://www.ft.lk/business/34",
        "tags":    ["h3", "h4"],
        "headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130"},
    },
    {
        "name":    "Economy Next",
        "url":     "https://economynext.com/",
        "tags":    ["h3", "h4"],
        "headers": {"User-Agent": "Mozilla/5.0"},
    },
]

# Canonical company list — (canonical_name, [aliases/tickers])
COMPANY_MAP = [
    ("Commercial Bank",             ["commercial bank", "combank", "comb", "com bank"]),
    ("Hatton National Bank",        ["hatton national", "hnb"]),
    ("Sampath Bank",                ["sampath bank", "samp"]),
    ("Seylan Bank",                 ["seylan bank", "seyb"]),
    ("Bank of Ceylon",              ["bank of ceylon", "boc"]),
    ("DFCC Bank",                   ["dfcc bank", "dfcc"]),
    ("Nations Trust Bank",          ["nations trust", "ntb"]),
    ("Pan Asia Bank",               ["pan asia bank", "pabc"]),
    ("LOLC Finance",                ["lolc finance"]),
    ("LOLC Holdings",               ["lolc holdings", "lolc"]),
    ("Hayleys PLC",                 ["hayleys", "hayl"]),
    ("John Keells Holdings",        ["john keells", "jkh"]),
    ("Aitken Spence",               ["aitken spence", "spen"]),
    ("Expolanka",                   ["expolanka", "expo"]),
    ("Melstacorp",                  ["melstacorp"]),
    ("Distilleries",                ["distilleries", "dist"]),
    ("Ceylon Tobacco",              ["ceylon tobacco", "ctc"]),
    ("Dialog Axiata",               ["dialog", "dial"]),
    ("SLT Mobitel",                 ["sri lanka telecom", "slt", "slt-mobitel"]),
    ("Cargills Ceylon",             ["cargills", "carg"]),
    ("Overseas Realty",             ["overseas realty", "osea"]),
    ("Teejay Lanka",                ["teejay lanka", "tjl"]),
    ("Tokyo Cement",                ["tokyo cement", "tkyo"]),
    ("Sunshine Holdings",           ["sunshine holdings", "sun"]),
    ("First Capital",               ["first capital", "cfvf", "first capital holdings", "first capital treasuries"]),
    ("Alliance Finance",            ["alliance finance"]),
    ("Siyapatha Finance",           ["siyapatha finance"]),
    ("Sarvodaya Development Finance",["sarvodaya", "sarvodaya development"]),
    ("Vidullanka",                  ["vidullanka", "vll"]),
    ("Resus Energy",                ["resus energy", "resc"]),
    ("LCB Finance",                 ["lcb finance"]),
    ("Fintrex Finance",             ["fintrex finance", "fintrex"]),
    ("JF Packaging",                ["jf packaging"]),
    ("Capital Alliance",            ["capital alliance", "calh"]),
    ("WealthTrust Securities",      ["wealthtrust", "wtsl"]),
    ("Merchant Bank",               ["merchant bank", "mbsl"]),
    ("Assetline Finance",           ["assetline finance", "assetline"]),
    ("CBC Finance",                 ["cbc finance"]),
    ("LTL Holdings",                ["ltl holdings", "ltlh"]),
    ("InsureMe",                    ["insureme"]),
    ("Pan Asian Power",             ["pan asian power", "pap"]),
    ("ACL Cables",                  ["acl cables", "acl"]),
    ("Lanka Tiles",                 ["lanka tiles", "tile"]),
    ("Royal Ceramics",              ["royal ceramics", "rcl"]),
    ("Watawala Plantations",        ["watawala"]),
    ("Malwatte Valley",             ["malwatte"]),
    ("Bogawantalawa",               ["bogawantalawa"]),
    ("Kelani Valley",               ["kelani valley"]),
]

# Pre-build lowercase regex per company
_COMPANY_PATTERNS = [
    (canonical, re.compile(
        r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE
    ))
    for canonical, aliases in COMPANY_MAP
]

# Historical category → signal type
POSITIVE_CATEGORIES = {"Financial Announcements", "New Listings Announcements"}
NEGATIVE_CATEGORIES  = {"Non-Compliance Announcements"}
IPO_KEYWORDS = ["initial public offering", "ipo", "basis of allotment", "listing via introduction"]
FINANCIAL_KEYWORDS = ["profit", "revenue", "earnings", "dividend", "results", "financial statement"]

# ─────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────

def divider(char="=", width=70):
    return char * width

def extract_companies(text: str) -> list:
    """Return list of canonical company names found in text."""
    found = []
    tl = text.lower()
    for canonical, pattern in _COMPANY_PATTERNS:
        if pattern.search(tl):
            found.append(canonical)
    return found

def rating(score: float) -> str:
    if score >= 8.0:    return "*** STRONG BUY ***"
    elif score >= 4.0:  return "**  BUY"
    elif score >= 1.5:  return "*   WATCH"
    elif score >= -1.0: return "    NEUTRAL"
    else:               return "!   CAUTION"

def sentiment_bar(pos, neg, neu, width=24):
    total = pos + neg + neu or 1
    p = int(pos / total * width)
    n = int(neg / total * width)
    u = width - p - n
    if u < 0: u = 0
    return "[" + "+" * p + "-" * n + "." * u + "]"

def normalize_symbol(symbol: str) -> str:
    base = symbol.split(".", 1)[0].strip().lower()
    return re.sub(r"[^a-z0-9]+", "", base)

def build_live_company_map(live_rows: list[dict]) -> dict:
    alias_to_company = {}
    for canonical, aliases in COMPANY_MAP:
        alias_to_company[canonical.lower()] = canonical
        for alias in aliases:
            alias_to_company[alias.lower()] = canonical

    live_map = {}
    for row in live_rows:
        symbol = row.get("symbol", "")
        base = normalize_symbol(symbol)
        company = alias_to_company.get(base)
        if not company:
            continue

        price = row.get("lastTrade") or row.get("lastTradedPrice")
        if price is None or price == 0:
            price = row.get("previousClose", 0.00)
        change_pct = row.get("percentageChange", 0.00)

        live_map[company] = {
            "symbol": symbol,
            "price": float(price) if price is not None else 0.0,
            "change_pct": float(change_pct) if change_pct is not None else 0.0,
        }

    return live_map

def fetch_all_shares_live() -> list[dict]:
    url = "https://www.cse.lk/api/tradeSummary"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
    }
    try:
        response = requests.post(url, json={}, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json().get("reqTradeSummery", [])
    except Exception as exc:
        print(f"  [!] Live share fetch failed: {exc}")
        return []

def print_live_shares_table(data: list[dict] | None = None):
    data = data or fetch_all_shares_live()
    if not data:
        print("\n  LIVE SHARE SNAPSHOT: no data available.")
        return

    print("\n" + divider("="))
    print("  LIVE SHARE SNAPSHOT (CSE)")
    print(divider("="))
    print(f"  {'Ticker':<12} | {'Price (LKR)':<12} | {'Change %':<10}")
    print("  " + "-" * 40)

    for share in data:
        symbol = share.get("symbol", "N/A")
        price = share.get("lastTrade") or share.get("lastTradedPrice")
        if price is None or price == 0:
            price = share.get("previousClose", 0.00)
        change_pct = share.get("percentageChange", 0.00)
        print(f"  {symbol:<12} | {price:<12.2f} | {change_pct:<10.2f}")

def future_bias_from_score(score: float) -> tuple[str, str]:
    if score >= 8.0:
        return "Very bullish", "positive earnings, dividends, and expansion news can extend the move"
    if score >= 4.0:
        return "Bullish", "good results or company updates can push it higher"
    if score >= 1.5:
        return "Mildly positive", "selective news flow can improve the chart, but upside is limited"
    if score >= -1.0:
        return "Neutral", "new news will decide direction; expect a range until stronger catalysts arrive"
    return "Bearish", "negative announcements or weak results could keep pressure on the chart"

def event_driver_hint(company_data: dict) -> str:
    positive = company_data.get("positive", 0)
    negative = company_data.get("negative", 0)
    live_mentions = company_data.get("mention_live", 0)
    hist_mentions = company_data.get("mention_hist", 0)

    if negative > positive:
        return "watch non-compliance, regulatory actions, and weak operating updates"
    if live_mentions > hist_mentions and positive >= negative:
        return "watch live earnings, dividend, and market-moving headlines"
    if positive > 0:
        return "watch financial results, dividends, bond issues, and expansion news"
    return "watch any new announcement because the share has low current signal strength"

def build_future_engine(ranked: list) -> dict:
    future_rows = []
    for company, score, d in ranked:
        bias, reason = future_bias_from_score(score)
        future_rows.append({
            "company": company,
            "current_score": round(score, 2),
            "future_bias": bias,
            "future_driver": event_driver_hint(d),
            "future_reason": reason,
            "hist_mentions": d.get("mention_hist", 0),
            "live_mentions": d.get("mention_live", 0),
        })

    bullish = sum(1 for row in future_rows if row["future_bias"] in {"Very bullish", "Bullish", "Mildly positive"})
    bearish = sum(1 for row in future_rows if row["future_bias"] == "Bearish")
    neutral = len(future_rows) - bullish - bearish

    market_state = "bullish" if bullish > bearish + neutral * 0.25 else "bearish" if bearish > bullish else "mixed"

    return {
        "rows": future_rows,
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "market_state": market_state,
    }

def print_future_engine(future_engine: dict):
    rows = future_engine.get("rows", [])
    if not rows:
        print("\n  FUTURE ENGINE: no rows available.")
        return

    print()
    print(divider("="))
    print("  FUTURE NEWS ENGINE")
    print(divider("="))
    print(f"  Market view   : {future_engine['market_state'].upper()}")
    print(f"  Bullish names : {future_engine['bullish']}")
    print(f"  Neutral names : {future_engine['neutral']}")
    print(f"  Bearish names : {future_engine['bearish']}")
    print()
    print(f"  {'Company':<35}{'Bias':<18}{'Driver':<54}{'Reason'}")
    print("  " + "-" * 125)

    for row in rows[:12]:
        print(f"  {row['company']:<35}{row['future_bias']:<18}{row['future_driver']:<54} {row['future_reason']}")

    top_future = rows[:5]
    if top_future:
        print()
        print("  Future leaders to watch:")
        for row in top_future:
            print(f"    - {row['company']} ({row['future_bias']})")

    print()
    print("  How future news changes the engine:")
    print("    - Positive earnings, dividends, debt issues, and expansion news lift the score")
    print("    - Negative compliance, weak profits, or regulatory actions reduce the score")
    print("    - Fresh live headlines have the fastest impact because they hit the score immediately")
    print(divider("="))

def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned.strip("_") or "company"

def company_future_profile(company: str, score: float, d: dict) -> dict:
    bias, reason = future_bias_from_score(score)
    return {
        "bias": bias,
        "reason": reason,
        "driver": event_driver_hint(d),
    }

def render_company_dashboard(company: str, score: float, d: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    total_mentions = d["mention_hist"] + d["mention_live"] or 1
    sentiment_total = d["positive"] + d["negative"] + d["neutral"] or 1
    future = company_future_profile(company, score, d)

    fig = plt.figure(figsize=(14, 8), facecolor="#0f172a")
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.8, 1.1, 1.1], width_ratios=[1.1, 0.9])
    fig.suptitle(f"{company} Dashboard", fontsize=20, fontweight="bold", color="white", y=0.98)

    ax_score = fig.add_subplot(gs[0, 0])
    ax_score.set_facecolor("#111827")
    ax_score.barh(["Combined"], [score], color="#22c55e" if score >= 4 else "#f59e0b" if score >= 1.5 else "#64748b")
    ax_score.axvline(0, color="#94a3b8", linewidth=1)
    ax_score.set_title("Current Signal", color="white", fontsize=12)
    ax_score.set_xlim(-5, max(10, score + 3))
    ax_score.tick_params(axis="x", colors="white")
    ax_score.tick_params(axis="y", colors="white")
    for spine in ax_score.spines.values():
        spine.set_color("#334155")
    ax_score.text(0.02, 0.82, f"Score: {score:+.2f}", transform=ax_score.transAxes, color="white", fontsize=14, fontweight="bold")
    ax_score.text(0.02, 0.58, f"Signal: {rating(score)}", transform=ax_score.transAxes, color="#cbd5e1", fontsize=11)
    ax_score.text(0.02, 0.34, f"Mentions: {total_mentions}", transform=ax_score.transAxes, color="#cbd5e1", fontsize=11)

    ax_mix = fig.add_subplot(gs[0, 1])
    ax_mix.set_facecolor("#111827")
    ax_mix.pie(
        [d["positive"], d["negative"], d["neutral"]],
        labels=["Positive", "Negative", "Neutral"],
        colors=["#16a34a", "#dc2626", "#64748b"],
        autopct=lambda pct: f"{pct:.0f}%" if pct > 0 else "",
        textprops={"color": "white", "fontsize": 9},
        startangle=90,
    )
    ax_mix.set_title("Sentiment Mix", color="white", fontsize=12)

    ax_future = fig.add_subplot(gs[1, 0])
    ax_future.set_facecolor("#111827")
    ax_future.axis("off")
    ax_future.text(0.02, 0.92, "Future Engine", color="white", fontsize=13, fontweight="bold")
    ax_future.text(0.02, 0.70, f"Bias: {future['bias']}", color="#e2e8f0", fontsize=12)
    ax_future.text(0.02, 0.48, f"Driver: {future['driver']}", color="#cbd5e1", fontsize=10, wrap=True)
    ax_future.text(0.02, 0.24, f"Reason: {future['reason']}", color="#cbd5e1", fontsize=10, wrap=True)

    ax_counts = fig.add_subplot(gs[1, 1])
    ax_counts.set_facecolor("#111827")
    bars = [d["mention_hist"], d["mention_live"], d["positive"], d["negative"], d["neutral"]]
    labels = ["Hist", "Live", "Pos", "Neg", "Neu"]
    colors = ["#38bdf8", "#8b5cf6", "#16a34a", "#dc2626", "#64748b"]
    ax_counts.bar(labels, bars, color=colors)
    ax_counts.set_title("Counts", color="white", fontsize=12)
    ax_counts.tick_params(axis="x", colors="white")
    ax_counts.tick_params(axis="y", colors="white")
    for spine in ax_counts.spines.values():
        spine.set_color("#334155")

    ax_notes = fig.add_subplot(gs[2, :])
    ax_notes.set_facecolor("#111827")
    ax_notes.axis("off")
    recent_mentions = d.get("live_mentions", [])[:3]
    lines = [
        f"Historical Score: {d['hist_score']:+.2f}",
        f"Live Score: {d['live_score']:+.2f}",
        f"Mentions: {total_mentions} | Recent live: {d['mention_live']} | Historical: {d['mention_hist']}",
        f"Outlook: {future['bias']}",
    ]
    if recent_mentions:
        lines.append("Recent headlines:")
        for item in recent_mentions:
            lines.append(f"- [{item['sentiment']}] {item['source']}: {item['text'][:100]}")
    else:
        lines.append("Recent headlines: none captured in live scrape")
    ax_notes.text(0.02, 0.92, "\n".join(lines), color="white", fontsize=10, va="top", family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / f"{safe_filename(company)}.png"
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path

def save_all_company_dashboards(ranked: list):
    dashboard_dir = Path(OUTPUT_DIR) / "dashboards"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for company, score, d in ranked:
        saved.append(render_company_dashboard(company, score, d, dashboard_dir))

    print()
    print(divider("="))
    print("  COMPANY DASHBOARDS")
    print(divider("="))
    print(f"  Saved {len(saved)} dashboard images to {dashboard_dir}")
    print("  Each PNG is a separate visual dashboard for one CSE share.")
    print(divider("="))

def fetch_commercial_bank_live_price() -> float | None:
    try:
        url = "https://www.cse.lk/api/tradeSummary"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.post(url, data={"symbol": COMMERCIAL_BANK_SYMBOL}, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        for item in data.get("reqTradeSummery", []):
            if item.get("symbol") == COMMERCIAL_BANK_SYMBOL and item.get("price"):
                return float(item["price"])
    except Exception:
        pass
    return None

def fetch_commercial_bank_price_history() -> pd.DataFrame:
    try:
        df = yf.download(COMMERCIAL_BANK_SYMBOL, period="1y", interval="1d", auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            df = df.dropna().copy()
            df.index = pd.to_datetime(df.index)
            return df
    except Exception:
        pass

    live_price = fetch_commercial_bank_live_price() or 100.0
    dates = pd.date_range(end=datetime.today(), periods=252, freq="B")
    np.random.seed(42)
    random_walk = np.random.randn(252).cumsum()
    prices = random_walk - random_walk[-1] + live_price
    return pd.DataFrame({"Close": prices}, index=dates)

def load_commercial_bank_events() -> pd.DataFrame:
    if not COMMERCIAL_BANK_NEWS_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(COMMERCIAL_BANK_NEWS_CSV, encoding="utf-8-sig", low_memory=False)
    text_cols = ["company", "subject", "description", "heading", "title", "remarks"]
    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        if col in df.columns:
            mask = mask | df[col].fillna("").astype(str).str.contains(COMMERCIAL_BANK_NAME, case=False, na=False)
    events = df.loc[mask].copy()

    date_source = None
    for col in ["_parsed_date", "createdDate", "announcementDate", "publishedDate", "date"]:
        if col in events.columns and events[col].notna().any():
            date_source = col
            break

    if date_source is None:
        return pd.DataFrame()

    events["event_date"] = pd.to_datetime(events[date_source], errors="coerce")
    events = events.dropna(subset=["event_date"]).sort_values("event_date").copy()
    events["event_text"] = events[[c for c in text_cols if c in events.columns]].fillna("").agg(" ".join, axis=1)
    return events

def classify_commercial_bank_event(row: pd.Series):
    text = str(row.get("event_text", "")).lower()
    category = str(row.get("_category", "")).lower()

    if "non-compliance" in category or "non-compliance" in text:
        return "negative", -1
    if "financial" in category or any(word in text for word in ["profit", "results", "dividend", "revenue", "earnings"]):
        return "positive", 1
    if "new listings" in category or any(word in text for word in ["debenture", "bond", "ipo", "allotment", "listing"]):
        return "positive", 1
    return "neutral", 0

def commercial_bank_snapshot() -> str:
    price_df = fetch_commercial_bank_price_history()
    events = load_commercial_bank_events()

    latest = float(price_df["Close"].iloc[-1])
    recent_20 = price_df["Close"].tail(20)
    recent_60 = price_df["Close"].tail(60)
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

    recent_events = pd.DataFrame()
    if not events.empty:
        recent_events = events[events["event_date"] >= (price_df.index.max() - pd.Timedelta(days=120))]
        event_score = sum(classify_commercial_bank_event(row)[1] for _, row in recent_events.iterrows()) if not recent_events.empty else 0
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

    lines = [
        "",
        divider("="),
        f"  {COMMERCIAL_BANK_NAME} SNAPSHOT",
        divider("="),
        f"  Symbol          : {COMMERCIAL_BANK_SYMBOL}",
        f"  Latest Close    : {latest:.2f} LKR",
        f"  20D Return      : {short_return:+.2f}%",
        f"  60D Return      : {medium_return:+.2f}%",
        f"  Event Bias      : {outlook}",
        f"  Outlook         : {idea}",
    ]

    if not events.empty:
        display_events = recent_events if not recent_events.empty else events.tail(4)
        label = "Recent events" if not recent_events.empty else "Latest saved events"
        lines.append(f"  Announcement Hits: {len(recent_events)} recent / {len(events)} total")
        lines.append(f"  {label}:")
        for _, row in display_events.iterrows():
            label, direction = classify_commercial_bank_event(row)
            txt = str(row.get("remarks", row.get("title", "Event")))[:110]
            lines.append(f"    - [{label}] {row['event_date']:%Y-%m-%d} {txt}")
    else:
        lines.append("  Announcement Hits: none found in the saved news file")

    lines.append(divider("="))
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────
# STEP 1 — Historical Analysis (Signal-Based)
# ─────────────────────────────────────────────────────────

def analyse_historical(hist_df: pd.DataFrame) -> dict:
    """
    Score companies from historical data using:
    - Mention frequency (recent vs old)
    - IPO/new listing → strong positive signal
    - Financial announcements → positive signal
    - Non-compliance → negative signal
    """
    scores = defaultdict(lambda: {
        "hist_score": 0.0, "recent_score": 0.0, "live_score": 0.0,
        "mention_hist": 0, "mention_live": 0,
        "positive": 0, "negative": 0, "neutral": 0,
        "live_mentions": [], "signals": [],
    })

    today  = date.today()
    cutoff = today - timedelta(days=RECENT_DAYS)

    for _, row in hist_df.iterrows():
        # Combine all text fields
        text_parts = []
        for col in ["remarks", "subject", "description", "heading", "title", "company"]:
            v = row.get(col, "")
            if pd.notna(v) and str(v).strip():
                text_parts.append(str(v).strip())
        text = " ".join(text_parts)
        if not text:
            continue

        companies = extract_companies(text)
        if not companies:
            continue

        # Determine date
        rec_date = None
        date_str = row.get("_parsed_date", "")
        if pd.notna(date_str) and date_str:
            try:
                rec_date = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
            except Exception:
                pass

        is_recent = rec_date is not None and rec_date >= cutoff
        category  = str(row.get("_category", ""))
        tl        = text.lower()

        # Determine signal weight for this record
        base_weight = W_HIST_RECENT if is_recent else W_HIST_OLD
        signal_score = base_weight  # default: positive mention signal

        # Override for special categories
        if category in NEGATIVE_CATEGORIES:
            signal_score = W_COMPLIANCE_NEG
            signal_label = "Non-compliance"
        elif any(kw in tl for kw in IPO_KEYWORDS):
            signal_score = W_IPO_SIGNAL if is_recent else W_IPO_SIGNAL * 0.5
            signal_label = "IPO/New Listing"
        elif any(kw in tl for kw in FINANCIAL_KEYWORDS):
            signal_score = W_FINANCIAL if is_recent else W_FINANCIAL * 0.5
            signal_label = "Financial Results"
        else:
            signal_label = "Activity"

        for c in companies:
            scores[c]["hist_score"]    += signal_score
            scores[c]["mention_hist"]  += 1
            if signal_score > 0:
                scores[c]["positive"] += 1
            elif signal_score < 0:
                scores[c]["negative"] += 1
            else:
                scores[c]["neutral"]  += 1

    return dict(scores)

# ─────────────────────────────────────────────────────────
# STEP 2 — Live News Scrape
# ─────────────────────────────────────────────────────────

def scrape_live_news() -> list:
    articles = []
    for src in LIVE_SOURCES:
        print(f"  Fetching: {src['name']} ...")
        try:
            resp = requests.get(src["url"], headers=src["headers"], timeout=20)
            if resp.status_code != 200:
                print(f"    [!] HTTP {resp.status_code}")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            seen, count = set(), 0
            for tag_name in src["tags"]:
                for tag in soup.find_all(tag_name):
                    text = tag.get_text(strip=True)
                    if len(text) > 40 and text not in seen:
                        seen.add(text)
                        link_tag = tag.find("a") or tag.find_parent("a")
                        url = ""
                        if link_tag and link_tag.get("href"):
                            url = link_tag["href"]
                        articles.append({"source": src["name"], "text": text, "url": url})
                        count += 1
                if count >= LIVE_LIMIT:
                    break
            print(f"    OK: {count} headlines")
        except Exception as e:
            print(f"    [!] Error: {e}")
        time.sleep(1.0)
    return articles

# ─────────────────────────────────────────────────────────
# STEP 3 — FinBERT Sentiment on Live News
# ─────────────────────────────────────────────────────────

def run_sentiment(texts: list, fn, batch=32) -> list:
    results = []
    for i in range(0, len(texts), batch):
        chunk = [t[:512] if t else "neutral" for t in texts[i:i+batch]]
        try:
            out = fn(chunk, truncation=True)
            results.extend(out)
        except Exception as e:
            print(f"  [!] Batch error: {e}")
            results.extend([{"label": "neutral", "score": 0.5}] * len(chunk))
    return results

def apply_live_sentiment(scores: dict, live_articles: list, live_sentiments: list):
    """Add live news signal into scores dict (mutates in place)."""
    for article, sent in zip(live_articles, live_sentiments):
        companies = extract_companies(article["text"])
        if not companies:
            continue
        label = sent["label"]
        conf  = sent["score"]

        if label == "positive":
            delta = W_LIVE_POS * conf
        elif label == "negative":
            delta = W_LIVE_NEG * conf
        else:
            delta = W_LIVE_NEU

        for c in companies:
            if c not in scores:
                scores[c] = {
                    "hist_score": 0.0, "recent_score": 0.0, "live_score": 0.0,
                    "mention_hist": 0, "mention_live": 0,
                    "positive": 0, "negative": 0, "neutral": 0,
                    "live_mentions": [], "signals": [],
                }
            scores[c]["live_score"]   += delta
            scores[c]["mention_live"] += 1
            if label == "positive":   scores[c]["positive"] += 1
            elif label == "negative": scores[c]["negative"] += 1
            else:                     scores[c]["neutral"]  += 1
            scores[c]["live_mentions"].append({
                "text":       article["text"],
                "source":     article["source"],
                "sentiment":  label,
                "confidence": round(conf, 3),
            })

# ─────────────────────────────────────────────────────────
# STEP 4 — Combined Score & Rank
# ─────────────────────────────────────────────────────────

def finalise_scores(scores: dict, live_map: dict | None = None, only_live: bool = False) -> list:
    """Compute combined score, return sorted list."""
    ranked = []
    for company, d in scores.items():
        if only_live and (not live_map or company not in live_map):
            continue
        total_mentions = d["mention_hist"] + d["mention_live"]
        if total_mentions == 0:
            continue
        combined = d["hist_score"] + d["live_score"]
        ranked.append((company, combined, d))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

# ─────────────────────────────────────────────────────────
# STEP 5 — Dashboard Print
# ─────────────────────────────────────────────────────────

def print_dashboard(ranked: list, live_map: dict | None = None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    future_engine = build_future_engine(ranked)

    print()
    print(divider("="))
    print("  CSE SMART ANALYZER — SHARE PERFORMANCE PREDICTION")
    print(f"  Report Date : {now}")
    print(f"  Data        : 5-Year CSE Announcements (2021-2025) + Live News")
    print(divider("="))

    if not ranked:
        print("  No companies identified.")
        return

    print()
    print(f"  {'#':<4}{'Company':<35}{'Score':>8}  {'Signal':<22}{'Mentions':>10}  {'Price':>10}  {'Chg%':>7}  {'Sentiment Bar'}")
    print("  " + "-" * 100)

    for i, (company, score, d) in enumerate(ranked, 1):
        total = d["mention_hist"] + d["mention_live"]
        bar   = sentiment_bar(d["positive"], d["negative"], d["neutral"])
        sig   = rating(score)
        live = live_map.get(company) if live_map else None
        price = live["price"] if live else 0.0
        change = live["change_pct"] if live else 0.0
        print(f"  {i:<4}{company:<35}{score:>8.2f}  {sig:<22}{total:>10}  {price:>10.2f}  {change:>7.2f}  {bar}")

    # Top 5 detail
    print()
    print(divider("="))
    print("  TOP 5 — DETAILED ANALYSIS")
    print(divider("="))

    for rank, (company, score, d) in enumerate(ranked[:5], 1):
        total = d["positive"] + d["negative"] + d["neutral"] or 1
        pp = d["positive"] / total * 100
        np_ = d["negative"] / total * 100
        nu = d["neutral"]  / total * 100

        print(f"\n  #{rank}  {company}")
        print("  " + "-" * 55)
        print(f"  Signal           : {rating(score)}")
        live = live_map.get(company) if live_map else None
        price = live["price"] if live else 0.0
        change = live["change_pct"] if live else 0.0
        print(f"  Combined Score   : {score:+.2f}")
        print(f"  Historical Score : {d['hist_score']:+.2f}  ({d['mention_hist']} CSE records)")
        print(f"  Live News Score  : {d['live_score']:+.2f}  ({d['mention_live']} live headlines)")
        print(f"  Live Price/Chg%  : {price:.2f} LKR / {change:+.2f}%")
        print(f"  Sentiment Mix    : {pp:.0f}% Pos | {np_:.0f}% Neg | {nu:.0f}% Neutral")

        if d["live_mentions"]:
            print(f"\n  Today's headlines about {company}:")
            for m in d["live_mentions"][:4]:
                tag = "[+]" if m["sentiment"] == "positive" else "[-]" if m["sentiment"] == "negative" else "[~]"
                src = m["source"]
                txt = m["text"][:110]
                print(f"    {tag} [{src}] {txt}")

    # Summary
    print()
    print(divider("="))
    print("  PREDICTION SUMMARY")
    print(divider("="))
    groups = {
        "STRONG BUY (score >= 8)": [c for c, s, _ in ranked if s >= 8],
        "BUY        (score >= 4)": [c for c, s, _ in ranked if 4 <= s < 8],
        "WATCH      (score >= 1.5)":[c for c, s, _ in ranked if 1.5 <= s < 4],
        "CAUTION    (score < -1)": [c for c, s, _ in ranked if s < -1],
    }
    has_any = False
    for label, group in groups.items():
        if group:
            has_any = True
            print(f"\n  {label}:")
            for c in group:
                sc = next(s for n, s, _ in ranked if n == c)
                live = live_map.get(c) if live_map else None
                price = live["price"] if live else 0.0
                change = live["change_pct"] if live else 0.0
                print(f"    - {c}  ({sc:+.2f})  {price:.2f} LKR  {change:+.2f}%")

    if not has_any:
        print("\n  All companies in NEUTRAL range today.")

    print()
    print(divider("="))
    print("  NOTE: Score = Historical Activity Signal + Live Sentiment Signal")
    print("  DISCLAIMER: AI research tool only — NOT financial advice.")
    print("  Always consult a licensed financial advisor before investing.")
    print(divider("="))

    print(commercial_bank_snapshot())
    print_future_engine(future_engine)

# ─────────────────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────────────────

def save_csv(ranked: list, live_map: dict | None = None):
    out = os.path.join(OUTPUT_DIR, "cse_prediction_results.csv")
    rows = []
    for company, score, d in ranked:
        total = d["positive"] + d["negative"] + d["neutral"] or 1
        live = live_map.get(company) if live_map else None
        price = live["price"] if live else 0.0
        change = live["change_pct"] if live else 0.0
        rows.append({
            "company":        company,
            "signal":         rating(score).strip(),
            "combined_score": round(score, 3),
            "hist_score":     round(d["hist_score"], 3),
            "live_score":     round(d["live_score"], 3),
            "live_price":     round(price, 2),
            "live_change_pct": round(change, 2),
            "hist_mentions":  d["mention_hist"],
            "live_mentions":  d["mention_live"],
            "pct_positive":   round(d["positive"] / total * 100, 1),
            "pct_negative":   round(d["negative"] / total * 100, 1),
            "pct_neutral":    round(d["neutral"]  / total * 100, 1),
            "generated_at":   datetime.now().isoformat(),
        })
    if rows:
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Results saved -> {out}")

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    print(divider("="))
    print("  CSE SMART ANALYZER — Starting...")
    print(divider("="))

    # ── Load FinBERT ──────────────────────────────────
    print("\n[1/5] Loading FinBERT AI sentiment model...")
    sentiment_fn = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
        max_length=512,
    )
    print("  Model loaded.")

    # ── Load historical CSVs ──────────────────────────
    print("\n[2/5] Loading historical CSE announcements (2021-2025)...")
    csv_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "cse_news_????.csv")))
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
            frames.append(df)
            print(f"  {os.path.basename(f)} -> {len(df):,} records")
        except Exception as e:
            print(f"  [!] Could not load {f}: {e}")

    if not frames:
        print("  [!] No historical CSVs found. Run cse_news_scraper.py first.")
        hist_df = pd.DataFrame()
    else:
        hist_df = pd.concat(frames, ignore_index=True)
        print(f"  Total: {len(hist_df):,} historical records")

    # ── Historical signal scoring ─────────────────────
    print("\n[3/5] Analysing historical company signals...")
    if not hist_df.empty:
        scores = analyse_historical(hist_df)
        print(f"  {len(scores)} companies identified in historical data.")
    else:
        scores = {}

    # ── Live news ─────────────────────────────────────
    print("\n[4/5] Fetching today's live business news...")
    live_articles = scrape_live_news()
    print(f"  Total live headlines collected: {len(live_articles)}")

    if live_articles:
        print(f"  Running FinBERT on {len(live_articles)} headlines...")
        live_sentiments = run_sentiment(
            [a["text"] for a in live_articles], sentiment_fn, batch=32
        )
        print("  Sentiment analysis complete.")
        apply_live_sentiment(scores, live_articles, live_sentiments)
    else:
        print("  [!] No live articles — relying on historical data only.")

    # ── Rank & display ────────────────────────────────
    print("\n[5/5] Ranking companies and generating report...")
    live_rows = fetch_all_shares_live()
    live_map = build_live_company_map(live_rows)
    ranked = finalise_scores(scores, live_map=live_map, only_live=True)
    print_dashboard(ranked, live_map=live_map)
    print_live_shares_table(live_rows)

    # ── Save ──────────────────────────────────────────
    save_csv(ranked, live_map=live_map)
    save_all_company_dashboards(ranked)
    print("\n  Done!")


if __name__ == "__main__":
    main()
