import os

import joblib
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load Paths
CSV_PATH = "cse_output/cse_combined_sources.csv"
SHARES_LIST = "cse_output/all_share_details.csv"
START_DATE = "2022-01-01"
REPORT_PATH = "cse_output/cse_model_report.txt"

df_news = pd.read_csv(CSV_PATH)
if "sentiment" not in df_news.columns:
    raise ValueError("Missing 'sentiment' column in combined sources CSV.")

df_news = df_news[df_news["source_type"] == "news"].copy()
df_news["published_date"] = pd.to_datetime(df_news["published_date"], errors="coerce")
df_news = df_news.dropna(subset=["published_date", "sentiment"]).copy()
df_news["date"] = df_news["published_date"].dt.date
daily_sentiment = df_news.groupby("date")["sentiment"].mean().to_frame()

all_data = []

def fetch_with_retry(symbol: str) -> pd.DataFrame:
    base = symbol.split(".")[0]
    candidates = [symbol, f"{base}.N0000.LK", f"{base}.N.LK", f"{base}.LK"]

    for fmt in candidates:
        if not fmt:
            continue
        print(f"Trying {fmt}...")
        df = yf.download(fmt, start=START_DATE, progress=False)
        if not df.empty and len(df) > 10:
            return df
    return pd.DataFrame()

# Loop through and train
shares_df = pd.read_csv(SHARES_LIST)
for _, row in shares_df.iterrows():
    yahoo_symbol = None
    if "Yahoo" in shares_df.columns and pd.notna(row.get("Yahoo")):
        yahoo_symbol = str(row.get("Yahoo")).strip()
    symbol = yahoo_symbol or str(row.get("Symbol", "")).strip()
    if not symbol:
        continue

    prices = fetch_with_retry(symbol)
    
    if prices.empty:
        print(f"  [!] Failed to find any Yahoo data for {symbol}")
        continue

    # Flatten columns for merge compatibility
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(-1)

    if "Close" not in prices.columns:
        print(f"  [!] Missing Close column for {symbol}")
        continue

    prices.index = pd.to_datetime(prices.index).date
    prices["Return"] = prices["Close"].pct_change()
    prices["MA5"] = prices["Close"].rolling(window=5).mean()
    prices["Target"] = (prices["Close"].shift(-1) > prices["Close"]).astype(int)

    # Merge with Sentiment
    merged = prices.join(daily_sentiment, how="inner").dropna()
    if not merged.empty:
        print(f"  [✓] Successfully merged {len(merged)} days for {symbol}")
        all_data.append(merged)

# Train the Model
if all_data:
    final_df = pd.concat(all_data)
    X = final_df[["Close", "Return", "MA5", "sentiment"]]
    y = final_df["Target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "cse_master_ai.pkl")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    matrix = confusion_matrix(y_test, y_pred)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("CSE Model Evaluation\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(matrix))

    print("\nSUCCESS: Model trained and saved as cse_master_ai.pkl")
    print(f"Evaluation report saved to {REPORT_PATH}")
else:
    print("\nERROR: No data found. Check your internet or if the tickers have changed again.")
    