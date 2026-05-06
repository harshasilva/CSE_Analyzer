import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


COMBINED_CSV = "cse_output/cse_combined_sources.csv"
LABELED_CSV = "data/cse_sentiment_labeled.csv"
REPORT_PATH = "cse_output/cse_sentiment_model_report.txt"
MODEL_PATH = "cse_sentiment_model.pkl"


def sentiment_to_label(value: float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    if value >= 0.2:
        return "positive"
    if value <= -0.2:
        return "negative"
    return "neutral"


def load_labeled_data() -> pd.DataFrame:
    if not os.path.exists(LABELED_CSV):
        raise FileNotFoundError(f"Missing labeled data: {LABELED_CSV}")
    df = pd.read_csv(LABELED_CSV)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Labeled CSV must include 'text' and 'label' columns.")
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["source"] = "labeled"
    return df[["text", "label", "source"]]


def load_combined_data() -> pd.DataFrame:
    if not os.path.exists(COMBINED_CSV):
        raise FileNotFoundError(f"Missing combined data: {COMBINED_CSV}")
    df = pd.read_csv(COMBINED_CSV)
    if "sentiment" not in df.columns:
        raise ValueError("Combined CSV must include 'sentiment' column.")

    df = df[df["source_type"].isin(["news", "announcement"])].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].where(df["text"].str.len() > 0, df["title"])
    df["label"] = df["sentiment"].apply(sentiment_to_label)
    df = df.dropna(subset=["text", "label"]).copy()
    df = df[df["text"].str.len() > 5]
    df["source"] = "combined"
    return df[["text", "label", "source"]]


def main() -> None:
    labeled_df = load_labeled_data()
    combined_df = load_combined_data()
    final_df = pd.concat([labeled_df, combined_df], ignore_index=True)

    if final_df.empty:
        print("\nERROR: No data available for training.")
        return

    X = final_df["text"]
    y = final_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    matrix = confusion_matrix(y_test, y_pred)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("CSE Sentiment Model Evaluation\n")
        f.write(f"Total samples: {len(final_df)}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(matrix))

    print("\nSUCCESS: Sentiment model trained and saved as cse_sentiment_model.pkl")
    print(f"Evaluation report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
    import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Load the SCORED news data
SCORED_CSV = "cse_output/cse_scored_sources.csv"
SHARES_LIST = "cse_output/all_share_details.csv"

if not os.path.exists(SCORED_CSV):
    print(f"Error: {SCORED_CSV} not found!")
    exit()

df_news = pd.read_csv(SCORED_CSV)

# --- THE FIX: DYNAMIC COLUMN DETECTION ---
possible_cols = ['sentiment_score', 'sentiment', 'label']
found_col = next((c for c in possible_cols if c in df_news.columns), None)

if found_col:
    print(f"Found sentiment data in column: '{found_col}'")
    if df_news[found_col].dtype == 'object':
        mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        df_news['sentiment_score'] = df_news[found_col].str.lower().map(mapping).fillna(0)
    else:
        df_news['sentiment_score'] = df_news[found_col]
else:
    print(f"Error: No sentiment column found.")
    exit()

df_news['date'] = pd.to_datetime(df_news['published_date']).dt.date
daily_scores = df_news.groupby('date')['sentiment_score'].mean().to_frame()

# 2. Download Prices & Merge (Simplified yfinance call)
shares_df = pd.read_csv(SHARES_LIST)
all_data = []

print("\n--- Starting Price Collection ---")

for _, row in shares_df.iterrows():
    ticker = row['Yahoo']
    print(f"Downloading {ticker}...")
    
    # We stop passing 'session=session' and let yfinance handle it automatically
    df = yf.download(ticker, start="2024-01-01", progress=False)
    
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.index = pd.to_datetime(df.index).date
        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        merged = df.join(daily_scores, how='inner').dropna()
        if not merged.empty:
            all_data.append(merged)
            print(f"  [✓] Linked {len(merged)} patterns for {ticker}")

# 3. Train the Final AI
if all_data:
    final_dataset = pd.concat(all_data)
    X = final_dataset[['Close', 'Return', 'sentiment_score']]
    y = final_dataset['Target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, "cse_master_ai.pkl")
    print("\n" + "="*50)
    print("SUCCESS: cse_master_ai.pkl is fully built and saved!")
    print("="*50)
else:
    print("\nError: No overlap found between news and prices.")