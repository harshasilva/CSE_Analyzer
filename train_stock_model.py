import pandas as pd
import numpy as np
import yfinance as yf
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import tqdm

# 1. Configuration
NEWS_DATA = Path("cse_output/cse_news_ALL_2021_2026.csv")
SYMBOL = "COMB.N0000"  # Target symbol to train on (Commercial Bank)
MODEL_NAME = "ProsusAI/finbert"

def prepare_training_data(symbol):
    print(f"--- Preparing Data for {symbol} ---")
    
    # Load news
    if not NEWS_DATA.exists():
        print("Error: News CSV not found. Run your scraper first!")
        return None
    
    df_news = pd.read_csv(NEWS_DATA)
    # Filter for specific symbol and ensure date is datetime
    df_news = df_news[df_news['symbol'] == symbol].copy()
    df_news['date'] = pd.to_datetime(df_news['_parsed_date']).dt.date
    
    # 2. Sentiment Analysis (using your FinBERT approach)
    print("Analyzing sentiment...")
    sentiment_pipe = pipeline("sentiment-analysis", model=MODEL_NAME)
    
    def get_sentiment_score(text):
        # Map: positive -> 1, neutral -> 0, negative -> -1
        res = sentiment_pipe(str(text)[:512])[0]
        score = res['score']
        if res['label'] == 'positive': return score
        if res['label'] == 'negative': return -score
        return 0

    # Aggregate daily sentiment
    df_news['sentiment_val'] = df_news['title'].apply(get_sentiment_score)
    daily_sentiment = df_news.groupby('date')['sentiment_val'].mean().reset_index()

    # 3. Fetch Price Data
    print("Fetching historical prices...")
    ticker = yf.Ticker(symbol)
    df_price = ticker.history(start="2021-01-01")
    df_price.index = df_price.index.date
    
    # 4. Merge News + Prices
    combined = df_price.join(daily_sentiment.set_index('date'), how='left').fillna(0)
    
    # 5. Feature Engineering
    # We want to use TODAY'S features to predict TOMORROW'S movement
    combined['Return'] = combined['Close'].pct_change()
    combined['MA5'] = combined['Close'].rolling(window=5).mean()
    combined['Target'] = (combined['Close'].shift(-1) > combined['Close']).astype(int) # 1 if up tomorrow
    
    # Clean up
    combined = combined.dropna()
    return combined

def train_model(data):
    # Select features
    features = ['Close', 'Return', 'MA5', 'sentiment_val']
    X = data[features]
    y = data['Target']
    
    # Time-series split (don't use random split for stocks!)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, preds))
    print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
    
    return model

if __name__ == "__main__":
    data = prepare_training_data(SYMBOL)
    if data is not None:
        model = train_model(data)
        # Save the model
        import joblib
        joblib.dump(model, "stock_model.pkl")
        print("Model saved as stock_model.pkl")