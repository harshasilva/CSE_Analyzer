import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# 1. Setup the AI Sentiment Model (FinBERT - optimized for finance)
print("Loading AI Model... (This may take a minute the first time)")
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_market_insights():
    # Use Adaderana Biz - very stable in May 2026
    url = "https://bizenglish.adaderana.lk/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    print(f"Scraping latest news from {url}...")
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # In 2026, headlines are mostly in <h4> tags on this site
        headlines = soup.find_all('h4')
        
        print("\n" + "="*50)
        print("CSE AI SENTIMENT DASHBOARD - MAY 2026")
        print("="*50)
        
        count = 0
        for h in headlines:
            text = h.text.strip()
            if len(text) > 35: # Ignore small UI text
                # 2. Process with AI
                result = sentiment_pipeline(text)[0]
                label = result['label']
                score = round(result['score'], 2)
                
                # Extract date
                parent_div = h.find_parent('div', class_='story-text')
                date_str = ""
                if parent_div:
                    parts = parent_div.text.split('|')
                    if len(parts) > 1:
                        date_str = parts[-1].strip()
                
                date_display = f" | Release Date: {date_str}" if date_str else ""
                
                # Format output based on sentiment
                icon = "📈" if label == "positive" else "📉" if label == "negative" else "😐"
                
                print(f"\n[{icon} {label.upper()}] (Confidence: {score}){date_display}")
                print(f"News: {text}")
                
                count += 1
                if count == 5: break # Let's analyze the top 5
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_market_insights()