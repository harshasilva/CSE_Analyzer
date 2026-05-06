import kagglehub
import pandas as pd
import os

# Download latest version of the dataset
path = kagglehub.dataset_download("ivantha/sri-lanka-news-dataset")
print("Path to dataset files:", path)

# List the files in the downloaded dataset path
files = os.listdir(path)
print("Files in dataset:", files)

# Find the first CSV file in the dataset directory
csv_file = next((f for f in files if f.endswith('.csv')), None)
if csv_file:
    df = pd.read_csv(os.path.join(path, csv_file))
    print(df.head())
    
    # Filter for CSE related keywords. Assuming the column is 'headline'. We'll check columns first.
    print("Columns:", df.columns)
    
    # We will adjust the column name based on actual data
    if 'headline' in df.columns:
        text_col = 'headline'
    elif 'title' in df.columns:
        text_col = 'title'
    else:
        text_col = df.columns[0] # Fallback
        
    # Handle NaN values before applying string matching
    df[text_col] = df[text_col].fillna('')
        
    cse_news = df[df[text_col].str.contains("CSE|Stock|Colombo", case=False)]
    print(f"Found {len(cse_news)} CSE related news articles.")
else:
    print("No CSV file found in the dataset.")
    cse_news = pd.DataFrame()

# Load FinBERT
from transformers import pipeline
from tqdm import tqdm
print("Loading FinBERT sentiment analysis pipeline...")
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Analyze sentiment of headlines
print("Analyzing sentiment of the headlines (this may take a few minutes)...")
# Process in smaller chunks with a progress bar
results = []
headlines = cse_news[text_col].tolist()
batch_size = 32

for i in tqdm(range(0, len(headlines), batch_size), desc="Analyzing Sentiment"):
    batch = headlines[i:i+batch_size]
    # Handle NaN or non-string values
    batch = [str(text)[:512] if pd.notna(text) else "neutral" for text in batch]
    batch_results = sentiment_pipeline(batch, truncation=True)
    results.extend(batch_results)

# Add results to DataFrame
cse_news['sentiment'] = [r['label'] for r in results]

# Determine the date column
date_col = 'published_date' if 'published_date' in cse_news.columns else 'date' if 'date' in cse_news.columns else cse_news.columns[1]

# Calculate daily sentiment
daily_sentiment = cse_news.groupby(date_col)['sentiment'].value_counts(normalize=True).unstack()

# Plotting
import matplotlib.pyplot as plt
daily_sentiment.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('CSE Sentiment Over Time')
plt.ylabel('Percentage')
plt.show()