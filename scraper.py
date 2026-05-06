import requests
from bs4 import BeautifulSoup

# Stable URL for Business News in 2026
URL = "https://www.ft.lk/business/34" 

def get_market_news():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
    }
    
    print(f"Connecting to {URL}...")
    try:
        response = requests.get(URL, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # In 2026, Daily FT uses 'h3' tags for their business news titles
            headlines = soup.find_all('h3') 
            
            print("\n--- Latest Sri Lankan Market Headlines ---")
            news_list = []
            for h in headlines:
                headline_text = h.text.strip()
                if len(headline_text) > 30: # Filters out menus and small text
                    # Extract date from parent element
                    parent_text = h.parent.text.strip()
                    date_text = ""
                    if '\n' in parent_text:
                        date_text = parent_text.split('\n')[-1].strip()
                    
                    if date_text:
                        news_list.append(f"{headline_text} (Release Date: {date_text})")
                    else:
                        news_list.append(headline_text)
            
            # Print the top 5
            for i, news in enumerate(news_list[:5], 1):
                print(f"{i}. {news}")
            
            return news_list
        else:
            print(f"Failed to connect. Status Code: {response.status_code}")
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    get_market_news()