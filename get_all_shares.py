import requests
import pandas as pd
import time

def get_full_cse_list():
    url = "https://www.cse.lk/api/getTradeSummary"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    print("Connecting to CSE to fetch all listed shares...")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            # These columns are usually returned: symbol, companyName, lastTradedPrice, etc.
            df.to_csv("cse_output/all_share_details.csv", index=False)
            print(f"Successfully saved {len(df)} companies to cse_output/all_share_details.csv")
            return df
        else:
            print(f"Server error: {response.status_code}. The CSE API might be down.")
    except Exception as e:
        print(f"Connection failed: {e}")
    return None

if __name__ == "__main__":
    get_full_cse_list()
    