import requests

def get_all_shares_live():
    url = "https://www.cse.lk/api/tradeSummary"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest"
    }

    try:
        # Use an empty JSON body for POST in 2026
        response = requests.post(url, json={}, headers=headers, timeout=10)
        data = response.json().get('reqTradeSummery', [])
        
        print(f"{'Ticker':<12} | {'Price (LKR)':<12} | {'Change %':<10}")
        print("-" * 40)

        for share in data:
            symbol = share.get('symbol', 'N/A')
            
            # 2026 Price field check: Try lastTrade first, then lastTradedPrice
            price = share.get('lastTrade') or share.get('lastTradedPrice')
            
            # If price is still None (no trades today), use Previous Close
            if price is None or price == 0:
                price = share.get('previousClose', 0.00)
            
            change_pct = share.get('percentageChange', 0.00)
            
            print(f"{symbol:<12} | {price:<12.2f} | {change_pct:<10.2f}")

    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    get_all_shares_live()