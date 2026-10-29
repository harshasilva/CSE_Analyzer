import pandas as pd
import os

# List of major CSE tickers formatted for Yahoo Finance
cse_tickers = [
    {"Symbol": "COMB.N0000", "Yahoo": "COMB-N0000.CM", "Name": "Commercial Bank"},
    {"Symbol": "SAMP.N0000", "Yahoo": "SAMP-N0000.CM", "Name": "Sampath Bank"},
    {"Symbol": "HNB.N0000",  "Yahoo": "HNB-N0000.CM",  "Name": "Hatton National Bank"},
    {"Symbol": "JKH.N0000",  "Yahoo": "JKH-N0000.CM",  "Name": "John Keells Holdings"},
    {"Symbol": "DIAL.N0000", "Yahoo": "DIAL-N0000.CM", "Name": "Dialog Axiata"},
    {"Symbol": "LOLC.N0000", "Yahoo": "LOLC-N0000.CM", "Name": "LOLC Holdings"},
    {"Symbol": "HAYL.N0000", "Yahoo": "HAYL-N0000.CM", "Name": "Hayleys PLC"},
    {"Symbol": "DIST.N0000", "Yahoo": "DIST-N0000.CM", "Name": "Distilleries Company"},
    {"Symbol": "VONE.N0000", "Yahoo": "VONE-N0000.CM", "Name": "Vallibel One"},
    {"Symbol": "ACL.N0000",  "Yahoo": "ACL-N0000.CM",  "Name": "ACL Cables"}
]

os.makedirs("cse_output", exist_ok=True)
df = pd.DataFrame(cse_tickers)
df.to_csv("cse_output/all_share_details.csv", index=False)
print("Created local master list in cse_output/all_share_details.csv")
