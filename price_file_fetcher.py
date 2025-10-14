import numpy as np
import pandas as pd
import os, sys
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.data_process_toolkit import month_to_quarter
from utils.data_preparation_toolkit import situatuion_judgement
from API.api import get_price_marketcap

path = "data/ern"
price_data_path = "data/price"

price_start = "2014-01-01"
price_end = "2025-10-1"

price_list_df = pd.read_csv("data/equity_list2.csv")
equity_list = price_list_df["Equity Name"].tolist()

for equity_name in tqdm(equity_list):
    try:
        price_df = get_price_marketcap(equity_name, price_start, price_end)
        price_df["PB Ratio"] = None
    except Exception as e:
        print(f"[Warning] {equity_name}'s data fetching failed", e)
        continue
    price_df.to_csv(price_data_path + "/" + equity_name + ".csv", index=False)


    