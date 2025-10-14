import numpy as np
import pandas as pd
import os, sys
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.data_process_toolkit import month_to_quarter
from utils.data_preparation_toolkit import situatuion_judgement
from API.api import get_stock_price_data_boolmberg_start_end_period, get_marketcap

path = "data/ern"
price_start = "2005-01-01"
price_end = "2023-12-31"


for file in os.listdir(path):
    equity_name = file[:-5]
    print(equity_name)