import numpy as np
import pandas as pd
import os, sys
from datetime import datetime
from utils.data_preparation_toolkit import create_security_revenue_data_beat_analysis
from utils.data_visualiztion_toolkit import beat_analysis_data_to_xlsx
import matplotlib.pyplot as plt
import shutup

shutup.please()
# parameters
surprise_beat_threshold = 0.1
influence_period = 20 # trading days
lower_date_boundry = "2015-01-01" 
upper_date_boundry = "2024-10-01"
output_ols_report = True


path = "data/ern"
for file in os.listdir(path):
    df = pd.read_excel(os.path.join(path, file), engine="openpyxl")
    columns = df.columns
    if "P/E" not in columns:
        print(file)
        sys.exit(1)
    else:
        print(f"{file} check passed")

        




# output_folder_path = "output/semi/"
# security_revenue_data_list = []
# for equity in os.listdir(path):
#     equity_name = equity[:-5]
#     data = pd.read_excel(os.path.join(path, equity), engine="openpyxl")
#     data["Ann Date"] = pd.to_datetime(data["Ann Date"], errors='coerce')
#     data["Next Ann Date"] = data["Ann Date"].shift(1)
#     start_date_index = (data["Ann Date"] - pd.to_datetime(lower_date_boundry)).abs().idxmin()
#     end_date_index = (data["Ann Date"] - pd.to_datetime(upper_date_boundry)).abs().idxmin()
#     revenue_estimate = data[["Ann Date", "Per", "Per End", "Reported", "Estimate", "%Surp", "Next Ann Date"]].iloc[end_date_index:start_date_index,:].dropna().reset_index(drop=True)
#     revenue_estimate["%Surp"] = revenue_estimate["%Surp"].replace("N.M.", "0")
#     revenue_estimate["%Surp"] = revenue_estimate["%Surp"].str.rstrip("%").astype(float) / 100
#     revenue_estimate["equity_name"] = equity_name
#     security_revenue_data_list.append(revenue_estimate)

# security_revenue_data = pd.concat(security_revenue_data_list, ignore_index=True)
# surprise = security_revenue_data["%Surp"]
# beat_series_index = surprise[surprise > surprise_beat_threshold].index
# security_revenue_data = security_revenue_data.loc[beat_series_index,:]
# print(f"start to calculate revenue data: {datetime.now()}")
# security_revenue_data = create_security_revenue_data_beat_analysis(security_revenue_data,
#                                                                     surprise_beat_threshold,
#                                                                     influence_period)
# print(f"finish calculating revenue data: {datetime.now()}")
# beat_analysis_data_to_xlsx(security_revenue_data, sector, influence_period, output_folder_path, surprise_beat_threshold)
