import numpy as np
import pandas as pd
import os
from datetime import datetime
from utils.data_preparation_toolkit import create_security_revenue_data, create_security_revenue_data_consecutive_beat
from utils.data_visualiztion_toolkit import (path_check,
                                             output_beat_miss_statistical_feature, 
                                             plot_alpha_figure, 
                                             output_beat_miss_beta, 
                                             plot_alpha_beatmiss_ratio_figure,plot_stock_price, 
                                             report_overall_feature_cross_region_sector,
                                             plot_retrace_distribution,
                                             report_retrace_statistics,
                                             plot_retrace_beatsize,
                                             security_revenue_data_to_xlsx)
from utils.data_process_toolkit import calculate_excess_return, calculate_stock_return
import matplotlib.pyplot as plt
import shutup

shutup.please()
# parameters
surprise_beat_threshold = 0.2
surprise_miss_threshold = 0.1
influence_period = 20 # trading days
avg_alpha_period = 3 # trading days
lower_date_boundry = "2015-01-01" 
upper_date_boundry = "2024-10-01"
# start_quarter = "Q1 24" 
# end_quarter = "Q1 19"
output_ols_report = True
# beat_definition_period = 3
# miss_definition_period = 3
retrace_range_up = 0.10
retrace_range_down = 0.02

# Switch Region Sector
option = 3 # 1:usa semiconductor, 2:usa it, 3:mixed semiconductor
if option == 1:
    region = "USA" 
    sector = "Semicondctor"
    sheetname = "semiconductor"
    import config.usa_semi_config as config
if option == 2:
    region = "USA" 
    sector = "IT"
    sheetname = "IT"
    import config.usa_it_config as config
if option ==3:
    region = "mixed"
    sector = "Semiconductor"
    sheetname = "semiconductor"
    import config.mix_semi_config as config
    output_folder_path  = f"output/"
path_check(output_folder_path)

# path
ern_folder_path = config.ern_folder_path
security_list_path = config.security_list_path

# get security list
security_info = pd.read_excel(security_list_path, engine="openpyxl", sheet_name=sheetname)[["Equity_name"]]
security_list = security_info["Equity_name"].to_list()
security_revenue_data = []

# prepare security_revenue_data
for security in security_list:
    
    erning_path = os.path.join(ern_folder_path, security+".xlsx")
    data = pd.read_excel(erning_path, engine="openpyxl")

    data["Ann Date"] = pd.to_datetime(data["Ann Date"], errors='coerce')
    data["Next Ann Date"] = data["Ann Date"].shift(1)

    # search the nearest start date and end date
    start_date_index = (data["Ann Date"] - pd.to_datetime(lower_date_boundry)).abs().idxmin()
    end_date_index = (data["Ann Date"] - pd.to_datetime(upper_date_boundry)).abs().idxmin()

    revenue_estimate = data[["Ann Date", "Per", "Per End", "Reported", "Estimate", "%Surp", "Next Ann Date"]].iloc[end_date_index:start_date_index,:].dropna().reset_index(drop=True)

    security_revenue_data.append({"name":security, "revenue_data": revenue_estimate})
print(f"start to calculate revenue data: {datetime.now()}")
security_revenue_data = create_security_revenue_data_consecutive_beat(security_revenue_data,
                                                                      surprise_beat_threshold,
                                                                      surprise_miss_threshold,
                                                                      influence_period,
                                                                      retrace_range_up,
                                                                      retrace_range_down)
print(f"finish calculating revenue data: {datetime.now()}")
# Generate an Excel
security_revenue_data_to_xlsx(security_revenue_data, region, sector, influence_period, output_folder_path, surprise_beat_threshold, surprise_miss_threshold)