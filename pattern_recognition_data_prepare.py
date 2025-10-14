import numpy as np
import pandas as pd
import os, sys
from datetime import datetime
import matplotlib.pyplot as plt
import shutup
from tqdm import tqdm
from utils.data_process_toolkit import month_to_quarter
from utils.data_preparation_toolkit import situatuion_judgement
from API.api import get_stock_price_data_boolmberg_start_end_period, get_marketcap

shutup.please()
# parameters
influence_period = 20 # trading days
lower_date_boundry = "2020-01-01" 
upper_date_boundry = "2025-05-01"
output_ols_report = True
num_total_quarter = 8
price_prev_window = 20
price_after_window = 20
random_seed = 42
num_of_parts = 10



PE_check_flag = False
Raw_data_process = False
bbg_data_collect = False
final_dataset_construction = True



path = "data/ern"
sector_df_path = "data/spx_sector.csv"
total_equity_df_prefix = "data/total_equity_df"
bbg_data_collect_path = "data/bbg_data_collect"
final_dataset_path = "data/final_dataset"

########################################### Run PE Check ###########################################

if PE_check_flag:
    for file in os.listdir(path):
        df = pd.read_excel(os.path.join(path, file), engine="openpyxl")
        columns = df.columns
        if "P/E" not in columns:
            print(f"[Error] {file} does not have P/E column")
            sys.exit(1)
        else:
            print(f"{file} check passed")

########################################### Raw Data Process ###########################################
if Raw_data_process:
    total_equity_df_list = []
    sector_df = pd.read_csv(sector_df_path)
    sector_dic = dict(zip(sector_df["Ticker"], sector_df["Sector-Code"]))
    for equity in tqdm(os.listdir(path), desc="Raw Data Processing"):  
        columns = []
        equity_name = equity[:-5]
        data = pd.read_excel(os.path.join(path, equity), 
                            engine="openpyxl")
        data["Equity name"] = equity_name
        data["Ann Date"] = pd.to_datetime(data["Ann Date"], errors='coerce')
        data["Prev Ann Date"] = data["Ann Date"].shift(-1)
        data["Next Ann Date"] = data["Ann Date"].shift(1)

        data["EPS"] = data["Comp"]
        try:
            data["Surprise"] = data["%Surp"].replace("N.M.", "0").str.rstrip("%").astype(float) / 100
        except Exception:
            print(f"{equity_name} encountered a problem while transforming Surprise")
            sys.exit(1)

        for i in range(len(data)):
            pe = data.loc[i, "P/E"]
            if type(pe) is str:
                if 'k' in pe:
                    data.loc[i, "P/E"] = float(pe.strip("k")) * 1000
                elif pe == '':
                    pass
                else:
                    tqdm.write(f"{equity_name} has wrong value in PE")
                    sys.exit(1)
        data["PE"] = data["P/E"]
        data["PE Change"] = data["PE"].pct_change(periods=-1)

        data["%Px Chg"] = data["%Px Chg"].replace("N.M.", "0").str.rstrip("%").astype(float) / 100
        # Up: 1, Down: 0
        data["Up Down Flag"] = data["%Px Chg"].apply(lambda x: 1 if x > 0 else 0)
        # Beat: 1, Miss: 0
        data["Beat Miss Flag"] = data["Surprise"].apply(lambda x: 1 if x>0 else 0)

        try:
            data["Sector"] = sector_dic[equity_name]
        except KeyError as e:
            print(f"{equity_name} has no sector code")
            sys.exit(1)

        data["Quarter"] = data["Per End"].apply(month_to_quarter)

        columns.extend(["Equity name", "Ann Date", "Prev Ann Date", "Next Ann Date", "EPS", "Surprise", "PE", "PE Change", "Up Down Flag", "Beat Miss Flag", "Sector", "Quarter"])
        for i in range(num_total_quarter):
            col = f"Surprise {8-i}"
            data[col] = data["Ann Date"].shift(8-i)
            columns.append(col)
        
        start_date_index = (data["Ann Date"] - pd.to_datetime(lower_date_boundry)).abs().idxmin()
        end_date_index = (data["Ann Date"] - pd.to_datetime(upper_date_boundry)).abs().idxmin()
        data = data.iloc[end_date_index:start_date_index,:]

        equity_df = data[columns]
        equity_df.dropna(how="any", inplace=True)
        if not equity_df.empty:
            total_equity_df_list.append(equity_df)

    total_equity_df = pd.concat(total_equity_df_list).reset_index(drop=True)
    total_equity_df = total_equity_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"number of total data is {len(total_equity_df)}")
    parts = np.array_split(total_equity_df, num_of_parts)
    for i, part in enumerate(parts):
        part.to_csv(f"{total_equity_df_prefix}_{i}.csv", index=False)

########################################### Price and other data collection from BBG ###########################################

#### 把data/total_equity_df下的文件移动到bbg_data_collect中，然后进行处理
if bbg_data_collect:
    for file in os.listdir(bbg_data_collect_path):
        path = os.path.join(bbg_data_collect_path, file)
        total_equity_df= pd.read_csv(path)
        for idx, row in tqdm(total_equity_df.iterrows(), 
                     total=total_equity_df.shape[0], 
                     desc=f"BBG Data Collection Processing {file}"):
            equity_name = row["Equity name"]
            ann_date = row["Ann Date"]
            prev_ann_date = row["Prev Ann Date"]
            next_ann_date = row["Next Ann Date"]
            try:
                full_price_seq = get_stock_price_data_boolmberg_start_end_period(equity_name, prev_ann_date, next_ann_date)
                market_cap = get_marketcap(equity_name, ann_date)
                total_equity_df.loc[idx, "Data Availability"] = True
                total_equity_df.loc[idx, "total price seq"] = full_price_seq
                total_equity_df.loc[idx, "market cap"] = market_cap
            except Exception:
                full_price_seq = pd.DataFrame()
                market_cap = None
                total_equity_df.loc[idx, "Data Availability"] = False
                total_equity_df.loc[idx, "total price seq"] = full_price_seq
                total_equity_df.loc[idx, "market cap"] = market_cap
                print(f"{equity_name}, {ann_date} has no price sequence or market cap")
        total_equity_df.to_csv(os.path.join(final_dataset_path, file)) 
                
########################################### Final Dataset Construction ###########################################

if final_dataset_construction:
    file_list = []
    for file in os.listdir(final_dataset_path):
        file_path = os.path.join(final_dataset_path, file)
        file_list.append(pd.read_csv(file_path))
    
    final_dataset = pd.concat(file_list, ignore_index=True)
    final_dataset = final_dataset[(final_dataset["Data Availability"] == True) & (final_dataset["Beat Miss Flag"] == 1)]
    final_dataset = final_dataset.reset_index(drop=True)
    print(f"number of final data is {len(final_dataset)}")

    for idx, row in final_dataset.iterrows():
        equity_name = row["Equity name"]
        ann_date = row["Ann Date"]
        prev_ann_date = row["Prev Ann Date"]
        next_ann_date = row["Next Ann Date"]
        full_price_seq = row["total price seq"]
        print(full_price_seq)
        price_idx = full_price_seq.index[full_price_seq["Date"] == ann_date][0]

        price_seq_prev = full_price_seq["Price"][price_idx+1-price_prev_window:price_idx+1]
        price_seq_after = full_price_seq["Price"][price_idx:]

        situation_flag, situation_details = situatuion_judgement(price_seq_after, price_after_window)
        final_dataset.loc[idx, "situation flag"] = situation_flag
        final_dataset.loc[idx, "situation details"] = situation_details
    final_dataset.to_csv("data/final_dataset.csv", index=False)
    
    
    # data["Next Ann Date"] = data["Ann Date"].shift(1)
    # start_date_index = (data["Ann Date"] - pd.to_datetime(lower_date_boundry)).abs().idxmin()
    # end_date_index = (data["Ann Date"] - pd.to_datetime(upper_date_boundry)).abs().idxmin()
    # revenue_estimate = data[["Ann Date", "Per", "Per End", "Reported", "Estimate", "%Surp", "Next Ann Date"]].iloc[end_date_index:start_date_index,:].dropna().reset_index(drop=True)
    # revenue_estimate["%Surp"] = revenue_estimate["%Surp"].replace("N.M.", "0")
    # revenue_estimate["%Surp"] = revenue_estimate["%Surp"].str.rstrip("%").astype(float) / 100
    # revenue_estimate["equity_name"] = equity_name
    # if PE_check_flag:
    #     pe_series = data["P/E"].iloc[end_date_index:start_date_index].dropna().reset_index(drop=True)
    #     if len(pe_series) != len(revenue_estimate):
    #         print(f"[Error] {equity} P/E length {len(pe_series)} does not match revenue length {len(revenue_estimate)}")
    #         sys.exit(1)
    #     revenue_estimate["P/E"] = pe_series
    # security_revenue_data_list.append(revenue_estimate)



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
