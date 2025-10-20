import numpy as np
import pandas as pd
import os, sys
from datetime import datetime
import matplotlib.pyplot as plt
import shutup
from tqdm import tqdm
from utils.data_process_toolkit import month_to_quarter
from utils.data_preparation_toolkit import situatuion_judgement, situatuion_judgement2
from API.api import get_price_from_csv, get_market_cap_from_csv

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
num_of_parts = 1
beat_threshold = 0.1


PE_check_flag = False
Raw_data_process = False
beat_up_dataset_construction = False
beat_analysis_excel_construction = False
beat_up_dataset_construction_strategy2 = True


path = "data/ern"
sector_df_path = "data/spx_sector.csv"
total_equity_df_prefix = "data/total_equity_df"
bbg_data_collect_path = "data/bbg_data_collect"
final_dataset_path = "data/final_dataset"
price_path = "data/price"

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

        columns.extend(["Equity name", "Ann Date", "Prev Ann Date", "Next Ann Date", "EPS", "Surprise", "PE", "PE Change", "Up Down Flag", "Beat Miss Flag", "Sector", "Quarter", "%Px Chg"])
        for i in range(num_total_quarter):
            col = f"Surprise {8-i}"
            data[col] = data["Surprise"].shift(8-i)
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
    total_equity_df = total_equity_df.reset_index(drop=True)
    print(f"number of total data is {len(total_equity_df)}")
    parts = np.array_split(total_equity_df, num_of_parts)
    for i, part in enumerate(parts):
        part.to_csv(f"{total_equity_df_prefix}_{i}.csv", index=False)

########################################### Beat Up dataset consturction ###########################################

#### 把data/total_equity_df下的文件移动到bbg_data_collect中，然后进行处理
if beat_up_dataset_construction:
    for file in os.listdir(bbg_data_collect_path):
        path = os.path.join(bbg_data_collect_path, file)
        total_equity_df= pd.read_csv(path)
        total_equity_df["total price seq"] = None
        total_equity_df["market cap"] = None
        total_equity_df["situation flag"] = None
        total_equity_df["situation details"] = None
        total_equity_df["price_prev"] = None
        total_equity_df = total_equity_df[(total_equity_df["Beat Miss Flag"] ==1)&(total_equity_df["Up Down Flag"] == 1)&(total_equity_df["Surprise"] >=beat_threshold)].reset_index(drop=True)
        # total_equity_df = total_equity_df[total_equity_df["Beat Miss Flag"] ==1].reset_index(drop=True)
        print(f"number of total data in {file} is {len(total_equity_df)}")
        for idx, row in tqdm(total_equity_df.iterrows(), 
                     total=total_equity_df.shape[0], 
                     desc=f"Beat Up Dataset Construction {file}"):
            equity_name = row["Equity name"]
            ann_date = row["Ann Date"]
            prev_ann_date = row["Prev Ann Date"]
            next_ann_date = row["Next Ann Date"]
            px_change = row["%Px Chg"]
            full_price_seq = get_price_from_csv(equity_name,prev_ann_date, next_ann_date, price_path).reset_index(drop=True)
            market_cap = get_market_cap_from_csv(equity_name, ann_date, price_path)
            try:
                day0_idx = full_price_seq.index[full_price_seq["Date"] == ann_date][0]
                day0_price = full_price_seq.loc[day0_idx, "Price"]
                day1_price = full_price_seq.loc[day0_idx+1, "Price"]
                day1_price_change = (day1_price - day0_price) / day0_price
                if abs(day1_price_change - px_change) > 0.001:
                    price_idx = day0_idx - 1
                else:
                    price_idx = day0_idx
                price_seq_prev = full_price_seq["Price"][price_idx+1-price_prev_window:price_idx+1].tolist()
                price_seq_after = full_price_seq[["Price"]][price_idx:price_idx+price_after_window+1]
            except Exception as e:
                total_equity_df.loc[idx, "situation flag"] = None
                total_equity_df.at[idx, "total price seq"] = None
                total_equity_df.loc[idx, "market cap"] = None
                total_equity_df.loc[idx, "price_prev"] = None
                continue

            if len(price_seq_prev) < price_prev_window or len(price_seq_after) < price_after_window:
                total_equity_df.loc[idx, "situation flag"] = None
                total_equity_df.at[idx, "total price seq"] = None
                total_equity_df.loc[idx, "market cap"] = None
                total_equity_df.loc[idx, "price_prev"] = None
                continue
            else:
                situation_flag, situation_details = situatuion_judgement(price_seq_after, price_after_window)
                if situation_flag == 4:
                    print(equity_name, ann_date, prev_ann_date, next_ann_date, full_price_seq, price_seq_after.head(5))

            total_equity_df.loc[idx, "situation flag"] = situation_flag
            total_equity_df.at[idx, "total price seq"] = full_price_seq["Price"].tolist()
            total_equity_df.loc[idx, "market cap"] = market_cap
            total_equity_df.at[idx, "price_prev"] = price_seq_prev
        print(f"number of data befroe dropna: {len(total_equity_df)}")
        total_equity_df.dropna(subset="situation flag", inplace=True)
        print(f"number of validated in {file} is {len(total_equity_df)}")
        total_equity_df.to_parquet(os.path.join(final_dataset_path, "beat_dataset",file[:-4]+".parquet"), engine="pyarrow", index=False)
    

if beat_analysis_excel_construction:
    for file in os.listdir(bbg_data_collect_path):
        path = os.path.join(bbg_data_collect_path, file)
        total_equity_df= pd.read_csv(path)
        total_equity_df["market cap"] = None
        total_equity_df["situation flag"] = None
        total_equity_df["retrace_date"] = None
        total_equity_df["trough_date"] = None
        total_equity_df["trough_loss"] = None
        total_equity_df["peak_date"] = None
        total_equity_df["peak_gain"] = None
        total_equity_df["full_time_peak_date"] = None
        total_equity_df["full_time_peak_gain"] = None
        for i in range(price_after_window+1):
            total_equity_df[f"day_{i}"] = None
        # total_equity_df = total_equity_df[(total_equity_df["Beat Miss Flag"] ==1)&(total_equity_df["Up Down Flag"] == 1)].reset_index(drop=True)
        total_equity_df = total_equity_df[(total_equity_df["Beat Miss Flag"] ==1)&(total_equity_df["Surprise"] >=beat_threshold)].reset_index(drop=True)
        print(f"number of total data in {file} is {len(total_equity_df)}")
        for idx, row in tqdm(total_equity_df.iterrows(), 
                     total=total_equity_df.shape[0], 
                     desc=f"BBG Data Collection Processing {file}"):
            equity_name = row["Equity name"]
            ann_date = row["Ann Date"]
            prev_ann_date = row["Prev Ann Date"]
            next_ann_date = row["Next Ann Date"]
            px_change = row["%Px Chg"]
            full_price_seq = get_price_from_csv(equity_name,prev_ann_date, next_ann_date, price_path).reset_index(drop=True)
            market_cap = get_market_cap_from_csv(equity_name, ann_date, price_path)
            try:
                #### Day 0 calibration
                day0_idx = full_price_seq.index[full_price_seq["Date"] == ann_date][0]
                day0_price = full_price_seq.loc[day0_idx, "Price"]
                day1_price = full_price_seq.loc[day0_idx+1, "Price"]
                day1_price_change = (day1_price - day0_price) / day0_price
                if abs(day1_price_change - px_change) > 0.001:
                    price_idx = day0_idx - 1
                else:
                    price_idx = day0_idx
                price_seq_prev = full_price_seq["Price"][price_idx+1-price_prev_window:price_idx+1].tolist()
                price_seq_after = full_price_seq[["Price"]][price_idx:price_idx+price_after_window+1]
            except Exception as e:
                total_equity_df.loc[idx, "situation flag"] = None
                total_equity_df.loc[idx, "market cap"] = None
                continue

            if len(price_seq_prev) < price_prev_window or len(price_seq_after) < price_after_window:
                total_equity_df.loc[idx, "situation flag"] = None
                total_equity_df.loc[idx, "market cap"] = None
                continue
            else:
                situation_flag, situation_details = situatuion_judgement(price_seq_after, price_after_window)
                # if situation_flag == 4:
                    # print(equity_name, ann_date, prev_ann_date, next_ann_date, full_price_seq, price_seq_after.head(5))
                    # print("situation 4 detected")

            total_equity_df.loc[idx, "situation flag"] = situation_flag
            total_equity_df.loc[idx, "market cap"] = market_cap
            for i in range(price_after_window+1):
                total_equity_df.loc[idx, f"day_{i}"] = price_seq_after["Price"].reset_index(drop=True)[i]
            if situation_flag == 1:
                total_equity_df.loc[idx, "retrace_date"] = situation_details["retrace_date"]
                total_equity_df.loc[idx, "trough_date"] = situation_details["trough_date"]
                total_equity_df.loc[idx, "trough_loss"] = situation_details["trough_loss"]
                total_equity_df.loc[idx, "peak_date"] = situation_details["peak_date"]
                total_equity_df.loc[idx, "peak_gain"] = situation_details["peak_gain"]
                total_equity_df.loc[idx, "full_time_peak_date"] = situation_details["full_time_peak_date"]
                total_equity_df.loc[idx, "full_time_peak_gain"] = situation_details["full_time_peak_gain"]

        print(f"number of data befroe dropna: {len(total_equity_df)}")
        total_equity_df.dropna(subset="situation flag", inplace=True)
        print(f"number of validated in {file} is {len(total_equity_df)}")
        situation_1 = total_equity_df[total_equity_df["situation flag"]==1].reset_index(drop=True)
        situation_2 = total_equity_df[total_equity_df["situation flag"]==2].reset_index(drop=True)
        situation_3 = total_equity_df[total_equity_df["situation flag"]==3].reset_index(drop=True)
        situation_4 = total_equity_df[total_equity_df["situation flag"]==4].reset_index(drop=True)
        output_path = os.path.join(final_dataset_path, "beat_analysis", f"beat_analysis_{beat_threshold*100}.xlsx")

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            situation_1.to_excel(writer, sheet_name="situation 1", index=False)
            situation_2.to_excel(writer, sheet_name="situation 2", index=False)
            situation_3.to_excel(writer, sheet_name="situation 3", index=False)
            situation_4.to_excel(writer, sheet_name="situation 4", index=False)

        print(f"number of situation 1 in {file} is {len(situation_1)}")
        print(f"number of situation 2 in {file} is {len(situation_2)}")
        print(f"number of situation 3 in {file} is {len(situation_3)}")
        print(f"number of situation 4 in {file} is {len(situation_4)}")