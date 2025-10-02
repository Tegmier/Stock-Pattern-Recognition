import numpy as np
import pandas as pd
from utils.strategy import (beat_judgement,
                            miss_judgement,
                            up_down_judgement,
                            up_down_judgement_adjusted,
                            basic_up_down_judgement,
                            calculate_retrace,
                            calculate_retrace_by_range)
from API.api import get_stock_price_data_boolmberg_start_end,get_stock_price_data_boolmberg_start_end_period
import sys

def create_security_revenue_data(security_revenue_data, 
                                 surprise_beat_threshold, 
                                 surprise_miss_threshold, 
                                 influence_period):
    for security in security_revenue_data:
        security_name = security["name"]
        security_data = security["revenue_data"]
        security_data["%Surp"] = security_data["%Surp"].replace("N.M.", "0")
        security_data["%Surp"] = security_data["%Surp"].str.rstrip("%").astype(float) / 100
        surprise = security_data["%Surp"]

        beat_up_list, beat_down_list, miss_up_list, miss_down_list = [], [], [], []
        # Beat calculation
        beat_series_index = surprise[surprise >= surprise_beat_threshold].index
        beat_data = security_data.loc[beat_series_index,:]  
        for i in range(beat_data.shape[0]):
            item = beat_data.iloc[i,:]
            beat_ann_date = item["Ann Date"]
            next_beat_ann_date = item["Next Ann Date"]
            stock_price_data = get_stock_price_data_boolmberg_start_end(security_name, beat_ann_date, next_beat_ann_date)
            # positional_index_start = security_stock_price_data[security_stock_price_data["Date"] == beat_ann_date].index.to_list()[0]
            # positional_index_end = security_stock_price_data[security_stock_price_data["Date"] == next_beat_ann_date].index.to_list()[0]
            # stock_range = security_stock_price_data.iloc[positional_index_start:positional_index_start+influence_period]
            print(beat_ann_date)
            print(next_beat_ann_date)
            print(stock_price_data)
            security_stock_price_data = 0
            if security_stock_price_data["Price"][positional_index_start+1] >= security_stock_price_data["Price"][positional_index_start]:
                # call a beat up
                beat_up_list.append({"Ann Date":beat_ann_date, "Per":item["Per"], "%Surp":item["%Surp"], "beta":item["beta"], "Next Ann Date":item["Next Ann Date"], "Stock_Price_Period":stock_range})
            else:
                # call a beat down
                beat_down_list.append({"Ann Date":beat_ann_date, "Per":item["Per"], "%Surp":item["%Surp"], "beta":item["beta"], "Next Ann Date":item["Next Ann Date"], "Stock_Price_Period":stock_range})

        # miss calculation
        miss_series_index = surprise[surprise <= -surprise_miss_threshold].index
        miss_data = security_data.loc[miss_series_index,:]
        for i in range(miss_data.shape[0]):
            item = miss_data.iloc[i,:]
            miss_ann_date = item["Ann Date"]
            next_miss_ann_date = item["Next Ann Date"]
            positional_index_start = security_stock_price_data[security_stock_price_data["Date"] == miss_ann_date].index.to_list()[0]
            positional_index_end = security_stock_price_data[security_stock_price_data["Date"] == next_miss_ann_date].index.to_list()[0]
            stock_range = security_stock_price_data.iloc[positional_index_start:positional_index_start+influence_period]
            if security_stock_price_data["Price"][positional_index_start+1] >= security_stock_price_data["Price"][positional_index_start]:
                # call a miss up
                miss_up_list.append({"Ann Date":miss_ann_date, "Per":item["Per"], "%Surp":item["%Surp"], "beta":item["beta"], "Next Ann Date":item["Next Ann Date"], "Stock_Price_Period":stock_range})
            else:
                # call  miss down
                miss_down_list.append({"Ann Date":miss_ann_date, "Per":item["Per"], "%Surp":item["%Surp"], "beta":item["beta"], "Next Ann Date":item["Next Ann Date"], "Stock_Price_Period":stock_range})
        

        # avg_beat & avg_miss
        beat_surprise = surprise[surprise>=0]
        miss_surprise = surprise[surprise<0]

        beat_surprise_max = beat_surprise.max()
        beat_surprise_min = beat_surprise.min()
        miss_surprise_max = miss_surprise.min()
        miss_surprise_min = miss_surprise.max()

        beat_surprise_avg = beat_surprise.mean()
        miss_surprise_avg = miss_surprise.mean()
        beat_surprise_var = beat_surprise.var()
        miss_surprise_var = miss_surprise.var()

        statistics = {}
        statistics["beat_surprise_avg"] = beat_surprise_avg
        statistics["miss_surprise_avg"] = miss_surprise_avg
        statistics["beat_surprise_var"] = beat_surprise_var
        statistics["miss_surprise_var"] = miss_surprise_var
        statistics["beat_surprise_max"] = beat_surprise_max
        statistics["beat_surprise_min"] = beat_surprise_min
        statistics["miss_surprise_max"] = miss_surprise_max
        statistics["miss_surprise_min"] = miss_surprise_min

        
        security["beat_up"] = beat_up_list
        security["beat_down"] = beat_down_list
        security["miss_up"] = miss_up_list
        security["miss_down"] = miss_down_list
        security["statistics"] = statistics

    return security_revenue_data
    
def create_security_revenue_data_consecutive_beat(security_revenue_data, surprise_beat_threshold, surprise_miss_threshold, influence_period, retrace_range_up, retrace_range_down):
    for security in security_revenue_data:
        security_name = security["name"]
        security_data = security["revenue_data"]
        security_data["%Surp"] = security_data["%Surp"].replace("N.M.", "0")
        security_data["%Surp"] = security_data["%Surp"].str.rstrip("%").astype(float) / 100
        surprise = security_data["%Surp"]

        beat_up_list, beat_down_list, miss_up_list, miss_down_list, fluctuate_list = [], [], [], [], []
        # Beat calculation
        beat_series_index = surprise[surprise > surprise_beat_threshold].index
        beat_data = security_data.loc[beat_series_index,:]  
        for i in range(beat_data.shape[0]):
            item = beat_data.iloc[i,:]
            beat_ann_date = item["Ann Date"]
            next_beat_ann_date = item["Next Ann Date"]
            stock_price_data = get_stock_price_data_boolmberg_start_end_period(security_name, beat_ann_date, next_beat_ann_date, influence_period)
            up_down_flag = basic_up_down_judgement(stock_price_data)
            dic_beat = {"Ann Date":beat_ann_date, "Per":item["Per"], "%Surp":item["%Surp"], "Next Ann Date":item["Next Ann Date"], "Stock_Price_Period":stock_price_data, "Retrace":retrace_list_preparation(stock_price_data, up_down_flag, influence_period)}
            if up_down_flag == 0:
                # call a beat up
                beat_up_list.append(dic_beat)
            else:
                # call a beat down
                beat_down_list.append(dic_beat)

        miss_series_index = surprise[surprise <= -surprise_miss_threshold].index
        miss_data = security_data.loc[miss_series_index,:]
        for i in range(miss_data.shape[0]):
            item = miss_data.iloc[i,:]
            miss_ann_date = item["Ann Date"]
            next_miss_ann_date = item["Next Ann Date"]
            stock_price_data = get_stock_price_data_boolmberg_start_end_period(security_name, miss_ann_date, next_miss_ann_date, influence_period)
            up_down_flag = basic_up_down_judgement(stock_price_data)
            dic_miss = {"Ann Date":miss_ann_date, "Per":item["Per"], "%Surp":item["%Surp"], "Next Ann Date":item["Next Ann Date"], "Stock_Price_Period":stock_price_data, "Retrace":retrace_list_preparation(stock_price_data, up_down_flag, influence_period)}
            if up_down_flag == 0:
                # call a miss up
                miss_up_list.append(dic_miss)
            else:
                # call a miss down
                miss_down_list.append(dic_miss)
        # avg_beat & avg_miss
        beat_surprise = surprise[surprise>=0]
        miss_surprise = surprise[surprise<0]

        beat_surprise_max = beat_surprise.max()
        beat_surprise_min = beat_surprise.min()
        miss_surprise_max = miss_surprise.min()
        miss_surprise_min = miss_surprise.max()

        beat_surprise_avg = beat_surprise.mean()
        miss_surprise_avg = miss_surprise.mean()
        beat_surprise_var = beat_surprise.var()
        miss_surprise_var = miss_surprise.var()

        statistics = {}
        statistics["beat_surprise_avg"] = beat_surprise_avg
        statistics["miss_surprise_avg"] = miss_surprise_avg
        statistics["beat_surprise_var"] = beat_surprise_var
        statistics["miss_surprise_var"] = miss_surprise_var
        statistics["beat_surprise_max"] = beat_surprise_max
        statistics["beat_surprise_min"] = beat_surprise_min
        statistics["miss_surprise_max"] = miss_surprise_max
        statistics["miss_surprise_min"] = miss_surprise_min

        
        security["beat_up"] = beat_up_list
        security["beat_down"] = beat_down_list
        security["miss_up"] = miss_up_list
        security["miss_down"] = miss_down_list
        security["statistics"] = statistics

    return security_revenue_data


def retrace_list_preparation(stock_price_data, up_down_flag, influence_period):
    # 0->up, 1->down
    day0_price = stock_price_data["Price"][0]
    price_seq = stock_price_data["Price"].to_numpy() #有可能day1是最大值
    if up_down_flag == 0:
        retrace_date = np.argmax(price_seq)
        retrace_rate = (price_seq[retrace_date]-day0_price)/day0_price
        find_recover = price_seq[retrace_date:] <= day0_price
        recover_date = -1 if np.argmax(find_recover) == 0 else retrace_date+np.argmax(find_recover)
        trend = 0 if retrace_date < influence_period*0.9 else -1
        return {"retrace_date":retrace_date, "retrace_rate":retrace_rate, "recover_date":recover_date, "trend":trend}
    
    if up_down_flag ==1:
        retrace_date = np.argmin(price_seq)
        retrace_rate = (day0_price-price_seq[retrace_date])/day0_price
        find_recover = price_seq[retrace_date:] >= day0_price
        recover_date = -1 if np.argmax(find_recover) == 0 else retrace_date+np.argmax(find_recover)
        trend = 0 if retrace_date < influence_period*0.9 else -1
        return {"retrace_date":retrace_date, "retrace_rate":retrace_rate, "recover_date":recover_date, "trend":trend}
    
def create_security_revenue_data_beat_analysis(security_revenue_data, surprise_beat_threshold, influence_period, ):
    surprise = security_revenue_data["%Surp"]
    beat_series_index = surprise[surprise > surprise_beat_threshold].index
    security_revenue_data = security_revenue_data.loc[beat_series_index,:]

    equity_name_list = []

    ann_date_list = []
    next_ann_date_list = []
    per_list = []
    per_end_list = []
    reported_list = []
    estimate_list = []
    sup_list = []
    stock_price_list = []
    beat_detail_list = []
    situation_flag_list = []

    for idx,row in security_revenue_data.iterrows():
        equity_name = row["equity_name"]
        beat_ann_date = row["Ann Date"]
        next_beat_ann_date = row["Next Ann Date"]
        per = row["Per"]
        per_end = row["Per End"]
        reported = row["Reported"]
        estimate = row["Estimate"]
        sup = row["%Surp"]
        try:
            stock_price_data = get_stock_price_data_boolmberg_start_end(equity_name, beat_ann_date, next_beat_ann_date)
            if len(stock_price_data) <= 20:
                raise ValueError("长度返回出错")
        except ValueError as e:
            print(f"equity_name: {equity_name}")
            print(f"Ann date: {beat_ann_date}")
            continue
        situation_flag, beat_detail = situatuion_judgement(stock_price_data, influence_period)
        equity_name_list.append(equity_name)
        ann_date_list.append(beat_ann_date)
        next_ann_date_list.append(next_beat_ann_date)
        per_list.append(per)
        per_end_list.append(per_end)
        reported_list.append(reported)
        estimate_list.append(estimate)
        sup_list.append(sup)
        stock_price_list.append(stock_price_data)
        situation_flag_list.append(situation_flag)
        beat_detail_list.append(beat_detail)

    result=pd.DataFrame({"equity_name":equity_name_list,
                         "ann_date":ann_date_list,
                         "next_ann_date":next_ann_date_list,
                         "per":per_list,
                         "per_end":per_end_list,
                         "reported":reported_list,
                         "estimate":estimate_list,
                         "sup":sup_list,
                         "stock_price":stock_price_list,
                         "situation_flag":situation_flag_list,
                         "beat_detail":beat_detail_list})
    result = result.drop_duplicates(subset=["equity_name", "ann_date"])
    return result

def situatuion_judgement(stock_price_data, influence_period):
    full_price_series = stock_price_data["Price"].to_numpy()
    price_series = full_price_series[:influence_period+1]
    day_0_price = price_series[0]
    day_1_price = price_series[1]
    if day_1_price <= day_0_price:
        # a beat down
        return 4, {}
    else:
        # a beat up
        serires_from_day_1 = price_series[1:]
        if serires_from_day_1[1:].min() > day_1_price:
            return 3, {}
        else:
            idx = np.argmin(serires_from_day_1[1:]>day_1_price) + 1 #idx相对于serires_from_day_1
            series_recover = serires_from_day_1[idx:]
            if np.any(series_recover > day_1_price):
                retrace_date = idx+1
                series_by_recover = series_recover[:np.argmax(series_recover>day_1_price)]
                trough_date = retrace_date + np.argmin(series_by_recover)
                trough_price = serires_from_day_1[trough_date-1]
                trough_loss = (day_1_price-trough_price)/day_1_price
                peak_date = 1 + idx + np.argmax(series_recover)
                peak_price = serires_from_day_1[peak_date-1]
                peak_gain = (peak_price-day_1_price)/day_1_price
                full_time_peak_date = np.argmax(full_price_series[trough_date:]) + trough_date
                full_time_peak_price = full_price_series[full_time_peak_date]
                full_time_peak_gain = (full_time_peak_price - day_1_price)/day_1_price
                return 1, {"retrace_date": retrace_date, 
                           "trough_date":trough_date, 
                           "trough_loss":trough_loss, 
                           "peak_date":peak_date, 
                           "peak_gain":peak_gain, 
                           "full_time_peak_date":full_time_peak_date, 
                           "full_time_peak_gain":full_time_peak_gain}
            else:
                return 2, {}



