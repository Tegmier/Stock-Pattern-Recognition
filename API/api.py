import yfinance as yf
import pandas as pd
from xbbg import blp
import datetime
import numpy as np
import os

def get_stock_data_yf(ticker, start_date, trading_days):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date)
    data = data.iloc[:trading_days]
    return data

def get_stock_price_data_boolmberg_start_period(ticker, start_date, number_of_trading_days):
    # start_date is a date type time value
    end_date = start_date + pd.DateOffset(months=3)
    df = blp.bdh(
    [ticker],
    ["PX_LAST"],
    start_date=start_date,
    end_date=end_date)
    df = df.reset_index()
    df.columns = ["Date", "Price"]
    return df.iloc[:number_of_trading_days]

def get_stock_price_data_boolmberg_start_end_period(ticker, start_date, end_date):
    # start_date is a date type time value
    # 获取的数据是从以前到现在排列的
    # 返回dataframe类型的数据
    df = blp.bdh(
    [ticker],
    ["PX_LAST"],
    start_date=start_date,
    end_date=end_date)
    df = df.reset_index()
    df.columns = ["Date", "Price"]
    return df

def get_stock_price_data_boolmberg_start_end(ticker, start_date, end_date):
    # start_date is a date type time value
    # 获取的数据是从以前到现在排列的
    # 返回dataframe类型的数据
    df = blp.bdh(
    [ticker],
    ["PX_LAST"],
    start_date=start_date,
    end_date=end_date)
    df = df.reset_index()
    df.columns = ["Date", "Price"]
    return df

def get_one_day_indicator_data(ticker, current_date):
    indicator_dic = {}
    try:
        df = blp.bdh(
            [ticker],
            ["CUR_MKT_CAP", "PE_RATIO", "PX_TO_BOOK_RATIO"],
            start_date=current_date,
            end_date=current_date)
        df = df.reset_index()
        df.columns= ["Date", "Market Cap", "PE Ratio", "PB Ratio"]
    except Exception as e:
        print("[Warning]", e)
        return {}
    else:
        indicator_dic["Date"] = df["Date"][0]
        indicator_dic["Market Cap"] = df["Market Cap"][0]
        indicator_dic["PE Ratio"] = df["PE Ratio"][0]
        indicator_dic["PB Ratio"] = df["PB Ratio"][0]
    return indicator_dic

def get_marketcap(ticker, current_date):
    df = blp.bdh(
        [ticker],
        ["CUR_MKT_CAP"],
        start_date=current_date,
        end_date=current_date)
    df = df.reset_index()
    df.columns= ["Date", "Market Cap"]
    return df["Market Cap"][0]

def get_price_marketcap(ticker, start_date, end_date):
    df = blp.bdh(
        [ticker],
        ["PX_LAST", "CUR_MKT_CAP", "PE_RATIO"],
        start_date=start_date,
        end_date=end_date)
    df = df.reset_index()
    df.columns= ["Date", "Price", "Market Cap", "PE Ratio"]
    return df

def get_price_from_csv(ticker, start_date, end_date, csv_path):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    file_path = os.path.join(csv_path, ticker + ".csv")
    info_df = pd.read_csv(file_path, parse_dates=["Date"])
    info_df = info_df[["Date", "Price"]]
    start_index = abs(info_df["Date"] - start_date).idxmin()
    end_index = abs(info_df["Date"] - end_date).idxmin()
    return info_df.loc[start_index:end_index]

def get_market_cap_from_csv(ticker, date, csv_path):
    date = pd.to_datetime(date)
    file_path = os.path.join(csv_path, ticker + ".csv")
    info_df = pd.read_csv(file_path, parse_dates=["Date"])
    idx = abs(info_df["Date"] - date).idxmin()
    return info_df.loc[idx, "Market Cap"]





# # test
# ticker = "AZO US Equity"
# start_date = datetime.date(2014,1,1)
# end_date = datetime.date(2025,10,1)
# # earnings_date = datetime.date(2025,8,27)
# df = get_price_marketcap(ticker, start_date, end_date)
# print(df)
# # df.to_csv("azo_price_marketcap.csv", index=False)
# # print(get_one_day_indicator_data(ticker, earnings_date))





# # test
# ticker = "NVDA US Equity"
# start_date = datetime.date(2025,10,1)
# end_date = datetime.date(2025,11,30)
# # earnings_date = datetime.date(2025,8,27)
# print(get_stock_price_data_boolmberg_start_end(ticker, start_date, end_date))
# # print(get_one_day_indicator_data(ticker, earnings_date))

# # test
# ticker = "ADP US Equity"
# start_date = datetime.date(2021,4,28)
# end_date = datetime.date(2021,10,27)
# ann_date = datetime.date(2021,7,28)
# # earnings_date = datetime.date(2025,8,27)
# print(get_stock_price_data_boolmberg_start_end_period(ticker, start_date, end_date))
# print(get_marketcap(ticker, start_date))