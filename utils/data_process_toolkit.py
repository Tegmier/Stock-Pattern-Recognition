import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm

def calculate_excess_return(security, situation, security_info, avg_alpha_period):
    # return a list of alpha for each beat/miss for each equity
    excess_return_list, beta_list = [], []
    beatmiss_size_alpha = []
    security_name = security["name"]
    # beta = security_info["beta"][security_info["Equity_name"] == security_name].iloc[0]
    for every_beat_up in security[situation]:
        ann_date = every_beat_up["Ann Date"]
        stock_price_period = every_beat_up["Stock_Price_Period"]
        every_beat_up_stock_price=stock_price_period[["Date", "Price"]]
        every_beat_up_index_price=stock_price_period[["Date","Index_Price"]]
        every_beat_up_stock_price_0 = every_beat_up_stock_price["Price"][every_beat_up_stock_price["Date"] == ann_date].iloc[0]
        every_beat_up_index_price_0 = every_beat_up_index_price["Index_Price"][every_beat_up_index_price["Date"] == ann_date].iloc[0]

        every_beat_up_stock_return = (every_beat_up_stock_price["Price"] - every_beat_up_stock_price_0)/every_beat_up_stock_price_0
        every_beat_up_index_return = (every_beat_up_index_price["Index_Price"] - every_beat_up_index_price_0)/every_beat_up_index_price_0

        # cal alpha, alpha = R(stock) - beta * R(index)
        beta = every_beat_up["beta"]
        excess_return = every_beat_up_stock_return - every_beat_up_index_return * beta
        excess_return_list.append(excess_return.values)

        every_beat_up_beta = np.cov(every_beat_up_stock_return, every_beat_up_index_return, bias=True)[0, 1] / np.var(every_beat_up_index_return, ddof=0)
        beta_list.append(every_beat_up_beta)

        # beatmiss_size/alpha
        beatmiss_size = np.abs(every_beat_up["%Surp"])
        avg_alpha = np.mean(excess_return.values[-(avg_alpha_period + 1):-1]) # 从尾部取平均值
        beatmiss_size_alpha.append([avg_alpha, beatmiss_size])
    return excess_return_list, beta_list, beatmiss_size_alpha

def generate_ols_report(X, y):
    X = sm.add_constant(X)  # 在 alpha_list 前加一列 1
    model = sm.OLS(y, X).fit()
    return model

def calculate_stock_return(stock_data):
    """
        Input: stock_data
        Type: Numpy
        Output: return seqence
        Type: Numpy
    """
    day0 = stock_data[0] 
    return (stock_data-day0)/day0