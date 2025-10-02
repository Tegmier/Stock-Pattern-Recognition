import numpy as np
import pandas as pd

def beat_judgement(stock_range):
    # beat up返回0， beat down 返回1， fluctuate返回2
    # beat up 策略：1.day1大于0  2.前四个点的avg大于0  3.后两个点均大于0
    # beat down 策略：1.day1大于0  2.前四个点的avg小于0  3.后两个点均小于0
    first_five = stock_range["Price"].iloc[0:5].to_numpy()
    first_five = (first_five-first_five[0])/first_five[0]
    
    avg = np.mean(first_five)
    is_last_two_over_zero = first_five[3]>0 and first_five[4]>0
    is_last_two_below_zero = first_five[2]<0 and first_five[3]<0
    if first_five[1] > 0 and avg > 0 and is_last_two_over_zero:
        return 0
    elif first_five[1] < 0 and avg < 0 and is_last_two_below_zero:
        return 1
    else:
        return 2
    
def miss_judgement(stock_range):
    # miss up返回0， miss down 返回1， fluctuate返回2
    # miss up 策略：1.day1<0  2.前四个点的avg>0  3.后两个点>0
    # miss down 策略：1.day1<0  2.前四个点的avg小于0  3.后两个点均<0
    first_five = stock_range["Price"].iloc[0:5].to_numpy()
    first_five = (first_five-first_five[0])/first_five[0]
    
    avg = np.mean(first_five)
    is_last_two_over_zero = first_five[3]>0 and first_five[4]>0
    is_last_two_below_zero = first_five[3]<0 and first_five[4]<0
    if first_five[1] < 0 and avg > 0 and is_last_two_over_zero:
        return 0
    elif first_five[1] < 0 and avg < 0 and is_last_two_below_zero:
        return 1
    else:
        return 2
    
def basic_up_down_judgement(stock_price_data):
    # up返回0, down返回1
    day0 = stock_price_data["Price"][0]
    day1 = stock_price_data["Price"][1]
    if day1>=day0:
        return 0
    else:
        return 1

def up_down_judgement(stock_range):
    # up返回0， down 返回1， fluctuate返回2
    # up 策略：1.前四个点的avg>0  2.有3个点>0  3.后两个点>0
    # down 策略：1.前四个点的avg<0  2.有3个点<0  3.后两个点<0
    first_five = stock_range["Price"].iloc[0:5].to_numpy()
    first_five = (first_five-first_five[0])/first_five[0]

    avg = np.mean(first_five)
    count_above_zero = np.sum(first_five > 0)
    count_below_zero = np.sum(first_five < 0)
    is_last_two_over_zero = first_five[3]>0 and first_five[4]>0
    is_last_two_below_zero = first_five[3]<0 and first_five[4]<0
    if avg>0 and count_above_zero >=3 and is_last_two_over_zero:
        return 0
    elif avg<0 and count_below_zero>=3 and is_last_two_below_zero:
        return 1
    else:
        return 2

def up_down_judgement_adjusted(stock_range):
    # up返回0， down 返回1， fluctuate返回2
    # up 策略：1.前四个点的avg>0  2.有3个点>0  3.后两个点>0
    # down 策略：1.前四个点的avg<0  2.有3个点<0  3.后两个点<0
    first_five = stock_range["Price"].iloc[0:5].to_numpy()
    first_five = (first_five-first_five[0])/first_five[0]

    avg = np.mean(first_five)
    count_above_zero = np.sum(first_five > 0)
    count_below_zero = np.sum(first_five < 0)
    # is_last_two_over_zero = first_five[3]>0 and first_five[4]>0
    # is_last_two_below_zero = first_five[3]<0 and first_five[4]<0
    if avg>0 and count_above_zero >=3:
        return 0
    elif avg<0 and count_below_zero>=3:
        return 1
    else:
        return 2

def calculate_retrace(stock_range, up_down_flag):
    # up_down_flag 0:up 1:down
    seq = stock_range["Price"].to_numpy()
    seq = (seq - seq[0])/seq[0]
    trend_range = seq[5:]
    if up_down_flag == 0:
        idx = np.argmax(trend_range <= 0) + 5 if np.any(trend_range <= 0) else -1
        return idx
    if up_down_flag == 1:
        idx = np.argmax(trend_range >= 0) + 5 if np.any(trend_range >= 0) else -1
        return idx
    
def calculate_retrace_by_range(stock_price_seq, up_down_flag, retrace_range_up, retrace_range_down):
    # the retrace will happen on this day
    # up:0 down:1 fluctuate:2
    # 从trading day1开始
    stock_price_seq = stock_price_seq.to_numpy()[1:]
    if up_down_flag == 0:
        # 创建数组，这个数组的每一个位置t代表day0到dayt的最大股价
        stock_price_max_seq = np.maximum.accumulate(stock_price_seq)
        retrace_rate = (stock_price_seq-stock_price_max_seq)/stock_price_seq
        # 回撤时间点的回撤率设置为回撤率的绝对值，非回撤时间的的回撤率设置为0
        retrace_rate[retrace_rate>=0] = 0
        retrace_rate[retrace_rate<0] = np.abs(retrace_rate[retrace_rate<0])
        idx = np.where(retrace_rate > retrace_range_up)[0]
        return idx[0]+2 if idx.size>0 else -1
    if up_down_flag == 1:
        # 创建数组，这个数组的每一个位置t代表day0到dayt的最小股价
        stock_price_min_seq = np.minimum.accumulate(stock_price_seq)
        retrace_rate = (stock_price_seq-stock_price_min_seq)/stock_price_seq
        # 回撤时间点的回撤率设置为回撤率的绝对值，非回撤时间的的回撤率设置为0
        retrace_rate[retrace_rate<=0] = 0
        retrace_rate[retrace_rate>0] = np.abs(retrace_rate[retrace_rate>0])
        idx = np.where(retrace_rate > retrace_range_down)[0]
        return idx[0]+2 if idx.size>0 else -1
    if up_down_flag == 2:
        return 0


# # Test case for calculate_retrace_by_range
# seq = [102, 100, 101, 100, 96, 97, 98, 102, 105]
# # seq = [100, 102, 111, 110,109.99, 109.98, 105, 108, 122, 100]
# seq = pd.Series(seq)

# print(calculate_retrace_by_range(seq, 1, 0.03, 0.03))


