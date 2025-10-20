import numpy as np
import pandas as pd
    
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
            
def situatuion_judgement2(stock_price_data, end_date):
    full_price_series = stock_price_data["Price"].to_numpy()
    price_series = full_price_series[:end_date+1]
    day_0_price = price_series[0]
    day_1_price = price_series[1]
    if day_1_price <= day_0_price:
        # a beat down
        return 3, {}
    # Define situation 1: above day1 price on the final day
    # Define situation 2: below day1 price on the final day
    else:
        # a beat up
        if price_series[-1] >= day_1_price:
            return 1, {}
        else:
            return 2, {}