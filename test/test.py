import pandas as pd
import numpy as np
def situatuion_judgement(stock_price_data):
    price_series = stock_price_data["Price"].to_numpy()
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
                return 1, {"retrace_date": retrace_date, "trough_date":trough_date, "trough_loss":trough_loss, "peak_date":peak_date, "peak_gain":peak_gain}
            else:
                return 2, {}

test = pd.DataFrame({
    "Price": [98,100, 105, 106, 107 ,103, 100, 98, 96, 102,104, 88,103]
})

print(situatuion_judgement(test))