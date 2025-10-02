import pandas as pd
from xbbg import blp

# df = blp.bdh(
#     ["TSLA US Equity"],
#     ["PX_LAST","IVOL_Delta"],#,"OPT_DELTA_MID_RT"
#     start_date="20210801",
#     end_date="20210819"
# )

df = blp.bdh(
    ["NVDA US Equity"],
    ["LATEST_ANNOUNCEMENT_DT"],#,"OPT_DELTA_MID_RT"
    start_date="20240101",
    end_date="20240131"
)
# print(df[('NVDA US Equity', 'LATEST_ANNOUNCEMENT_DT')])
print(df)
