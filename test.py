import numpy as np
import pandas as pd
import os
from datetime import datetime

import matplotlib.pyplot as plt
import shutup

shutup.please()
# parameters

path = "data/spx"
sector = "SPX"
for equity in os.listdir(path):
    data = pd.read_excel(os.path.join(path, equity), engine="openpyxl")
    if "P/E" not in data.columns:
        print(equity)