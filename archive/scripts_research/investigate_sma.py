import pandas as pd
import numpy as np
from scripts.optimize_exclude_overheated import load_history_data

history_data = load_history_data()
df = history_data['创业板']
df.index = pd.to_datetime(df.index)
prices = df['adj_close'].tail(20).values
sma20 = np.mean(prices)

print("Current Price:", prices[-1])
print("SMA20:", sma20)
