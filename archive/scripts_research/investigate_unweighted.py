import pandas as pd
import numpy as np
from scripts.optimize_exclude_overheated import load_history_data

history_data = load_history_data()
df = history_data['创业板']
df.index = pd.to_datetime(df.index)
prices = df['adj_close'].tail(20).values
log_prices = np.log(prices)

window = 20
x = np.arange(window)

coeffs = np.polyfit(x, log_prices, 1)
slope = coeffs[0]

print("Unweighted Slope:", slope)

# Also check simple return
simple_return = prices[-1] / prices[0] - 1
print("Simple 20-day return:", simple_return)
