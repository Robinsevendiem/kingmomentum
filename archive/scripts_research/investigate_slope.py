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
x_norm = np.linspace(0, 1, window)
weights = 1 + x_norm ** 2

coeffs = np.polyfit(x, log_prices, 1, w=weights)
slope = coeffs[0]

y_pred = np.polyval(coeffs, x)
sse = np.sum(weights * (log_prices - y_pred)**2)
y_mean = np.average(log_prices, weights=weights)
sst = np.sum(weights * (log_prices - y_mean)**2)
r2 = 1 - sse / sst if sst != 0 else 0

score = (np.exp(slope * 252) - 1) * r2 * 100
print("20-day Prices:", prices)
print("Slope:", slope)
print("R2:", r2)
print("Score:", score)
