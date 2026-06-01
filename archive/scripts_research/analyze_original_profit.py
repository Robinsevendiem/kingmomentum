import pandas as pd
import numpy as np

trades = pd.read_csv('/Users/robin-macmini/Documents/trae_projects/foolreveal/data/reference/2017.8.1-2026.3.23调仓.csv')
trades['调仓时间'] = pd.to_datetime(trades['调仓时间'].str.replace(' 开盘', ''))
trades = trades.sort_values('调仓时间').reset_index(drop=True)

# Calculate trade returns
# A trade consists of: Buy at trade T-1, Sell at trade T
trades['prev_nav'] = trades['调仓后净值'].shift(1).fillna(1.0)
trades['trade_return'] = trades['调仓后净值'] / trades['prev_nav'] - 1

print("Top 5 best trades:")
print(trades.sort_values('trade_return', ascending=False)[['调仓时间', '卖出', 'trade_return']].head(5))

print("\nTop 5 worst trades:")
print(trades.sort_values('trade_return', ascending=True)[['调仓时间', '卖出', 'trade_return']].head(5))

print("\nAverage return by asset:")
print(trades.groupby('卖出')['trade_return'].mean().sort_values(ascending=False))

print("\nTotal compounded return by asset (approximate):")
print(trades.groupby('卖出')['trade_return'].apply(lambda x: (1+x).prod() - 1).sort_values(ascending=False))
