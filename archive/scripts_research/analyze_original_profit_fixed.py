import pandas as pd
import numpy as np

trades = pd.read_csv('/Users/robin-macmini/Documents/trae_projects/foolreveal/data/reference/2017.8.1-2026.3.23调仓.csv')
trades['调仓时间'] = pd.to_datetime(trades['调仓时间'].str.replace(' 开盘', ''))
trades = trades.sort_values('调仓时间').reset_index(drop=True)

# Fix the bug in the data where '卖出' is sometimes incorrectly '现金' instead of the previously bought asset
corrected_sells = []
current_holding = '现金'

for i, row in trades.iterrows():
    # What we actually sold should be what we were currently holding
    corrected_sells.append(current_holding)
    
    # What we actually bought is '买入'. If the row says '现金' for both, it means it sold the asset and bought cash
    if row['买入'] == '现金' and row['卖出'] == '现金':
        current_holding = '现金'
    elif row['买入'] == '现金' and row['卖出'] != '现金':
        current_holding = '现金' # Shouldn't happen based on the bug pattern but just in case
    elif row['卖出'] == '现金' and row['买入'] != '现金':
        # This is a buy from cash.
        current_holding = row['买入']
    else:
        current_holding = row['买入']
        
trades['corrected_sell'] = corrected_sells
trades['corrected_buy'] = trades['买入']
trades.loc[(trades['买入'] == '现金') & (trades['卖出'] == '现金'), 'corrected_buy'] = '现金'

trades['prev_nav'] = trades['调仓后净值'].shift(1).fillna(1.0)
trades['trade_return'] = trades['调仓后净值'] / trades['prev_nav'] - 1

print("Top 5 best trades (fixed):")
print(trades.sort_values('trade_return', ascending=False)[['调仓时间', 'corrected_sell', 'corrected_buy', 'trade_return']].head(5))

print("\nTop 5 worst trades (fixed):")
print(trades.sort_values('trade_return', ascending=True)[['调仓时间', 'corrected_sell', 'corrected_buy', 'trade_return']].head(5))

print("\nAverage return by asset (fixed):")
print(trades.groupby('corrected_sell')['trade_return'].mean().sort_values(ascending=False))

print("\nTotal compounded return by asset (approximate, fixed):")
print(trades.groupby('corrected_sell')['trade_return'].apply(lambda x: (1+x).prod() - 1).sort_values(ascending=False))

print("\nAsset Win Rate:")
win_rates = trades.groupby('corrected_sell').apply(lambda x: (x['trade_return'] > 0).mean())
print(win_rates.sort_values(ascending=False))

# Count trades
print("\nNumber of trades per asset (sells):")
print(trades['corrected_sell'].value_counts())
