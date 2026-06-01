import pandas as pd
import os

trades = pd.read_csv('/Users/robin-macmini/Documents/trae_projects/foolreveal/data/reference/2017.8.1-2026.3.23调仓.csv')
trades['调仓时间'] = pd.to_datetime(trades['调仓时间'].str.replace(' 开盘', ''))
trades = trades.sort_values('调仓时间').reset_index(drop=True)

cash_buys_indices = trades.index[(trades['买入'] == '现金') | ((trades['买入'] == '现金') & (trades['卖出'] == '现金'))]

mapping = {
    '创业板': 'data/159915.SZ_创业板ETF_history.csv',
    '南方原油': 'data/501018.SH_南方原油(LOF)_history.csv',
    '上证180': 'data/510180.SH_180ETF_history.csv',
    '30年国债': 'data/511090.SH_30年国债ETF_history.csv',
    '港股科技': 'data/513020.SH_港股科技ETF_history.csv',
    '纳指100': 'data/51310import pandas as pd
import os

trades = pd.read_cTFimport os

trades 13
trades ???rades['调仓时间'] = pd.to_datetime(trades['调仓时间'].str.replace(' 开盘', 'trades = trades.sort_values('"调仓时间"').reset_index(drop=True)

cash_buys_indices = trades.index[(trades['"os
cash_buys_indices = trades.index[(trades['买入'] == '现?t