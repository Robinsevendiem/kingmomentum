import pandas as pd
import numpy as np

holdings = pd.read_csv('/Users/robin-macmini/Documents/trae_projects/foolreveal/data/reference/2017.8.1-2026.3.23持仓.csv')
holdings['日期'] = pd.to_datetime(holdings['日期'])
holdings = holdings.sort_values('日期')

cumulative_max = holdings['净值'].cummax()
drawdown = (holdings['净值'] - cumulative_max) / cumulative_max
max_drawdown = drawdown.min()

print('Final Net Value:', holdings['净值'].iloc[-1])
print('Max Drawdown:', max_drawdown)
print('Annualized Return:', (holdings['净值'].iloc[-1] ** (252 / len(holdings))) - 1)

holdings['is_cash'] = (holdings['ETF名称'] == '现金')
cash_groups = (holdings['is_cash'] != holdings['is_cash'].shift()).cumsum()
cash_periods = holdings[holdings['is_cash']].groupby(cash_groups).size()
print('Longest cash holding period (days):', cash_periods.max() if not cash_periods.empty else 0)

print('Total days:', len(holdings))
print('Days in cash:', holdings['is_cash'].sum())

