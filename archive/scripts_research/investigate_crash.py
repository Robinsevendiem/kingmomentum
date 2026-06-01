import pandas as pd
from scripts.optimize_exclude_overheated import load_history_data

history_data = load_history_data()
df = history_data['创业板']
df.index = pd.to_datetime(df.index)

pct_chg = df['adj_close'].pct_change()
print("创业板 daily returns around 2026-03-23:")
print(pct_chg.loc['2026-03-19':'2026-03-23'])
