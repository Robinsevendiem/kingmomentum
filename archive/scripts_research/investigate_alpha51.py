import pandas as pd
from scripts.optimize_exclude_overheated import load_history_data
from Home import calculate_alpha51

history_data = load_history_data()
df = history_data['创业板']
df.index = pd.to_datetime(df.index)

alpha51 = calculate_alpha51(df, window=10)
print("Alpha51 for 创业板 around 2026-03-23:")
print(alpha51.loc['2026-03-19':'2026-03-23'])
