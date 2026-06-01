import pandas as pd
from scripts.optimize_exclude_overheated import load_history_data

history_data = load_history_data()
df = history_data['创业板']
df.index = pd.to_datetime(df.index)

w = 10
p = df['adj_close']
diff = (p.shift(w) - p.shift(2*w))/w - (p - p.shift(w))/w
diff = diff.dropna()

print("Alpha51 diff for 创业板 around 2026-03-23:")
print(diff.loc['2026-03-19':'2026-03-23'])
