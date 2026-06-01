import pandas as pd
from scripts.optimize_exclude_overheated import load_history_data

history_data = load_history_data()
df = history_data['创业板']
df.index = pd.to_datetime(df.index)

print("创业板 prices last 25 days:")
print(df['adj_close'].tail(25))
