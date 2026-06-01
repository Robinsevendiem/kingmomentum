import pandas as pd
from scripts.optimize_exclude_overheated import load_history_data, precalculate_all_scores
from test_crash_filter import run_backtest_with_crash_filter, calculate_metrics

history_data = load_history_data()
for name, df in history_data.items():
    df.index = pd.to_datetime(df.index)
raw_scores_df = precalculate_all_scores(history_data, window=20)

base_params = {
    'start_date': pd.to_datetime('2017-08-10'),
    'end_date': pd.to_datetime('2026-03-23'),
    'initial_capital': 1.0,
    'global_cutoff': 600,
    'buffer_score': 5,
    'min_score': 0,
    'exclude_overheated_from_norm': True,
    'crash_filter_enabled': True,
    'crash_window': 2,
    'crash_threshold': 0.07
}

df_test = run_backtest_with_crash_filter(history_data, raw_scores_df, base_params)
ret, dd, sharpe = calculate_metrics(df_test)
print(f'Optimal Crash Filter (Window: 2, Threshold: 7%):')
print(f'Ann Return: {ret*100:.2f}%, Max Drawdown: {dd*100:.2f}%, Sharpe: {sharpe:.2f}')
