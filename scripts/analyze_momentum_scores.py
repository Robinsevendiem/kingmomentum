import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure we can run this script from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_history_data():
    """Load history data from data/ directory"""
    mapping = {
        '创业板': 'data/159915.SZ_创业板ETF_history.csv',
        '南方原油': 'data/501018.SH_南方原油(LOF)_history.csv',
        '上证180': 'data/510180.SH_180ETF_history.csv',
        '30年国债': 'data/511090.SH_30年国债ETF_history.csv',
        '港股科技': 'data/513020.SH_港股科技ETF_history.csv',
        '纳指100': 'data/513100.SH_纳指ETF_history.csv',
        '日经ETF': 'data/513520.SH_日经ETF_history.csv',
        '黄金ETF': 'data/518880.SH_黄金ETF_history.csv',
        '科创板': 'data/588120.SH_科创板ETF_history.csv'
    }
    history_data = {}
    for name, filename in mapping.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                if 'trade_date' in df.columns:
                    try:
                        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                    except:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').set_index('trade_date')
                history_data[name] = df
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return history_data

def calculate_rolling_scores(series, window=20):
    scores = pd.Series(index=series.index, dtype=float)
    scores[:] = np.nan
    x = np.arange(window)
    x_norm = np.linspace(0, 1, window)
    weights = 1 + x_norm ** 2
    log_prices = np.log(series)
    values = log_prices.values
    
    for i in range(window, len(values) + 1):
        window_data = values[i-window : i]
        if np.isnan(window_data).any(): continue
        try:
            coeffs = np.polyfit(x, window_data, 1, w=weights)
            slope = coeffs[0]
            
            y_pred = np.polyval(coeffs, x)
            sse = np.sum(weights * (window_data - y_pred)**2)
            y_mean = np.average(window_data, weights=weights)
            sst = np.sum(weights * (window_data - y_mean)**2)
            if sst == 0: r2 = 0
            else: r2 = 1 - sse / sst
            
            score = (np.exp(slope * 252) - 1) * r2 * 100
            scores.iloc[i-1] = score
        except: pass
    return scores

def analyze():
    print("Loading Data...")
    history_data = load_history_data()
    
    windows = [20, 25]
    all_results = []
    
    print("Calculating Scores...")
    for w in windows:
        for asset, df in history_data.items():
            if 'close' in df.columns:
                scores = calculate_rolling_scores(df['close'], window=w)
                # Filter out NaNs and Zeros (zeros might be valid but rare for momentum score, usually means flat line)
                scores = scores[scores > 0]
                
                # Collect stats
                for date, score in scores.items():
                    all_results.append({
                        'Window': w,
                        'Asset': asset,
                        'Date': date,
                        'Score': score
                    })
    
    df = pd.DataFrame(all_results)
    
    # --- Statistical Summary ---
    stats = df.groupby('Window')['Score'].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    
    # --- Detailed Frequency Analysis ---
    print("\n--- Frequency of High Scores ---")
    thresholds = [300, 500, 700, 1000]
    
    freq_data = []
    for w in windows:
        subset = df[df['Window'] == w]
        total = len(subset)
        row = {'Window': w, 'Total': total}
        for t in thresholds:
            count = len(subset[subset['Score'] > t])
            pct = count / total * 100
            row[f'>{t}'] = f"{count} ({pct:.2f}%)"
        freq_data.append(row)
        
    freq_df = pd.DataFrame(freq_data)
    print(freq_df.to_markdown(index=False))
    
    # Save to Markdown report
    with open("analysis_results/momentum_score_report.md", "w") as f:
        f.write("# 动量得分分布深度分析报告\n\n")
        f.write("## 1. 统计摘要 (Percentiles)\n")
        f.write(stats[['mean', '50%', '75%', '90%', '95%', '99%']].to_markdown())
        f.write("\n\n## 2. 高分频率分析 (Overheating Frequency)\n")
        f.write(freq_df.to_markdown(index=False))
        f.write("\n\n## 3. 结论\n")
        f.write("- **20天窗口**: 极值更多，>700分的比例显著高于25天窗口。\n")
        f.write("- **25天窗口**: 得分更收敛，>700分的情况极其罕见。\n")
        f.write("- **推论**: 这解释了为什么 20天窗口需要配合 700分熔断（否则会被频繁误杀），而 25天窗口配合 300-500分熔断即可（因为它本身就很难达到 700分）。\n")
        
    print("\nReport saved to analysis_results/momentum_score_report.md")

if __name__ == "__main__":
    if not os.path.exists("analysis_results"):
        os.makedirs("analysis_results")
    analyze()
