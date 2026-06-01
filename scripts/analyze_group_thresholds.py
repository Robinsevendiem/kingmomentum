import pandas as pd
import numpy as np
import os
import sys

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

def calculate_stats(series, window=25):
    x = np.arange(window)
    weights = 1 + (np.linspace(0, 1, window) ** 2)
    
    scores = []
    future_returns = []
    
    values = series.values
    
    for i in range(window, len(values) - 20):
        window_data = values[i-window : i]
        
        if np.isnan(window_data).any(): continue
        
        try:
            # Score
            y = np.log(window_data)
            coeffs = np.polyfit(x, y, 1, w=weights)
            slope = coeffs[0]
            
            y_pred = np.polyval(coeffs, x)
            sse = np.sum(weights * (y - y_pred)**2)
            y_mean = np.average(y, weights=weights)
            sst = np.sum(weights * (y - y_mean)**2)
            r2 = 1 - sse/sst if sst != 0 else 0
            
            score = (np.exp(slope * 252) - 1) * r2 * 100
            
            # Future Return (20d)
            curr_price = values[i-1]
            future_price = values[i-1+20]
            ret = future_price / curr_price - 1
            
            scores.append(score)
            future_returns.append(ret)
            
        except:
            pass
            
    return pd.DataFrame({
        'score': scores,
        'future_return': future_returns
    })

def analyze_groups():
    print("Loading Data...")
    data = load_history_data()
    window = 25
    
    # Define Bins
    bins = [0, 100, 300, 500, 800, 1000, float('inf')]
    labels = ['0-100', '100-300', '300-500', '500-800', '800-1000', '>1000']
    
    asset_stats = []
    
    for asset, df in data.items():
        # Prefer adjusted close
        if 'adj_close' in df.columns:
            series = df['adj_close']
        elif 'close' in df.columns:
            series = df['close']
        else:
            continue
            
        stats_df = calculate_stats(series, window)
        
        if stats_df.empty: continue
        
        stats_df['bin'] = pd.cut(stats_df['score'], bins=bins, labels=labels)
        
        row = {'Asset': asset}
        
        # Calculate stats per bin
        # Manual groupby to avoid KeyError for missing bins
        for bin_label in labels:
            group = stats_df[stats_df['bin'] == bin_label]
            count = len(group)
            
            if count > 10: # Min sample size
                avg_ret = group['future_return'].mean()
                row[f'{bin_label}_Ret'] = avg_ret
            else:
                row[f'{bin_label}_Ret'] = np.nan
        
        # Determine "Turning Point" (Max Return Bin)
        max_ret = -float('inf')
        best_bin = "N/A"
        
        for bin_label in labels:
            val = row[f'{bin_label}_Ret']
            if not np.isnan(val) and val > max_ret:
                max_ret = val
                best_bin = bin_label
        
        row['Best_Zone'] = best_bin
        
        # Categorize
        # Check specific zones
        ret_300_500 = row['300-500_Ret']
        ret_500_800 = row['500-800_Ret']
        
        if not np.isnan(ret_500_800) and ret_500_800 > 0:
            category = "High_Vol (Use 800+)"
        elif not np.isnan(ret_300_500) and ret_300_500 > 0:
            category = "Med_Vol (Use 500)"
        else:
            category = "Standard (Use 300)"
            
        # Format for display (after logic)
        for bin_label in labels:
            if not np.isnan(row[f'{bin_label}_Ret']):
                row[f'{bin_label}_Ret'] = f"{row[f'{bin_label}_Ret']:.2%}"
            else:
                row[f'{bin_label}_Ret'] = "N/A"
                
        row['Category'] = category
        asset_stats.append(row)
        
    res_df = pd.DataFrame(asset_stats)
    
    # Reorder columns
    cols = ['Asset', 'Category', 'Best_Zone'] + [c for c in res_df.columns if '_Ret' in c]
    res_df = res_df[cols]
    
    print("\n--- Asset Threshold Group Analysis (Mean Future 20d Return) ---")
    print(res_df.to_markdown(index=False))
    
    # Save detailed report
    with open("analysis_results/threshold_group_report.md", "w") as f:
        f.write("# 分组阈值深度分析报告\n\n")
        f.write(f"**分析窗口**: {window}天\n")
        f.write("**指标**: 未来20日平均收益率 (Mean Future 20d Return)\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n## 结论建议\n")
        
        groups = res_df.groupby('Category')['Asset'].apply(list)
        for cat, assets in groups.items():
            f.write(f"### {cat}\n")
            f.write(f"- **标的**: {', '.join(assets)}\n")
            f.write("- **特征**: ...\n\n")

if __name__ == "__main__":
    analyze_groups()
