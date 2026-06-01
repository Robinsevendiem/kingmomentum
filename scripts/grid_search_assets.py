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

def calculate_rolling_scores(series, window):
    # Vectorized calculation
    x = np.arange(window)
    weights = 1 + (np.linspace(0, 1, window) ** 2)
    
    scores = pd.Series(index=series.index, dtype=float)
    scores[:] = np.nan
    
    values = series.values
    log_prices = np.log(values)
    
    # We can't fully vectorize polyfit easily, so loop is fine for this scale
    # But we can optimize by pre-allocating
    
    for i in range(window, len(values) + 1):
        window_data = log_prices[i-window : i]
        
        if np.isnan(window_data).any(): continue
        
        try:
            coeffs = np.polyfit(x, window_data, 1, w=weights)
            slope = coeffs[0]
            
            y_pred = np.polyval(coeffs, x)
            sse = np.sum(weights * (window_data - y_pred)**2)
            y_mean = np.average(window_data, weights=weights)
            sst = np.sum(weights * (window_data - y_mean)**2)
            
            r2 = 1 - sse/sst if sst != 0 else 0
            score = (np.exp(slope * 252) - 1) * r2 * 100
            scores.iloc[i-1] = score
        except:
            pass
            
    return scores

def run_grid_search():
    print("Loading Data...")
    data = load_history_data()
    
    # Grid
    windows = [20, 25, 30]
    cutoffs = [300, 400, 500, 600, 800, 1000, 100000] # 100000 = No Cutoff
    
    results = []
    
    print(f"Grid Search Space: Windows={windows}, Cutoffs={cutoffs}")
    
    for asset, df in data.items():
        print(f"Analyzing {asset}...")
        
        # Prefer adjusted close
        if 'adj_close' in df.columns:
            prices = df['adj_close']
        elif 'close' in df.columns:
            prices = df['close']
        else:
            continue
            
        daily_ret = prices.pct_change()
        
        # 1. Loop Windows (Expensive part)
        for w in windows:
            scores = calculate_rolling_scores(prices, w)
            
            # 2. Loop Cutoffs (Cheap part)
            for c in cutoffs:
                # Signal: 1 if Score > 0 and Score <= Cutoff
                # Shift 1 day to apply to next day returns
                signal = ((scores > 0) & (scores <= c)).astype(int).shift(1)
                
                # Strategy Returns
                strat_ret = signal * daily_ret
                
                # Metrics
                # Only consider periods where data exists (after window)
                valid_ret = strat_ret.dropna()
                if len(valid_ret) < 100: continue
                
                ann_ret = valid_ret.mean() * 252
                ann_vol = valid_ret.std() * np.sqrt(252)
                sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
                
                # Win Rate (of days held)
                held_days = valid_ret[signal == 1]
                if len(held_days) > 0:
                    win_rate = (held_days > 0).mean()
                else:
                    win_rate = 0
                
                results.append({
                    'Asset': asset,
                    'Window': w,
                    'Cutoff': c if c < 100000 else 'None',
                    'Ann_Return': ann_ret,
                    'Sharpe': sharpe,
                    'Win_Rate': win_rate
                })
                
    df_res = pd.DataFrame(results)
    
    # Find Best Params per Asset (by Sharpe)
    print("\n--- Best Parameters per Asset (Sorted by Sharpe) ---")
    
    # We want to group by Window first (20 vs 25)
    target_windows = [20, 25]
    
    with open("analysis_results/grid_search_report_grouped.md", "w") as f:
        f.write("# 分组资产参数网格搜索报告\n\n")
        f.write("**优化目标**: Sharpe Ratio (夏普比率)\n\n")
        
        for w_target in target_windows:
            f.write(f"## {w_target}天动量窗口\n\n")
            print(f"\n--- Best Cutoffs for Window = {w_target} ---")
            
            # Filter for this window
            df_w = df_res[df_res['Window'] == w_target]
            
            best_params = []
            for asset in df_w['Asset'].unique():
                subset = df_w[df_w['Asset'] == asset]
                best = subset.sort_values('Sharpe', ascending=False).iloc[0]
                best_params.append(best)
            
            df_best = pd.DataFrame(best_params)
            
            # Format
            df_display = df_best.copy()
            df_display['Ann_Return'] = df_display['Ann_Return'].map('{:.2%}'.format)
            df_display['Win_Rate'] = df_display['Win_Rate'].map('{:.2%}'.format)
            df_display['Sharpe'] = df_display['Sharpe'].map('{:.2f}'.format)
            
            # Markdown table
            table = df_display[['Asset', 'Cutoff', 'Ann_Return', 'Sharpe', 'Win_Rate']].to_markdown(index=False)
            f.write(table)
            f.write("\n\n")
            print(table)

    print("\nReport saved to analysis_results/grid_search_report_grouped.md")
    
    # Also save full results
    df_res.to_csv("analysis_results/grid_search_results.csv", index=False)
    print("\nFull results saved to analysis_results/grid_search_results.csv")

if __name__ == "__main__":
    run_grid_search()
