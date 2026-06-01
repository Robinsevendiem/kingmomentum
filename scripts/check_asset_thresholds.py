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
    # Prepare inputs
    x = np.arange(window)
    weights = 1 + (np.linspace(0, 1, window) ** 2)
    
    # We need to iterate
    scores = []
    future_returns = []
    dates = []
    
    values = series.values
    idx = series.index
    
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
            dates.append(idx[i-1])
            
        except:
            pass
            
    return pd.DataFrame({
        'date': dates,
        'score': scores,
        'future_return': future_returns
    })

def analyze_assets():
    print("Loading Data...")
    data = load_history_data()
    
    results = []
    global_cutoff = 300
    window = 25
    
    print(f"Analyzing for Window={window}, Global Cutoff={global_cutoff}...")
    
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
        
        # Percentiles
        p50 = np.percentile(stats_df['score'], 50)
        p90 = np.percentile(stats_df['score'], 90)
        p95 = np.percentile(stats_df['score'], 95)
        p99 = np.percentile(stats_df['score'], 99)
        max_score = np.max(stats_df['score'])
        
        # Overheat Analysis (> 300)
        overheat = stats_df[stats_df['score'] > global_cutoff]
        count = len(overheat)
        pct_time = count / len(stats_df) * 100
        
        if count > 0:
            avg_ret = overheat['future_return'].mean()
            # Win rate (positive return)
            win_rate = len(overheat[overheat['future_return'] > 0]) / count
        else:
            avg_ret = np.nan
            win_rate = np.nan
            
        # Normal Regime Analysis (100 - 300)
        normal = stats_df[(stats_df['score'] > 100) & (stats_df['score'] <= 300)]
        if len(normal) > 0:
            normal_avg_ret = normal['future_return'].mean()
        else:
            normal_avg_ret = np.nan
            
        results.append({
            'Asset': asset,
            'P95': p95,
            'P99': p99,
            'Max': max_score,
            'Time > 300': f"{pct_time:.1f}%",
            'Avg Ret (>300)': avg_ret,
            'Avg Ret (100-300)': normal_avg_ret
        })
        
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('P99', ascending=False)
    
    print("\n--- Asset Specific Analysis ---")
    print(res_df.to_markdown(index=False, floatfmt=".2f"))
    
    # Save
    res_df.to_csv("analysis_results/asset_threshold_analysis.csv", index=False)

if __name__ == "__main__":
    analyze_assets()
