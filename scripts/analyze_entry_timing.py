
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import logic if needed, 
# but we will just copy the score calculation for standalone execution.

def load_history_data():
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
            except Exception:
                pass
    return history_data

def calculate_rolling_scores(series, window=20):
    scores = pd.Series(index=series.index, dtype=float)
    scores[:] = np.nan
    x = np.arange(window)
    weights = 1 + (np.linspace(0, 1, window) ** 2)
    
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
            r2 = 1 - sse/sst if sst != 0 else 0
            
            score = (np.exp(slope * 252) - 1) * r2 * 100
            scores.iloc[i-1] = score
        except:
            pass
    return scores

def analyze_entry_timing():
    data = load_history_data()
    all_scores = pd.DataFrame()
    price_data = pd.DataFrame()
    
    print("Calculating scores...")
    for asset, df in data.items():
        col = 'adj_close' if 'adj_close' in df.columns else 'close'
        series = df[col]
        scores = calculate_rolling_scores(series, window=20)
        scores.name = asset
        all_scores = pd.merge(all_scores, scores, left_index=True, right_index=True, how='outer')
        
        series.name = asset
        price_data = pd.merge(price_data, series, left_index=True, right_index=True, how='outer')
        
    # Analyze "Potential Entries"
    # An entry signal is generated when an asset becomes Rank 1
    
    results = []
    
    dates = all_scores.index.sort_values()
    
    for i in range(20, len(dates)-20):
        date = dates[i]
        today_scores = all_scores.loc[date].dropna()
        if today_scores.empty: continue
        
        # Get Top Asset
        top_asset = today_scores.idxmax()
        top_score = today_scores.max()
        
        if top_score <= 0: continue # No entry
        
        # Condition 1: Absolute Score Value
        # Condition 2: Score Trend (vs 5 days ago)
        prev_date_idx = i - 5
        if prev_date_idx >= 0:
            prev_date = dates[prev_date_idx]
            if prev_date in all_scores.index:
                prev_score = all_scores.loc[prev_date, top_asset]
            else:
                prev_score = np.nan
        else:
            prev_score = np.nan
            
        score_trend = "Rising" if (not np.isnan(prev_score) and top_score > prev_score) else "Falling"
        
        # Outcome: Next 10 days return
        next_date_idx = i + 10
        if next_date_idx < len(dates):
            # Calculate return
            p_curr = price_data.loc[date, top_asset]
            # Use 'asof' logic implicitly by index location if contiguous, but safest to lookup
            # We assume price exists if score exists
            
            # Find next valid price 10 days later
            # This is rough, using index + 10 trading days
            future_date = dates[next_date_idx]
            p_future = price_data.loc[future_date, top_asset]
            
            if pd.notnull(p_curr) and pd.notnull(p_future):
                ret_10d = p_future / p_curr - 1
                
                results.append({
                    'Date': date,
                    'Asset': top_asset,
                    'Score': top_score,
                    'Score_Trend': score_trend,
                    'Return_10d': ret_10d,
                    'Win': ret_10d > 0
                })
                
    df = pd.DataFrame(results)
    
    print("\n--- Analysis Report: Entry Conditions vs 10-Day Win Rate ---")
    
    # Bin by Score
    bins = [0, 50, 100, 150, 200, 300, 500, 1000]
    df['Score_Bin'] = pd.cut(df['Score'], bins=bins)
    
    stats = df.groupby('Score_Bin')['Win'].agg(['count', 'mean'])
    stats['mean'] = stats['mean'] * 100
    stats.columns = ['Trade Count', 'Win Rate (%)']
    print("\n1. Win Rate by Initial Score:")
    print(stats)
    
    # Bin by Trend
    print("\n2. Win Rate by Score Trend (Rising vs Falling):")
    stats_trend = df.groupby('Score_Trend')['Win'].agg(['count', 'mean'])
    stats_trend['mean'] = stats_trend['mean'] * 100
    print(stats_trend)
    
    # Combined
    print("\n3. Combined Strategy (Score > X AND Rising):")
    # Test a few thresholds
    for threshold in [0, 50, 100, 150]:
        subset = df[(df['Score'] > threshold) & (df['Score_Trend'] == 'Rising')]
        wr = subset['Win'].mean()
        count = len(subset)
        print(f"Entry when Score > {threshold} AND Rising: Win Rate = {wr:.1%} (Samples: {count})")
        
        subset_bad = df[(df['Score'] > threshold) & (df['Score_Trend'] == 'Falling')]
        wr_bad = subset_bad['Win'].mean()
        print(f"Entry when Score > {threshold} AND Falling: Win Rate = {wr_bad:.1%}")

if __name__ == "__main__":
    analyze_entry_timing()
