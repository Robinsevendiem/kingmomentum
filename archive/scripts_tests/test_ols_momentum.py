import pandas as pd
import numpy as np
import os
from scipy.stats import linregress

# File paths
RECORD_DIR = 'record'
TRADE_FILE = os.path.join(RECORD_DIR, '2017-08-10 至 2026-02-26调仓记录.csv')

def load_history_data():
    """Load all asset history files into a dictionary of DataFrames"""
    mapping = {
        '创业板': '159915.SZ_创业板ETF_history.csv',
        '南方原油': '501018.SH_南方原油(LOF)_history.csv',
        '上证180': '510180.SH_180ETF_history.csv',
        '30年国债': '511090.SH_30年国债ETF_history.csv',
        '港股科技': '513020.SH_港股科技ETF_history.csv',
        '纳指100': '513100.SH_纳指ETF_history.csv',
        '日经ETF': '513520.SH_日经ETF_history.csv',
        '黄金ETF': '518880.SH_黄金ETF_history.csv',
        '科创板': '588120.SH_科创板ETF_history.csv'
    }
    
    history_data = {}
    for name, filename in mapping.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('trade_date').set_index('trade_date')
            history_data[name] = df
    return history_data

def calculate_ols_momentum(series, window_size=25):
    """
    Calculate OLS Momentum as specified:
    1. Log Price: y = ln(P_t)
    2. Regression: y = alpha + beta * t
    3. R_trend = exp(beta * 250) - 1
    4. R^2 = Goodness of Fit
    5. Score = R_trend * R^2
    
    Note: Window size is specified as 25 days in the prompt.
    Previously we used 20 days. Let's test 25 days specifically as requested.
    But also test 20 days to see which one fits better.
    """
    if len(series) < 5: return None
    
    prices = series.values
    
    # 1. Log Price
    y = np.log(prices)
    x = np.arange(len(y))
    
    # 2. Linear Regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # 3. Trend Strength (R_trend) - Annualized using 250 days
    # Note: exp(slope * 250) approx (1 + slope)^250 approx 1 + slope*250
    # Formula: exp(beta * 250) - 1
    r_trend = np.exp(slope * 250) - 1
    
    # 4. Trend Stability (R^2)
    r_squared = r_value ** 2
    
    # 5. Final Score
    score = r_trend * r_squared
    
    return {
        'r_trend': r_trend,
        'r_squared': r_squared,
        'score': score
    }

def test_new_formula():
    print("--- 基于OLS回归的动量修正体系测试 (New OLS Momentum Test) ---")
    
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', ''))
    df_trade = df_trade.sort_values('调仓时间')
    
    history_data = load_history_data()
    
    # We will test two window sizes: 20 days (previous best) and 25 days (new hypothesis)
    windows = [20, 25]
    
    dataset = []
    
    for _, row in df_trade.iterrows():
        trade_date = row['调仓时间']
        bought = row['买入']
        if bought == '现金' or bought not in history_data: continue
        
        row_data = {'date': trade_date, 'winner': bought, 'assets': {}}
        
        for asset, df in history_data.items():
            try:
                prev_date = df.index[df.index < trade_date][-1]
                prev_loc = df.index.get_loc(prev_date)
                
                # Calculate for both windows
                for w in windows:
                    if prev_loc >= w-1:
                        series = df.iloc[prev_loc-w+1 : prev_loc+1]['close']
                        res = calculate_ols_momentum(series, window_size=w)
                        if res:
                            row_data['assets'][f'{asset}_{w}'] = res
            except: pass
            
        if len(row_data['assets']) > 0:
            dataset.append(row_data)

    # Analyze Accuracy
    for w in windows:
        matches = 0
        total = 0
        
        # Also track Top 2 matches
        matches_top2 = 0
        
        for item in dataset:
            scores = {}
            # Extract scores for this window size
            for key, val in item['assets'].items():
                if key.endswith(f'_{w}'):
                    asset_name = key.rsplit('_', 1)[0]
                    scores[asset_name] = val['score']
            
            if not scores: continue
            
            # Sort Descending
            sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Check Top 1
            if sorted_assets[0][0] == item['winner']:
                matches += 1
                
            # Check Top 2
            if len(sorted_assets) >= 2:
                if item['winner'] in [sorted_assets[0][0], sorted_assets[1][0]]:
                    matches_top2 += 1
            elif len(sorted_assets) == 1:
                 if item['winner'] == sorted_assets[0][0]:
                    matches_top2 += 1
            
            total += 1
            
        acc = matches / total if total > 0 else 0
        acc2 = matches_top2 / total if total > 0 else 0
        
        print(f"\n[窗口周期: {w}日]")
        print(f"Top 1 准确率: {acc:.2%} ({matches}/{total})")
        print(f"Top 2 覆盖率: {acc2:.2%} ({matches_top2}/{total})")

if __name__ == "__main__":
    test_new_formula()
