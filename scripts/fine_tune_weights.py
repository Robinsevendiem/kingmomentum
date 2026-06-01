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

def calculate_wls_momentum(prices, weights):
    """Calculate Weighted Linear Regression Momentum"""
    y = np.log(prices)
    x = np.arange(len(y))
    
    # WLS using numpy.polyfit with weights
    coeffs, cov = np.polyfit(x, y, 1, w=weights, cov=True)
    slope = coeffs[0]
    
    # Calculate R^2 for WLS
    y_pred = np.polyval(coeffs, x)
    sse = np.sum(weights * (y - y_pred)**2)
    y_mean = np.average(y, weights=weights)
    sst = np.sum(weights * (y - y_mean)**2)
    
    if sst == 0:
        r2 = 0
    else:
        r2 = 1 - sse / sst
        
    # Annualized Trend * R2
    # Use 252 for annualization (doesn't affect rank)
    score = (np.exp(slope * 252) - 1) * r2
    return score

def generate_weights(n, param):
    """
    Generate weights for N points using Power formula.
    w = 1 + x^param
    x is normalized time from 0 to 1.
    """
    x = np.linspace(0, 1, n)
    w = 1 + x ** param
    return w

def fine_tune_weights():
    print("--- 权重参数精细化优化 (Fine-tuning Weights) ---")
    
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', ''))
    df_trade = df_trade.sort_values('调仓时间')
    
    history_data = load_history_data()
    
    # Collect Data
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
                if prev_loc >= 19:
                    series = df.iloc[prev_loc-19 : prev_loc+1]['close']
                    row_data['assets'][asset] = series.values
            except: pass
            
        if len(row_data['assets']) > 0:
            dataset.append(row_data)

    # Optimization Loop
    best_acc = 0
    best_param = 0
    
    # Fine-grained Power Parameters
    # From 1.0 to 4.0 with step 0.1
    params = np.arange(1.0, 4.1, 0.1)
    
    print(f"测试范围: 1.0 - 4.0 (步长 0.1)")
    
    results = []
    
    for p in params:
        matches = 0
        matches_top2 = 0
        total = 0
        
        for item in dataset:
            scores = {}
            for asset, prices in item['assets'].items():
                w = generate_weights(len(prices), p)
                s = calculate_wls_momentum(prices, w)
                scores[asset] = s
            
            sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Top 1 Match
            if sorted_assets[0][0] == item['winner']: 
                matches += 1
                
            # Top 2 Match
            if len(sorted_assets) >= 2:
                if item['winner'] in [sorted_assets[0][0], sorted_assets[1][0]]:
                    matches_top2 += 1
            elif len(sorted_assets) == 1:
                 if item['winner'] == sorted_assets[0][0]:
                    matches_top2 += 1
            
            total += 1
            
        acc = matches / total if total > 0 else 0
        acc2 = matches_top2 / total if total > 0 else 0
        
        results.append({'param': p, 'acc': acc, 'acc2': acc2})
        
        if acc > best_acc:
            best_acc = acc
            best_param = p
            # print(f"New Best: {acc:.2%} (Top2: {acc2:.2%}) -> Power: {p:.1f}")

    # Find Top 3 Params
    sorted_results = sorted(results, key=lambda x: x['acc'], reverse=True)
    
    print(f"\n[最佳参数 Top 3]")
    for i in range(3):
        res = sorted_results[i]
        print(f"排名 {i+1}: Power={res['param']:.1f}, Top1={res['acc']:.2%}, Top2={res['acc2']:.2%}")
        
    print(f"\n[参数敏感性分析]")
    # Check if there is a stable range
    # Print accuracy for integer steps
    for p in [1.0, 2.0, 3.0, 4.0]:
        res = next((r for r in results if abs(r['param'] - p) < 0.01), None)
        if res:
            print(f"Power {p:.1f}: {res['acc']:.2%}")

if __name__ == "__main__":
    fine_tune_weights()
