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
    # r_trend = np.exp(slope * 252) - 1
    # For ranking, slope * r2 is sufficient and more stable
    # But to be precise with "Annualized * R2", we use that.
    
    score = (np.exp(slope * 252) - 1) * r2
    return score

def generate_weights(n, method='linear', param=1.0):
    """
    Generate weights for N points.
    methods:
    - 'linear': linear increase from 1 to param. (e.g. 1 to 2)
    - 'exponential': exp increase with decay rate param.
    - 'power': power increase t^param.
    """
    x = np.linspace(0, 1, n)
    
    if method == 'linear':
        # Linear from 1 to param (e.g. 2)
        # y = 1 + (param - 1) * x
        w = 1 + (param - 1) * x
    elif method == 'exponential':
        # Exp from 1 to param? No, usually exp decay.
        # Decay factor alpha. w_t = (1-alpha)^(N-1-t)
        # Or just exp(k * t).
        # Let's use exp(param * x)
        w = np.exp(param * x)
    elif method == 'power':
        # x^param
        # Usually t^2 or t^3.
        # We want weights to be non-zero, so (x + epsilon)^param?
        # Or 1 + x^param
        w = 1 + x ** param
        
    return w

def optimize_wls():
    print("--- 时间加权回归参数优化 (WLS Optimization) ---")
    
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
    best_config = {}
    
    # 1. Linear Weights: Slope from 1 to X
    # Range: 1.0 (Equal) to 5.0 (Steep)
    for slope in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        matches = 0
        total = 0
        
        for item in dataset:
            scores = {}
            for asset, prices in item['assets'].items():
                w = generate_weights(len(prices), 'linear', slope)
                s = calculate_wls_momentum(prices, w)
                scores[asset] = s
            
            sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_assets[0][0] == item['winner']: matches += 1
            total += 1
            
        acc = matches / total if total > 0 else 0
        if acc > best_acc:
            best_acc = acc
            best_config = {'method': 'linear', 'param': slope}
            # print(f"New Best (Linear): {acc:.2%} -> Slope: {slope}")

    # 2. Exponential Weights
    # Param: rate. exp(rate * x)
    for rate in [0.5, 1.0, 2.0, 3.0, 5.0]:
        matches = 0
        total = 0
        
        for item in dataset:
            scores = {}
            for asset, prices in item['assets'].items():
                w = generate_weights(len(prices), 'exponential', rate)
                s = calculate_wls_momentum(prices, w)
                scores[asset] = s
            
            sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_assets[0][0] == item['winner']: matches += 1
            total += 1
            
        acc = matches / total if total > 0 else 0
        if acc > best_acc:
            best_acc = acc
            best_config = {'method': 'exponential', 'param': rate}
            # print(f"New Best (Exp): {acc:.2%} -> Rate: {rate}")
            
    # 3. Power Weights
    # Param: power. 1 + x^p
    for p in [2, 3, 4, 5]:
        matches = 0
        total = 0
        
        for item in dataset:
            scores = {}
            for asset, prices in item['assets'].items():
                w = generate_weights(len(prices), 'power', p)
                s = calculate_wls_momentum(prices, w)
                scores[asset] = s
            
            sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_assets[0][0] == item['winner']: matches += 1
            total += 1
            
        acc = matches / total if total > 0 else 0
        if acc > best_acc:
            best_acc = acc
            best_config = {'method': 'power', 'param': p}
            # print(f"New Best (Power): {acc:.2%} -> Power: {p}")
            
    print(f"\n[最佳加权方案]")
    print(f"准确率: {best_acc:.2%}")
    print(f"配置: {best_config}")
    
    # Compare with standard OLS (Linear 1.0)
    # Re-run OLS specifically
    matches_ols = 0
    total = 0
    for item in dataset:
        scores = {}
        for asset, prices in item['assets'].items():
            w = np.ones(len(prices))
            s = calculate_wls_momentum(prices, w)
            scores[asset] = s
        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_assets[0][0] == item['winner']: matches_ols += 1
        total += 1
    print(f"标准OLS准确率: {matches_ols/total:.2%}")

if __name__ == "__main__":
    optimize_wls()
