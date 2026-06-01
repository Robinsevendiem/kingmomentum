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

def calculate_wls_momentum(prices):
    """Calculate Weighted Linear Regression Momentum (Power=2)"""
    y = np.log(prices)
    x = np.arange(len(y))
    x_norm = np.linspace(0, 1, len(y))
    weights = 1 + x_norm ** 2
    
    coeffs, cov = np.polyfit(x, y, 1, w=weights, cov=True)
    slope = coeffs[0]
    
    y_pred = np.polyval(coeffs, x)
    sse = np.sum(weights * (y - y_pred)**2)
    y_mean = np.average(y, weights=weights)
    sst = np.sum(weights * (y - y_mean)**2)
    
    if sst == 0:
        r2 = 0
    else:
        r2 = 1 - sse / sst
        
    score = (np.exp(slope * 252) - 1) * r2
    return score

def calculate_max_drawdown(prices):
    """Calculate Max Drawdown"""
    cum_max = np.maximum.accumulate(prices)
    drawdowns = (prices - cum_max) / cum_max
    max_dd = drawdowns.min()
    return max_dd

def calculate_volatility(prices):
    """Calculate Annualized Volatility"""
    prices_series = pd.Series(prices)
    daily_rets = prices_series.pct_change().dropna()
    vol = daily_rets.std() * np.sqrt(252)
    return vol

def optimize_multi_factor():
    print("--- 多因子综合打分优化 (Multi-Factor Optimization) ---")
    
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
                    prices = series.values
                    
                    mom = calculate_wls_momentum(prices)
                    dd = calculate_max_drawdown(prices)
                    vol = calculate_volatility(prices)
                    
                    row_data['assets'][asset] = {'mom': mom, 'dd': dd, 'vol': vol}
            except: pass
            
        if len(row_data['assets']) > 0:
            dataset.append(row_data)

    # Optimization Loop
    # Score = w_mom * Norm(Mom) + w_dd * Norm(DD) - w_vol * Norm(Vol)
    # Norm: MinMax (0-100)
    
    best_acc = 0
    best_weights = {}
    
    # Grid Search
    # w_mom: Fixed at 1.0 (Base)
    # w_dd: 0 to 2.0
    # w_vol: 0 to 2.0
    
    # We suspect small positive weights for DD/Vol penalty?
    # Or maybe large?
    
    for w_dd in [0, 0.5, 1.0, 1.5, 2.0]:
        for w_vol in [0, 0.5, 1.0, 1.5, 2.0]:
            matches = 0
            matches_top2 = 0
            total = 0
            
            for item in dataset:
                # Normalize Factors for current assets
                raw_values = {'mom': [], 'dd': [], 'vol': []}
                assets = list(item['assets'].keys())
                
                for asset in assets:
                    raw_values['mom'].append(item['assets'][asset]['mom'])
                    raw_values['dd'].append(item['assets'][asset]['dd'])
                    raw_values['vol'].append(item['assets'][asset]['vol'])
                
                norm_scores = {}
                for k in raw_values:
                    arr = np.array(raw_values[k])
                    mn, mx = np.min(arr), np.max(arr)
                    if mx != mn:
                        # MinMax 0-100
                        norm_scores[k] = (arr - mn) / (mx - mn) * 100
                    else:
                        norm_scores[k] = np.full_like(arr, 50)
                
                final_scores = {}
                for i, asset in enumerate(assets):
                    # For Mom: Higher is Better -> + Mom
                    # For DD: MaxDD is negative (e.g. -0.2). Min is -0.5, Max is -0.1.
                    # Norm: -0.5 -> 0, -0.1 -> 100.
                    # So Higher Norm Score means Smaller Drawdown (Better).
                    # So we should ADD this term (Reward Small Drawdown).
                    
                    # For Vol: Min is 0.1, Max is 0.5.
                    # Norm: 0.1 -> 0, 0.5 -> 100.
                    # So Higher Norm Score means Higher Volatility (Worse).
                    # So we should SUBTRACT this term (Penalize High Vol).
                    
                    s = norm_scores['mom'][i] + w_dd * norm_scores['dd'][i] - w_vol * norm_scores['vol'][i]
                    final_scores[asset] = s
                    
                sorted_assets = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
                
                if sorted_assets[0][0] == item['winner']:
                    matches += 1
                    
                if len(sorted_assets) >= 2:
                    if item['winner'] in [sorted_assets[0][0], sorted_assets[1][0]]:
                        matches_top2 += 1
                elif len(sorted_assets) == 1:
                     if item['winner'] == sorted_assets[0][0]:
                        matches_top2 += 1
                
                total += 1
                
            acc = matches / total if total > 0 else 0
            acc2 = matches_top2 / total if total > 0 else 0
            
            if acc >= best_acc: # Prefer simpler model if equal
                # If equal accuracy, check Top2?
                if acc > best_acc:
                    best_acc = acc
                    best_weights = {'w_dd': w_dd, 'w_vol': w_vol, 'acc2': acc2}
                    # print(f"New Best: {acc:.2%} (Top2: {acc2:.2%}) -> w_dd: {w_dd}, w_vol: {w_vol}")
                elif acc == best_acc:
                    # If equal, prefer model with higher Top2
                    if 'acc2' in best_weights and acc2 > best_weights['acc2']:
                        best_weights = {'w_dd': w_dd, 'w_vol': w_vol, 'acc2': acc2}
                        # print(f"New Best (Top2): {acc:.2%} (Top2: {acc2:.2%}) -> w_dd: {w_dd}, w_vol: {w_vol}")

    print(f"\n[最佳多因子模型]")
    print(f"Top 1 准确率: {best_acc:.2%}")
    print(f"Top 2 覆盖率: {best_weights.get('acc2', 0):.2%}")
    print(f"参数: {best_weights}")
    
    # Check if this beats the pure Mom model (62.79%)
    # Pure Mom is w_dd=0, w_vol=0.

if __name__ == "__main__":
    optimize_multi_factor()
