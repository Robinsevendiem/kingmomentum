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

def calculate_weighted_linear_regression(prices, weights=None):
    """
    Weighted Linear Regression Momentum
    y = ln(P)
    x = t
    w = weights
    """
    y = np.log(prices)
    x = np.arange(len(y))
    
    if weights is None:
        # Standard OLS
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r2 = r_value ** 2
    else:
        # Weighted Least Squares (WLS)
        # Weights should sum to 1? Or just relative weights.
        # Sklearn or numpy polyfit with weights
        # Let's use numpy polyfit
        coeffs, cov = np.polyfit(x, y, 1, w=weights, cov=True)
        slope = coeffs[0]
        
        # Calculate R^2 for WLS
        # R^2 = 1 - SSE / SST
        # Predicted y
        y_pred = np.polyval(coeffs, x)
        # Weighted SSE
        sse = np.sum(weights * (y - y_pred)**2)
        # Weighted Mean
        y_mean = np.average(y, weights=weights)
        # Weighted SST
        sst = np.sum(weights * (y - y_mean)**2)
        r2 = 1 - sse / sst
        
    # Annualized Trend
    r_trend = np.exp(slope * 252) - 1
    
    return {
        'wls_slope': slope,
        'wls_r2': r2,
        'wls_score': r_trend * r2
    }

def calculate_efficiency_ratio_momentum(series):
    """
    Efficiency Ratio Momentum
    Momentum = Total Return * Efficiency Ratio
    Efficiency Ratio = |Net Change| / Sum(|Daily Changes|)
    Or "位移 / 路程"
    
    Net Change (Displacement) = ln(P_end / P_start) or (P_end - P_start)
    Path Length (Distance) = Sum(abs(ln(P_t / P_t-1))) or Sum(abs(P_t - P_t-1))
    
    Usually ER = Abs(Close_N - Close_0) / Sum(Abs(Close_i - Close_i-1))
    
    The prompt says: "Actual Total Return * Trend Unilaterality"
    "Identify actual effective trend return"
    Formula: Final Score = Annualized Return * ER
    """
    prices = series.values
    
    # 1. Actual Total Return (Annualized)
    # Log return is better for additivity
    total_log_ret = np.log(prices[-1] / prices[0])
    ann_ret = np.exp(total_log_ret / len(prices) * 252) - 1
    
    # 2. Efficiency Ratio (ER)
    # Displacement = Abs(Log Return) ? No, ER is directionless usually (0 to 1).
    # But here we want Momentum, so Direction matters.
    # The prompt says "Momentum * Efficiency Coefficient".
    # So Direction comes from Return. ER is just a quality filter (0-1).
    
    # Path: Sum of absolute daily log returns
    log_prices = np.log(prices)
    daily_log_rets = np.diff(log_prices)
    path_length = np.sum(np.abs(daily_log_rets))
    
    displacement = np.abs(total_log_ret)
    
    if path_length == 0:
        er = 0
    else:
        er = displacement / path_length
        
    # Score
    # "Total Return * ER"
    # Or "Annualized Return * ER"
    score = ann_ret * er
    
    return {
        'er_ret': ann_ret,
        'er_ratio': er,
        'er_score': score
    }

def calculate_all_variants(series):
    if len(series) < 20: return None
    
    results = {}
    prices = series.values
    
    # 1. Weighted Linear Regression (Linear weights 1 -> 2)
    # Weights: linspace(1, 2, N)
    n = len(prices)
    weights = np.linspace(1, 2, n)
    wls_res = calculate_weighted_linear_regression(prices, weights)
    results.update(wls_res)
    
    # 2. Standard OLS (Equal weights)
    ols_res = calculate_weighted_linear_regression(prices, weights=None)
    results['ols_slope'] = ols_res['wls_slope']
    results['ols_r2'] = ols_res['wls_r2']
    results['ols_score'] = ols_res['wls_score']
    
    # 3. Efficiency Ratio Method
    er_res = calculate_efficiency_ratio_momentum(series)
    results.update(er_res)
    
    # 4. Pure Return
    results['ret_simple'] = prices[-1] / prices[0] - 1
    
    return results

def test_momentum_variants():
    print("--- 动量计算方式全遍历测试 (Momentum Variants Test) ---")
    
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', ''))
    df_trade = df_trade.sort_values('调仓时间')
    
    history_data = load_history_data()
    
    dataset = []
    
    for _, row in df_trade.iterrows():
        trade_date = row['调仓时间']
        bought = row['买入']
        if bought == '现金' or bought not in history_data: continue
        
        row_data = {'date': trade_date, 'winner': bought, 'assets': {}}
        
        for asset, df in history_data.items():
            try:
                # 20-day window (Confirmed best)
                prev_date = df.index[df.index < trade_date][-1]
                prev_loc = df.index.get_loc(prev_date)
                
                if prev_loc >= 19:
                    series = df.iloc[prev_loc-19 : prev_loc+1]['close']
                    res = calculate_all_variants(series)
                    if res:
                        row_data['assets'][asset] = res
            except: pass
            
        if len(row_data['assets']) > 0:
            dataset.append(row_data)

    # Compare Metrics
    metrics = ['wls_score', 'ols_score', 'er_score', 'ret_simple']
    
    print(f"\n[测试结果对比 (Top 1 准确率)]")
    for m in metrics:
        matches = 0
        matches_top2 = 0
        total = 0
        
        for item in dataset:
            scores = {}
            for asset, res in item['assets'].items():
                scores[asset] = res[m]
            
            sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
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
        print(f"{m}: Top1={acc:.2%}, Top2={acc2:.2%}")

if __name__ == "__main__":
    test_momentum_variants()
