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

def calculate_mixed_factors(series_20, series_25):
    """Calculate factors using different windows"""
    
    # 1. 20-day factors
    # Momentum (Slope * R^2)
    y = np.log(series_20.values)
    x = np.arange(len(y))
    slope, _, r_value, _, _ = linregress(x, y)
    linreg_mom_20 = (np.exp(slope * 252) - 1) * (r_value ** 2)
    ret_20 = series_20.iloc[-1] / series_20.iloc[0] - 1
    
    # 2. 25-day factors
    # Max Drawdown
    prices_25 = series_25.values
    cum_max = np.maximum.accumulate(prices_25)
    drawdowns = (prices_25 - cum_max) / cum_max
    max_dd_25 = drawdowns.min()
    
    # Volatility
    daily_rets_25 = series_25.pct_change().dropna()
    vol_25 = daily_rets_25.std() * np.sqrt(252)
    
    # Also calculate 20-day versions for comparison
    prices_20 = series_20.values
    cum_max_20 = np.maximum.accumulate(prices_20)
    drawdowns_20 = (prices_20 - cum_max_20) / cum_max_20
    max_dd_20 = drawdowns_20.min()
    
    daily_rets_20 = series_20.pct_change().dropna()
    vol_20 = daily_rets_20.std() * np.sqrt(252)
    
    return {
        'linreg_mom_20': linreg_mom_20,
        'ret_20': ret_20,
        'max_dd_25': max_dd_25,
        'vol_25': vol_25,
        'max_dd_20': max_dd_20,
        'vol_20': vol_20
    }

def test_mixed_windows():
    print("--- 25日预热窗口假设验证 (Mixed Window Test) ---")
    
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', ''))
    df_trade = df_trade.sort_values('调仓时间')
    
    history_data = load_history_data()
    
    # Collect data
    dataset = []
    
    for _, row in df_trade.iterrows():
        trade_date = row['调仓时间']
        bought = row['买入']
        if bought == '现金': continue
        if bought not in history_data: continue
        
        row_data = {'date': trade_date, 'winner': bought, 'assets': {}}
        
        for asset, df in history_data.items():
            try:
                # Need up to 25 days before trade_date
                prev_date = df.index[df.index < trade_date][-1]
                prev_loc = df.index.get_loc(prev_date)
                
                if prev_loc >= 24: # Need 25 points (0 to 24)
                    series_25 = df.iloc[prev_loc-24 : prev_loc+1]['close'] # 25 days
                    series_20 = df.iloc[prev_loc-19 : prev_loc+1]['close'] # 20 days
                    
                    factors = calculate_mixed_factors(series_20, series_25)
                    row_data['assets'][asset] = factors
            except: pass
            
        if len(row_data['assets']) > 0:
            dataset.append(row_data)

    # Optimization Loop
    # We want to compare:
    # Model A: Score = Mom(20) + w_dd * MaxDD(20) - w_vol * Vol(20) (Base)
    # Model B: Score = Mom(20) + w_dd * MaxDD(25) - w_vol * Vol(25) (Mixed)
    
    best_acc = 0
    best_params = {}
    
    # Grid search
    # Mom Type: linreg_mom_20
    # DD Source: 20 or 25
    # Vol Source: 20 or 25
    
    # We found previously that w_dd=0, w_vol=0 was best for 20-day model (60.47%).
    # Let's see if adding 25-day DD/Vol improves it.
    
    for dd_source in ['max_dd_20', 'max_dd_25']:
        for vol_source in ['vol_20', 'vol_25']:
            for w_dd in [0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]:
                for w_vol in [0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]:
                    
                    matches = 0
                    total = 0
                    
                    for item in dataset:
                        scores = {}
                        for asset, f in item['assets'].items():
                            # Score = Mom(20) + w_dd * DD_Source - w_vol * Vol_Source
                            # Note: MaxDD is negative, so adding w_dd * MaxDD penalizes it.
                            s = f['linreg_mom_20'] + w_dd * f[dd_source] - w_vol * f[vol_source]
                            scores[asset] = s
                            
                        if not scores: continue
                        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        if sorted_assets[0][0] == item['winner']:
                            matches += 1
                        total += 1
                        
                    acc = matches / total if total > 0 else 0
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_params = {
                            'dd_source': dd_source,
                            'vol_source': vol_source,
                            'w_dd': w_dd,
                            'w_vol': w_vol
                        }
                        # print(f"New Best: {acc:.2%} -> DD: {dd_source}, Vol: {vol_source}, w_dd: {w_dd}, w_vol: {w_vol}")

    print(f"\n[最佳混合窗口模型]")
    print(f"准确率: {best_acc:.2%}")
    print(f"参数: {best_params}")
    
    # Check specifically if 25-day factors help explain the exceptions?
    # Compare with Base Model (Mom 20 only)
    base_matches = 0
    mixed_matches = 0
    total = 0
    
    for item in dataset:
        scores_base = {}
        scores_mixed = {}
        
        for asset, f in item['assets'].items():
            # Base: Mom 20 only
            scores_base[asset] = f['linreg_mom_20']
            
            # Mixed: Best Params
            s = f['linreg_mom_20'] + best_params['w_dd'] * f[best_params['dd_source']] - best_params['w_vol'] * f[best_params['vol_source']]
            scores_mixed[asset] = s
            
        sorted_base = sorted(scores_base.items(), key=lambda x: x[1], reverse=True)
        sorted_mixed = sorted(scores_mixed.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_base[0][0] == item['winner']: base_matches += 1
        if sorted_mixed[0][0] == item['winner']: mixed_matches += 1
        total += 1
        
    print(f"\n[模型对比]")
    print(f"基础模型 (仅20日动量): {base_matches}/{total} ({base_matches/total:.2%})")
    print(f"混合模型 (20日动量 + 25日因子): {mixed_matches}/{total} ({mixed_matches/total:.2%})")

if __name__ == "__main__":
    test_mixed_windows()
