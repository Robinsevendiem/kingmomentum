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

def calculate_factors(series):
    """Calculate Return, MaxDD, Volatility, Slope, R2"""
    if len(series) < 20: return None
    
    # 20-day factors
    prices = series.values
    
    # 1. Return
    ret = prices[-1] / prices[0] - 1
    
    # 2. Linear Regression Slope & R2
    y = np.log(prices)
    x = np.arange(len(y))
    slope, _, r_value, _, _ = linregress(x, y)
    r2 = r_value ** 2
    
    # Annualized slope
    ann_slope = np.exp(slope * 252) - 1
    
    # 3. Max Drawdown
    cum_max = np.maximum.accumulate(prices)
    drawdowns = (prices - cum_max) / cum_max
    max_dd = drawdowns.min()
    
    # 4. Volatility
    daily_rets = series.pct_change().dropna()
    vol = daily_rets.std() * np.sqrt(252)
    
    return {
        'ret': ret,
        'slope': ann_slope,
        'r2': r2,
        'slope_r2': ann_slope * r2,
        'max_dd': max_dd,
        'vol': vol
    }

def normalize_factors(assets_data):
    """Normalize factors across assets using different methods"""
    if not assets_data: return {}
    
    # Extract raw values
    keys = list(assets_data.keys())
    raw_values = {k: [] for k in assets_data[keys[0]].keys()}
    
    for asset, factors in assets_data.items():
        for k, v in factors.items():
            raw_values[k].append(v)
            
    # Calculate stats for normalization
    stats = {}
    for k, v in raw_values.items():
        arr = np.array(v)
        stats[k] = {
            'min': np.min(arr),
            'max': np.max(arr),
            'mean': np.mean(arr),
            'std': np.std(arr)
        }
        
    normalized_assets = {}
    
    for asset, factors in assets_data.items():
        normalized_assets[asset] = {}
        for k, v in factors.items():
            # 1. Min-Max Normalization (0-100)
            if stats[k]['max'] != stats[k]['min']:
                min_max = (v - stats[k]['min']) / (stats[k]['max'] - stats[k]['min']) * 100
            else:
                min_max = 50
            
            # 2. Z-Score
            if stats[k]['std'] != 0:
                z_score = (v - stats[k]['mean']) / stats[k]['std']
            else:
                z_score = 0
                
            normalized_assets[asset][f'{k}_minmax'] = min_max
            normalized_assets[asset][f'{k}_z'] = z_score
            normalized_assets[asset][k] = v # Keep raw
            
    return normalized_assets

def test_normalization():
    print("--- 跨市场因子归一化验证 (Cross-Market Normalization Test) ---")
    
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', ''))
    df_trade = df_trade.sort_values('调仓时间')
    
    history_data = load_history_data()
    
    dataset = []
    
    for _, row in df_trade.iterrows():
        trade_date = row['调仓时间']
        bought = row['买入']
        if bought == '现金': continue
        if bought not in history_data: continue
        
        # Calculate raw factors for ALL assets
        current_assets_factors = {}
        
        for asset, df in history_data.items():
            try:
                prev_date = df.index[df.index < trade_date][-1]
                prev_loc = df.index.get_loc(prev_date)
                if prev_loc >= 19:
                    series = df.iloc[prev_loc-19 : prev_loc+1]['close']
                    f = calculate_factors(series)
                    if f:
                        current_assets_factors[asset] = f
            except: pass
            
        if len(current_assets_factors) > 1: # Need at least 2 assets to normalize
            normalized = normalize_factors(current_assets_factors)
            dataset.append({'winner': bought, 'assets': normalized})

    # Optimization: Find best combination of Normalized Factors
    # Score = w1 * Norm(Ret/Slope) + w2 * Norm(MaxDD) - w3 * Norm(Vol)
    # Norm method: MinMax (0-100) or Z-Score?
    # Official doc says "Score normalized to 0-100", implies MinMax or Rank-based scaling.
    # But maybe factors themselves are normalized first? Or the final weighted sum is normalized?
    # "得分归一化至 0 到 100 之间" -> Usually means Final Score is mapped to 0-100.
    # But the INPUT factors (Vol, DD) are very different scales.
    # So we should normalize inputs first to make weights meaningful.
    
    best_acc = 0
    best_config = {}
    
    # Grid Search
    # Method: MinMax vs Z-Score
    # Mom Factor: Slope vs Slope*R2
    # Weights for DD and Vol
    
    normalization_methods = ['minmax', 'z']
    mom_factors = ['slope', 'slope_r2']
    
    for norm in normalization_methods:
        for mom in mom_factors:
            # Factor keys
            k_mom = f'{mom}_{norm}'
            k_dd = f'max_dd_{norm}'
            k_vol = f'vol_{norm}'
            
            # Weights
            # Since we normalized, DD (negative raw) -> MinMax (0 is worst, 100 is best? No, 0 is min raw (worst dd), 100 is max raw (best dd, close to 0)).
            # Wait, MaxDD is negative (e.g. -0.2). Max is -0.01 (better), Min is -0.5 (worse).
            # So MinMax(MaxDD): 100 is Best (Smallest Drawdown), 0 is Worst.
            # So we should ADD this term (positive weight).
            
            # Volatility is positive. Min is 0.1 (better), Max is 0.5 (worse).
            # MinMax(Vol): 100 is Highest Vol (Bad), 0 is Lowest Vol (Good).
            # So we should SUBTRACT this term (negative weight).
            
            # Z-Score:
            # DD: Higher Z (closer to 0) is better. ADD.
            # Vol: Higher Z (higher vol) is worse. SUBTRACT.
            
            for w_dd in [0, 0.5, 1.0, 2.0]:
                for w_vol in [0, 0.5, 1.0, 2.0]:
                    
                    matches = 0
                    total = 0
                    
                    for item in dataset:
                        scores = {}
                        for asset, f in item['assets'].items():
                            # Score calculation
                            s = f[k_mom] + w_dd * f[k_dd] - w_vol * f[k_vol]
                            scores[asset] = s
                            
                        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        if sorted_assets[0][0] == item['winner']:
                            matches += 1
                        total += 1
                        
                    acc = matches / total if total > 0 else 0
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_config = {
                            'norm': norm,
                            'mom': mom,
                            'w_dd': w_dd,
                            'w_vol': w_vol
                        }
                        # print(f"New Best: {acc:.2%} -> Norm: {norm}, Mom: {mom}, w_dd: {w_dd}, w_vol: {w_vol}")

    print(f"\n[最佳归一化模型]")
    print(f"准确率: {best_acc:.2%}")
    print(f"配置: {best_config}")
    
    # Compare with raw Slope*R2 model (previous best ~60-62%)
    # If this is significantly better (e.g. > 65%), then normalization is key.
    
if __name__ == "__main__":
    test_normalization()
