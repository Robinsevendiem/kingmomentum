import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

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

def calculate_enhanced_dmom(series, market_series=None):
    """
    Enhanced Directional Momentum (D-MOM)
    P(r > 0) = delta0 + delta1 * IV + delta2 * r_m + delta3 * r_i + ...
    
    Variables:
    1. IV: Idiosyncratic Volatility (Variance of residuals)
    2. r_m: Lagged Market Return (20 days)
    3. r_i: Lagged Individual Return (20 days)
    4. P: Max Positive Streak (20 days)
    5. N: Max Negative Streak (20 days)
    
    Since we don't have trained coefficients (delta), we will:
    1. Use a simple weighted sum (Score) based on typical directional signs:
       Score = r_i (Pos) + P (Pos) - N (Pos) - IV (Pos/Neg?) + r_m (Pos)
       IV sign is tricky. High IV usually bad for momentum. So -IV.
       
    2. Or better, use Logistic Regression on the fly? 
       Train on past data (expanding window) to predict NEXT day return sign > 0?
       Or next 20-day return sign > 0?
       The strategy rotates every ~14 days. So prediction target should be "Return over next holding period > 0".
       
       However, doing a full ML backtest is complex.
       Let's stick to factor construction. The user asked for "Calculation Formula".
       We will construct the raw factors and see if any simple linear combination explains the trades.
       
       Focus on the new variable: r_i (Lagged Return).
       Wait, r_i is just "Momentum".
       The enhanced model is essentially: Momentum + Volatility + Streaks.
       
       Let's calculate all these components and see if a linear combination works.
    """
    if len(series) < 20: return None
    
    returns = series.pct_change().dropna()
    
    # 1. r_i: Lagged Individual Return (Total return over window)
    r_i = series.iloc[-1] / series.iloc[0] - 1
    
    # 2. r_m: Market Return
    if market_series is not None:
        common_idx = returns.index.intersection(market_series.index)
        if len(common_idx) > 0:
            mkt_slice = market_series.loc[common_idx]
            r_m = (mkt_series.iloc[-1] + 1) / (mkt_series.iloc[0] + 1) - 1 # Assuming market_series are returns? No, let's assume market_series is pct_change
            # Wait, market_returns passed in is mean(pct_change).
            # So cumulate it.
            r_m = np.prod(1 + mkt_slice) - 1
            
            # IV Calculation
            y = returns.loc[common_idx]
            X = mkt_slice.values.reshape(-1, 1)
            # Simple regression for residuals
            beta = np.cov(y, mkt_slice)[0, 1] / np.var(mkt_slice)
            alpha = np.mean(y) - beta * np.mean(mkt_slice)
            residuals = y - (alpha + beta * mkt_slice)
            iv = residuals.var()
        else:
            r_m = 0
            iv = returns.var()
    else:
        r_m = 0
        iv = returns.var()

    # 3. P and N
    pos_streak = 0
    max_pos_streak = 0
    neg_streak = 0
    max_neg_streak = 0
    
    for r in returns:
        if r > 0:
            pos_streak += 1
            neg_streak = 0
        elif r < 0:
            neg_streak += 1
            pos_streak = 0
        else:
            pos_streak = 0
            neg_streak = 0
        max_pos_streak = max(max_pos_streak, pos_streak)
        max_neg_streak = max(max_neg_streak, neg_streak)
        
    return {
        'dmom_r_i': r_i,      # The new enhanced factor
        'dmom_r_m': r_m,
        'dmom_iv': iv,
        'dmom_P': max_pos_streak,
        'dmom_N': max_neg_streak
    }

def test_enhanced_dmom():
    print("--- 增强型方向动量 (Enhanced D-MOM) 测试 ---")
    
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', ''))
    df_trade = df_trade.sort_values('调仓时间')
    
    history_data = load_history_data()
    
    # Market Proxy
    all_returns = pd.DataFrame()
    for asset, df in history_data.items():
        all_returns[asset] = df['close'].pct_change()
    market_returns = all_returns.mean(axis=1)
    
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
                    mkt_series = market_returns.loc[series.index]
                    
                    res = calculate_enhanced_dmom(series, mkt_series)
                    if res:
                        row_data['assets'][asset] = res
            except: pass
            
        if len(row_data['assets']) > 0:
            dataset.append(row_data)

    # Test the "Enhanced" combination
    # Score = r_i + ...
    # We know r_i (Simple Return) has ~55% accuracy.
    # Can we improve it by adding IV, P, N?
    # Score = w_r * r_i + w_p * P - w_n * N - w_iv * IV
    
    # We will normalize factors first to combine them
    
    best_acc = 0
    best_weights = {}
    
    # Simplified Grid Search
    # w_r is fixed at 1.0 (Base)
    # w_iv: 0, 0.5, 1.0
    # w_streak: 0, 0.5, 1.0 (Net streak P-N)
    
    for w_iv in [0, 1.0, 5.0, 10.0]:
        for w_streak in [0, 0.1, 0.5, 1.0]:
            matches = 0
            total = 0
            
            for item in dataset:
                scores = {}
                # Normalize for this day
                vals = {'r_i': [], 'iv': [], 'net_streak': []}
                assets = list(item['assets'].keys())
                for asset in assets:
                    vals['r_i'].append(item['assets'][asset]['dmom_r_i'])
                    vals['iv'].append(item['assets'][asset]['dmom_iv'])
                    vals['net_streak'].append(item['assets'][asset]['dmom_P'] - item['assets'][asset]['dmom_N'])
                
                # MinMax
                norm = {}
                for k in vals:
                    arr = np.array(vals[k])
                    mn, mx = np.min(arr), np.max(arr)
                    if mx != mn:
                        norm[k] = (arr - mn) / (mx - mn)
                    else:
                        norm[k] = np.zeros_like(arr)
                        
                for i, asset in enumerate(assets):
                    # Score: Higher r_i is better. Lower IV is better (so 1-NormIV). Higher Streak is better.
                    s = norm['r_i'][i] + w_streak * norm['net_streak'][i] - w_iv * norm['iv'][i] # Raw IV is small, normalized is 0-1.
                    # Wait, Normalized IV: 1 is Max IV (Worst). So subtract.
                    scores[asset] = s
                    
                sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                if sorted_assets[0][0] == item['winner']:
                    matches += 1
                total += 1
                
            acc = matches / total if total > 0 else 0
            if acc > best_acc:
                best_acc = acc
                best_weights = {'w_iv': w_iv, 'w_streak': w_streak}
                
    print(f"\n[增强型 D-MOM 最佳组合]")
    print(f"准确率: {best_acc:.2%}")
    print(f"参数: {best_weights}")
    print(f"对比: 最佳WLS模型准确率 ~62.8%")

if __name__ == "__main__":
    test_enhanced_dmom()
