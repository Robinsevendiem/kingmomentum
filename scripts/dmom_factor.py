import pandas as pd
import numpy as np
import os
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

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

def calculate_dmom(series, market_series=None):
    """
    Directional Momentum (D-MOM)
    Model: P(r > 0) = delta0 + delta1 * IV + delta2 * r_m + beta+ * P + beta- * N
    
    Simplified Implementation for Factor Testing:
    We need to calculate the probability score based on the components.
    Usually this requires training a Logit/Probit model on historical data.
    But for a simple factor test without retraining every day, we can approximate the score 
    by combining the components with typical signs found in the paper.
    
    Paper findings (typical):
    - IV (Idiosyncratic Volatility): Negative impact? Or Positive?
      Usually high vol implies lower future returns in some contexts (IV puzzle).
      But here it's about DIRECTION. 
      Let's assume we just calculate the raw components first.
    
    Components:
    1. IV: Idiosyncratic Volatility (Variance of CAPM residuals)
    2. r_m: Market Return (Lagged)
    3. P: Max Streak of Positive Returns (in window)
    4. N: Max Streak of Negative Returns (in window)
    
    We need a Market Proxy. For this global portfolio, maybe 'ACWI' or just Equal Weight of the 9 assets?
    Let's use Equal Weight Index of the 9 assets as Market Proxy if not provided.
    """
    if len(series) < 20: return None
    
    returns = series.pct_change().dropna()
    
    # 1. Market Return
    # If market_series is None, use mean of available assets? 
    # For single asset calculation without external market data, we can't do CAPM IV properly.
    # Let's approximate IV as Total Volatility for now, or skip IV if market missing.
    # Actually, IV is a key component. 
    # Let's assume Market = Average of the 9 ETFs.
    
    if market_series is not None:
        # Align dates
        common_idx = returns.index.intersection(market_series.index)
        if len(common_idx) < 15: return None
        
        y = returns.loc[common_idx]
        X = market_series.loc[common_idx].values.reshape(-1, 1)
        
        # CAPM Regression
        reg = LinearRegression().fit(X, y)
        residuals = y - reg.predict(X)
        iv = residuals.var()
    else:
        # Fallback: Total Variance
        iv = returns.var()
        
    # 2. P and N (Streaks)
    # Max consecutive positive days
    # Max consecutive negative days
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
        
    # 3. Probability Score
    # We don't have the coefficients (delta, beta) without training.
    # But we can test the raw components as factors to see if they explain the strategy.
    # Or construct a naive score: D-MOM = P - N (Net Streak)
    # Or D-MOM = P / (P + N)
    # The paper suggests "Probability Score".
    # Let's return the raw components for analysis.
    
    return {
        'dmom_iv': iv,
        'dmom_P': max_pos_streak,
        'dmom_N': max_neg_streak,
        'dmom_net_streak': max_pos_streak - max_neg_streak,
        'dmom_ratio': max_pos_streak / (max_pos_streak + max_neg_streak) if (max_pos_streak + max_neg_streak) > 0 else 0
    }

def test_dmom_factor():
    print("--- 方向动量因子 (D-MOM) 测试 ---")
    
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', ''))
    df_trade = df_trade.sort_values('调仓时间')
    
    history_data = load_history_data()
    
    # Create Market Proxy (Equal Weight of all available assets)
    # Calculate daily returns for all assets first
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
                    # 20-day window
                    series = df.iloc[prev_loc-19 : prev_loc+1]['close']
                    
                    # Market series for same period
                    mkt_series = market_returns.loc[series.index]
                    
                    res = calculate_dmom(series, mkt_series)
                    if res:
                        row_data['assets'][asset] = res
            except: pass
            
        if len(row_data['assets']) > 0:
            dataset.append(row_data)

    # Analyze
    factors = ['dmom_iv', 'dmom_P', 'dmom_N', 'dmom_net_streak', 'dmom_ratio']
    
    print(f"\n[D-MOM 因子表现 (Top 1 准确率)]")
    for f in factors:
        matches = 0
        total = 0
        
        for item in dataset:
            scores = {}
            for asset, res in item['assets'].items():
                scores[asset] = res[f]
            
            # IV and N should be minimized? Or N is negative signal.
            # IV: usually low IV is better for "Stability". So Reverse=False (Ascending).
            # P: Higher is better. Reverse=True.
            # N: Lower is better. Reverse=False.
            # Net Streak: Higher is better. Reverse=True.
            
            is_reverse = True
            if f in ['dmom_iv', 'dmom_N']:
                is_reverse = False
                
            sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=is_reverse)
            
            if sorted_assets[0][0] == item['winner']:
                matches += 1
            total += 1
            
        acc = matches / total if total > 0 else 0
        print(f"{f}: {acc:.2%}")

    # Compare with Previous Best (WLS Score ~62%)
    # Is D-MOM better?
    # Net Streak is likely the best proxy for "Directional Momentum" without full regression.

if __name__ == "__main__":
    test_dmom_factor()
