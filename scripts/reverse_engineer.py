import pandas as pd
import numpy as np
import os
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

# File paths
RECORD_DIR = 'record'
TRADE_FILE = os.path.join(RECORD_DIR, '2017-08-10 至 2026-02-26调仓记录.csv')
POSITION_FILE = os.path.join(RECORD_DIR, '2017-08-10 至 2026-02-26持仓记录.csv')

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

def calculate_linear_regression_momentum(series):
    """
    Calculate momentum based on the slope of linear regression
    slope = (n * sum(xy) - sum(x)sum(y)) / (n * sum(x^2) - (sum(x))^2)
    Then normalize by price to get % change per day? Or just R^2 * slope?
    Common 'linear regression momentum' is (slope / price) or just R^2 * slope.
    Given description: "Weighted Linear Regression Momentum Strategy"
    Maybe it means: Return * w1 + MaxDD * w2 + Vol * w3
    BUT description says: "Based on weighted linear regression momentum strategy" AND "Score based on Return, MaxDD, Volatility"
    Maybe the "Momentum" itself is calculated via LinReg slope of log prices?
    Let's calculate 3 factors for 20 days:
    1. Return (or Slope of LinReg)
    2. Max Drawdown
    3. Volatility
    """
    # Use log prices for regression
    y = np.log(series.values)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # Annualized slope = (exp(slope) ^ 252) - 1 approx slope * 252
    return slope * 252 # Annualized return based on regression

def calculate_factors(series):
    # 1. Return: Simple return or LinReg Slope?
    # Description says "Based on weighted linear regression momentum". 
    # Usually this implies the momentum factor IS the regression slope * R^2 (adjusted slope).
    # Let's try:
    # Factor A: Linear Regression Slope (annualized) * R^2 (quality of trend)
    # Factor B: Max Drawdown (negative)
    # Factor C: Volatility (negative)
    
    # Or maybe simply: Return (Simple), MaxDD, Volatility.
    # Let's try to calculate raw metrics first.
    
    # 20-day window
    if len(series) < 20: return None
    
    prices = series.values
    
    # 1. Return (Simple)
    ret = prices[-1] / prices[0] - 1
    
    # 2. Linear Regression Slope
    y = np.log(prices)
    x = np.arange(len(y))
    slope, _, r_value, _, _ = linregress(x, y)
    linreg_mom = (np.exp(slope * len(series)) - 1) * (r_value ** 2) # Adjusted Slope Momentum
    
    # 3. Max Drawdown
    cum_max = np.maximum.accumulate(prices)
    drawdowns = (prices - cum_max) / cum_max
    max_dd = drawdowns.min()
    
    # 4. Volatility
    daily_rets = series.pct_change().dropna()
    vol = daily_rets.std() * np.sqrt(252)
    
    return {
        'ret': ret,
        'slope': slope,
        'r2': r_value**2,
        'linreg_mom': linreg_mom, # Slope * R2
        'max_dd': max_dd,
        'vol': vol
    }

def reverse_engineer_strategy():
    print("--- 策略逆向工程 (Reverse Engineering) ---")
    
    # 1. Load Data
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', ''))
    df_trade = df_trade.sort_values('调仓时间')
    
    history_data = load_history_data()
    
    # 2. Collect Training Data for Weights
    # For each trade date, we have a "Winner" (the asset bought).
    # We want to find weights w1, w2, w3 such that:
    # Score = w1 * Factor1 + w2 * Factor2 + w3 * Factor3
    # And Score(Winner) is MAX among all assets.
    # Or at least Rank 1.
    
    # Factors candidates:
    # 1. Return (20d)
    # 2. LinReg Slope (20d)
    # 3. LinReg Slope * R^2 (20d)
    # 4. Max Drawdown (20d)
    # 5. Volatility (20d)
    
    # We can use a simple brute force or optimization to find best weights.
    # Since description says "Score based on Return, MaxDD, Volatility", let's assume:
    # Score = w_ret * Ret + w_dd * MaxDD + w_vol * Vol
    # Note: MaxDD and Vol are usually negative factors, so weights should be positive if we use -MaxDD, -Vol?
    # Or weights can be negative.
    
    correct_predictions = []
    
    # Let's collect the factor data for all assets on all trade dates
    dataset = []
    
    for _, row in df_trade.iterrows():
        trade_date = row['调仓时间']
        bought = row['买入']
        if bought == '现金': continue # Skip cash for weight learning (cash logic is separate)
        if bought not in history_data: continue
        
        # Calculate factors for ALL assets
        row_data = {'date': trade_date, 'winner': bought, 'assets': {}}
        
        for asset, df in history_data.items():
            try:
                # Data up to T-1
                prev_date = df.index[df.index < trade_date][-1]
                prev_loc = df.index.get_loc(prev_date)
                
                if prev_loc >= 19: # Need 20 points
                    series = df.iloc[prev_loc-19 : prev_loc+1]['close'] # 20 days
                    factors = calculate_factors(series)
                    if factors:
                        row_data['assets'][asset] = factors
            except: pass
            
        if len(row_data['assets']) > 0:
            dataset.append(row_data)

    # 3. Test different weight combinations
    # Grid search for weights:
    # w_ret: 0.5 to 2.0
    # w_dd: 0 to 2.0 (usually penalty)
    # w_vol: 0 to 2.0 (penalty)
    
    best_acc = 0
    best_weights = {}
    
    # Let's try 3 models:
    # Model A: Score = Ret + w_dd * MaxDD + w_vol * Vol
    # Model B: Score = LinReg_Mom + ...
    # Model C: Score = Slope + ...
    
    # Simplified grid search
    # We normalize factors first? Or just raw?
    # Description says "Score normalized to 0-100". This implies the final score is normalized.
    # But ranking depends on raw score.
    
    # Try weights
    for w_mom_type in ['ret', 'slope', 'linreg_mom']:
        for w_dd in [0, 0.5, 1.0, 1.5, 2.0, 3.0]: # Penalty weight (positive value, subtracted)
            for w_vol in [0, 0.5, 1.0, 1.5, 2.0, 3.0]: # Penalty weight
                
                matches = 0
                total = 0
                
                for item in dataset:
                    scores = {}
                    for asset, f in item['assets'].items():
                        # Score = Momentum - w_dd * |MaxDD| - w_vol * Vol
                        # MaxDD is negative number usually (e.g. -0.05). So + w_dd * MaxDD is a penalty if w_dd > 0?
                        # Wait, MaxDD is negative. To penalize, we add it (reduce score).
                        # So Score = Mom + w_dd * MaxDD (since MaxDD < 0) - w_vol * Vol
                        
                        mom = f[w_mom_type]
                        # Score formula
                        s = mom + w_dd * f['max_dd'] - w_vol * f['vol']
                        scores[asset] = s
                    
                    # Rank
                    if not scores: continue
                    sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    
                    # Check if winner is Top 1
                    if sorted_assets[0][0] == item['winner']:
                        matches += 1
                    total += 1
                
                acc = matches / total if total > 0 else 0
                if acc > best_acc:
                    best_acc = acc
                    best_weights = {
                        'mom_type': w_mom_type,
                        'w_dd': w_dd,
                        'w_vol': w_vol
                    }
                    print(f"New Best: {acc:.2%} -> Mom: {w_mom_type}, w_dd: {w_dd}, w_vol: {w_vol}")

    print("\n[最佳参数模型]")
    print(f"准确率: {best_acc:.2%}")
    print(f"参数: {best_weights}")
    
    # 4. Analyze Cash Logic with Best Model
    # "If no ETF meets standard, hold cash"
    # What is the standard? Score > 0? Or Score > Threshold?
    # Or "Score normalized to 0-100" means relative?
    # Usually "Score > 0" is the filter if Momentum is the driver.
    
    print("\n[空仓逻辑验证]")
    # Check cash trades with best model
    cash_violations = 0
    cash_trades_count = 0
    
    for _, row in df_trade.iterrows():
        if row['买入'] != '现金': continue
        
        trade_date = row['调仓时间']
        cash_trades_count += 1
        
        # Calculate max score on this day
        max_score = -999
        
        # Get asset data
        for asset, df in history_data.items():
            try:
                prev_date = df.index[df.index < trade_date][-1]
                prev_loc = df.index.get_loc(prev_date)
                if prev_loc >= 19:
                    series = df.iloc[prev_loc-19 : prev_loc+1]['close']
                    f = calculate_factors(series)
                    
                    # Calculate score
                    mom = f[best_weights['mom_type']]
                    s = mom + best_weights['w_dd'] * f['max_dd'] - best_weights['w_vol'] * f['vol']
                    
                    if s > max_score:
                        max_score = s
            except: pass
            
        print(f"  {trade_date.date()} 切入现金. 当时最高得分: {max_score:.4f}")
        if max_score > 0: # Assuming 0 is the threshold
            # print("    违反: 存在得分 > 0 的标的")
            cash_violations += 1
            
    print(f"空仓时最高得分 > 0 的次数: {cash_violations} / {cash_trades_count}")

if __name__ == "__main__":
    reverse_engineer_strategy()
