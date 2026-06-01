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

def calculate_all_factors(series):
    """
    Calculate all common variants of Momentum, Drawdown, and Volatility.
    Input: Series of prices (Close) for 20 days (or N days).
    """
    if len(series) < 5: return None
    
    prices = series.values
    n = len(prices)
    
    results = {}
    
    # --- 1. Momentum (动量) ---
    # 1.1 Simple Return (简单收益率)
    # Formula: P_end / P_start - 1
    results['mom_simple_ret'] = prices[-1] / prices[0] - 1
    
    # 1.2 Log Return (对数收益率)
    # Formula: ln(P_end / P_start)
    results['mom_log_ret'] = np.log(prices[-1] / prices[0])
    
    # 1.3 Linear Regression Slope (Raw Slope)
    # Regression of Log Price vs Time (0, 1, ..., n-1)
    # Model: ln(P) = alpha + beta * t
    y = np.log(prices)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    results['mom_slope'] = slope
    
    # 1.4 Annualized Slope (年化斜率)
    # Formula: exp(slope * 252) - 1
    results['mom_ann_slope'] = np.exp(slope * 252) - 1
    
    # 1.5 Slope * R^2 (调整后动量)
    # Formula: Slope * (R^2)
    # This penalizes high volatility trends
    results['mom_slope_r2'] = slope * (r_value ** 2)
    
    # 1.6 Annualized Slope * R^2
    results['mom_ann_slope_r2'] = (np.exp(slope * 252) - 1) * (r_value ** 2)
    
    # 1.7 Average Daily Return (平均日收益率)
    daily_rets = series.pct_change().dropna()
    results['mom_avg_daily_ret'] = daily_rets.mean()
    
    # --- 2. Drawdown (回撤) ---
    # 2.1 Max Drawdown (最大回撤)
    # Formula: Min((Price - CumMax) / CumMax)
    cum_max = np.maximum.accumulate(prices)
    drawdowns = (prices - cum_max) / cum_max
    results['dd_max'] = drawdowns.min()
    
    # 2.2 Average Drawdown (平均回撤)
    results['dd_avg'] = drawdowns.mean()
    
    # 2.3 End Drawdown (当前回撤)
    # Drawdown at the last day
    results['dd_current'] = drawdowns[-1]
    
    # 2.4 Ulcer Index (溃疡指数)
    # Sqrt(Mean(Drawdown^2))
    results['dd_ulcer'] = np.sqrt(np.mean(drawdowns ** 2))
    
    # --- 3. Volatility (波动率) ---
    # 3.1 Standard Deviation of Returns (标准差)
    # Formula: Std(DailyRet) * Sqrt(252)
    vol_daily = daily_rets.std()
    results['vol_std'] = vol_daily * np.sqrt(252)
    
    # 3.2 Downside Deviation (下行波动率)
    # Only consider negative returns
    neg_rets = daily_rets[daily_rets < 0]
    if len(neg_rets) > 0:
        downside_vol = np.sqrt(np.mean(neg_rets ** 2)) * np.sqrt(252)
    else:
        downside_vol = 0
    results['vol_downside'] = downside_vol
    
    # 3.3 Average True Range (ATR) - Simplified
    # Since we only have Close, we can use High-Low if available, or just Abs(Close - PrevClose)
    # But here input is just Series (Close). So we can only do Volatility of Close.
    # ATR usually requires High/Low. We'll skip ATR for now unless we load full OHLC.
    
    # 3.4 Coefficient of Variation (变异系数)
    # Std / Mean
    if np.mean(prices) != 0:
        results['vol_cv'] = np.std(prices) / np.mean(prices)
    else:
        results['vol_cv'] = 0
        
    return results

def test_factor_formulas():
    print("--- 因子公式精确化测试 (Factor Precision Test) ---")
    
    # Load Data
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
                # Use 20-day window (as confirmed previously)
                prev_date = df.index[df.index < trade_date][-1]
                prev_loc = df.index.get_loc(prev_date)
                
                if prev_loc >= 19:
                    # Series of Close prices
                    series = df.iloc[prev_loc-19 : prev_loc+1]['close']
                    
                    # Calculate ALL factor variants
                    factors = calculate_all_factors(series)
                    if factors:
                        row_data['assets'][asset] = factors
            except: pass
            
        if len(row_data['assets']) > 0:
            dataset.append(row_data)

    # Analyze: Which single factor correlates best with "Winner"?
    # We check each factor individually first.
    
    factor_performance = {}
    
    # Get list of all calculated factor keys
    sample_keys = list(dataset[0]['assets'][list(dataset[0]['assets'].keys())[0]].keys())
    
    for key in sample_keys:
        matches = 0
        total = 0
        
        for item in dataset:
            scores = {}
            for asset, f in item['assets'].items():
                scores[asset] = f[key]
            
            # For Drawdown and Volatility, we usually want to MINIMIZE them.
            # So if we sort Descending (Reverse=True), we expect "Least Negative" for MaxDD (since MaxDD is negative).
            # But Volatility is positive. So for Volatility, "Best" is Smallest.
            # So we need to handle sorting direction.
            
            is_reverse = True # Default for Momentum and MaxDD (since -0.1 > -0.5)
            if 'vol' in key or 'ulcer' in key:
                is_reverse = False # Smaller is better
            
            sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=is_reverse)
            
            # Check if Winner is Top 1
            if sorted_assets[0][0] == item['winner']:
                matches += 1
            total += 1
            
        acc = matches / total if total > 0 else 0
        factor_performance[key] = acc
        
    print("\n[单因子解释力排名]")
    sorted_factors = sorted(factor_performance.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_factors:
        print(f"{k}: {v:.2%}")
        
    # Check Combinations?
    # We suspect "Score = Mom + w * DD - w * Vol"
    # Let's try combining the best Mom, best DD, best Vol.
    
    best_mom = sorted_factors[0][0] # Likely mom_slope_r2 or mom_ann_slope_r2
    
    # Find best DD and Vol from the list
    best_dd = next((k for k, v in sorted_factors if 'dd' in k), 'dd_max')
    best_vol = next((k for k, v in sorted_factors if 'vol' in k), 'vol_std')
    
    print(f"\n[最佳候选因子组合]")
    print(f"动量: {best_mom}")
    print(f"回撤: {best_dd}")
    print(f"波动: {best_vol}")
    
    # Quick Grid Search for Weights with these specific factors
    # Assuming normalized (MinMax) because raw scales differ too much to combine directly without weights.
    # But for "Precision", we want to see if raw combination works?
    # No, we must normalize as per strategy description.
    
    print("\n[因子组合测试 (MinMax归一化)]")
    
    best_combo_acc = 0
    best_combo_weights = {}
    
    # We use MinMax normalization for the chosen factors
    # For Vol/Ulcer, MinMax(0=Best, 100=Worst)? No, usually MinMax maps Min->0, Max->100.
    # So for Vol, 0 is Best (Lowest Vol).
    # So Score = Mom_Score + DD_Score - Vol_Score
    # Where Mom_Score: 100 is Best.
    # DD_Score: MaxDD is negative. Min is -0.5, Max is -0.01. MinMax maps -0.5->0, -0.01->100. So 100 is Best.
    # Vol_Score: Min is 0.1, Max is 0.5. MinMax maps 0.1->0, 0.5->100. So 0 is Best.
    # So Formula: Score = w_mom * Mom_Norm + w_dd * DD_Norm - w_vol * Vol_Norm
    
    for w_dd in [0, 0.5, 1.0, 1.5]:
        for w_vol in [0, 0.5, 1.0, 1.5]:
            matches = 0
            total = 0
            
            for item in dataset:
                # Normalize these 3 factors for current assets
                raw_values = {'mom': [], 'dd': [], 'vol': []}
                assets = list(item['assets'].keys())
                
                for asset in assets:
                    raw_values['mom'].append(item['assets'][asset][best_mom])
                    raw_values['dd'].append(item['assets'][asset][best_dd])
                    raw_values['vol'].append(item['assets'][asset][best_vol])
                
                # MinMax
                norm_scores = {}
                for k in raw_values:
                    arr = np.array(raw_values[k])
                    mn, mx = np.min(arr), np.max(arr)
                    if mx != mn:
                        norm = (arr - mn) / (mx - mn) * 100
                    else:
                        norm = np.full_like(arr, 50)
                    norm_scores[k] = norm
                
                # Calculate Final Score
                final_scores = {}
                for i, asset in enumerate(assets):
                    # Score = Mom + w_dd * DD - w_vol * Vol
                    s = norm_scores['mom'][i] + w_dd * norm_scores['dd'][i] - w_vol * norm_scores['vol'][i]
                    final_scores[asset] = s
                    
                sorted_assets = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
                if sorted_assets[0][0] == item['winner']:
                    matches += 1
                total += 1
            
            acc = matches / total if total > 0 else 0
            if acc > best_combo_acc:
                best_combo_acc = acc
                best_combo_weights = {'w_dd': w_dd, 'w_vol': w_vol}
                
    print(f"最佳组合准确率: {best_combo_acc:.2%}")
    print(f"参数: {best_combo_weights}")

if __name__ == "__main__":
    test_factor_formulas()
