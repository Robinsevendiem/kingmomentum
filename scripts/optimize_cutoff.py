import pandas as pd
import numpy as np
import os
from scipy.stats import linregress

# File paths
RECORD_DIR = 'record'
# We need history data
def load_history_data():
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

def calculate_score(prices):
    """
    Calculate Quadratic Weighted Linear Regression Momentum Score.
    Score = Annualized Slope * R^2 * 100
    """
    if len(prices) < 20: return -999
    
    y = np.log(prices)
    x = np.arange(len(y))
    x_norm = np.linspace(0, 1, len(y))
    weights = 1 + x_norm ** 2
    
    coeffs = np.polyfit(x, y, 1, w=weights)
    slope = coeffs[0]
    
    y_pred = np.polyval(coeffs, x)
    sse = np.sum(weights * (y - y_pred)**2)
    y_mean = np.average(y, weights=weights)
    sst = np.sum(weights * (y - y_mean)**2)
    if sst == 0: r2 = 0
    else: r2 = 1 - sse / sst
    
    score = (np.exp(slope * 252) - 1) * r2 * 100
    return score

def run_backtest_with_cutoff(cutoff_score):
    """Run backtest with a specific cutoff score"""
    history_data = load_history_data()
    assets = list(history_data.keys())
    
    all_dates = sorted(list(set().union(*[df.index for df in history_data.values()])))
    start_date = pd.Timestamp('2017-08-01')
    timeline = [d for d in all_dates if d >= start_date]
    
    cash = 100000.0
    holdings = {}
    current_asset = '现金'
    FEE_RATE = 0.0005
    BUFFER_SCORE_DIFF = 5.0
    
    trade_log = []
    value_history = []
    
    target_asset = '现金'
    
    for i, date in enumerate(timeline):
        # 1. Execute Trade at Open (based on target determined yesterday)
        can_sell = True
        can_buy = True
        
        if current_asset != '现金':
            if date not in history_data[current_asset].index: can_sell = False
        if target_asset != '现金':
            if date not in history_data[target_asset].index: can_buy = False
            
        # Sell
        if current_asset != target_asset and current_asset != '现金' and can_sell:
            price = history_data[current_asset].loc[date, 'open']
            proceeds = holdings[current_asset] * price * (1 - FEE_RATE)
            cash += proceeds
            del holdings[current_asset]
            current_asset = '现金'
            
        # Buy
        if current_asset == '现金' and target_asset != '现金' and can_buy:
            price = history_data[target_asset].loc[date, 'open']
            shares = cash / (price * (1 + FEE_RATE))
            cost = shares * price * (1 + FEE_RATE)
            cash -= cost
            holdings[target_asset] = shares
            current_asset = target_asset
            
        # 2. Portfolio Value
        day_value = cash
        for asset, shares in holdings.items():
            if date in history_data[asset].index:
                price = history_data[asset].loc[date, 'close']
            else:
                idx = history_data[asset].index.get_indexer([date], method='pad')[0]
                if idx >= 0: price = history_data[asset].iloc[idx]['close']
                else: price = 0
            day_value += shares * price
        value_history.append({'date': date, 'value': day_value})
        
        # 3. Generate Signal for Tomorrow
        scores = {}
        for asset in assets:
            df = history_data[asset]
            if date in df.index:
                loc = df.index.get_loc(date)
                if loc >= 19:
                    series = df.iloc[loc-19 : loc+1]['close']
                    s = calculate_score(series.values)
                    scores[asset] = s
        
        if not scores:
            next_target = '现金'
        else:
            # MinMax Normalization
            vals = np.array(list(scores.values()))
            mn, mx = np.min(vals), np.max(vals)
            norm_scores = {}
            for k, v in scores.items():
                if mx != mn: norm_scores[k] = (v - mn) / (mx - mn) * 100
                else: norm_scores[k] = 50
                
            sorted_scores = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
            top1_asset = sorted_scores[0][0]
            top1_norm = sorted_scores[0][1]
            top1_raw = scores[top1_asset]
            
            # --- CUTOFF LOGIC ---
            # If Top 1 Raw Score > Cutoff, it is "Overheated".
            # What to do? 
            # Option A: Skip Top 1, check Top 2?
            # Option B: Hold Cash? (Avoid risk)
            # Usually "Overheated" means "Don't Buy".
            # If we hold it, maybe Sell?
            # Let's assume: If Top 1 Raw > Cutoff, we treat it as invalid (Cash).
            # Or do we pick Top 2?
            # "不进入 ETF 排序" implies it is removed from candidates.
            # So we check Top 2. If Top 2 is also > Cutoff, check Top 3...
            # If all valid > Cutoff, Cash.
            
            valid_candidates = [a for a in scores if scores[a] <= cutoff_score]
            
            if not valid_candidates:
                next_target = '现金'
            else:
                # Re-rank valid candidates
                # Note: Normalization was based on ALL. Should we re-normalize?
                # Probably not necessary for relative rank. Just pick best valid raw score.
                # Or use normalized score of best valid raw.
                
                # Find best valid asset
                # Sort valid by raw score (same order as normalized)
                sorted_valid = sorted([(a, scores[a]) for a in valid_candidates], key=lambda x: x[1], reverse=True)
                
                best_valid_asset = sorted_valid[0][0]
                best_valid_raw = sorted_valid[0][1]
                
                # Check negative score rule
                if best_valid_raw < 0:
                    next_target = '现金'
                else:
                    # Apply buffer rule with best_valid_asset
                    # If current is valid and in Top 2 of valid list...
                    # Let's simplify: Just switch to best valid. Buffer complicates "validity".
                    # If current is > Cutoff, we MUST sell.
                    # If current <= Cutoff, we can keep it if close to best valid.
                    
                    if current_asset in valid_candidates:
                        # Current is valid. Is it close to best valid?
                        curr_raw = scores[current_asset]
                        # Convert to norm scale for comparison?
                        # Norm scale depends on min/max of ALL.
                        curr_norm = norm_scores[current_asset]
                        best_valid_norm = norm_scores[best_valid_asset]
                        
                        if best_valid_norm - curr_norm < BUFFER_SCORE_DIFF:
                            next_target = current_asset
                        else:
                            next_target = best_valid_asset
                    else:
                        next_target = best_valid_asset
                        
        target_asset = next_target
        
    # Stats
    df_res = pd.DataFrame(value_history).set_index('date')
    total_ret = df_res['value'].iloc[-1] / df_res['value'].iloc[0] - 1
    cum_max = df_res['value'].cummax()
    max_dd = ((df_res['value'] - cum_max) / cum_max).min()
    vol = df_res['value'].pct_change().std() * np.sqrt(252)
    sharpe = ((1+total_ret)**(252/len(df_res))-1 - 0.02) / vol if vol != 0 else 0
    
    return {
        'cutoff': cutoff_score,
        'return': total_ret,
        'max_dd': max_dd,
        'sharpe': sharpe
    }

def optimize_cutoff():
    print("--- 动量得分上限阈值优化 (Score Cutoff Optimization) ---")
    
    # Range of scores?
    # WLS Score = AnnSlope * R2 * 100.
    # AnnSlope for 20 days can be high. e.g. 10% in 20 days -> (1.1)^(252/20) - 1 ~ 200%?
    # Let's check distribution of Top 1 Scores first.
    # But we want to find optimal cutoff.
    # Let's test: 50, 80, 100, 150, 200, 300, 500, 1000 (No Limit)
    
    cutoffs = [50, 80, 100, 120, 150, 200, 300, 500, 9999]
    
    results = []
    for c in cutoffs:
        res = run_backtest_with_cutoff(c)
        results.append(res)
        print(f"阈值: {c} -> 收益: {res['return']:.2%}, 回撤: {res['max_dd']:.2%}, 夏普: {res['sharpe']:.2f}")
        
    # Find Best Sharpe
    best_sharpe = max(results, key=lambda x: x['sharpe'])
    print(f"\n[最佳夏普配置]")
    print(f"阈值: {best_sharpe['cutoff']}")
    print(f"夏普: {best_sharpe['sharpe']:.2f}")
    print(f"收益: {best_sharpe['return']:.2%}")
    print(f"回撤: {best_sharpe['max_dd']:.2%}")

if __name__ == "__main__":
    optimize_cutoff()
