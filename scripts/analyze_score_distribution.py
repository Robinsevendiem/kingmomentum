import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from scipy.stats import linregress

warnings.filterwarnings('ignore')

# --- 1. Data & Score Calculation ---

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
            try:
                df = pd.read_csv(filename)
                if 'trade_date' in df.columns:
                    try:
                        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                    except:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').set_index('trade_date')
                history_data[name] = df
            except: pass
    return history_data

def calculate_rolling_scores(series, window=20):
    scores = pd.Series(index=series.index, dtype=float)
    scores[:] = np.nan
    x = np.arange(window)
    x_norm = np.linspace(0, 1, window)
    weights = 1 + x_norm ** 2
    log_prices = np.log(series)
    values = log_prices.values
    
    for i in range(window, len(values) + 1):
        window_data = values[i-window : i]
        if np.isnan(window_data).any(): continue
        try:
            coeffs = np.polyfit(x, window_data, 1, w=weights)
            slope = coeffs[0]
            y_pred = np.polyval(coeffs, x)
            sse = np.sum(weights * (window_data - y_pred)**2)
            y_mean = np.average(window_data, weights=weights)
            sst = np.sum(weights * (window_data - y_mean)**2)
            if sst == 0: r2 = 0
            else: r2 = 1 - sse / sst
            score = (np.exp(slope * 252) - 1) * r2 * 100
            scores.iloc[i-1] = score
        except: pass
    return scores

def precalculate_all_scores(history_data):
    all_scores = pd.DataFrame()
    for asset, df in history_data.items():
        if 'close' in df.columns:
            scores = calculate_rolling_scores(df['close'])
            scores.name = asset
            all_scores = pd.merge(all_scores, scores, left_index=True, right_index=True, how='outer')
    return all_scores

# --- 2. Distribution Analysis ---

def analyze_distributions(all_scores):
    print("\n--- 动量分数分布统计 (Score Statistics) ---")
    stats = []
    
    # Filter only positive scores for cutoff analysis? 
    # Or all scores? Cutoff usually applies to high positive scores.
    # Let's look at positive scores only.
    
    for col in all_scores.columns:
        s = all_scores[col].dropna()
        pos_s = s[s > 0]
        
        if len(pos_s) > 0:
            p90 = np.percentile(pos_s, 90)
            p95 = np.percentile(pos_s, 95)
            p99 = np.percentile(pos_s, 99)
            mx = pos_s.max()
            mean = pos_s.mean()
            std = pos_s.std()
            
            stats.append({
                'Asset': col,
                'Mean': mean,
                'Std': std,
                'P90': p90,
                'P95': p95,
                'P99': p99,
                'Max': mx
            })
            
    df_stats = pd.DataFrame(stats)
    print(df_stats.to_string(float_format="{:.2f}".format))
    return df_stats

# --- 3. Dynamic Cutoff Backtest ---

def run_backtest_dynamic(history_data, raw_scores_df, cutoff_config):
    """
    cutoff_config: {
        'type': 'global' | 'asset_specific',
        'value': float (fixed score) or dict {asset: score}
    }
    """
    # Fixed Params
    BUFFER_SCORE_DIFF = 8.0
    FEE_RATE = 0.0005
    # Crash Filter: Window 3, Threshold 3% (from previous optimization)
    CRASH_WINDOW = 3
    CRASH_THRESHOLD = 0.03
    
    start_date = pd.Timestamp('2017-08-01')
    timeline = [d for d in raw_scores_df.index if d >= start_date]
    timeline = sorted(timeline)
    
    cash = 100000.0
    holdings = {}
    current_asset = '现金'
    target_asset = '现金'
    
    daily_returns = {}
    for asset, df in history_data.items():
        daily_returns[asset] = df['close'].pct_change()
    
    price_open = {asset: df['open'] for asset, df in history_data.items()}
    price_close = {asset: df['close'] for asset, df in history_data.items()}
    
    value_history = []
    
    for date in timeline:
        # A. Execution
        can_sell = True
        can_buy = True
        
        if current_asset != '现金':
            if date not in price_open[current_asset].index: can_sell = False
        if target_asset != '现金':
            if date not in price_open[target_asset].index: can_buy = False
            
        if current_asset != target_asset and current_asset != '现金' and can_sell:
            price = price_open[current_asset].loc[date]
            proceeds = holdings[current_asset] * price * (1 - FEE_RATE)
            cash += proceeds
            del holdings[current_asset]
            current_asset = '现金'
            
        if current_asset == '现金' and target_asset != '现金' and can_buy:
            price = price_open[target_asset].loc[date]
            shares = cash / (price * (1 + FEE_RATE))
            cost = shares * price * (1 + FEE_RATE)
            cash -= cost
            holdings[target_asset] = shares
            current_asset = target_asset
            
        # B. Valuation
        day_value = cash
        for asset, shares in holdings.items():
            if date in price_close[asset].index:
                price = price_close[asset].loc[date]
            else:
                try: price = price_close[asset].asof(date)
                except: price = 0
            day_value += shares * price
        value_history.append({'date': date, 'value': day_value})
        
        # C. Signal
        if date not in raw_scores_df.index:
            next_target = '现金'
        else:
            today_scores = raw_scores_df.loc[date].dropna()
            
            if today_scores.empty:
                next_target = '现金'
            else:
                # 1. Crash Filter
                valid_after_crash = []
                for asset in today_scores.index:
                    is_crashed = False
                    if asset in daily_returns:
                        try:
                            idx = daily_returns[asset].index.get_loc(date)
                            start_idx = max(0, idx - CRASH_WINDOW + 1)
                            recent_rets = daily_returns[asset].iloc[start_idx : idx+1]
                            if recent_rets.min() < -CRASH_THRESHOLD:
                                is_crashed = True
                        except: pass
                    if not is_crashed:
                        valid_after_crash.append(asset)
                
                filtered_scores = today_scores[today_scores.index.isin(valid_after_crash)]
                
                # 2. Cutoff Logic (Dynamic)
                valid_candidates = []
                for asset, score in filtered_scores.items():
                    if score <= 0: continue
                    
                    # Determine Cutoff
                    if cutoff_config['type'] == 'global':
                        limit = cutoff_config['value']
                    elif cutoff_config['type'] == 'asset_specific':
                        limit = cutoff_config['value'].get(asset, 300) # Default 300
                    else:
                        limit = 300
                        
                    if score <= limit:
                        valid_candidates.append(asset)
                        
                candidate_scores = filtered_scores[filtered_scores.index.isin(valid_candidates)]
                
                if candidate_scores.empty:
                    next_target = '现金'
                else:
                    # Normalize
                    vals = filtered_scores.values # Normalize relative to potential pool
                    if len(vals) > 0:
                        mn, mx = np.min(vals), np.max(vals)
                        if mx == mn:
                            norm_scores = pd.Series(50, index=filtered_scores.index)
                        else:
                            norm_scores = (filtered_scores - mn) / (mx - mn) * 100
                    else:
                        norm_scores = pd.Series()
                        
                    best_valid_asset = candidate_scores.idxmax()
                    best_valid_norm = norm_scores[best_valid_asset]
                    
                    if current_asset not in candidate_scores.index:
                        next_target = best_valid_asset
                    else:
                        curr_norm = norm_scores[current_asset]
                        if best_valid_norm - curr_norm > BUFFER_SCORE_DIFF:
                            next_target = best_valid_asset
                        else:
                            next_target = current_asset
                            
        target_asset = next_target
        
    # Stats
    df_res = pd.DataFrame(value_history).set_index('date')
    total_ret = df_res['value'].iloc[-1] / df_res['value'].iloc[0] - 1
    cum_max = df_res['value'].cummax()
    max_dd = ((df_res['value'] - cum_max) / cum_max).min()
    vol = df_res['value'].pct_change().std() * np.sqrt(252)
    sharpe = ((1+total_ret)**(252/len(df_res))-1 - 0.02) / vol if vol != 0 else 0
    
    return {'sharpe': sharpe, 'return': total_ret, 'max_dd': max_dd}

# --- 4. Main Optimization ---

def optimize_dynamic_cutoff():
    history = load_history_data()
    scores = precalculate_all_scores(history)
    
    # 1. Stats
    stats = analyze_distributions(scores)
    
    # 2. Test Configurations
    print("\n--- Testing Cutoff Strategies ---")
    
    # A. Global Fixed (Baseline)
    res_global = run_backtest_dynamic(history, scores, {'type': 'global', 'value': 300})
    print(f"Global Fixed (300): Sharpe={res_global['sharpe']:.2f}, Ret={res_global['return']:.2%}, MaxDD={res_global['max_dd']:.2%}")
    
    # B. Asset Specific - Percentile Based (e.g., P95)
    # Why P95? "Overheating" is extreme.
    
    for p_label, p_col in [('P90', 'P90'), ('P95', 'P95'), ('P99', 'P99')]:
        cutoff_map = dict(zip(stats['Asset'], stats[p_col]))
        res_p = run_backtest_dynamic(history, scores, {'type': 'asset_specific', 'value': cutoff_map})
        print(f"Asset Specific ({p_label}): Sharpe={res_p['sharpe']:.2f}, Ret={res_p['return']:.2%}, MaxDD={res_p['max_dd']:.2%}")
        
    # C. Asset Specific - Max (No Cutoff effectively per asset history?)
    # Or maybe "Mean + 2 Std"?
    stats['Mean+2Std'] = stats['Mean'] + 2 * stats['Std']
    cutoff_map_std = dict(zip(stats['Asset'], stats['Mean+2Std']))
    res_std = run_backtest_dynamic(history, scores, {'type': 'asset_specific', 'value': cutoff_map_std})
    print(f"Asset Specific (Mean+2Std): Sharpe={res_std['sharpe']:.2f}, Ret={res_std['return']:.2%}, MaxDD={res_std['max_dd']:.2%}")
    
    # D. Hybrid / Relaxed Global
    # Maybe global 300 is too strict for high-vol assets like Crude/Nasdaq?
    # Let's see stats first.

if __name__ == "__main__":
    optimize_dynamic_cutoff()
