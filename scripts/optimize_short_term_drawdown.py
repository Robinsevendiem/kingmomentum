import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import linregress

warnings.filterwarnings('ignore')

# --- 1. Data Loading & Pre-calculation ---

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
                # Flexible parsing
                if 'trade_date' in df.columns:
                    try:
                        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                    except:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').set_index('trade_date')
                history_data[name] = df
            except:
                pass
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
    print("Pre-calculating scores...")
    all_scores = pd.DataFrame()
    for asset, df in history_data.items():
        if 'close' in df.columns:
            scores = calculate_rolling_scores(df['close'])
            scores.name = asset
            all_scores = pd.merge(all_scores, scores, left_index=True, right_index=True, how='outer')
    return all_scores

# --- 2. Backtest Engine with Crash Filter ---

def run_backtest_with_crash_filter(history_data, raw_scores_df, crash_params):
    """
    crash_params: {
        'window': int,    # Check last N days
        'threshold': float # If any daily drop > X% (positive value), exclude
    }
    """
    # Base Params
    CUTOFF_SCORE = 300
    BUFFER_SCORE_DIFF = 8.0
    FEE_RATE = 0.0005
    start_date = pd.Timestamp('2017-08-01')
    
    timeline = [d for d in raw_scores_df.index if d >= start_date]
    timeline = sorted(timeline)
    
    cash = 100000.0
    holdings = {} 
    current_asset = '现金'
    target_asset = '现金'
    
    value_history = []
    trade_count = 0
    
    # Pre-calculate Daily Returns for Crash Check
    daily_returns = {}
    for asset, df in history_data.items():
        daily_returns[asset] = df['close'].pct_change()
    
    price_open = {asset: df['open'] for asset, df in history_data.items()}
    price_close = {asset: df['close'] for asset, df in history_data.items()}
    
    for date in timeline:
        # A. Execution
        can_sell = True
        can_buy = True
        
        if current_asset != '现金':
            if date not in price_open[current_asset].index: can_sell = False
        if target_asset != '现金':
            if date not in price_open[target_asset].index: can_buy = False
            
        # Sell
        if current_asset != target_asset and current_asset != '现金' and can_sell:
            price = price_open[current_asset].loc[date]
            proceeds = holdings[current_asset] * price * (1 - FEE_RATE)
            cash += proceeds
            del holdings[current_asset]
            current_asset = '现金'
            trade_count += 1
            
        # Buy
        if current_asset == '现金' and target_asset != '现金' and can_buy:
            price = price_open[target_asset].loc[date]
            shares = cash / (price * (1 + FEE_RATE))
            cost = shares * price * (1 + FEE_RATE)
            cash -= cost
            holdings[target_asset] = shares
            current_asset = target_asset
            trade_count += 1
            
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
                # 1. Apply Crash Filter
                # Exclude assets that had a daily drop > threshold in last N days
                # Note: Signal generated at Close. So check drops up to today?
                # Yes, "recent few days" includes today.
                
                valid_assets_after_crash_filter = []
                
                for asset in today_scores.index:
                    is_crashed = False
                    if asset in daily_returns:
                        # Get last N days returns ending today
                        # Need to find location of date
                        try:
                            # Use slicing on time index directly if possible, or get loc
                            # getting loc is safer
                            idx = daily_returns[asset].index.get_loc(date)
                            start_idx = max(0, idx - crash_params['window'] + 1)
                            # Slice: start_idx to idx (inclusive)
                            # iloc is exclusive on end, so idx+1
                            recent_rets = daily_returns[asset].iloc[start_idx : idx+1]
                            
                            # Check if any return < -threshold
                            # threshold is positive e.g. 0.03 (3%)
                            # so we check if min < -0.03
                            if recent_rets.min() < -crash_params['threshold']:
                                is_crashed = True
                        except:
                            pass # If data missing, assume safe? or skip? Assume safe.
                    
                    if not is_crashed:
                        valid_assets_after_crash_filter.append(asset)
                
                # Filter scores
                filtered_scores = today_scores[today_scores.index.isin(valid_assets_after_crash_filter)]
                
                # 2. Filter Valid Candidates (Score > 0 & <= 300)
                valid_candidates = filtered_scores[
                    (filtered_scores <= CUTOFF_SCORE) & (filtered_scores > 0)
                ]
                
                if valid_candidates.empty:
                    next_target = '现金'
                else:
                    # Normalize
                    vals = filtered_scores.values # Normalize based on available filtered pool? Or original?
                    # Usually normalize based on "Candidate Pool". If we exclude crashed, they are not candidates.
                    # So normalize filtered_scores.
                    if len(vals) > 0:
                        mn, mx = np.min(vals), np.max(vals)
                        if mx == mn:
                            norm_scores = pd.Series(50, index=filtered_scores.index)
                        else:
                            norm_scores = (filtered_scores - mn) / (mx - mn) * 100
                    else:
                         norm_scores = pd.Series()

                    # Best Valid
                    best_valid_asset = valid_candidates.idxmax()
                    best_valid_norm = norm_scores[best_valid_asset]
                    
                    # Switching Logic (Buffer 8.0)
                    if current_asset not in valid_candidates.index:
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
    
    return {
        'return': total_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'trades': trade_count // 2
    }

# --- 3. Optimization Loop ---

def optimize_crash_filter():
    history = load_history_data()
    scores_df = precalculate_all_scores(history)
    
    print("\n--- Baseline (No Filter) ---")
    base_res = run_backtest_with_crash_filter(history, scores_df, {'window': 1, 'threshold': 9.99}) # Threshold > 100% effectively disables it
    print(f"Baseline: Sharpe={base_res['sharpe']:.2f}, Ret={base_res['return']:.2%}, MaxDD={base_res['max_dd']:.2%}")
    
    print("\n--- Optimizing Crash Filter ---")
    
    # Grid Search
    # Window: Check last 1, 2, 3, 5 days
    # Threshold: Drop > 3%, 4%, 5%, 6%, 7%, 8%
    
    windows = [1, 2, 3, 5]
    thresholds = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    
    results = []
    
    for w in windows:
        for t in thresholds:
            params = {'window': w, 'threshold': t}
            res = run_backtest_with_crash_filter(history, scores_df, params)
            res['window'] = w
            res['threshold'] = t
            results.append(res)
            print(f"Window={w} days, Drop > {t:.1%}: Sharpe={res['sharpe']:.2f}, Ret={res['return']:.2%}, MaxDD={res['max_dd']:.2%}")
            
    # Sort
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('sharpe', ascending=False)
    
    print("\n[Top 5 Configurations]")
    print(df_results.head(5).to_string(index=False))

if __name__ == "__main__":
    optimize_crash_filter()
