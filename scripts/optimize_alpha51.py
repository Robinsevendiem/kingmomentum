import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

def load_history_data():
    """Load history data from data/ directory"""
    mapping = {
        '创业板': 'data/159915.SZ_创业板ETF_history.csv',
        '南方原油': 'data/501018.SH_南方原油(LOF)_history.csv',
        '上证180': 'data/510180.SH_180ETF_history.csv',
        '30年国债': 'data/511090.SH_30年国债ETF_history.csv',
        '港股科技': 'data/513020.SH_港股科技ETF_history.csv',
        '纳指100': 'data/513100.SH_纳指ETF_history.csv',
        '日经ETF': 'data/513520.SH_日经ETF_history.csv',
        '黄金ETF': 'data/518880.SH_黄金ETF_history.csv',
        '科创板': 'data/588120.SH_科创板ETF_history.csv'
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
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"File not found: {filename}")
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

def precalculate_all_scores(history_data, window=20):
    all_scores = pd.DataFrame()
    for asset, df in history_data.items():
        if 'adj_close' in df.columns:
            series = df['adj_close']
        elif 'close' in df.columns:
            series = df['close']
        else:
            continue
            
        scores = calculate_rolling_scores(series, window=window)
        scores.name = asset
        all_scores = pd.merge(all_scores, scores, left_index=True, right_index=True, how='outer')
    return all_scores

def calculate_alpha51(df, window=10, diff_threshold=0.05):
    """
    Alpha 51: Trend Deceleration Identification Factor
    Parametric version.
    Original: (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05) ? 1 : ...
    
    Generalized:
    Part1 = (Close[t-10] - Close[t-20]) / 10  (Previous 10-day avg daily return)
    Part2 = (Close[t] - Close[t-10]) / 10     (Current 10-day avg daily return)
    
    Wait, original formula:
    delay(close, 20) - delay(close, 10) -> Price at t-20 minus Price at t-10. This is NEGATIVE of return t-20 to t-10.
    Let's trace carefully.
    delay(close, 20): P[t-20]
    delay(close, 10): P[t-10]
    
    Term 1: (P[t-20] - P[t-10]) / 10. 
    If P[t-10] > P[t-20] (Uptrend), this term is NEGATIVE.
    
    Term 2: (P[t-10] - P[t]) / 10.
    If P[t] > P[t-10] (Uptrend), this term is NEGATIVE.
    
    Diff = Term 1 - Term 2.
    
    Condition: Diff < -0.05
    
    Let's interpret.
    Suppose strict uptrend: 
    P[t-20]=100, P[t-10]=110, P[t]=120.
    Term 1 = (100 - 110)/10 = -1.
    Term 2 = (110 - 120)/10 = -1.
    Diff = -1 - (-1) = 0.
    
    Suppose Deceleration (Fast start, then slow):
    P[t-20]=100, P[t-10]=120 (Gain 20), P[t]=125 (Gain 5).
    Term 1 = (100 - 120)/10 = -2.
    Term 2 = (120 - 125)/10 = -0.5.
    Diff = -2 - (-0.5) = -1.5. 
    -1.5 < -0.05. TRUE. Signal 1.
    
    So the logic is: Previous rally was stronger than current rally. "Deceleration".
    
    Let's parameterize:
    - Window length (10 in original) -> w
    - Threshold (0.05 in original) -> th
    """
    # Use adjusted close if available
    if 'adj_close' in df.columns: c = df['adj_close']
    else: c = df['close']
    
    # Original uses 20 and 10. We can generalize to 2*w and w.
    w = window
    
    p_t = c
    p_t_w = c.shift(w)
    p_t_2w = c.shift(2*w)
    
    # (P[t-2w] - P[t-w]) / w
    term1 = (p_t_2w - p_t_w) / w
    
    # (P[t-w] - P[t]) / w
    term2 = (p_t_w - p_t) / w
    
    diff = term1 - term2
    
    # Condition: Diff < -threshold
    # Note: threshold is usually positive in config, formula uses -1 * 0.05.
    # So we check < -th.
    
    cond = diff < -diff_threshold
    
    # Return boolean series (True = Risk/Signal)
    return cond

def precalculate_alpha51_all(history_data, window=10, threshold=0.05):
    all_a51 = pd.DataFrame()
    for asset, df in history_data.items():
        # Returns boolean Series
        a51 = calculate_alpha51(df, window, threshold)
        a51.name = asset
        all_a51 = pd.merge(all_a51, a51, left_index=True, right_index=True, how='outer')
    return all_a51

def run_backtest(history_data, raw_scores_df, alpha51_df, params):
    # Filter Timeline
    timeline = [d for d in raw_scores_df.index if params['start_date'] <= d <= params['end_date']]
    timeline = sorted(timeline)
    if not timeline: return pd.DataFrame(), pd.DataFrame()

    # State
    cash = params['initial_capital']
    holdings = {} 
    current_asset = '现金'
    target_asset = '现金'
    
    # Cache Prices
    price_open = {}
    price_close = {}
    for asset, df in history_data.items():
        if 'adj_open' in df.columns and 'adj_close' in df.columns:
            price_open[asset] = df['adj_open']
            price_close[asset] = df['adj_close']
        else:
            price_open[asset] = df['open']
            price_close[asset] = df['close']
            
    # Daily returns for crash filter
    daily_returns = {}
    if params['crash_filter_enabled']:
        for asset, df in history_data.items():
            if asset in price_close:
                daily_returns[asset] = price_close[asset].pct_change()
    
    value_history = []
    trade_log = []
    
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
            shares = holdings[current_asset]
            proceeds = shares * price * (1 - params['fee_rate'])
            cash += proceeds
            del holdings[current_asset]
            trade_log.append({'date': date, 'action': '卖出', 'asset': current_asset, 'price': price, 'shares': shares, 'amount': proceeds})
            current_asset = '现金'
            
        # Buy
        if current_asset == '现金' and target_asset != '现金' and can_buy:
            price = price_open[target_asset].loc[date]
            invest_amount = cash
            shares = invest_amount / (price * (1 + params['fee_rate']))
            cost = shares * price * (1 + params['fee_rate'])
            cash -= cost
            holdings[target_asset] = shares
            trade_log.append({'date': date, 'action': '买入', 'asset': target_asset, 'price': price, 'shares': shares, 'amount': cost})
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
        value_history.append({'date': date, 'value': day_value, 'holding': current_asset})
        
        # C. Signal
        if date not in raw_scores_df.index:
            next_target = '现金'
        else:
            today_scores = raw_scores_df.loc[date].dropna()
            if today_scores.empty:
                next_target = '现金'
            else:
                valid_assets = today_scores.index.tolist()
                
                # 1. Crash Filter
                if params['crash_filter_enabled']:
                    valid_after_crash = []
                    for asset in valid_assets:
                        is_crashed = False
                        if asset in daily_returns:
                            try:
                                if date in daily_returns[asset].index:
                                    idx = daily_returns[asset].index.get_loc(date)
                                    start_idx = max(0, idx - params['crash_window'] + 1)
                                    recent_rets = daily_returns[asset].iloc[start_idx : idx+1]
                                    if recent_rets.min() < -params['crash_threshold']:
                                        is_crashed = True
                            except: pass
                        if not is_crashed: valid_after_crash.append(asset)
                    valid_assets = valid_after_crash
                
                # 2. Alpha 51 Filter (The New Logic)
                # If enabled, remove assets where Alpha51 is True (Risk Signal)
                if params['use_alpha51']:
                    valid_after_a51 = []
                    if date in alpha51_df.index:
                        a51_today = alpha51_df.loc[date]
                        for asset in valid_assets:
                            # If a51 is True (Risk), exclude.
                            # Handle NaN as False (No Risk)
                            is_risk = False
                            if asset in a51_today and a51_today[asset] == True:
                                is_risk = True
                            
                            if not is_risk:
                                valid_after_a51.append(asset)
                    else:
                        valid_after_a51 = valid_assets # No data, keep all
                    valid_assets = valid_after_a51
                
                pool_scores = today_scores[today_scores.index.isin(valid_assets)]
                
                if pool_scores.empty:
                    next_target = '现金'
                else:
                    # 3. Cutoff Logic
                    # If we use Alpha51 exclusively, we might set cutoff to Infinity (disable it)
                    # or keep it as a secondary check.
                    # User asked: "Try using Alpha51 ONLY, without cutoff".
                    # So we will set cutoff effectively infinite in optimization.
                    
                    cutoff = params.get('cutoff_score', 999999)
                    valid_candidates = pool_scores[
                        (pool_scores <= cutoff) & (pool_scores > 0)
                    ]
                    
                    if valid_candidates.empty:
                        next_target = '现金'
                    else:
                        # Normalization (Exclude Overheated logic only applies if we HAVE a cutoff)
                        # If cutoff is infinite, this logic is moot (nothing excluded).
                        # We proceed with standard normalization.
                        
                        vals = valid_candidates.values
                        mn, mx = np.min(vals), np.max(vals)
                        if mx == mn: norm_scores = pd.Series(50, index=valid_candidates.index)
                        else: norm_scores = (pool_scores - mn) / (mx - mn) * 100
                        
                        best_valid_asset = valid_candidates.idxmax()
                        best_valid_norm = norm_scores[best_valid_asset]
                        
                        if current_asset not in valid_candidates.index:
                            next_target = best_valid_asset
                        else:
                            curr_norm = norm_scores[current_asset]
                            if best_valid_norm - curr_norm > params['buffer_score']:
                                next_target = best_valid_asset
                            else:
                                next_target = current_asset
        target_asset = next_target
        
    return pd.DataFrame(value_history).set_index('date'), pd.DataFrame(trade_log)

def optimize_alpha51():
    print("Loading Data...")
    history_data = load_history_data()
    
    # 1. Base Scores (Momentum)
    # We fix Window=20 as per previous champion result
    print("Calculating Momentum Scores (Window=20)...")
    scores_df = precalculate_all_scores(history_data, window=20)
    
    # 2. Alpha 51 Grid Search
    # Parameters: Window (w), Threshold (th)
    # Original: w=10, th=0.05
    
    a51_windows = [5, 10, 15]
    a51_thresholds = [0.01, 0.03, 0.05, 0.07, 0.10]
    
    results = []
    
    total_combos = len(a51_windows) * len(a51_thresholds)
    print(f"Starting Alpha51 Optimization on {total_combos} combinations...")
    
    start_date = pd.Timestamp('2019-01-01')
    end_date = pd.Timestamp.now()
    
    count = 0
    for w in a51_windows:
        for th in a51_thresholds:
            count += 1
            
            # Calc Alpha51 with current params
            a51_df = precalculate_alpha51_all(history_data, window=w, threshold=th)
            
            params = {
                'start_date': start_date,
                'end_date': end_date,
                'use_alpha51': True,
                'cutoff_score': 600, # ENABLE CHAMPION CUTOFF
                'buffer_score': 5, # Use Champion Buffer
                'exclude_overheated_from_norm': True, # Use Champion Norm Logic
                'crash_filter_enabled': True, # Keep basic crash filter
                'crash_window': 3,
                'crash_threshold': 0.03,
                'fee_rate': 0.0005,
                'initial_capital': 100000
            }
            
            try:
                res_df, trades_df = run_backtest(history_data, scores_df, a51_df, params)
                
                if not res_df.empty:
                    total_ret = res_df['value'].iloc[-1] / res_df['value'].iloc[0] - 1
                    daily_ret = res_df['value'].pct_change().dropna()
                    vol = daily_ret.std() * np.sqrt(252)
                    days = (res_df.index[-1] - res_df.index[0]).days
                    ann_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
                    risk_free = 0.02
                    sharpe = (ann_ret - risk_free) / vol if vol != 0 else 0
                    max_dd = (res_df['value'] / res_df['value'].cummax() - 1).min()
                    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
                    
                    print(f"[{count}/{total_combos}] A51_W:{w} A51_Th:{th} -> AnnRet:{ann_ret:.1%} Sharpe:{sharpe:.2f} MaxDD:{max_dd:.1%}")
                    
                    results.append({
                        'a51_window': w,
                        'a51_threshold': th,
                        'ann_ret': ann_ret,
                        'sharpe': sharpe,
                        'max_dd': max_dd,
                        'calmar': calmar,
                        'trades': len(trades_df)
                    })
            except Exception as e:
                print(f"Error: {e}")
                
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print("\nTop 5 by Sharpe:")
        print(res_df.sort_values('sharpe', ascending=False).head(5).to_string(index=False))
        
        print("\nTop 5 by Return:")
        print(res_df.sort_values('ann_ret', ascending=False).head(5).to_string(index=False))
        
        res_df.to_csv('scripts/opt_alpha51_results.csv', index=False)

if __name__ == "__main__":
    optimize_alpha51()
