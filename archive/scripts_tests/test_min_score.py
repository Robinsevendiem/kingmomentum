import pandas as pd
import numpy as np
import os

def load_history_data():
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
                    try: df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                    except: df['trade_date'] = pd.to_datetime(df['trade_date'])
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

def precalculate_all_scores(history_data, window=20):
    all_scores = pd.DataFrame()
    for asset, df in history_data.items():
        if 'adj_close' in df.columns: series = df['adj_close']
        else: series = df['close']
        scores = calculate_rolling_scores(series, window=window)
        scores.name = asset
        all_scores = pd.merge(all_scores, scores, left_index=True, right_index=True, how='outer')
    return all_scores

def calculate_alpha51(df, window=10, threshold=0.01):
    if 'adj_close' in df.columns: c = df['adj_close']
    else: c = df['close']
    w = window
    speed_prev = (c.shift(w) - c.shift(2*w)) / w
    speed_curr = (c - c.shift(w)) / w
    diff = speed_curr - speed_prev
    cond = diff < -threshold
    return cond

def precalculate_alpha51_all(history_data, window=10, threshold=0.01):
    all_a51 = pd.DataFrame()
    for asset, df in history_data.items():
        a51 = calculate_alpha51(df, window, threshold)
        a51.name = asset
        all_a51 = pd.merge(all_a51, a51, left_index=True, right_index=True, how='outer')
    return all_a51

def run_backtest(history_data, raw_scores_df, alpha51_df, params):
    timeline = [d for d in raw_scores_df.index if params['start_date'] <= d <= params['end_date']]
    timeline = sorted(timeline)
    if not timeline: return pd.DataFrame()

    cash = params['initial_capital']
    holdings = {} 
    current_asset = '现金'
    target_asset = '现金'
    
    price_open = {}
    price_close = {}
    for asset, df in history_data.items():
        if 'adj_open' in df.columns and 'adj_close' in df.columns:
            price_open[asset] = df['adj_open']
            price_close[asset] = df['adj_close']
        else:
            price_open[asset] = df['open']
            price_close[asset] = df['close']
            
    daily_returns = {}
    if params['crash_filter_enabled']:
        for asset, df in history_data.items():
            if asset in price_close:
                daily_returns[asset] = price_close[asset].pct_change()
    
    value_history = []
    
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
            current_asset = '现金'
            
        # Buy
        if current_asset == '现金' and target_asset != '现金' and can_buy:
            price = price_open[target_asset].loc[date]
            invest_amount = cash
            shares = invest_amount / (price * (1 + params['fee_rate']))
            cost = shares * price * (1 + params['fee_rate'])
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
                
                # 2. Alpha 51 Filter
                if params['use_alpha51']:
                    valid_after_a51 = []
                    if date in alpha51_df.index:
                        a51_today = alpha51_df.loc[date]
                        for asset in valid_assets:
                            is_risk = False
                            if asset in a51_today and a51_today[asset] == True:
                                is_risk = True
                            if not is_risk: valid_after_a51.append(asset)
                    else:
                        valid_after_a51 = valid_assets
                    valid_assets = valid_after_a51
                
                pool_scores = today_scores[today_scores.index.isin(valid_assets)]
                
                if pool_scores.empty:
                    next_target = '现金'
                else:
                    # 3. Minimum Score Filter (The new feature being tested)
                    min_score = params.get('min_score', 0)
                    
                    # Also apply the upper cutoff
                    cutoff = params.get('cutoff_score', 600)
                    
                    valid_candidates = pool_scores[
                        (pool_scores <= cutoff) & (pool_scores > min_score)
                    ]
                    
                    if valid_candidates.empty:
                        next_target = '现金'
                    else:
                        # Normalization
                        if params.get('exclude_overheated_from_norm', True):
                            norm_basis = valid_candidates
                        else:
                            norm_basis = pool_scores
                            
                        vals = norm_basis.values
                        if len(vals) == 0:
                            mn, mx = 0, 0
                        else:
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
        
    return pd.DataFrame(value_history).set_index('date')

def test_min_score():
    print("Loading Data...")
    history_data = load_history_data()
    
    print("Calculating Momentum Scores (Window=20)...")
    scores_df = precalculate_all_scores(history_data, window=20)
    
    print("Calculating Alpha 51 (W=10, Th=0.01)...")
    alpha51_df = precalculate_alpha51_all(history_data, window=10, threshold=0.01)
    
    # Range of min scores to test
    # 0 is the baseline (current strategy)
    min_scores = [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    
    results = []
    start_date = pd.Timestamp('2019-01-01')
    end_date = pd.Timestamp.now()
    
    print("Testing Minimum Score Thresholds...")
    
    for ms in min_scores:
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'window': 20,
            'cutoff_score': 600,
            'buffer_score': 5,
            'exclude_overheated_from_norm': True,
            'use_alpha51': True,
            'crash_filter_enabled': True,
            'crash_window': 3,
            'crash_threshold': 0.03,
            'fee_rate': 0.0005,
            'initial_capital': 100000,
            'min_score': ms  # The parameter being tested
        }
        
        try:
            res_df = run_backtest(history_data, scores_df, alpha51_df, params)
            
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
                
                # Count cash days
                cash_days = (res_df['holding'] == '现金').sum()
                cash_ratio = cash_days / len(res_df)
                
                print(f"Min Score: {ms:5.1f} | AnnRet: {ann_ret:6.2%} | Sharpe: {sharpe:4.2f} | MaxDD: {max_dd:6.2%} | Cash Ratio: {cash_ratio:5.1%}")
                
                results.append({
                    'min_score': ms,
                    'ann_ret': ann_ret,
                    'sharpe': sharpe,
                    'max_dd': max_dd,
                    'calmar': calmar,
                    'cash_ratio': cash_ratio
                })
        except Exception as e:
            print(f"Error for Min Score {ms}: {e}")

    # Output best
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        best_sharpe = res_df.loc[res_df['sharpe'].idxmax()]
        print("\n--- Best Threshold by Sharpe ---")
        print(f"Min Score: {best_sharpe['min_score']}")
        print(f"Sharpe:    {best_sharpe['sharpe']:.2f}")
        print(f"Ann Ret:   {best_sharpe['ann_ret']:.2%}")
        print(f"Max DD:    {best_sharpe['max_dd']:.2%}")

if __name__ == "__main__":
    test_min_score()
