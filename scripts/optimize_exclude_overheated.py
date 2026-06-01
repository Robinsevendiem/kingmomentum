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
        # Prefer adjusted close
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

def run_backtest(history_data, raw_scores_df, params):
    # Filter Timeline
    timeline = [d for d in raw_scores_df.index if params['start_date'] <= d <= params['end_date']]
    timeline = sorted(timeline)
    
    if not timeline: return pd.DataFrame(), pd.DataFrame()

    # State
    cash = params['initial_capital']
    holdings = {} 
    current_asset = '现金'
    target_asset = '现金'
    
    # Cache Prices (Adjusted preferred for backtest return calculation)
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
            # Use close price for crash detection
            if asset in price_close:
                daily_returns[asset] = price_close[asset].pct_change()
    
    value_history = []
    trade_log = []
    
    for date in timeline:
        # A. Execution (At Open)
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
            
            trade_log.append({
                'date': date,
                'action': '卖出',
                'asset': current_asset,
                'price': price,
                'shares': shares,
                'amount': proceeds
            })
            current_asset = '现金'
            
        # Buy
        if current_asset == '现金' and target_asset != '现金' and can_buy:
            price = price_open[target_asset].loc[date]
            invest_amount = cash
            shares = invest_amount / (price * (1 + params['fee_rate']))
            cost = shares * price * (1 + params['fee_rate'])
            cash -= cost
            holdings[target_asset] = shares
            
            trade_log.append({
                'date': date,
                'action': '买入',
                'asset': target_asset,
                'price': price,
                'shares': shares,
                'amount': cost
            })
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
        
        # C. Signal Generation
        if date not in raw_scores_df.index:
            next_target = '现金'
        else:
            today_scores = raw_scores_df.loc[date].dropna()
            if today_scores.empty:
                next_target = '现金'
            else:
                # 1. Crash Filter
                valid_assets_pool = today_scores.index.tolist()
                
                if params['crash_filter_enabled']:
                    valid_after_crash = []
                    for asset in valid_assets_pool:
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
                    valid_assets_pool = valid_after_crash
                
                pool_scores = today_scores[today_scores.index.isin(valid_assets_pool)]
                
                if pool_scores.empty:
                    next_target = '现金'
                else:
                    # 2. Candidate Filter
                    # Support global cutoff for optimization
                    cutoff = params.get('cutoff_score', 300)
                    valid_candidates = pool_scores[
                        (pool_scores <= cutoff) & (pool_scores > 0)
                    ]
                    
                    if valid_candidates.empty:
                        next_target = '现金'
                    else:
                        # 3. Normalization
                        exclude_overheated = params.get('exclude_overheated_from_norm', False)
                        
                        if exclude_overheated:
                            norm_basis = valid_candidates
                        else:
                            norm_basis = pool_scores
                            
                        vals = norm_basis.values
                        if len(vals) == 0: # Should not happen if valid_candidates not empty
                             mn, mx = 0, 0
                        else:
                             mn, mx = np.min(vals), np.max(vals)
                        
                        if mx == mn:
                            norm_scores = pd.Series(50, index=pool_scores.index)
                        else:
                            norm_scores = (pool_scores - mn) / (mx - mn) * 100
                        
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

def optimize():
    print("Loading Data...")
    history_data = load_history_data()
    if not history_data:
        print("No history data loaded.")
        return
        
    # Parameters to optimize
    windows = [15, 20, 25, 30]
    cutoffs = [300, 400, 500, 600, 700, 800, 1000]
    buffers = [5, 8, 10, 15]
    
    # Fixed params
    exclude_overheated = True
    crash_filter = True
    start_date = pd.Timestamp('2019-01-01') # Standardize start
    end_date = pd.Timestamp.now()
    
    results = []
    total_combos = len(windows) * len(cutoffs) * len(buffers)
    print(f"Starting Grid Search on {total_combos} combinations...")
    print(f"Mode: Exclude Overheated = {exclude_overheated}")
    
    count = 0
    
    # Cache scores for each window to avoid re-calc
    score_cache = {}
    for w in windows:
        print(f"Pre-calculating scores for window {w}...")
        score_cache[w] = precalculate_all_scores(history_data, window=w)
        
    for w in windows:
        raw_scores_df = score_cache[w]
        for c in cutoffs:
            for b in buffers:
                count += 1
                params = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'window': w,
                    'cutoff_score': c,
                    'buffer_score': b,
                    'exclude_overheated_from_norm': exclude_overheated,
                    'crash_filter_enabled': crash_filter,
                    'crash_window': 3,
                    'crash_threshold': 0.03,
                    'fee_rate': 0.0005,
                    'initial_capital': 100000
                }
                
                try:
                    res_df, trades_df = run_backtest(history_data, raw_scores_df, params)
                    
                    if not res_df.empty:
                        total_ret = res_df['value'].iloc[-1] / res_df['value'].iloc[0] - 1
                        
                        daily_ret = res_df['value'].pct_change().dropna()
                        vol = daily_ret.std() * np.sqrt(252)
                        
                        days = (res_df.index[-1] - res_df.index[0]).days
                        ann_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
                        
                        risk_free = 0.02
                        sharpe = (ann_ret - risk_free) / vol if vol != 0 else 0
                        
                        cum_max = res_df['value'].cummax()
                        drawdown = (res_df['value'] - cum_max) / cum_max
                        max_dd = drawdown.min()
                        
                        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
                        
                        trade_count = len(trades_df)
                        
                        # Progress log
                        if count % 10 == 0:
                            print(f"[{count}/{total_combos}] W:{w} C:{c} B:{b} -> AnnRet:{ann_ret:.1%} Sharpe:{sharpe:.2f} MaxDD:{max_dd:.1%}")
                        
                        results.append({
                            'window': w,
                            'cutoff': c,
                            'buffer': b,
                            'ann_ret': ann_ret,
                            'sharpe': sharpe,
                            'max_dd': max_dd,
                            'calmar': calmar,
                            'trades': trade_count
                        })
                except Exception as e:
                    print(f"Error: {e}")
                    
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Sort by Sharpe
        top_sharpe = res_df.sort_values('sharpe', ascending=False).head(10)
        print("\nTop 10 by Sharpe Ratio:")
        print(top_sharpe.to_string(index=False))
        
        # Sort by Return
        top_ret = res_df.sort_values('ann_ret', ascending=False).head(10)
        print("\nTop 10 by Annualized Return:")
        print(top_ret.to_string(index=False))
        
        # Sort by Calmar
        top_calmar = res_df.sort_values('calmar', ascending=False).head(10)
        print("\nTop 10 by Calmar Ratio:")
        print(top_calmar.to_string(index=False))
        
        res_df.to_csv('scripts/opt_exclude_overheated_results.csv', index=False)

if __name__ == "__main__":
    optimize()
