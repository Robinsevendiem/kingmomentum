import pandas as pd
import numpy as np
import os
import sys
from scipy import stats # For R2

# Ensure we can run this script from project root
# If running as `python scripts/fine_tune_strategy.py`, CWD is project root.
# We will use relative paths assuming CWD is project root.

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
    
    # Vectorized loop (still python loop but optimized inner ops)
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
        if 'close' in df.columns:
            scores = calculate_rolling_scores(df['close'], window=window)
            scores.name = asset
            all_scores = pd.merge(all_scores, scores, left_index=True, right_index=True, how='outer')
    return all_scores

def run_backtest(history_data, raw_scores_df, params):
    # Filter Timeline
    timeline = [d for d in raw_scores_df.index if params['start_date'] <= d <= params['end_date']]
    timeline = sorted(timeline)
    
    if not timeline: return pd.DataFrame(), pd.DataFrame(), [], {}

    # State
    cash = params['initial_capital']
    holdings = {} 
    current_asset = '现金'
    target_asset = '现金'
    
    price_open = {asset: df['open'] for asset, df in history_data.items()}
    price_close = {asset: df['close'] for asset, df in history_data.items()}
    
    daily_returns = {}
    if params['crash_filter_enabled']:
        for asset, df in history_data.items():
            daily_returns[asset] = df['close'].pct_change()
    
    value_history = []
    trade_log = []
    last_signal_info = {}
    cost_basis = {}
    
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
            
            # Calculate trade return
            trade_return_pct = 0.0
            if current_asset in cost_basis:
                buy_price = cost_basis[current_asset]
                if buy_price > 0:
                    trade_return_pct = (price - buy_price) / buy_price
                del cost_basis[current_asset]
            
            trade_log.append({
                'date': date,
                'action': '卖出',
                'asset': current_asset,
                'price': price,
                'shares': shares,
                'amount': proceeds,
                'fee': shares * price * params['fee_rate'],
                'trade_return': trade_return_pct
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
            cost_basis[target_asset] = price
            
            trade_log.append({
                'date': date,
                'action': '买入',
                'asset': target_asset,
                'price': price,
                'shares': shares,
                'amount': cost,
                'fee': shares * price * params['fee_rate']
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
                    valid_candidates = pool_scores[
                        (pool_scores <= params['cutoff_score']) & (pool_scores > 0)
                    ]
                    
                    if valid_candidates.empty:
                        next_target = '现金'
                    else:
                        vals = pool_scores.values
                        mn, mx = np.min(vals), np.max(vals)
                        if mx == mn: norm_scores = pd.Series(50, index=pool_scores.index)
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
        
        # Capture last signal
        if date == timeline[-1]:
            last_signal_info = {
                'date': date,
                'next_holding': target_asset
            }
        
    return pd.DataFrame(value_history).set_index('date'), pd.DataFrame(trade_log), timeline, last_signal_info

# --- 2. Optimization Logic ---

def load_ground_truth():
    """Load the original holding records"""
    path = 'data/record/2017-08-10 至 2026-02-26持仓记录.csv'
    if not os.path.exists(path):
        print(f"Record file not found at {path}")
        return None
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading record file: {e}")
        return None
        
    # Columns: ,日期,ETF名称,净值,涨跌幅,累计收益
    
    # Clean up date
    if '日期' not in df.columns:
        print("Column '日期' not found in record file.")
        print(df.columns)
        return None
        
    df['date'] = pd.to_datetime(df['日期']).dt.normalize()
    df = df.set_index('date')
    
    name_map = {
        '纳指ETF': '纳指100',
        '纳指100': '纳指100',
        '日经ETF': '日经ETF',
        '港股科技': '港股科技',
        '港股科技ETF': '港股科技',
        '180ETF': '上证180',
        '上证180': '上证180',
        '科创板': '科创板',
        '科创板ETF': '科创板',
        '创业板': '创业板',
        '创业板ETF': '创业板',
        '南方原油': '南方原油',
        '原油': '南方原油',
        '黄金ETF': '黄金ETF',
        '黄金': '黄金ETF',
        '30年国债': '30年国债',
        '国债': '30年国债',
        '现金': '现金'
    }
    
    if 'ETF名称' not in df.columns:
        print("Column 'ETF名称' not found.")
        return None
        
    df['holding_clean'] = df['ETF名称'].map(name_map).fillna(df['ETF名称'])
    return df[['holding_clean']]

def evaluate(history_data, ground_truth, params):
    # Pre-calc scores
    scores_df = precalculate_all_scores(history_data, window=params['window'])
    
    # Run Backtest
    backtest_res, trade_log_df, _, _ = run_backtest(history_data, scores_df, params) # Unpack full result
    
    if backtest_res.empty:
        return 0, 0, 0, 0, 0, 0
    
    # Join with ground truth
    combined = backtest_res.join(ground_truth, how='inner', lsuffix='_bt', rsuffix='_gt')
    
    # Calculate match rate
    if 'holding' not in combined.columns or 'holding_clean' not in combined.columns:
        match_rate = 0
    else:
        matches = combined[combined['holding'] == combined['holding_clean']]
        match_rate = len(matches) / len(combined) if len(combined) > 0 else 0
    
    # --- Financial Metrics ---
    if len(backtest_res) > 0:
        total_ret = backtest_res['value'].iloc[-1] / backtest_res['value'].iloc[0] - 1
        
        # Daily Returns
        daily_ret = backtest_res['value'].pct_change().dropna()
        
        # Volatility
        vol = daily_ret.std() * np.sqrt(252)
        
        # Annualized Return
        days = (backtest_res.index[-1] - backtest_res.index[0]).days
        ann_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
        
        # Sharpe
        risk_free = 0.02
        sharpe = (ann_ret - risk_free) / vol if vol != 0 else 0
        
        # Sortino
        downside_ret = daily_ret[daily_ret < 0]
        downside_std = downside_ret.std() * np.sqrt(252)
        sortino = (ann_ret - risk_free) / downside_std if downside_std != 0 else 0
        
        # Max Drawdown
        cum_max = backtest_res['value'].cummax()
        drawdown = (backtest_res['value'] - cum_max) / cum_max
        max_dd = drawdown.min()
        
        # Calmar Ratio (Annualized Return / Abs(MaxDD))
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        # Trade Metrics
        trade_count = len(trade_log_df)
        
        # Win Rate & PL Ratio
        if trade_count > 0:
            # Filter for sell trades which have 'trade_return'
            sell_trades = trade_log_df[trade_log_df['action'] == '卖出']
            if not sell_trades.empty and 'trade_return' in sell_trades.columns:
                wins = sell_trades[sell_trades['trade_return'] > 0]
                losses = sell_trades[sell_trades['trade_return'] <= 0]
                
                win_rate = len(wins) / len(sell_trades)
                
                avg_win = wins['trade_return'].mean() if not wins.empty else 0
                avg_loss = abs(losses['trade_return'].mean()) if not losses.empty else 0
                pl_ratio = avg_win / avg_loss if avg_loss != 0 else 0
            else:
                win_rate = 0
                pl_ratio = 0
        else:
            win_rate = 0
            pl_ratio = 0
            
        # Equity R2
        try:
            y = np.log(backtest_res['value'].values)
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            equity_r2 = r_value**2
        except:
            equity_r2 = 0
            
    else:
        total_ret = 0
        sharpe = 0
        sortino = 0
        max_dd = 0
        calmar = 0
        trade_count = 0
        win_rate = 0
        pl_ratio = 0
        equity_r2 = 0
    
    return match_rate, total_ret, sharpe, sortino, max_dd, calmar, trade_count, win_rate, pl_ratio, equity_r2

def optimize():
    print("Loading Data...")
    history_data = load_history_data()
    if not history_data:
        print("No history data loaded.")
        return
        
    ground_truth = load_ground_truth()
    
    if ground_truth is None:
        print("No ground truth data loaded.")
        return

    # Filter ground truth to available history range
    start_date = pd.Timestamp('2017-08-10')
    end_date = pd.Timestamp('2026-02-26')
    
    # Parameter Grid
    # We can tune: Window, Cutoff, Buffer, CrashFilter
    
    # Narrow search for speed
    windows = [20, 25] 
    cutoffs = [300, 500, 700] 
    buffers = [5, 8, 10]
    crash_filters = [False, True] 
    
    results = []
    
    total_combos = len(windows) * len(cutoffs) * len(buffers) * len(crash_filters)
    print(f"Starting Grid Search on {total_combos} combinations...")
    print(f"Eval Period: {start_date.date()} - {end_date.date()}")
    
    count = 0
    for w in windows:
        for c in cutoffs:
            for b in buffers:
                for cf in crash_filters:
                    count += 1
                    params = {
                        'start_date': start_date,
                        'end_date': end_date,
                        'window': w,
                        'cutoff_score': c,
                        'buffer_score': b,
                        'crash_filter_enabled': cf,
                        'crash_window': 3,
                        'crash_threshold': 0.03,
                        'fee_rate': 0.0005,
                        'initial_capital': 100000
                    }
                    
                    try:
                        match_rate, ret, sharpe, sortino, max_dd, calmar, trades, win_rate, pl_ratio, r2 = evaluate(history_data, ground_truth, params)
                        print(f"[{count}/{total_combos}] W:{w} C:{c} B:{b} CF:{cf} -> Ret:{ret:.1%} Shp:{sharpe:.2f} Sort:{sortino:.2f} Win:{win_rate:.1%}")
                        
                        results.append({
                            'window': w,
                            'cutoff': c,
                            'buffer': b,
                            'crash_filter': cf,
                            'match_rate': match_rate,
                            'return': ret,
                            'sharpe': sharpe,
                            'sortino': sortino,
                            'max_dd': max_dd,
                            'calmar': calmar,
                            'trade_count': trades,
                            'win_rate': win_rate,
                            'pl_ratio': pl_ratio,
                            'equity_r2': r2
                        })
                    except Exception as e:
                        print(f"Error in combo {params}: {e}")
    
    # Save results
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values('return', ascending=False)
        
        print("\nTop 5 Parameter Sets by Return:")
        print(res_df.head(5))
        
        res_df.to_csv('optimization_results.csv', index=False)
        print("\nResults saved to optimization_results.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    optimize()
