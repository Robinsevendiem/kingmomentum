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
            df = pd.read_csv(filename)
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('trade_date').set_index('trade_date')
            history_data[name] = df
    return history_data

def calculate_rolling_scores(series, window=20):
    """
    Vectorized or Rolling calculation of Quadratic WLS Score.
    Since rolling apply with complex custom function is slow/hard to vectorize perfectly,
    we use a loop over the series for clarity and correctness, 
    but we do it once per asset.
    """
    scores = pd.Series(index=series.index, dtype=float)
    scores[:] = np.nan
    
    # Pre-compute weights
    x = np.arange(window)
    x_norm = np.linspace(0, 1, window)
    weights = 1 + x_norm ** 2
    
    # We need log prices
    log_prices = np.log(series)
    
    # Optimization: Use strided view or just loop? 
    # Loop is fine for ~2000 days.
    
    values = log_prices.values
    dates = log_prices.index
    
    for i in range(window, len(values) + 1):
        window_data = values[i-window : i]
        
        # Check for NaNs
        if np.isnan(window_data).any():
            continue
            
        # WLS
        # polyfit is relatively slow in loop. 
        # Manual matrix math is faster: (X.T W X)^-1 X.T W Y
        # X is [1, t]. 
        # But let's stick to numpy for reliability unless too slow.
        try:
            coeffs = np.polyfit(x, window_data, 1, w=weights)
            slope = coeffs[0]
            
            # R2
            y_pred = np.polyval(coeffs, x)
            sse = np.sum(weights * (window_data - y_pred)**2)
            y_mean = np.average(window_data, weights=weights)
            sst = np.sum(weights * (window_data - y_mean)**2)
            
            if sst == 0: r2 = 0
            else: r2 = 1 - sse / sst
            
            score = (np.exp(slope * 252) - 1) * r2 * 100
            scores.iloc[i-1] = score
        except:
            pass
            
    return scores

def precalculate_all_scores(history_data):
    print("Pre-calculating scores for all assets...")
    all_scores = pd.DataFrame()
    
    for asset, df in history_data.items():
        # Use Close price for signal
        scores = calculate_rolling_scores(df['close'])
        scores.name = asset
        all_scores = pd.merge(all_scores, scores, left_index=True, right_index=True, how='outer')
        
    print("Score calculation complete.")
    return all_scores

# --- 2. Backtest Engine (Fast) ---

def run_backtest_fast(history_data, raw_scores_df, buffer_config):
    """
    buffer_config: dict
        type: 'score_diff', 'pct_diff', 'consecutive'
        value: float/int
    """
    # Parameters
    CUTOFF_SCORE = 300
    FEE_RATE = 0.0005
    
    # Prepare Timeline
    # Start from 2017-08-01
    start_date = pd.Timestamp('2017-08-01')
    timeline = [d for d in raw_scores_df.index if d >= start_date]
    timeline = sorted(timeline)
    
    # State
    cash = 100000.0
    holdings = {} # {asset: shares}
    current_asset = '现金'
    target_asset = '现金' # Signal from yesterday
    
    # Streak State
    best_candidate_streak = 0
    prev_best_candidate = None
    
    value_history = []
    trade_count = 0
    
    # Cache Open/Close prices for fast lookup
    # Creating a panel or dict of series is faster than df lookup in loop
    price_open = {asset: df['open'] for asset, df in history_data.items()}
    price_close = {asset: df['close'] for asset, df in history_data.items()}
    
    for date in timeline:
        # --- A. Execution (At Open) ---
        # Execute trade based on 'target_asset' determined yesterday
        
        can_sell = True
        can_buy = True
        
        # Check tradability
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
            trade_count += 1 # Count Sell as half-trade? Or Buy+Sell = 1 trade? User asked for pairs usually.
            
        # Buy
        if current_asset == '现金' and target_asset != '现金' and can_buy:
            price = price_open[target_asset].loc[date]
            shares = cash / (price * (1 + FEE_RATE))
            cost = shares * price * (1 + FEE_RATE)
            cash -= cost
            holdings[target_asset] = shares
            current_asset = target_asset
            trade_count += 1
            
        # --- B. Valuation (At Close) ---
        day_value = cash
        for asset, shares in holdings.items():
            if date in price_close[asset].index:
                price = price_close[asset].loc[date]
            else:
                # Fallback to last close (slow, but robust)
                # For speed, we assume forward fill or just 0 if missing (shouldn't happen often for major ETFs)
                # Let's use asof
                try:
                    price = price_close[asset].asof(date)
                except:
                    price = 0
            day_value += shares * price
            
        value_history.append({'date': date, 'value': day_value})
        
        # --- C. Signal Generation (At Close) ---
        # 1. Get scores for today
        if date not in raw_scores_df.index:
            # No scores today?
            next_target = '现金'
        else:
            today_scores = raw_scores_df.loc[date]
            # Filter NaNs
            today_scores = today_scores.dropna()
            
            if today_scores.empty:
                next_target = '现金'
                # Streak reset
                best_candidate_streak = 0
                prev_best_candidate = None
            else:
                # 2. Identify Valid Candidates (Score <= 300 & Score > 0)
                # Note: Strategy says "positive score" for ranking? 
                # Cutoff says "Score <= 300".
                # Negative scores -> Cash.
                
                valid_candidates = today_scores[
                    (today_scores <= CUTOFF_SCORE) & (today_scores > 0)
                ]
                
                if valid_candidates.empty:
                    next_target = '现金'
                    best_candidate_streak = 0
                    prev_best_candidate = None
                else:
                    # 3. Rank Valid Candidates
                    # Need Normalized Scores for comparison?
                    # Normalization usually done on ALL scores, then filtered?
                    # Or filtered then normalized?
                    # Strategy doc: "得分归一化至 0 到 100". Usually implies normalization step.
                    # Normalization context: All 9 assets (even invalid ones? or only valid?)
                    # Usually Normalize ALL available scores for the day.
                    
                    # Normalize ALL available today_scores
                    vals = today_scores.values
                    mn, mx = np.min(vals), np.max(vals)
                    if mx == mn:
                        norm_scores = pd.Series(50, index=today_scores.index)
                    else:
                        norm_scores = (today_scores - mn) / (mx - mn) * 100
                    
                    # Now pick Best Valid
                    # Best Valid is the one with highest raw score among valid ones.
                    # (Monotonic with norm score)
                    
                    best_valid_asset = valid_candidates.idxmax()
                    best_valid_raw = valid_candidates.max()
                    best_valid_norm = norm_scores[best_valid_asset]
                    
                    # --- D. Buffer Logic ---
                    
                    # Streak Logic Update
                    if best_valid_asset == prev_best_candidate:
                        best_candidate_streak += 1
                    else:
                        best_candidate_streak = 1
                    prev_best_candidate = best_valid_asset
                    
                    # Switching Decision
                    # If current holding is invalid (not in valid_candidates or is '现金'):
                    # MUST switch to best valid.
                    if current_asset not in valid_candidates.index:
                        next_target = best_valid_asset
                    else:
                        # Current is valid. Apply Buffer.
                        
                        should_switch = False
                        
                        curr_norm = norm_scores[current_asset]
                        
                        if buffer_config['type'] == 'score_diff':
                            # Rule: New - Current > Threshold
                            diff = best_valid_norm - curr_norm
                            if diff > buffer_config['value']:
                                should_switch = True
                                
                        elif buffer_config['type'] == 'pct_diff':
                            # Rule: New > Current * (1 + Threshold)
                            # Use Raw or Norm? User said "percent".
                            # Let's use Norm Score for consistency.
                            if curr_norm > 0:
                                if best_valid_norm > curr_norm * (1 + buffer_config['value']):
                                    should_switch = True
                            else:
                                should_switch = True # Current is 0 or negative (shouldn't be here if valid>0)
                                
                        elif buffer_config['type'] == 'consecutive':
                            # Rule: Best Candidate has been best for N days
                            # Note: We only switch if we DON'T hold the best.
                            if best_valid_asset != current_asset:
                                if best_candidate_streak >= buffer_config['value']:
                                    should_switch = True
                            else:
                                should_switch = False # We hold the best
                                
                        else:
                            # No buffer
                            should_switch = (best_valid_asset != current_asset)
                            
                        if should_switch:
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
        'trades': trade_count // 2 # Pairs
    }

# --- 3. Optimization Loop ---

def optimize_buffer():
    print("Loading Data...")
    history = load_history_data()
    print("Pre-calculating Scores...")
    scores_df = precalculate_all_scores(history)
    
    results = []
    
    # 1. Score Difference Buffer
    print("\n--- Optimizing Score Difference Buffer ---")
    for diff in [0, 2, 5, 8, 10, 15, 20]:
        cfg = {'type': 'score_diff', 'value': diff}
        res = run_backtest_fast(history, scores_df, cfg)
        res['type'] = 'Score Diff'
        res['param'] = diff
        results.append(res)
        print(f"Diff={diff}: Sharpe={res['sharpe']:.2f}, Ret={res['return']:.2%}, Trades={res['trades']}")
        
    # 2. Percentage Difference Buffer
    print("\n--- Optimizing Percentage Buffer ---")
    for pct in [0.0, 0.05, 0.10, 0.15, 0.20]:
        cfg = {'type': 'pct_diff', 'value': pct}
        res = run_backtest_fast(history, scores_df, cfg)
        res['type'] = 'Pct Diff'
        res['param'] = f"{pct:.0%}"
        results.append(res)
        print(f"Pct={pct:.0%}: Sharpe={res['sharpe']:.2f}, Ret={res['return']:.2%}, Trades={res['trades']}")
        
    # 3. Consecutive Days Buffer
    print("\n--- Optimizing Consecutive Days Buffer ---")
    for days in [1, 2, 3, 4, 5]:
        cfg = {'type': 'consecutive', 'value': days}
        res = run_backtest_fast(history, scores_df, cfg)
        res['type'] = 'Consecutive'
        res['param'] = days
        results.append(res)
        print(f"Days={days}: Sharpe={res['sharpe']:.2f}, Ret={res['return']:.2%}, Trades={res['trades']}")
        
    # Summary
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('sharpe', ascending=False)
    
    print("\n[Optimization Results - Top 5 by Sharpe]")
    print(df_results.head(5).to_string(index=False))
    
    # Save to CSV for analysis
    df_results.to_csv('buffer_optimization_results.csv', index=False)

if __name__ == "__main__":
    optimize_buffer()
