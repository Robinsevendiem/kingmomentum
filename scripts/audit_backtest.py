import pandas as pd
import numpy as np
import os

# File paths
RECORD_DIR = 'record'

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
    if len(prices) < 20: return -999
    y = np.log(prices)
    x = np.arange(len(y))
    weights = 1 + (np.linspace(0, 1, len(y)) ** 2)
    coeffs = np.polyfit(x, y, 1, w=weights)
    slope = coeffs[0]
    y_pred = np.polyval(coeffs, x)
    sse = np.sum(weights * (y - y_pred)**2)
    y_mean = np.average(y, weights=weights)
    sst = np.sum(weights * (y - y_mean)**2)
    r2 = 1 - sse/sst if sst != 0 else 0
    return (np.exp(slope * 252) - 1) * r2 * 100

def audit_backtest_logic():
    print("--- 回测逻辑审计 (Backtest Logic Audit) ---")
    
    history_data = load_history_data()
    assets = list(history_data.keys())
    
    # Timeline
    all_dates = sorted(list(set().union(*[df.index for df in history_data.values()])))
    start_date = pd.Timestamp('2017-08-01')
    timeline = [d for d in all_dates if d >= start_date]
    
    # State
    cash = 100000.0
    holdings = {} # {asset: shares}
    current_asset = '现金'
    target_asset = '现金' # Decision from previous day
    
    FEE_RATE = 0.0005
    BUFFER_SCORE_DIFF = 5.0
    CUTOFF_SCORE = 300.0
    
    trade_log = []
    
    # We will simulate step-by-step and print detailed logs for a few trades
    
    print(f"审计开始资金: {cash}")
    
    for i, date in enumerate(timeline):
        # 1. Trade Execution at OPEN (based on target_asset determined at i-1 Close)
        
        # Check if assets are trading today
        can_sell = True
        can_buy = True
        
        if current_asset != '现金':
            if date not in history_data[current_asset].index: can_sell = False
        if target_asset != '现金':
            if date not in history_data[target_asset].index: can_buy = False
            
        # SELL Logic
        if current_asset != target_asset and current_asset != '现金':
            if can_sell:
                # Sell at Open Price
                sell_price = history_data[current_asset].loc[date, 'open']
                shares = holdings[current_asset]
                proceeds = shares * sell_price * (1 - FEE_RATE)
                
                # Audit Log
                prev_close = history_data[current_asset].loc[:date].iloc[-2]['close'] # Close of T-1 (approx)
                print(f"[{date.date()}] 卖出 {current_asset}")
                print(f"  卖出价格 (Open): {sell_price:.4f}")
                print(f"  卖出金额: {proceeds:.2f} (含手续费)")
                
                # Check PnL of this trade
                # We need to track cost basis properly if we want trade PnL.
                # But here we track Portfolio Value.
                
                cash += proceeds
                del holdings[current_asset]
                current_asset = '现金'
            else:
                print(f"[{date.date()}] 警告: 需卖出 {current_asset} 但今日停牌，推迟卖出")
                
        # BUY Logic
        if current_asset == '现金' and target_asset != '现金':
            if can_buy:
                # Buy at Open Price
                buy_price = history_data[target_asset].loc[date, 'open']
                invest_amount = cash
                shares = invest_amount / (buy_price * (1 + FEE_RATE))
                cost = shares * buy_price * (1 + FEE_RATE)
                
                cash -= cost
                holdings[target_asset] = shares
                current_asset = target_asset
                
                print(f"[{date.date()}] 买入 {target_asset}")
                print(f"  买入价格 (Open): {buy_price:.4f}")
                print(f"  持有份额: {shares:.2f}")
                print(f"  剩余现金: {cash:.2f}")
            else:
                print(f"[{date.date()}] 警告: 需买入 {target_asset} 但今日停牌，推迟买入")
                
        # 2. Portfolio Value Update (At Close)
        # This is for tracking curve, NOT for trading decision.
        # Value = Cash + Shares * Close
        
        # 3. Signal Generation (At Close) for NEXT day
        scores = {}
        for asset in assets:
            df = history_data[asset]
            if date in df.index:
                loc = df.index.get_loc(date)
                if loc >= 19:
                    series = df.iloc[loc-19 : loc+1]['close']
                    scores[asset] = calculate_score(series.values)
        
        # Decision Logic
        next_target = '现金'
        if scores:
            # Normalize
            vals = np.array(list(scores.values()))
            mn, mx = np.min(vals), np.max(vals)
            norm_scores = {}
            for k, v in scores.items():
                if mx != mn: norm_scores[k] = (v - mn) / (mx - mn) * 100
                else: norm_scores[k] = 50
            
            # Filter Overheated
            valid_candidates = [a for a in scores if scores[a] <= CUTOFF_SCORE]
            
            if valid_candidates:
                # Rank valid by Raw Score
                sorted_valid = sorted([(a, scores[a]) for a in valid_candidates], key=lambda x: x[1], reverse=True)
                best_valid = sorted_valid[0][0]
                best_raw = sorted_valid[0][1]
                
                if best_raw > 0:
                    # Buffer Check
                    if current_asset in valid_candidates:
                        curr_norm = norm_scores[current_asset]
                        best_norm = norm_scores[best_valid]
                        if best_norm - curr_norm < BUFFER_SCORE_DIFF:
                            next_target = current_asset
                        else:
                            next_target = best_valid
                    else:
                        next_target = best_valid
        
        target_asset = next_target
        
        # Stop after a few trades for inspection
        if len(trade_log) > 5: # Just print logic, not store log
            pass
            
    print("审计完成。")

if __name__ == "__main__":
    # We run a short version to print logs
    # Limit timeline inside the function or just run full and see output
    # Just run full is fine, we want to see the specific Open/Close logic.
    # The user question is about: Sell at Open, Buy at Open.
    # "If prev day held asset, sell at Open, buy new at Open."
    # "Return calculation: Sell Open Price vs Buy Price (previous Buy Open)."
    
    # My code does exactly this:
    # 1. Sell current at Open -> Update Cash.
    # 2. Buy target at Open -> Update Holdings.
    # 3. Value tracking at Close -> Uses Close price.
    
    # The concern might be: Does the "Day T" return include the "Sell at Open" action?
    # Day T Value = (Cash after Sell/Buy) + (Shares * Close).
    # If we swapped A to B at Open:
    # Day T Value = Shares_B * Close_B_T.
    # Day T-1 Value = Shares_A * Close_A_T-1.
    # Return T = (Shares_B * Close_B_T) / (Shares_A * Close_A_T-1) - 1.
    
    # Decomposition:
    # 1. Overnight Return (A): Close_A_T-1 -> Open_A_T. (Realized on Sell)
    # 2. Intraday Return (B): Open_B_T -> Close_B_T. (Unrealized on Hold)
    # The switch happens at Open.
    # Cost of A (Sell) = Open_A_T.
    # Cost of B (Buy) = Open_B_T.
    # The gap between Close_A_T-1 and Open_A_T is captured.
    # The gap between Open_B_T and Close_B_T is captured.
    # Is anything missing?
    # No. The cash balance continuity ensures PnL is correct.
    # Cash_after_sell = Shares_A * Open_A_T.
    # Shares_B = Cash_after_sell / Open_B_T.
    # Value_T = Shares_B * Close_B_T = (Shares_A * Open_A_T / Open_B_T) * Close_B_T.
    # Value_T / Value_T-1 = (Open_A_T / Close_A_T-1) * (Close_B_T / Open_B_T).
    # This correctly chains the returns: Overnight A * Intraday B.
    
    audit_backtest_logic()
