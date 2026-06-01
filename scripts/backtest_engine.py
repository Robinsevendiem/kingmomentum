import pandas as pd
import numpy as np
import os
from scipy.stats import linregress

# File paths
RECORD_DIR = 'record'
# We need history files
# Load all history data
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

def calculate_score(prices):
    """
    Calculate Quadratic Weighted Linear Regression Momentum Score.
    Score = Slope * R^2 * 100
    Weights: 1 + (t/19)^2
    """
    if len(prices) < 20: return -999
    
    y = np.log(prices)
    x = np.arange(len(y))
    x_norm = np.linspace(0, 1, len(y))
    weights = 1 + x_norm ** 2
    
    # WLS
    coeffs = np.polyfit(x, y, 1, w=weights)
    slope = coeffs[0]
    
    # R2
    y_pred = np.polyval(coeffs, x)
    sse = np.sum(weights * (y - y_pred)**2)
    y_mean = np.average(y, weights=weights)
    sst = np.sum(weights * (y - y_mean)**2)
    
    if sst == 0: r2 = 0
    else: r2 = 1 - sse / sst
    
    # Annualized slope * R2 * 100
    # Use 252 annualization
    score = (np.exp(slope * 252) - 1) * r2 * 100
    return score

def run_backtest():
    print("--- 策略全样本回测 (Full Backtest) ---")
    
    history_data = load_history_data()
    assets = list(history_data.keys())
    
    # Merge all dates to create a master timeline
    all_dates = sorted(list(set().union(*[df.index for df in history_data.values()])))
    # Filter dates from 2017-08-01 (Start of backtest limitation)
    start_date = pd.Timestamp('2017-08-01')
    timeline = [d for d in all_dates if d >= start_date]
    
    # Portfolio State
    cash = 100000.0
    holdings = {} # {asset: shares}
    total_value = cash
    
    # Tracking
    value_history = []
    trade_log = []
    
    current_asset = '现金'
    
    # Buffer logic parameters
    BUFFER_SCORE_DIFF = 5.0 # If new_score - current_score < 5, don't switch (if current is Top 2)
    
    # Fee
    FEE_RATE = 0.0005 # 0.05%
    
    print(f"回测区间: {timeline[0].date()} 至 {timeline[-1].date()}")
    print(f"初始资金: {cash}")
    
    for i, date in enumerate(timeline):
        # 1. Update Portfolio Value (Open Price Execution happens today based on Yesterday's signal)
        # But we simulate: At Close of Day T, calculate signal. Trade at Open of T+1.
        # So loop iterates "Day T".
        # We need to know "Tomorrow" (T+1) to execute trade.
        
        # Actually, simpler loop:
        # On Date T, we have Holdings from T-1.
        # We check T Open prices to execute any pending trades from T-1 signal.
        # Then at T Close, we calculate signal for T+1.
        
        # Let's align with the loop:
        # 'date' is today.
        
        # A. Execute Pending Trade (at Open)
        # We need a variable "target_asset" determined yesterday.
        # On Day 0, target is 'Cash' (default).
        
        if i == 0:
            target_asset = '现金' # Initial state
            
        # Get Open prices for execution
        # Note: Not all assets trade on 'date'.
        # If held asset doesn't trade, we can't sell. If target doesn't trade, we can't buy.
        
        # Logic:
        # If current != target:
        #   Sell current (if not Cash) -> Cash
        #   Buy target (if not Cash) -> Shares
        # Trade happens at 'Open'.
        
        trade_executed = False
        
        # Check tradability
        can_sell = True
        can_buy = True
        
        if current_asset != '现金':
            if date not in history_data[current_asset].index:
                can_sell = False
        
        if target_asset != '现金':
            if date not in history_data[target_asset].index:
                can_buy = False
                
        # Execute Sell
        if current_asset != target_asset and current_asset != '现金' and can_sell:
            # Sell all
            price = history_data[current_asset].loc[date, 'open']
            shares = holdings[current_asset]
            proceeds = shares * price * (1 - FEE_RATE)
            cash += proceeds
            del holdings[current_asset]
            
            trade_log.append({
                'date': date,
                'action': 'SELL',
                'asset': current_asset,
                'price': price,
                'value': proceeds
            })
            current_asset = '现金'
            trade_executed = True
            
        # Execute Buy
        if current_asset == '现金' and target_asset != '现金' and can_buy:
            # Buy target
            price = history_data[target_asset].loc[date, 'open']
            # Calculate shares
            invest_amount = cash
            shares = invest_amount / (price * (1 + FEE_RATE))
            cost = shares * price * (1 + FEE_RATE)
            
            cash -= cost
            holdings[target_asset] = shares
            
            trade_log.append({
                'date': date,
                'action': 'BUY',
                'asset': target_asset,
                'price': price,
                'value': cost
            })
            current_asset = target_asset
            trade_executed = True
            
        # B. Calculate Portfolio Value at Close
        # Used for Net Value curve
        day_value = cash
        for asset, shares in holdings.items():
            # If trading today, use Close. If not, use last available Close (Fill Forward)
            if date in history_data[asset].index:
                price = history_data[asset].loc[date, 'close']
            else:
                # Find last close
                idx = history_data[asset].index.get_indexer([date], method='pad')[0]
                if idx >= 0:
                    price = history_data[asset].iloc[idx]['close']
                else:
                    price = 0 # Should not happen if we bought it
            day_value += shares * price
            
        value_history.append({'date': date, 'value': day_value})
        
        # C. Signal Generation (at Close) for NEXT day (T+1)
        # Calculate scores for all assets using data up to 'date'
        
        scores = {}
        for asset in assets:
            df = history_data[asset]
            if date in df.index:
                loc = df.index.get_loc(date)
                if loc >= 19: # Need 20 days
                    # Check if today is trading day for this asset
                    series = df.iloc[loc-19 : loc+1]['close']
                    s = calculate_score(series.values)
                    scores[asset] = s
            else:
                # Asset not trading today. Use last data?
                # Usually strategy runs on trading days. If asset halted, maybe skip update?
                # Or use last available data?
                # Let's skip assets not trading today to avoid stale data signal.
                pass
                
        # Normalize
        if not scores:
            next_target = '现金'
        else:
            # MinMax
            vals = np.array(list(scores.values()))
            mn, mx = np.min(vals), np.max(vals)
            
            norm_scores = {}
            for k, v in scores.items():
                if mx != mn:
                    norm_scores[k] = (v - mn) / (mx - mn) * 100
                else:
                    norm_scores[k] = 50
            
            # Rank
            sorted_scores = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
            top1_asset = sorted_scores[0][0]
            top1_score = sorted_scores[0][1]
            top1_raw = scores[top1_asset]
            
            # Determine Target
            # 1. Cash Rule
            if top1_raw < 0:
                next_target = '现金'
            else:
                # 2. Buffer Rule
                # If we hold an asset, and it is in Top 2, and Top1 - Current < 5
                if current_asset in norm_scores:
                    curr_score = norm_scores[current_asset]
                    # Check rank
                    curr_rank = next((i for i, (a, s) in enumerate(sorted_scores) if a == current_asset), 999)
                    
                    if curr_rank <= 1: # Top 2 (Index 0 or 1)
                        if top1_score - curr_score < BUFFER_SCORE_DIFF:
                            next_target = current_asset # Stay
                        else:
                            next_target = top1_asset # Switch
                    else:
                        next_target = top1_asset # Switch
                else:
                    next_target = top1_asset
        
        target_asset = next_target

    # Analysis
    df_res = pd.DataFrame(value_history).set_index('date')
    df_res['return'] = df_res['value'].pct_change().fillna(0)
    
    total_ret = df_res['value'].iloc[-1] / df_res['value'].iloc[0] - 1
    ann_ret = (1 + total_ret) ** (252 / len(df_res)) - 1
    vol = df_res['return'].std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / vol if vol != 0 else 0
    
    cum_max = df_res['value'].cummax()
    dd = (df_res['value'] - cum_max) / cum_max
    max_dd = dd.min()
    
    print("-" * 30)
    print(f"回测结果:")
    print(f"总收益率: {total_ret:.2%}")
    print(f"年化收益率: {ann_ret:.2%}")
    print(f"年化波动率: {vol:.2%}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_dd:.2%}")
    print(f"交易次数: {len(trade_log) // 2}") # Approx buy+sell pairs

if __name__ == "__main__":
    run_backtest()
