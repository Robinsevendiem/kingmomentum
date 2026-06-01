import pandas as pd
import numpy as np
import os
import sys

from scripts.optimize_exclude_overheated import load_history_data, calculate_rolling_scores, precalculate_all_scores

# We will implement the exact run_backtest logic from Home.py here to ensure it's up to date.
def run_backtest(history_data, raw_scores_df, params):
    # Filter Timeline
    timeline = [d for d in raw_scores_df.index if params['start_date'] <= d <= params['end_date']]
    timeline = sorted(timeline)
    
    if not timeline: return pd.DataFrame(), pd.DataFrame()

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
        elif 'open' in df.columns and 'close' in df.columns:
            price_open[asset] = df['open']
            price_close[asset] = df['close']

    portfolio_value = []
    trade_records = []
    
    global_cutoff = params.get('global_cutoff', 600)
    buffer_score = params.get('buffer_score', 5.0)
    exclude_overheated = params.get('exclude_overheated_from_norm', True)
    min_score = params.get('min_score', 0.0)

    for i, date in enumerate(timeline):
        today_scores = raw_scores_df.loc[date].dropna()
        valid_candidates = today_scores[today_scores < global_cutoff]
        
        # 1. Process valid pool
        valid_assets_pool = valid_candidates.copy()
        
        # Minimum score filter
        valid_assets_pool = valid_assets_pool[valid_assets_pool >= min_score]
        
        # 2. Normalization
        if exclude_overheated:
            norm_basis = valid_candidates
        else:
            norm_basis = today_scores
            
        normalized_scores = pd.Series(dtype=float)
        if not norm_basis.empty and len(norm_basis) > 1:
            min_val = norm_basis.min()
            max_val = norm_basis.max()
            if max_val > min_val:
                for asset in valid_assets_pool.index:
                    normalized_scores[asset] = (valid_assets_pool[asset] - min_val) / (max_val - min_val) * 100
            else:
                for asset in valid_assets_pool.index:
                    normalized_scores[asset] = 100.0
        elif len(valid_assets_pool) == 1:
             normalized_scores[valid_assets_pool.index[0]] = 100.0
        
        # 3. Decision Logic
        if normalized_scores.empty:
            target_asset = '现金'
        else:
            best_asset = normalized_scores.idxmax()
            best_score = normalized_scores[best_asset]
            
            if current_asset == '现金':
                target_asset = best_asset
            elif current_asset in normalized_scores:
                current_score = normalized_scores[current_asset]
                if best_score > current_score + buffer_score:
                    target_asset = best_asset
                else:
                    target_asset = current_asset
            else:
                target_asset = best_asset
                
        # 4. Execute Trade (assuming next day open)
        if target_asset != current_asset:
            if i < len(timeline) - 1:
                next_date = timeline[i+1]
                trade_price_sell = 1.0
                trade_price_buy = 1.0
                
                if current_asset != '现金':
                    try:
                        trade_price_sell = price_open[current_asset].loc[next_date]
                        cash = holdings[current_asset] * trade_price_sell
                        del holdings[current_asset]
                    except KeyError:
                        pass # Price missing, assume flat
                        
                if target_asset != '现金':
                    try:
                        trade_price_buy = price_open[target_asset].loc[next_date]
                        holdings[target_asset] = cash / trade_price_buy
                        cash = 0
                    except KeyError:
                        target_asset = '现金'
                        
                trade_records.append({
                    '调仓时间': next_date,
                    '卖出': current_asset,
                    '买入': target_asset
                })
                current_asset = target_asset
                
        # 5. Record Value (End of day)
        current_val = cash
        if current_asset != '现金':
            try:
                current_val = holdings[current_asset] * price_close[current_asset].loc[date]
            except KeyError:
                try:
                    # Fallback to last known price if missing today
                    last_price = price_close[current_asset].loc[:date].iloc[-1]
                    current_val = holdings[current_asset] * last_price
                except:
                    pass
                    
        portfolio_value.append({
            '日期': date,
            'ETF名称': current_asset,
            '净值': current_val
        })
        
    df_val = pd.DataFrame(portfolio_value)
    df_trades = pd.DataFrame(trade_records)
    return df_val, df_trades

def main():
    history_data = load_history_data()
    for name, df in history_data.items():
        df.index = pd.to_datetime(df.index)
        
    raw_scores_df = precalculate_all_scores(history_data, window=20)
    
    params = {
        'start_date': pd.to_datetime('2017-08-10'),
        'end_date': pd.to_datetime('2026-03-23'),
        'initial_capital': 1.0,
        'global_cutoff': 600,
        'buffer_score': 5,
        'min_score': 0,
        'exclude_overheated_from_norm': True
    }
    
    df_val, df_trades = run_backtest(history_data, raw_scores_df, params)
    
    if not df_val.empty:
        final_val = df_val['净值'].iloc[-1]
        days = len(df_val)
        ann_ret = (final_val ** (252/days)) - 1
        
        cumulative_max = df_val['净值'].cummax()
        drawdown = (df_val['净值'] - cumulative_max) / cumulative_max
        max_dd = drawdown.min()
        
        print(f"Current Strategy Performance (2017-08-10 to 2026-03-23):")
        print(f"Final Value: {final_val:.4f}")
        print(f"Annualized Return: {ann_ret*100:.2f}%")
        print(f"Max Drawdown: {max_dd*100:.2f}%")
        print(f"Total Trades: {len(df_trades)}")
        
if __name__ == '__main__':
    main()
