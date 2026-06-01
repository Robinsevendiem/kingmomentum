import pandas as pd
import numpy as np
import os
from collections import Counter

# File paths
RECORD_DIR = 'record'
TRADE_FILE = os.path.join(RECORD_DIR, '2017-08-10 至 2026-02-26调仓记录.csv')
POSITION_FILE = os.path.join(RECORD_DIR, '2017-08-10 至 2026-02-26持仓记录.csv')

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

def analyze_momentum_rules():
    print("--- 调仓规则推断分析 ---")
    
    # 1. Load Data
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', ''))
    df_trade = df_trade.sort_values('调仓时间')
    
    history_data = load_history_data()
    all_assets = list(history_data.keys())
    
    # 2. Analyze Momentum Lookback Periods
    # Common momentum periods: 5, 10, 20, 60 days
    lookback_periods = [5, 10, 20, 25, 60]
    
    results = []
    
    for _, row in df_trade.iterrows():
        trade_date = row['调仓时间']
        bought_asset = row['买入']
        sold_asset = row['卖出']
        
        if bought_asset == '现金' or bought_asset not in history_data:
            continue
            
        # Get data up to trade date (excluding trade date itself for signal calculation usually, but strategy might use Open price of trade date)
        # Let's assume decision is made based on Close of previous day
        prev_date = trade_date - pd.Timedelta(days=1)
        
        # Calculate returns for ALL assets over various periods ending at prev_date
        # We need to find the latest available trading day before or on prev_date for each asset
        
        asset_returns = {}
        for asset, df in history_data.items():
            # Get data up to prev_date
            df_slice = df.loc[:trade_date] # Include trade date to check if we used Open? Or just Close of prev?
            # Usually strategies use Close[T-1] vs Close[T-1-N]
            
            if df_slice.empty:
                continue
                
            # Find the index of the trade date (or nearest before)
            try:
                # Use asof to find latest date <= trade_date - 1 day
                target_date = df_slice.index[df_slice.index < trade_date][-1]
                current_price = df_slice.loc[target_date, 'close'] # Close price before trade
                
                asset_returns[asset] = {}
                for period in lookback_periods:
                    # Find price N days ago (trading days)
                    if len(df_slice.loc[:target_date]) > period:
                        past_date = df_slice.loc[:target_date].index[-period-1]
                        past_price = df_slice.loc[past_date, 'close']
                        ret = (current_price - past_price) / past_price
                        asset_returns[asset][period] = ret
            except IndexError:
                pass
        
        # Rank assets by return for each period
        for period in lookback_periods:
            # Get returns for all assets for this period
            rets = {k: v.get(period, -999) for k, v in asset_returns.items()}
            # Sort descending
            sorted_assets = sorted(rets.items(), key=lambda x: x[1], reverse=True)
            
            # Check rank of bought asset
            rank = -1
            for i, (asset, ret) in enumerate(sorted_assets):
                if asset == bought_asset:
                    rank = i + 1
                    break
            
            results.append({
                'date': trade_date,
                'bought': bought_asset,
                'period': period,
                'rank': rank,
                'return': rets.get(bought_asset),
                'top1_asset': sorted_assets[0][0] if sorted_assets else None,
                'top1_return': sorted_assets[0][1] if sorted_assets else None
            })

    df_res = pd.DataFrame(results)
    
    # 3. Summarize Findings
    print("\n[动量周期推断]")
    for period in lookback_periods:
        # Check how often the bought asset was Rank 1
        subset = df_res[df_res['period'] == period]
        rank1_count = len(subset[subset['rank'] == 1])
        total = len(subset)
        ratio = rank1_count / total if total > 0 else 0
        print(f"周期 {period}日: 买入标的为收益率第1名的比例: {ratio:.2%} ({rank1_count}/{total})")
        
        # Check Rank 2?
        rank2_count = len(subset[subset['rank'] <= 2])
        ratio2 = rank2_count / total if total > 0 else 0
        print(f"          买入标的为前2名的比例: {ratio2:.2%}")

    # 4. Analyze "Gap" or Threshold
    # Is there a minimum return required?
    # Or is it RSRS (Resistance Support Relative Strength)?
    # Let's check 20-day return specifically as it's common (1 month)
    best_period = 20 # Hypothesis
    subset_20 = df_res[df_res['period'] == best_period]
    avg_rank = subset_20['rank'].mean()
    print(f"\n[20日动量深入分析]")
    print(f"买入标的平均排名: {avg_rank:.2f}")
    
    # Check if we buy when R > 0?
    positive_ret_count = len(subset_20[subset_20['return'] > 0])
    print(f"买入时20日收益率 > 0 的比例: {positive_ret_count/len(subset_20):.2%}")

    # 5. Check if sold asset was dropping in rank?
    # (Simplified: just inferring buy logic is usually enough for "rules")

    # 6. Check Moving Average logic (Price > MA20?)
    ma_results = []
    for _, row in df_trade.iterrows():
        trade_date = row['调仓时间']
        bought_asset = row['买入']
        if bought_asset == '现金' or bought_asset not in history_data:
            continue
            
        df = history_data[bought_asset]
        # Get data before trade
        df_slice = df.loc[:trade_date]
        if df_slice.empty: continue
        
        try:
            target_date = df_slice.index[df_slice.index < trade_date][-1]
            close = df_slice.loc[target_date, 'close']
            ma20 = df_slice.loc[:target_date, 'close'].rolling(20).mean().iloc[-1]
            
            ma_results.append(close > ma20)
        except:
            pass
            
    print(f"\n[均线逻辑]")
    print(f"买入时价格 > MA20 的比例: {sum(ma_results)/len(ma_results):.2%}")
    
if __name__ == "__main__":
    analyze_momentum_rules()
