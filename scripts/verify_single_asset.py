import pandas as pd
import numpy as np
import os

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

def analyze_single_asset_logic():
    print("--- 单一持仓策略逻辑深度分析 ---")
    
    # 1. Load Data
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', ''))
    df_trade = df_trade.sort_values('调仓时间')
    
    history_data = load_history_data()
    all_assets = list(history_data.keys())
    
    # 2. Analyze Rank of Held Asset (Is it always Top 1?)
    # We previously found:
    # 5-day: 38% Top 1
    # 10-day: 44% Top 1
    # 20-day: 52% Top 1
    # 25-day: 36% Top 1
    # 60-day: 20% Top 1
    
    # This suggests that "Top 1 by 20-day return" is the most likely candidate, 
    # BUT 52% is too low for a "Rule". 
    # If the rule is "Buy Top 1", why did it buy Rank 2 (30% of time) or Rank 3?
    
    # Hypothesis A: It's NOT simple return ranking. Maybe Risk-Adjusted Return (Sharpe)?
    # Hypothesis B: It's RSRS (Resistance Support Relative Strength)?
    # Hypothesis C: It's a "Winner Stay" logic? (Don't switch unless new Top 1 beats current by X%?)
    
    print("\n[假设验证 1: 是否存在'强者恒强'的换仓缓冲机制?]")
    # Check if we held Asset A, and Asset B became Top 1 but only slightly better, did we switch?
    # Or did we switch only when Asset B was MUCH better?
    
    switch_count = 0
    buffer_logic_count = 0
    
    rank_stats = []
    
    # Let's look at the cash holding periods first.
    # When did we hold cash?
    cash_trades = df_trade[df_trade['买入'] == '现金']
    print(f"\n[现金持仓分析] 共 {len(cash_trades)} 次切入现金")
    for _, row in cash_trades.iterrows():
        date = row['调仓时间']
        # Check returns of ALL assets on this date
        print(f"  {date.date()}: 切入现金")
        # Calculate max return of all assets
        max_ret = -999
        best_asset = ""
        
        for asset, df in history_data.items():
            try:
                # 20-day return
                d_idx = df.index.get_indexer([date], method='pad')[0]
                if d_idx < 20: continue
                
                # Close T-1 / Close T-21 - 1
                # Actually let's use the exact previous trading day
                prev_date = df.index[df.index < date][-1]
                prev_loc = df.index.get_loc(prev_date)
                
                if prev_loc >= 20:
                    p_curr = df.iloc[prev_loc]['close']
                    p_prev = df.iloc[prev_loc-20]['close']
                    ret = (p_curr - p_prev) / p_prev
                    
                    if ret > max_ret:
                        max_ret = ret
                        best_asset = asset
            except: pass
            
        print(f"    当时市场最优标的 ({best_asset}) 20日涨幅: {max_ret:.2%}")
        
    # Conclusion from cash analysis will likely show: All assets were negative or below threshold.
    
    # Now back to "Why Rank 2 or 3?"
    # Maybe the ranking is based on Volatility-adjusted return?
    # Or maybe we are looking at the wrong period?
    # Let's try to find a metric where the bought asset is ALWAYS Top 1.
    
    print("\n[寻找最佳排名指标]")
    # We will test: 
    # 1. Returns: 10, 15, 20, 25, 30 days
    # 2. Return / Volatility (Sharpe-like): 20 days
    
    periods = [10, 15, 20, 25, 30]
    
    best_metric_score = 0
    best_metric_name = ""
    
    for p in periods:
        # 1. Simple Return
        top1_matches = 0
        valid_trades = 0
        
        for _, row in df_trade.iterrows():
            if row['买入'] == '现金' or row['买入'] not in history_data: continue
            
            trade_date = row['调仓时间']
            bought = row['买入']
            valid_trades += 1
            
            # Rank all assets by Period Return
            asset_metrics = {}
            for asset, df in history_data.items():
                try:
                    prev_date = df.index[df.index < trade_date][-1]
                    prev_loc = df.index.get_loc(prev_date)
                    if prev_loc >= p:
                        ret = df.iloc[prev_loc]['close'] / df.iloc[prev_loc-p]['close'] - 1
                        asset_metrics[asset] = ret
                except: pass
                
            # Sort
            sorted_assets = sorted(asset_metrics.items(), key=lambda x: x[1], reverse=True)
            if sorted_assets and sorted_assets[0][0] == bought:
                top1_matches += 1
                
        score = top1_matches / valid_trades if valid_trades > 0 else 0
        print(f"  {p}日涨幅排名 Top 1 命中率: {score:.2%}")
        
        if score > best_metric_score:
            best_metric_score = score
            best_metric_name = f"{p}日涨幅"
            
        # 2. Risk Adjusted Return (Return / StdDev)
        top1_matches_risk = 0
        for _, row in df_trade.iterrows():
            if row['买入'] == '现金' or row['买入'] not in history_data: continue
            trade_date = row['调仓时间']
            bought = row['买入']
            
            asset_metrics = {}
            for asset, df in history_data.items():
                try:
                    prev_date = df.index[df.index < trade_date][-1]
                    prev_loc = df.index.get_loc(prev_date)
                    if prev_loc >= p:
                        # Slice last p days
                        # Calculate daily returns std dev
                        slice_df = df.iloc[prev_loc-p+1 : prev_loc+1]
                        daily_rets = slice_df['close'].pct_change().dropna()
                        std = daily_rets.std()
                        total_ret = df.iloc[prev_loc]['close'] / df.iloc[prev_loc-p]['close'] - 1
                        
                        if std > 0:
                            asset_metrics[asset] = total_ret / std
                        else:
                            asset_metrics[asset] = -999
                except: pass
                
            sorted_assets = sorted(asset_metrics.items(), key=lambda x: x[1], reverse=True)
            if sorted_assets and sorted_assets[0][0] == bought:
                top1_matches_risk += 1
                
        score_risk = top1_matches_risk / valid_trades if valid_trades > 0 else 0
        print(f"  {p}日夏普(Ret/Std)排名 Top 1 命中率: {score_risk:.2%}")

if __name__ == "__main__":
    analyze_single_asset_logic()
