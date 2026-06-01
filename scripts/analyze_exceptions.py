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

def analyze_exceptions():
    print("--- 调仓规则异常分析 (Exception Analysis) ---")
    
    # 1. Load Data
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', ''))
    df_trade = df_trade.sort_values('调仓时间')
    
    history_data = load_history_data()
    
    # 2. Analyze Exceptions for "20-day Return > 0" and "Price > MA20"
    exceptions = []
    total_valid_trades = 0
    
    for _, row in df_trade.iterrows():
        trade_date = row['调仓时间']
        bought_asset = row['买入']
        
        if bought_asset == '现金' or bought_asset not in history_data:
            continue
            
        total_valid_trades += 1
        
        # Get data up to trade date (assume Close[T-1] for decision)
        # We need to be careful: If trade date is Monday, T-1 is Friday.
        # history_data index is trade_date.
        
        df = history_data[bought_asset]
        df_slice = df.loc[:trade_date]
        
        if df_slice.empty: continue
        
        # Assume decision is based on Close price of the day BEFORE the trade (T-1)
        # Because trade executes at "Open" of T.
        try:
            # Find the latest date BEFORE trade_date
            decision_date = df_slice.index[df_slice.index < trade_date][-1]
            decision_close = df_slice.loc[decision_date, 'close']
            
            # 20-day Return Logic
            # P[T-1] / P[T-1-20] - 1
            if len(df_slice.loc[:decision_date]) > 20:
                past_date_20 = df_slice.loc[:decision_date].index[-21] # 20 days ago from T-1
                past_price_20 = df_slice.loc[past_date_20, 'close']
                ret_20 = (decision_close - past_price_20) / past_price_20
            else:
                ret_20 = None
                
            # MA20 Logic
            # MA20 on T-1 includes Close[T-1]
            ma20 = df_slice.loc[:decision_date, 'close'].rolling(20).mean().iloc[-1]
            
            # Check conditions
            is_positive_ret = ret_20 > 0 if ret_20 is not None else False
            is_above_ma20 = decision_close > ma20 if ma20 is not None else False
            
            if not is_positive_ret or not is_above_ma20:
                exceptions.append({
                    'date': trade_date,
                    'asset': bought_asset,
                    'price_T_minus_1': decision_close,
                    'ret_20': ret_20,
                    'ma20': ma20,
                    'is_pos_ret': is_positive_ret,
                    'is_above_ma': is_above_ma20
                })
                
        except (IndexError, KeyError) as e:
            # print(f"Error processing {trade_date} for {bought_asset}: {e}")
            pass

    # 3. Output Analysis
    print(f"总有效非现金买入次数: {total_valid_trades}")
    print(f"异常交易次数 (违反任一规则): {len(exceptions)}")
    
    if exceptions:
        print("\n[异常明细 (Top 10)]")
        df_ex = pd.DataFrame(exceptions)
        print(df_ex[['date', 'asset', 'ret_20', 'is_pos_ret', 'is_above_ma']].head(10))
        
        print("\n[异常统计]")
        neg_ret_count = len([x for x in exceptions if not x['is_pos_ret']])
        below_ma_count = len([x for x in exceptions if not x['is_above_ma']])
        print(f"违反'正收益'规则的次数: {neg_ret_count} ({neg_ret_count/total_valid_trades:.2%})")
        print(f"违反'均线'规则的次数: {below_ma_count} ({below_ma_count/total_valid_trades:.2%})")
        
        # Analyze the magnitude of violation
        neg_rets = [x['ret_20'] for x in exceptions if x['ret_20'] is not None and x['ret_20'] <= 0]
        if neg_rets:
            print(f"负收益最大值 (接近0的程度): {max(neg_rets):.4%}")
            print(f"负收益最小值 (严重违规): {min(neg_rets):.4%}")
            
    # 4. Check "Open Price" Hypothesis
    # Maybe they use the Open price of the trade day itself?
    print("\n[假设检验: 使用调仓日开盘价作为决策依据]")
    exceptions_open = []
    
    for _, row in df_trade.iterrows():
        trade_date = row['调仓时间']
        bought_asset = row['买入']
        if bought_asset == '现金' or bought_asset not in history_data: continue
        
        df = history_data[bought_asset]
        if trade_date not in df.index: continue
        
        open_price = df.loc[trade_date, 'open']
        
        # Calculate indicators based on T-1 Close but check if Open satisfies condition?
        # Usually strategy uses T-1 Close to generate signal for T Open.
        # But maybe they filter: "If Open < MA20, don't buy"?
        
        # Let's check if Open Price > MA20 (calculated on T-1)
        try:
            decision_date = df.loc[:trade_date].index[-2] # T-1
            ma20 = df.loc[:decision_date, 'close'].rolling(20).mean().iloc[-1]
            
            if open_price < ma20:
                exceptions_open.append(trade_date)
        except: pass
        
    print(f"使用开盘价判断，违反均线规则次数: {len(exceptions_open)}")

if __name__ == "__main__":
    analyze_exceptions()
