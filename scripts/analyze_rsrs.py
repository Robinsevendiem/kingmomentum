
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# 1. Load Data
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
                    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')
                    # Fallback
                    if df['trade_date'].isnull().all():
                         df['trade_date'] = pd.to_datetime(pd.read_csv(filename)['trade_date'])
                
                df = df.sort_values('trade_date').set_index('trade_date')
                history_data[name] = df
            except Exception as e:
                print(f"Error loading {name}: {e}")
    return history_data

# 2. RSRS Calculation
def calculate_rsrs(df, N=18, M=600):
    """
    N: Regression Window (Standard 18)
    M: Z-Score Window (Standard 600)
    """
    # Need High and Low
    # Use adj_high and adj_low if available? 
    # Usually RSRS uses raw High/Low because the ratio is similar, but splits affect it.
    # Adjusted is safer.
    
    if 'adj_high' in df.columns and 'adj_low' in df.columns:
        highs = df['adj_high']
        lows = df['adj_low']
    else:
        # Fallback to unadjusted but warn
        highs = df['high']
        lows = df['low']
        
    beta_series = []
    r2_series = []
    dates = []
    
    # Rolling regression
    # This is slow in pure python loop, but robust.
    values_high = highs.values
    values_low = lows.values
    index_dates = highs.index
    
    for i in range(N, len(df)):
        y = values_high[i-N:i]
        x = values_low[i-N:i]
        
        # Filter NaNs
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(valid_mask) < N * 0.8:
            beta_series.append(np.nan)
            r2_series.append(np.nan)
            continue
            
        x = x[valid_mask]
        y = y[valid_mask]
        
        try:
            x = sm.add_constant(x)
            model = sm.OLS(y, x)
            results = model.fit()
            beta = results.params[1]
            r2 = results.rsquared
            
            beta_series.append(beta)
            r2_series.append(r2)
        except:
            beta_series.append(np.nan)
            r2_series.append(np.nan)
            
    # Align indices (we started at N)
    rsrs_df = pd.DataFrame({
        'beta': beta_series,
        'r2': r2_series
    }, index=index_dates[N:])
    
    # Calculate RSRS_Score (Z-Score of Beta)
    # Note: Rolling mean/std requires M samples
    rsrs_df['beta_norm'] = (rsrs_df['beta'] - rsrs_df['beta'].rolling(M).mean()) / rsrs_df['beta'].rolling(M).std()
    
    # RSRS Right Skew Correction (RSRS_R2)
    # Adjusted Beta = Beta * R2
    rsrs_df['beta_adj'] = rsrs_df['beta'] * rsrs_df['r2']
    rsrs_df['beta_adj_norm'] = (rsrs_df['beta_adj'] - rsrs_df['beta_adj'].rolling(M).mean()) / rsrs_df['beta_adj'].rolling(M).std()
    
    return rsrs_df

def analyze():
    data = load_history_data()
    print(f"Loaded {len(data)} assets.")
    
    results = []
    
    for name, df in data.items():
        print(f"Analyzing {name}...")
        rsrs = calculate_rsrs(df)
        
        # Calculate Future Returns (20d)
        col = 'adj_close' if 'adj_close' in df.columns else 'close'
        prices = df[col]
        
        # Align
        common_idx = rsrs.index.intersection(prices.index)
        rsrs = rsrs.loc[common_idx]
        prices = prices.loc[common_idx]
        
        # Calculate returns
        future_ret = prices.pct_change(20).shift(-20)
        
        # Combine
        combined = rsrs.copy()
        combined['ret_20d'] = future_ret
        combined['asset'] = name
        
        results.append(combined)
        
    full_df = pd.concat(results)
    full_df = full_df.dropna()
    
    print("\n--- RSRS Signal Analysis (All Assets) ---")
    
    # Define Signal Thresholds
    # Buy: RSRS > 0.7
    # Sell: RSRS < -0.7
    
    thresholds = [0.7, 1.0]
    
    for t in thresholds:
        buy_signal = full_df[full_df['beta_norm'] > t]
        sell_signal = full_df[full_df['beta_norm'] < -t]
        
        avg_ret_buy = buy_signal['ret_20d'].mean()
        win_rate_buy = (buy_signal['ret_20d'] > 0).mean()
        
        avg_ret_sell = sell_signal['ret_20d'].mean()
        
        print(f"\nRSRS Threshold Z > {t}:")
        print(f"  Buy Signal Count: {len(buy_signal)}")
        print(f"  Avg Return (20d): {avg_ret_buy:.2%}")
        print(f"  Win Rate (20d):   {win_rate_buy:.1%}")
        print(f"  (Reference: Sell Signal Avg Return: {avg_ret_sell:.2%})")
        
    print("\n--- Correlation with Current Strategy? ---")
    # Does High RSRS correlate with High Returns better than just Momentum?
    # This is qualitative for now based on numbers.

if __name__ == "__main__":
    analyze()
