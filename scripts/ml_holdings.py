import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from scipy.stats import linregress

# File paths
RECORD_DIR = 'record'
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

def calculate_features(prices):
    """
    Calculate comprehensive features for 20-day window.
    Focus on WLS Slope, R2, and their product.
    Also include simple return, volatility, max drawdown for comparison.
    """
    if len(prices) < 20: return None
    
    # Use last 20 days
    prices = prices[-20:]
    
    y = np.log(prices)
    x = np.arange(len(y))
    
    # 1. Simple Return
    ret = prices[-1] / prices[0] - 1
    
    # 2. Volatility
    vol = pd.Series(prices).pct_change().std() * np.sqrt(252)
    
    # 3. Max Drawdown
    cum_max = np.maximum.accumulate(prices)
    dd = (prices - cum_max) / cum_max
    max_dd = dd.min()
    
    # 4. Standard OLS
    slope_ols, _, r_value_ols, _, _ = linregress(x, y)
    r2_ols = r_value_ols ** 2
    score_ols = slope_ols * r2_ols
    
    # 5. Weighted Linear Regression (Power=2)
    # Weights: 1 + (t/19)^2
    x_norm = np.linspace(0, 1, len(y))
    weights = 1 + x_norm ** 2
    
    coeffs_wls = np.polyfit(x, y, 1, w=weights)
    slope_wls = coeffs_wls[0]
    
    # R2 WLS
    y_pred = np.polyval(coeffs_wls, x)
    sse = np.sum(weights * (y - y_pred)**2)
    y_mean = np.average(y, weights=weights)
    sst = np.sum(weights * (y - y_mean)**2)
    if sst == 0: r2_wls = 0
    else: r2_wls = 1 - sse / sst
    
    score_wls = slope_wls * r2_wls
    
    return {
        'ret': ret,
        'vol': vol,
        'max_dd': max_dd,
        'slope_ols': slope_ols,
        'r2_ols': r2_ols,
        'score_ols': score_ols,
        'slope_wls': slope_wls,
        'r2_wls': r2_wls,
        'score_wls': score_wls
    }

def train_ml_on_holdings():
    print("--- 基于持仓记录的机器学习训练 (Training on Daily Holdings) ---")
    
    # Load Position File (Daily records)
    df_pos = pd.read_csv(POSITION_FILE)
    df_pos['日期'] = pd.to_datetime(df_pos['日期'])
    df_pos = df_pos.sort_values('日期')
    
    # Filter only trading days (skip weekends if any, usually position file has all days? or just trading days?)
    # Usually position file is daily.
    # Target: 'ETF名称' column.
    
    history_data = load_history_data()
    all_assets = list(history_data.keys())
    
    X = []
    y = []
    feature_names = None
    
    # To reduce noise, maybe we sample every 5 days? Or use all daily data.
    # Using all daily data might overfit to "holding periods" (autocorrelation).
    # But for "learning the rule", it's fine.
    
    print(f"原始持仓记录数: {len(df_pos)}")
    
    valid_samples = 0
    
    for _, row in df_pos.iterrows():
        date = row['日期']
        held_asset = row['ETF名称']
        
        if held_asset == '现金': continue # Skip cash days for ranking learning (or handle separately)
        if held_asset not in all_assets: continue
        
        # Calculate features for ALL assets on this day
        # Note: Position is End of Day. So features should be based on Close of THIS day (T)?
        # Strategy executes at T+1 Open based on T Close.
        # So Position at T Close reflects decision made at T-1 Close?
        # NO.
        # Trade record: "2026-02-06 Open" bought "Southern Oil".
        # Position record: "2026-02-06" shows "Southern Oil".
        # So on 2026-02-06, we hold Oil.
        # The decision to hold Oil on 2026-02-06 was made based on data up to 2026-02-05 Close (or 2026-02-06 Open).
        # Actually, if we rebalance at Open, the holding for the rest of the day is determined.
        # So for date T, the features should be calculated using data up to T-1 Close?
        # Or T Close?
        # If we trade at Open, we use T-1 data.
        # So for holding on Date T, the relevant features are from T-1.
        
        # Let's try T-1.
        # Need to find the trading day before 'date'.
        
        day_features = {}
        
        # Check if we have data for T-1
        # We can just look up date - 1 day? No, weekends.
        # Use history data index.
        
        try:
            # Pick one asset to find previous trading day
            ref_df = history_data[held_asset] # Held asset must exist
            if date not in ref_df.index: 
                # Maybe date is a weekend/holiday in Position file?
                # Find nearest previous trading day
                prev_dates = ref_df.index[ref_df.index < date]
                if len(prev_dates) == 0: continue
                calc_date = prev_dates[-1] # T-1
            else:
                # If date is trading day, and we trade at Open, we use T-1.
                loc = ref_df.index.get_loc(date)
                if loc == 0: continue
                calc_date = ref_df.index[loc-1]
                
            # Now calc features for all assets up to calc_date
            for asset in all_assets:
                if asset not in history_data: continue
                df = history_data[asset]
                
                # Find location of calc_date
                if calc_date in df.index:
                    loc = df.index.get_loc(calc_date)
                    if loc >= 19:
                        series = df.iloc[loc-19 : loc+1]['close'] # 20 days ending at calc_date
                        f = calculate_features(series.values)
                        if f:
                            day_features[asset] = f
                            
            if len(day_features) < 2: continue
            
            # Normalize Features (MinMax)
            df_day = pd.DataFrame(day_features).T
            if feature_names is None: feature_names = df_day.columns.tolist()
            
            df_norm = (df_day - df_day.min()) / (df_day.max() - df_day.min())
            df_norm = df_norm.fillna(0.5)
            
            # Add samples
            # We want to predict "Is Winner?"
            # For each asset, add a sample.
            for asset in df_norm.index:
                X.append(df_norm.loc[asset].values)
                y.append(1 if asset == held_asset else 0)
                
            valid_samples += 1
            
        except Exception as e:
            # print(e)
            pass
            
    print(f"有效训练样本数: {len(X)} (来自 {valid_samples} 个持仓日)")
    
    if len(X) == 0:
        print("Error: No samples generated.")
        return

    X = np.array(X)
    y = np.array(y)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print(f"RF 准确率: {accuracy_score(y_test, y_pred):.2%}")
    
    # Feature Importance
    print("\n[特征重要性排名]")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(len(feature_names)):
        print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
        
    # 2. XGBoost
    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X_train, y_train)
    print(f"XGBoost 准确率: {accuracy_score(y_test, model_xgb.predict(X_test)):.2%}")

if __name__ == "__main__":
    train_ml_on_holdings()
