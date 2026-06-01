import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# File paths
RECORD_DIR = 'record'
TRADE_FILE = os.path.join(RECORD_DIR, '2017-08-10 至 2026-02-26调仓记录.csv')

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
    Calculate a comprehensive set of features for ML models.
    Inputs: 20-day prices (or longer if needed, but strategy uses 20 days).
    Features:
    - Returns (5d, 10d, 20d)
    - Volatility (5d, 10d, 20d)
    - Max Drawdown (20d)
    - Linear Regression Slope (20d)
    - R^2 (20d)
    - Slope * R^2 (20d)
    - Moving Average distance (Price / MA20 - 1)
    - RSI (14d)
    """
    if len(prices) < 20: return None
    
    # Prices Series
    p = pd.Series(prices)
    
    feats = {}
    
    # 1. Returns
    feats['ret_5'] = p.iloc[-1] / p.iloc[-6] - 1
    feats['ret_10'] = p.iloc[-1] / p.iloc[-11] - 1
    feats['ret_20'] = p.iloc[-1] / p.iloc[0] - 1
    
    # 2. Volatility (Std Dev of returns)
    rets = p.pct_change().dropna()
    feats['vol_5'] = rets.tail(5).std()
    feats['vol_10'] = rets.tail(10).std()
    feats['vol_20'] = rets.std()
    
    # 3. Max Drawdown (20d)
    cum_max = np.maximum.accumulate(p.values)
    dd = (p.values - cum_max) / cum_max
    feats['max_dd_20'] = dd.min()
    
    # 4. Linear Regression (20d)
    y = np.log(p.values)
    x = np.arange(len(y))
    # Simple OLS
    slope, intercept, r_value, _, _ = pd.Series(y).sort_index().reset_index(drop=True).pipe(lambda s: np.polyfit(s.index, s.values, 1)) # Fast polyfit
    # Or use scipy linregress for consistency
    # from scipy.stats import linregress
    # slope, _, r_value, _, _ = linregress(x, y)
    
    feats['slope_20'] = slope
    feats['r2_20'] = r_value ** 2
    feats['slope_r2_20'] = slope * (r_value ** 2)
    
    # 5. MA Distance
    ma20 = p.mean()
    feats['ma_dist_20'] = p.iloc[-1] / ma20 - 1
    
    return feats

def ml_reverse_engineering():
    print("--- 机器学习策略破解 (ML Reverse Engineering) ---")
    
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', ''))
    df_trade = df_trade.sort_values('调仓时间')
    
    history_data = load_history_data()
    all_assets = list(history_data.keys())
    
    # Prepare Dataset
    # Each row is a trade decision.
    # Target: The asset that was bought (Winner).
    # Features: Features of ALL 9 assets on that day.
    # Since we need to predict "Which asset is the winner", this is a Multi-Class Classification problem.
    # OR Learning to Rank.
    # Let's simplify:
    # For each trade date, we have 9 samples (one for each asset).
    # Target = 1 if bought, 0 if not.
    # Features = Asset's own features.
    # But ranking depends on relative performance. So we should normalize features across assets for each date.
    
    X = []
    y = []
    dates = []
    asset_names = []
    
    feature_names = None
    
    for _, row in df_trade.iterrows():
        trade_date = row['调仓时间']
        bought = row['买入']
        if bought == '现金': continue
        
        # Calculate features for ALL assets
        day_features = {}
        valid_day = True
        
        for asset in all_assets:
            if asset not in history_data: continue
            df = history_data[asset]
            try:
                prev_date = df.index[df.index < trade_date][-1]
                prev_loc = df.index.get_loc(prev_date)
                if prev_loc >= 19:
                    series = df.iloc[prev_loc-19 : prev_loc+1]['close']
                    f = calculate_features(series.values)
                    if f:
                        day_features[asset] = f
            except: pass
            
        if len(day_features) < 2: continue # Need at least 2 assets to compare
        
        # Normalize Features for this day (Cross-sectional MinMax)
        # This is crucial for ranking models
        df_day = pd.DataFrame(day_features).T
        if feature_names is None: feature_names = df_day.columns.tolist()
        
        # MinMax Scaler per column
        df_norm = (df_day - df_day.min()) / (df_day.max() - df_day.min())
        df_norm = df_norm.fillna(0.5) # Handle constant columns
        
        for asset in df_norm.index:
            X.append(df_norm.loc[asset].values)
            y.append(1 if asset == bought else 0)
            dates.append(trade_date)
            asset_names.append(asset)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"样本总数: {len(X)} (交易日 x 资产数)")
    print(f"正样本数: {sum(y)}")
    
    # Split Train/Test
    # Time Series Split usually better, but for reverse engineering "static rules", random split is fine to check if rules are learnable.
    # Let's use 80/20 split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Random Forest (Feature Importance)
    print("\n[随机森林模型训练中...]")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"RF 测试集准确率: {acc:.2%}")
    
    # Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n[关键特征排名 (Top 5)]")
    for f in range(5):
        print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
        
    # 2. Neural Network (Non-linear relationships)
    print("\n[神经网络模型训练中...]")
    # Simple MLP
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp.predict(X_test_scaled)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    print(f"MLP 测试集准确率: {acc_mlp:.2%}")
    
    # 3. XGBoost (Gradient Boosting)
    print("\n[XGBoost模型训练中...]")
    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"XGBoost 测试集准确率: {acc_xgb:.2%}")
    
    # Check "Slope * R2" correlation specifically
    # If 'slope_r2_20' is the top feature, it confirms our manual finding.
    
if __name__ == "__main__":
    ml_reverse_engineering()
