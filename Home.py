import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import tushare as ts
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from data_update_utils import get_open_trade_end_date, update_history_file

warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(page_title="ETF 动量策略回测系统", layout="wide", page_icon="📈")

import statsmodels.api as sm

CUSTOM_ASSETS_PATH = "data/custom_assets.json"
ACTIVE_POOL_PATH = "data/active_pool.json"

BUILTIN_ASSETS = [
    {"code": "513520.SH", "name": "日经ETF", "start_date": "20190612", "asset_type": "FD", "file_path": "data/513520.SH_日经ETF_history.csv"},
    {"code": "513100.SH", "name": "纳指ETF", "start_date": "20130515", "asset_type": "FD", "file_path": "data/513100.SH_纳指ETF_history.csv"},
    {"code": "513020.SH", "name": "港股科技ETF", "start_date": "20220127", "asset_type": "FD", "file_path": "data/513020.SH_港股科技ETF_history.csv"},
    {"code": "510180.SH", "name": "180ETF", "start_date": "20060518", "asset_type": "FD", "file_path": "data/510180.SH_180ETF_history.csv"},
    {"code": "588120.SH", "name": "科创板ETF", "start_date": "20230908", "asset_type": "FD", "file_path": "data/588120.SH_科创板ETF_history.csv"},
    {"code": "159915.SZ", "name": "创业板ETF", "start_date": "20111209", "asset_type": "FD", "file_path": "data/159915.SZ_创业板ETF_history.csv"},
    {"code": "501018.SH", "name": "南方原油(LOF)", "start_date": "20160624", "asset_type": "FD", "file_path": "data/501018.SH_南方原油(LOF)_history.csv"},
    {"code": "518880.SH", "name": "黄金ETF", "start_date": "20130729", "asset_type": "FD", "file_path": "data/518880.SH_黄金ETF_history.csv"},
    {"code": "511090.SH", "name": "30年国债ETF", "start_date": "20230613", "asset_type": "FD", "file_path": "data/511090.SH_30年国债ETF_history.csv"},
]


def _read_json(path, default):
    import json

    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _write_json(path, obj):
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_custom_assets():
    data = _read_json(CUSTOM_ASSETS_PATH, [])
    if not isinstance(data, list):
        return []
    items = []
    for x in data:
        if not isinstance(x, dict):
            continue
        code = str(x.get("code", "")).strip()
        name = str(x.get("name", "")).strip()
        if not code or not name:
            continue
        items.append(x)
    return items


def save_custom_assets(items):
    _write_json(CUSTOM_ASSETS_PATH, items)


def load_active_pool_codes():
    data = _read_json(ACTIVE_POOL_PATH, {})
    if isinstance(data, dict) and isinstance(data.get("codes"), list):
        codes = [str(x).strip() for x in data["codes"] if str(x).strip()]
        if codes:
            return codes
    return [x["code"] for x in BUILTIN_ASSETS]


def save_active_pool_codes(codes):
    codes = [str(x).strip() for x in codes if str(x).strip()]
    _write_json(ACTIVE_POOL_PATH, {"codes": codes})


def get_all_assets_config():
    all_assets = []
    all_assets.extend(BUILTIN_ASSETS)
    all_assets.extend(load_custom_assets())
    by_code = {}
    for a in all_assets:
        code = str(a.get("code", "")).strip()
        if not code:
            continue
        by_code[code] = a
    return list(by_code.values())


def get_selected_assets_config():
    active = set(load_active_pool_codes())
    selected = [a for a in get_all_assets_config() if str(a.get("code", "")).strip() in active]
    if not selected:
        selected = BUILTIN_ASSETS[:]
    return selected


@st.cache_data
def lookup_asset_basic(_token: str, ts_code: str):
    ts_code = str(ts_code).strip().upper()
    if not ts_code:
        return None

    ts.set_token(_token)
    pro = ts.pro_api(_token)

    try:
        df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,list_date")
        hit = df[df["ts_code"] == ts_code]
        if not hit.empty:
            row = hit.iloc[0]
            return {"asset_type": "E", "name": str(row["name"]), "start_date": str(row["list_date"])}
    except Exception:
        pass

    try:
        df = pro.stock_basic(exchange="", list_status="D", fields="ts_code,name,list_date")
        hit = df[df["ts_code"] == ts_code]
        if not hit.empty:
            row = hit.iloc[0]
            return {"asset_type": "E", "name": str(row["name"]), "start_date": str(row["list_date"])}
    except Exception:
        pass

    for market in ("E", "O"):
        try:
            df = pro.fund_basic(market=market, status="L", fields="ts_code,name,list_date")
            hit = df[df["ts_code"] == ts_code]
            if not hit.empty:
                row = hit.iloc[0]
                return {"asset_type": "FD", "name": str(row["name"]), "start_date": str(row["list_date"])}
        except Exception:
            pass

    return None


def add_custom_asset(token: str, ts_code: str):
    ts_code = str(ts_code).strip().upper()
    if not ts_code:
        return None, "标的代码为空"

    existed = {str(a.get("code", "")).strip() for a in get_all_assets_config()}
    if ts_code in existed:
        return None, "标的已存在于资产库"

    basic = lookup_asset_basic(token, ts_code)
    if basic is None:
        end_date = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
        df_fd = ts.pro_bar(ts_code=ts_code, start_date="20200101", end_date=end_date, adj=None, asset="FD")
        df_e = ts.pro_bar(ts_code=ts_code, start_date="20200101", end_date=end_date, adj=None, asset="E")
        if df_fd is not None and not df_fd.empty:
            basic = {"asset_type": "FD", "name": ts_code, "start_date": "20200101"}
        elif df_e is not None and not df_e.empty:
            basic = {"asset_type": "E", "name": ts_code, "start_date": "20200101"}
        else:
            return None, "无法识别标的类型或无可用行情数据"

    name = str(basic["name"]).strip() or ts_code
    start_date = str(basic.get("start_date", "")).strip() or "20200101"
    asset_type = str(basic["asset_type"]).strip() or "FD"

    file_path = f"data/custom/{ts_code}_{name}_history.csv"

    custom_assets = load_custom_assets()
    item = {
        "code": ts_code,
        "name": name,
        "start_date": start_date,
        "asset_type": asset_type,
        "file_path": file_path,
        "source": "user",
    }
    custom_assets.append(item)
    save_custom_assets(custom_assets)
    return item, ""

# --- Helper: Data Update ---
def update_data(token, force=False, assets=None):
    """
    Update ETF data using Tushare.
    force: If True, re-download all data from scratch.
    """
    if not token or not str(token).strip():
        st.error("未配置 Tushare Token，无法更新数据。请在侧边栏输入 Token 或在 Streamlit Cloud Secrets 配置 TUSHARE_TOKEN。")
        return False

    try:
        ts.set_token(token)
        pro = ts.pro_api(token)
    except Exception as e:
        st.error(f"Tushare 初始化失败: {e}")
        return False

    target_assets = assets if isinstance(assets, list) and assets else get_all_assets_config()

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_area = st.empty()
    logs = []

    today = get_open_trade_end_date(pro)
    
    total_etfs = len(target_assets)
    
    for i, a in enumerate(target_assets):
        code = str(a.get("code", "")).strip()
        name = str(a.get("name", "")).strip()
        filename = str(a.get("file_path", "")).strip() or f"data/{code}_{name}_history.csv"
        start_date = str(a.get("start_date", "")).strip() or "19900101"
        asset_type = str(a.get("asset_type", "FD")).strip() or "FD"
        
        status_text.text(f"正在处理: {name} ({code})...")
        
        try:
            result = update_history_file(
                file_path=filename,
                ts_code=code,
                asset_type=asset_type,
                start_date=start_date,
                end_date=today,
                force=force,
                pro=pro,
            )
            if result["status"] == "no_data":
                logs.append(f"{name}: 未获取到可用数据。")
            elif result["status"] == "adj_refreshed":
                logs.append(
                    f"{name}: 无新增原始数据，已基于复权因子全量重建前复权列，最新日期 {result['new_max']}。"
                )
            else:
                logs.append(
                    f"{name}: 新增 {result['new_rows']} 条，已基于复权因子重建前复权列 {result['adj_rows']} 条，复权因子 {result['factor_rows']} 条，最新日期 {result['new_max']}。"
                )
        except Exception as e:
            logs.append(f"{name} 更新失败: {e}")
        
        progress_bar.progress((i + 1) / total_etfs)
        time.sleep(0.1) # Be nice to API
        
    status_text.text("数据更新完成！")
    with st.expander("查看更新日志"):
        st.write(logs)
    
    return True

def get_tushare_token():
    """
    Read Tushare token from Streamlit secrets or sidebar session override.
    """
    ts_token = ""

    try:
        if "TUSHARE_TOKEN" in st.secrets:
            ts_token = st.secrets["TUSHARE_TOKEN"]
    except:
        pass

    if not ts_token:
        ts_token = os.getenv("TUSHARE_TOKEN", "").strip()

    if 'tushare_token' in st.session_state and st.session_state['tushare_token']:
        ts_token = st.session_state['tushare_token']

    return ts_token

# --- 1. Data Loading ---

@st.cache_data
def load_history_data():
    history_data = {}
    for a in get_selected_assets_config():
        name = str(a.get("name", "")).strip()
        filename = str(a.get("file_path", "")).strip()
        if not name or not filename:
            continue
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                if 'trade_date' in df.columns:
                    try:
                        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                    except:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').set_index('trade_date')
                history_data[name] = df
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
    return history_data

@st.cache_data
def calculate_rolling_scores(series, window=20):
    """
    Calculate Quadratic Weighted Linear Regression Momentum Score.
    """
    scores = pd.Series(index=series.index, dtype=float)
    scores[:] = np.nan
    
    # Pre-compute weights
    x = np.arange(window)
    x_norm = np.linspace(0, 1, window)
    weights = 1 + x_norm ** 2
    
    # We need log prices
    log_prices = np.log(series)
    values = log_prices.values
    
    # Loop over the series
    for i in range(window, len(values) + 1):
        window_data = values[i-window : i]
        
        # Check for NaNs
        if np.isnan(window_data).any():
            continue
            
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

@st.cache_data
def precalculate_all_scores(history_data, window=20):
    all_scores = pd.DataFrame()
    for asset, df in history_data.items():
        # Prefer adjusted close
        if 'adj_close' in df.columns:
            series = df['adj_close']
        elif 'close' in df.columns:
            series = df['close']
        else:
            continue
            
        scores = calculate_rolling_scores(series, window=window)
        scores.name = asset
        all_scores = pd.merge(all_scores, scores, left_index=True, right_index=True, how='outer')
    return all_scores

@st.cache_data
def calculate_rsrs_score(df, N=18, M=600):
    """
    Calculate RSRS Z-Score.
    N: Regression Window
    M: Standardization Window
    """
    if 'adj_high' in df.columns and 'adj_low' in df.columns:
        highs = df['adj_high']
        lows = df['adj_low']
    elif 'high' in df.columns and 'low' in df.columns:
        highs = df['high']
        lows = df['low']
    else:
        return None
        
    values_high = highs.values
    values_low = lows.values
    
    beta_series = np.full(len(df), np.nan)
    
    # Calculate Betas (Rolling Regression)
    # We only need the last beta if M is large, but to calculate Z-Score we need M betas.
    # So we need to calculate at least M+N betas back.
    # For efficiency, if len(df) is huge, we can trim? No, cache handles it.
    
    for i in range(N, len(df) + 1):
        y = values_high[i-N:i]
        x = values_low[i-N:i]
        
        # Simple check for NaNs
        if np.isnan(x).any() or np.isnan(y).any():
            continue
            
        try:
            # Using numpy polyfit is faster than statsmodels for simple linear regression
            coeffs = np.polyfit(x, y, 1)
            beta = coeffs[0]
            beta_series[i-1] = beta
        except:
            pass
            
    # Calculate Z-Score
    betas = pd.Series(beta_series, index=df.index)
    mean_beta = betas.rolling(M).mean()
    std_beta = betas.rolling(M).std()
    z_score = (betas - mean_beta) / std_beta
    
    return z_score

@st.cache_data
def precalculate_all_rsrs(history_data):
    all_rsrs = pd.DataFrame()
    for asset, df in history_data.items():
        rsrs = calculate_rsrs_score(df)
        if rsrs is not None:
            rsrs.name = asset
            all_rsrs = pd.merge(all_rsrs, rsrs, left_index=True, right_index=True, how='outer')
    return all_rsrs

# --- Alpha Factors ---

@st.cache_data
def calculate_alpha55(history_data):
    """
    Alpha 55: Price-Volume Relation Factor
    Formula: -1 * correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)
    Note: Requires cross-sectional ranking across all assets.
    """
    # 1. Prepare DataFrames for Close, High, Low, Volume
    closes = pd.DataFrame()
    highs = pd.DataFrame()
    lows = pd.DataFrame()
    volumes = pd.DataFrame()
    
    for asset, df in history_data.items():
        # Prefer adjusted for price, but volume is usually raw (or adjusted volume)
        # Tushare 'vol' is volume.
        
        if 'adj_close' in df.columns: c = df['adj_close']
        else: c = df['close']
            
        if 'adj_high' in df.columns: h = df['adj_high']
        else: h = df['high']
            
        if 'adj_low' in df.columns: l = df['adj_low']
        else: l = df['low']
        
        if 'vol' in df.columns: v = df['vol']
        elif 'volume' in df.columns: v = df['volume']
        else: v = pd.Series(np.nan, index=df.index)
            
        c.name = asset
        h.name = asset
        l.name = asset
        v.name = asset
        
        closes = pd.merge(closes, c, left_index=True, right_index=True, how='outer')
        highs = pd.merge(highs, h, left_index=True, right_index=True, how='outer')
        lows = pd.merge(lows, l, left_index=True, right_index=True, how='outer')
        volumes = pd.merge(volumes, v, left_index=True, right_index=True, how='outer')
        
    # 2. Calculate K term
    # (close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))
    
    ll12 = lows.rolling(12).min()
    hh12 = highs.rolling(12).max()
    
    denom = hh12 - ll12
    # Avoid division by zero
    denom = denom.replace(0, np.nan)
    
    K = (closes - ll12) / denom
    
    # 3. Rank K (Cross-sectional)
    # axis=1 means rank across columns (assets) for each row (date)
    # pct=True to normalize to 0-1
    rank_K = K.rank(axis=1, pct=True)
    
    # 4. Rank Volume (Cross-sectional)
    rank_V = volumes.rank(axis=1, pct=True)
    
    # 5. Rolling Correlation (Time-series)
    # correlation(rank_K, rank_V, 6)
    # For each asset (column), calculate rolling corr of its rank series
    
    alpha55 = rank_K.rolling(6).corr(rank_V)
    
    # Multiply by -1
    alpha55 = alpha55 * -1
    
    return alpha55

# --- 2. Backtest Logic ---

def run_backtest(history_data, raw_scores_df, params):
    """
    params: {
        'start_date': datetime,
        'end_date': datetime,
        'cutoff_score': float,
        'buffer_score': float,
        'fee_rate': float,
        'initial_capital': float,
        'exclude_overheated_from_norm': bool,
    }
    """
    # Filter Timeline
    timeline = [d for d in raw_scores_df.index if params['start_date'] <= d <= params['end_date']]
    timeline = sorted(timeline)
    
    if not timeline:
        return None, None, None, None # Return 4 values now

    # State
    cash = params['initial_capital']
    holdings = {} # {asset: shares}
    target_assets = [] # Signal from yesterday (T+1)
    
    # Cache Prices & Returns for Speed (Use Adjusted if available for Backtest to avoid Split Crashes)
    # Note: Using Adjusted prices for backtest execution preserves % returns but "Price" in logs will be adjusted.
    price_open = {}
    price_close = {}
    price_high = {}
    price_low = {}
    
    for asset, df in history_data.items():
        if 'adj_open' in df.columns and 'adj_close' in df.columns:
            price_open[asset] = df['adj_open']
            price_close[asset] = df['adj_close']
            price_high[asset] = df['adj_high'] if 'adj_high' in df.columns else df['high']
            price_low[asset] = df['adj_low'] if 'adj_low' in df.columns else df['low']
        else:
            price_open[asset] = df['open']
            price_close[asset] = df['close']
            price_high[asset] = df['high']
            price_low[asset] = df['low']
    
    value_history = []
    trade_log = []
    
    last_signal_info = {} # To store the final T+1 signal
    
    cost_basis = {} # {asset: price_per_share}

    for date in timeline:
        # --- A. Execution (At Open) ---
        fee = params['fee_rate']
        current_assets = list(holdings.keys())
        target_list = [x for x in (target_assets or []) if x in history_data]

        nav_open = cash
        for a, sh in holdings.items():
            if date in price_open[a].index:
                nav_open += sh * price_open[a].loc[date]
            else:
                try:
                    nav_open += sh * price_close[a].asof(date)
                except Exception:
                    pass

        cum_ret_pct = (nav_open / params['initial_capital']) - 1

        min_trade_shares = 1e-8
        min_trade_amount = 1e-6

        def _sell(asset, shares_to_sell, price):
            nonlocal cash
            if shares_to_sell <= min_trade_shares or price <= 0:
                return
            gross_amount = shares_to_sell * price
            if gross_amount <= min_trade_amount:
                return
            proceeds = shares_to_sell * price * (1 - fee)
            cash += proceeds
            holdings[asset] = holdings.get(asset, 0) - shares_to_sell
            if holdings.get(asset, 0) <= min_trade_shares:
                holdings.pop(asset, None)

            pnl_amount = np.nan
            trade_return_pct = np.nan
            if asset in cost_basis:
                total_buy_cost = shares_to_sell * cost_basis[asset]
                pnl_amount = proceeds - total_buy_cost
                trade_return_pct = pnl_amount / total_buy_cost if total_buy_cost > 0 else np.nan
                if asset not in holdings:
                    cost_basis.pop(asset, None)

            trade_log.append({
                'date': date,
                'action': '卖出',
                'asset': asset,
                'price': price,
                'shares': shares_to_sell,
                'amount': proceeds,
                'fee': shares_to_sell * price * fee,
                'return_pct': cum_ret_pct,
                'trade_return': trade_return_pct,
                'pnl_amount': pnl_amount
            })

        def _buy(asset, shares_to_buy, price):
            nonlocal cash
            if shares_to_buy <= min_trade_shares or price <= 0:
                return
            max_shares = cash / (price * (1 + fee)) if price > 0 else 0
            shares_to_buy = min(shares_to_buy, max_shares)
            if shares_to_buy <= min_trade_shares:
                return

            cost = shares_to_buy * price * (1 + fee)
            if cost <= min_trade_amount:
                return
            cash -= cost

            prev_shares = holdings.get(asset, 0.0)
            holdings[asset] = prev_shares + shares_to_buy

            unit_cost = price * (1 + fee)
            if asset in cost_basis and prev_shares > 0:
                cost_basis[asset] = (cost_basis[asset] * prev_shares + unit_cost * shares_to_buy) / (prev_shares + shares_to_buy)
            else:
                cost_basis[asset] = unit_cost

            trade_log.append({
                'date': date,
                'action': '买入',
                'asset': asset,
                'price': price,
                'shares': shares_to_buy,
                'amount': cost,
                'fee': shares_to_buy * price * fee,
                'return_pct': cum_ret_pct,
                'trade_return': np.nan,
                'pnl_amount': np.nan
            })

        if not target_list:
            for a in current_assets:
                if date not in price_open[a].index:
                    continue
                _sell(a, holdings.get(a, 0), price_open[a].loc[date])
        else:
            target_n = len(target_list)
            desired_value = nav_open / target_n if target_n > 0 else 0

            for a in current_assets:
                if a in target_list:
                    continue
                if date not in price_open[a].index:
                    continue
                _sell(a, holdings.get(a, 0), price_open[a].loc[date])

            for a in list(holdings.keys()):
                if a not in target_list:
                    continue
                if date not in price_open[a].index:
                    continue
                p = price_open[a].loc[date]
                cur_val = holdings[a] * p
                if cur_val > desired_value:
                    desired_shares = desired_value / p if p > 0 else 0
                    _sell(a, max(0, holdings[a] - desired_shares), p)

            for a in target_list:
                if date not in price_open[a].index:
                    continue
                p = price_open[a].loc[date]
                cur_shares = holdings.get(a, 0.0)
                cur_val = cur_shares * p
                if cur_val < desired_value:
                    desired_shares = desired_value / p if p > 0 else 0
                    _buy(a, max(0, desired_shares - cur_shares), p)
            
        # --- B. Valuation (At Close) ---
        day_value = cash
        day_high = cash
        day_low = cash
        
        for asset, shares in holdings.items():
            # Close
            if date in price_close[asset].index:
                price = price_close[asset].loc[date]
            else:
                try:
                    price = price_close[asset].asof(date)
                except:
                    price = 0
            day_value += shares * price
            
            # High
            if date in price_high[asset].index:
                ph = price_high[asset].loc[date]
            else:
                try: ph = price_high[asset].asof(date)
                except: ph = 0
            day_high += shares * ph
            
            # Low
            if date in price_low[asset].index:
                pl = price_low[asset].loc[date]
            else:
                try: pl = price_low[asset].asof(date)
                except: pl = 0
            day_low += shares * pl
            
        holding_assets_str = "|".join(sorted(list(holdings.keys())))
        if not holding_assets_str:
            holding_label = '现金'
        else:
            holding_label = holding_assets_str if len(holdings) > 1 else list(holdings.keys())[0]

        value_history.append({
            'date': date, 
            'value': day_value, 
            'high': day_high,
            'low': day_low,
            'holding': holding_label,
            'holding_assets': holding_assets_str
        })
        
        # --- C. Signal Generation (At Close) ---
        pool_scores = pd.Series(dtype=float)
        if date not in raw_scores_df.index:
            next_target_assets = []
        else:
            today_scores = raw_scores_df.loc[date].dropna()
            
            if today_scores.empty:
                next_target_assets = []
            else:
                pool_scores = today_scores
                
                if pool_scores.empty:
                    next_target_assets = []
                else:
                    # 2. Filter Candidates (Score > 0 & <= Cutoff)
                    # Use Asset-Specific Cutoff
                    
                    def get_cutoff(asset_name):
                        name_to_code = params.get('name_to_code', {}) or {}
                        code = name_to_code.get(asset_name)
                        if code and code in params['user_cutoffs']:
                            return params['user_cutoffs'][code]
                        return float(params.get("fallback_cutoff", 300))
                        
                    # Vectorized cutoff check? Hard with dict lookup. Loop is easier for candidates.
                    # Or apply map to index
                    
                    current_cutoffs = pd.Series(pool_scores.index.map(get_cutoff), index=pool_scores.index)
                    
                    valid_candidates = pool_scores[
                        (pool_scores <= current_cutoffs) & (pool_scores > 0)
                    ]
                    
                    if valid_candidates.empty:
                        next_target_assets = []
                    else:
                        # 3. Normalize Valid Scores (Relative to Pool)
                        # Check normalization mode
                        exclude_overheated = params.get('exclude_overheated_from_norm', False)
                        
                        if exclude_overheated:
                            # Use only valid candidates (which are already filtered by cutoff) for normalization range
                            norm_basis = valid_candidates
                        else:
                            # Use entire pool (including overheated)
                            norm_basis = pool_scores
                            
                        vals = norm_basis.values
                        mn, mx = np.min(vals), np.max(vals)
                        
                        if mx == mn:
                            norm_scores = pd.Series(50, index=pool_scores.index)
                        else:
                            # Normalize all scores based on the chosen range
                            norm_scores = (pool_scores - mn) / (mx - mn) * 100
                            
                        hold_top_n = int(params.get("hold_top_n", 1) or 1)
                        hold_top_n = max(1, hold_top_n)
                        best_assets = valid_candidates.sort_values(ascending=False).index.tolist()[:hold_top_n]

                        current_set = set(holdings.keys())
                        valid_set = set(valid_candidates.index.tolist())

                        if not best_assets:
                            next_target_assets = []
                        elif not current_set:
                            next_target_assets = best_assets
                        elif any(a not in valid_set for a in current_set):
                            next_target_assets = best_assets
                        else:
                            if set(best_assets) == current_set:
                                next_target_assets = sorted(list(current_set), key=lambda x: best_assets.index(x) if x in best_assets else 10**9)
                            else:
                                curr_norms = norm_scores[list(current_set)]
                                min_curr_norm = float(curr_norms.min()) if not curr_norms.empty else -1e9
                                trigger = False
                                for a in best_assets:
                                    if a not in current_set and float(norm_scores.get(a, -1e9)) - min_curr_norm > params['buffer_score']:
                                        trigger = True
                                        break
                                next_target_assets = best_assets if trigger else list(current_set)
                            
        target_assets = next_target_assets
        
        # Capture last signal info
        if date == timeline[-1]:
            hold_top_n = int(params.get("hold_top_n", 1) or 1)
            if hold_top_n <= 1:
                next_holding = target_assets[0] if target_assets else '现金'
                last_signal_info = {
                    'date': date,
                    'next_holding': next_holding,
                    'score': pool_scores.get(next_holding, 0) if next_holding != '现金' else 0
                }
            else:
                next_holdings = target_assets[:]
                scores = {a: float(pool_scores.get(a, 0)) for a in next_holdings}
                last_signal_info = {
                    'date': date,
                    'next_holding': "|".join(next_holdings) if next_holdings else '现金',
                    'next_holdings': next_holdings,
                    'scores': scores
                }
        
    return pd.DataFrame(value_history).set_index('date'), pd.DataFrame(trade_log), timeline, last_signal_info

# --- 3. UI Layout ---

def plot_asset_trades(asset_name, df_ohlc, trades, start_date, end_date):
    """
    Generate Close Price chart with Buy/Sell markers.
    """
    # Filter OHLC by date range
    mask = (df_ohlc.index >= pd.Timestamp(start_date)) & (df_ohlc.index <= pd.Timestamp(end_date))
    chart_data = df_ohlc.loc[mask]
    
    if chart_data.empty:
        return None

    # Determine which columns to use (match backtest logic preference for Adjusted)
    if 'adj_close' in chart_data.columns:
        c = chart_data['adj_close']
        price_type = "(后复权)"
    else:
        c = chart_data['close']
        price_type = "(未复权)"

    # Line Chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=c,
        mode='lines',
        name=f'收盘价 {price_type}',
        line=dict(color='#1f77b4', width=2)
    ))

    # Buy Markers
    buy_trades = trades[trades['action'] == '买入']
    if not buy_trades.empty:
        fig.add_trace(go.Scatter(
            x=buy_trades['date'],
            y=buy_trades['price'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='red', line=dict(width=1, color='black')),
            name='买入点',
            hovertext=buy_trades['price'].apply(lambda x: f"买入价: {x:.3f}")
        ))

    # Sell Markers
    sell_trades = trades[trades['action'] == '卖出']
    if not sell_trades.empty:
        hover_texts = []
        for _, row in sell_trades.iterrows():
            ret_str = f"{row['trade_return']:.2%}" if pd.notnull(row['trade_return']) else "N/A"
            pnl_str = f"{row['pnl_amount']:.2f}" if pd.notnull(row['pnl_amount']) else "N/A"
            hover_texts.append(f"卖出价: {row['price']:.3f}<br>本次收益率: {ret_str}<br>本次盈亏额: {pnl_str}")
            
        fig.add_trace(go.Scatter(
            x=sell_trades['date'],
            y=sell_trades['price'], 
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='green', line=dict(width=1, color='black')),
            name='卖出点',
            hovertext=hover_texts
        ))

    fig.update_layout(
        title=f"{asset_name} 交易复盘 {price_type}",
        xaxis_title="日期",
        yaxis_title="价格",
        height=500,
        hovermode="closest"
    )
    return fig

def render_intro_page():
    st.title("📚 策略原理与交易者指南")
    
    st.markdown("""
    ### 1. 核心设计理念
    本策略是一个**基于动量的轮动策略**，旨在通过数学模型自动捕捉市场中趋势最强的资产，同时通过严格的风险控制机制避免由于市场过热或突发暴跌带来的损失。
    
    ### 2. 数据来源与处理
    - **数据源**: 策略使用 **Tushare Pro** 接口获取 ETF 的日线数据。
    - **复权处理**: 计算收益率时使用**后复权 (Adj Close)** 数据，以保证价格的连续性和收益计算的准确性。
    
    ### 3. 策略标的池详情
    本策略精选了 9 只具有代表性的 ETF，覆盖了不同的市场和资产类别，以实现低相关性的多元化配置。
    
    | 代码 | 名称 | 资产类别 | 典型特征 |
    | :--- | :--- | :--- | :--- |
    | **513100.SH** | 纳指ETF | 🇺🇸 美股科技 | 全球科技龙头，高成长高波动 |
    | **513520.SH** | 日经ETF | 🇯🇵 日本股市 | 亚洲发达市场，与A股相关性低 |
    | **513020.SH** | 港股科技ETF | 🇭🇰 港股科技 | 中国互联网巨头，估值弹性大 |
    | **510180.SH** | 180ETF | 🇨🇳 A股蓝筹 | 上海市场核心资产，金融地产占比高 |
    | **588120.SH** | 科创板ETF | 🇨🇳 A股硬科技 | 半导体、生物医药等硬核科技 |
    | **159915.SZ** | 创业板ETF | 🇨🇳 A股成长 | 新能源、医药等成长风格 |
    | **501018.SH** | 南方原油 | 🛢️ 商品原油 | 抗通胀，与股市相关性低 |
    | **518880.SH** | 黄金ETF | 🥇 商品黄金 | 避险资产，对抗货币贬值 |
    | **511090.SH** | 30年国债ETF | 🇨🇳 债券 | 防御性资产，股市下跌时的避风港 |
    
    ### 4. 核心因子：二次加权线性回归动量
    为了更精准地识别趋势，我们采用了**二次加权线性回归 (Quadratic Weighted Linear Regression)** 模型，而非简单的收益率排名。
    
    #### 计算公式
    $$
    Score = (e^{Slope \\times 252} - 1) \\times R^2 \\times 100
    $$
    
    其中：
    - **Slope (斜率)**: 通过对过去 20 天的对数价格进行加权线性回归得出，代表资产的**上涨速度**。
    - **R² (拟合优度)**: 代表价格走势的**平稳度**。$R^2$ 越接近 1，说明价格上涨越平稳，回撤越小。
    - **权重 ($w_t$)**: $w_t = 1 + (t/T)^2$，赋予最近的交易日更高的权重，使模型对趋势变化更敏感。
    
    > **核心逻辑**: 我们不仅追求“涨得快”（Slope），更追求“涨得稳”（$R^2$）。一个波动剧烈的大涨不如一个稳步向上的小涨得分高。
    
    ### 5. 关键参数解析
    
    #### (a) 动量窗口 (20天)
    - **设定**: 采用 20 个交易日（约一个月）作为动量计算窗口。
    - **原因**: 经测试，20天是捕捉中短期趋势的最佳平衡点。太短（如5-10天）容易被市场噪音干扰；太长（如60天）则对趋势反转反应迟钝。
    
    #### (b) 过热熔断阈值 (Score > 300)
    - **设定**: 当动量得分超过 300 分时，禁止开仓该标的，甚至强制卖出。
    - **深度分析**: 
        - 统计发现，得分与未来收益呈**倒 U 型曲线**关系。
        - 当得分适中（50-200）时，动量效应显著，未来大概率继续上涨。
        - 当得分极端高（>300）时，往往意味着资产价格呈指数级爆发（如年化收益率推算超过几百%），这种状态不可持续，极易发生均值回归或崩盘。
        - **结论**: 300分是“贪婪”与“危险”的分界线。
    
    #### (c) 换仓缓冲阈值 (Score Diff > 8)
    - **设定**: 只有当新标的的归一化得分比当前持仓标的高出 8 分以上时，才进行调仓。
    - **原因**: 避免“反复横跳”。如果两个标的得分相近，频繁切换只会徒增交易成本和滑点。8分的缓冲带确保了只有确定的“更强趋势”出现时才行动。
    
    ### 6. 风控机制：短期大跌剔除
    - **逻辑**: 如果某标的在最近 3 天内出现单日跌幅超过 3% 的情况，立即将其从候选池中剔除。
    - **目的**: “君子不立危墙之下”。在暴跌初期果断离场，规避可能发生的连续下跌风险。

    ### 7. 辅助分析工具：动量得分分布与预期收益分析
    我们提供了一个独立的分析工具，用于深入研究“动量得分”与“未来收益”之间的非线性关系（即倒 U 型曲线）。

    - **访问方式**: 请点击左侧侧边栏的 **"Momentum Analysis"** 页面。
    - **功能**: 
        - 可视化不同得分区间的未来收益分布。
        - 验证“过热熔断”阈值（300分）的合理性。

    ### 8. 参数统计与稳定性分析
    我们对策略参数进行了全量的网格搜索与统计分析，以验证策略的稳健性。

    - **访问方式**: 请点击左侧侧边栏的 **"Parameter Analysis"** 页面。
    - **核心结论**: 
        - 验证了参数平原的存在，排除了过拟合风险。
        - 确定了 **Window=25** 为策略还原的基石。
    """)

def get_strategy_params():
    """
    Render sidebar strategy controls and return params dict.
    """
    st.sidebar.header("⚙️ 策略参数设置")
    
    # Data Update Section
    st.sidebar.divider()
    st.sidebar.subheader("📥 数据更新")
    
    ts_token = get_tushare_token()
        
    # Allow manual override
    manual_token = st.sidebar.text_input("Tushare Token (可选)", value="", type="password", help="如果自动获取失败，请在此手动输入 Token")
    if manual_token:
        ts_token = manual_token
        st.session_state['tushare_token'] = manual_token
    
    if st.sidebar.button("🔄 更新数据"):
        # Add a force update checkbox logic or just detect if user wants to force?
        # Since button is stateless, we can add a checkbox before it.
        pass # Logic moved below

    force_update = st.sidebar.checkbox("强制全量更新 (修复数据分裂/缺失)", value=False, help="勾选后将删除现有数据并重新下载所有历史数据。如果发现价格图表有异常缺口，请使用此功能。")
    
    if st.sidebar.button("🔄 执行更新"):
        with st.spinner("正在连接 Tushare 更新数据..."):
            if not ts_token:
                st.error("未配置 Tushare Token，无法更新数据。")
                st.stop()
            if update_data(ts_token, force=force_update):
                st.success("数据更新成功！请刷新页面或重新回测。")
                # Clear cache to force reload
                load_history_data.clear()
                precalculate_all_scores.clear()
                st.rerun()
    
    st.sidebar.divider()

    st.sidebar.subheader("🧺 标的池管理")

    all_assets = get_all_assets_config()
    active_codes = load_active_pool_codes()
    code_to_asset = {str(a.get("code", "")).strip(): a for a in all_assets if str(a.get("code", "")).strip()}
    ordered_codes = [c for c in active_codes if c in code_to_asset] + [c for c in code_to_asset.keys() if c not in active_codes]

    label_to_code = {}
    default_labels = []
    for code in ordered_codes:
        a = code_to_asset.get(code)
        if not a:
            continue
        name = str(a.get("name", "")).strip()
        label = f"{name} ({code})"
        label_to_code[label] = code
        if code in active_codes:
            default_labels.append(label)

    selected_labels = st.sidebar.multiselect("参与轮动的标的", options=list(label_to_code.keys()), default=default_labels)
    selected_codes = [label_to_code[x] for x in selected_labels if x in label_to_code]

    col_pool_a, col_pool_b = st.sidebar.columns(2)
    with col_pool_a:
        if st.button("应用标的池"):
            if not selected_codes:
                st.error("轮动标的池不能为空")
                st.stop()
            save_active_pool_codes(selected_codes)
            load_history_data.clear()
            precalculate_all_scores.clear()
            st.rerun()
    with col_pool_b:
        if st.button("重置为默认"):
            save_active_pool_codes([x["code"] for x in BUILTIN_ASSETS])
            load_history_data.clear()
            precalculate_all_scores.clear()
            st.rerun()

    new_code = st.sidebar.text_input("新增标的代码", value="", help="示例：600519.SH / 513100.SH")
    if st.sidebar.button("➕ 添加标的并下载数据"):
        if not ts_token:
            st.error("未配置 Tushare Token，无法新增标的。")
            st.stop()
        item, err = add_custom_asset(ts_token, new_code)
        if err:
            st.error(err)
            st.stop()
        with st.spinner("正在下载新增标的历史数据..."):
            if not update_data(ts_token, force=True, assets=[item]):
                st.error("新增标的下载失败")
                st.stop()
        codes = load_active_pool_codes()
        if item["code"] not in codes:
            codes.append(item["code"])
        save_active_pool_codes(codes)
        load_history_data.clear()
        precalculate_all_scores.clear()
        st.success("新增标的已加入资产库并纳入轮动池")
        st.rerun()
    
    # Date Range (Only needed for backtest range, but useful to keep here or default)
    # For simplicity, we keep them here but Latest Holding might ignore start/end
    try:
        _hd = load_history_data()
        _mins = [df.index.min() for df in _hd.values() if df is not None and not df.empty]
        _maxs = [df.index.max() for df in _hd.values() if df is not None and not df.empty]
        min_date = min(_mins) if _mins else pd.Timestamp('2017-08-01')
        max_date = max(_maxs) if _maxs else pd.Timestamp.now()
    except Exception:
        min_date = pd.Timestamp('2017-08-01')
        max_date = pd.Timestamp.now()
    
    st.sidebar.subheader("📅 回测时间设置")
    time_range_option = st.sidebar.radio(
        "选择回测时长",
        ("最近1年", "最近2年", "最近3年", "最近4年", "最近5年", "自定义时间范围"),
        index=5
    )

    if time_range_option == "自定义时间范围":
        start_date = st.sidebar.date_input("开始日期", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("结束日期", max_date, min_value=min_date, max_value=max_date)
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
    else:
        # Calculate start date based on selection
        end_date = pd.Timestamp.now()
        years_map = {
            "最近1年": 1,
            "最近2年": 2,
            "最近3年": 3,
            "最近4年": 4,
            "最近5年": 5
        }
        years_back = years_map[time_range_option]
        start_date = end_date - pd.DateOffset(years=years_back)
        
        # If calculated start date is before min_date, clamp it
        if start_date < min_date:
            start_date = min_date
            
        # Display the calculated range for info
        st.sidebar.caption(f"范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    
    # Strategy Params
    st.sidebar.subheader("核心参数")
    
    window = st.sidebar.number_input("动量窗口 (天)", min_value=5, max_value=60, value=25, step=1)
    hold_top_n = st.sidebar.number_input("持有前 N 名 (等权)", min_value=1, max_value=max(1, len(get_selected_assets_config())), value=1, step=1)
    
    # Custom Cutoff Logic
    st.sidebar.markdown("**过热熔断阈值设置**")
    
    cutoff_mode = st.sidebar.radio(
        "阈值模式", 
        ["分标的独立设置", "全局统一设置"],
        index=1,
        help="选择'全局统一'将对所有标的使用相同阈值；选择'分标的独立'可为不同波动率的资产设置不同阈值。"
    )
    
    selected_assets = get_selected_assets_config()
    name_to_code = {str(a.get("name", "")).strip(): str(a.get("code", "")).strip() for a in selected_assets if str(a.get("name", "")).strip() and str(a.get("code", "")).strip()}
    code_to_name = {v: k for k, v in name_to_code.items()}
    user_cutoffs = {}
    fallback_cutoff = 300.0

    if cutoff_mode == "全局统一设置":
        global_cutoff = st.sidebar.number_input("全局熔断阈值", min_value=50, max_value=2000, value=500, step=50)
        for code in code_to_name.keys():
            user_cutoffs[code] = global_cutoff
        fallback_cutoff = float(global_cutoff)
    else:
        # Default initial values based on statistical analysis
        default_cutoffs = {
            '588120.SH': 500,  # 科创板
            '513100.SH': 600,  # 纳指100
            '513520.SH': 300,  # 日经ETF
            '159915.SZ': 1000, # 创业板
            '513020.SH': 400,  # 港股科技
            '510180.SH': 600,  # 上证180
            '518880.SH': 500,  # 黄金ETF
            '501018.SH': 1000, # 南方原油
            '511090.SH': 300,  # 30年国债
        }
        
        with st.sidebar.expander("自定义各标的阈值", expanded=True):
            for code, name in code_to_name.items():
                default_val = int(default_cutoffs.get(code, 500))
                val = st.number_input(f"{name} ({code})", min_value=50, max_value=2000, value=default_val, step=50)
                user_cutoffs[code] = val
            
    buffer_score = st.sidebar.number_input("换仓缓冲阈值 (分差)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    
    exclude_overheated_from_norm = st.sidebar.checkbox(
        "归一化时剔除过热标的", 
        value=True,
        help="勾选后，在计算归一化分数时，将先剔除超过熔断阈值的标的，再以剩余标的的最高分作为100分基准。这会放大剩余标的之间的分差，可能增加换仓频率。"
    )
    
    # Execution Params
    st.sidebar.subheader("交易参数")
    fee_rate = st.sidebar.number_input("交易费率 (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01) / 100
    initial_capital = st.sidebar.number_input("初始资金", min_value=10000, value=100000, step=10000)
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'window': window,
        'hold_top_n': int(hold_top_n),
        'user_cutoffs': user_cutoffs,
        'name_to_code': name_to_code,
        'fallback_cutoff': fallback_cutoff,
        'buffer_score': buffer_score,
        'exclude_overheated_from_norm': exclude_overheated_from_norm,
        'fee_rate': fee_rate,
        'initial_capital': initial_capital
    }

def render_latest_holding_page():
    st.title("🔔 最新持仓信号")
    
    # Hidden Link Button
    st.markdown("""
    <style>
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns([0.8, 0.2])
    with col_b:
        # Use a generic label and open in new tab via JS to "hide" URL in status bar partially
        # But Streamlit link_button shows URL. 
        # To truly hide, we can use a small hack or just a generic link text.
        # "External Resource"
        st.link_button("🌐 外部数据源", "https://168.nbjiadao.com/")
    
    # Sidebar
    params = get_strategy_params()
    
    st.info("点击下方按钮，系统将获取最新数据，并根据当前策略参数计算下一个交易日的建议持仓。")
    
    if st.button("🔍 检查并获取最新信号", type="primary"):
        # 1. Update Data
        ts_token = get_tushare_token()
        if not ts_token:
            st.error("未配置 Tushare Token，无法更新数据与计算最新信号。请在侧边栏输入 Token 或在 Streamlit Cloud Secrets 配置 TUSHARE_TOKEN。")
            return
        
        with st.spinner("正在同步最新市场数据..."):
            # Default to incremental update for "Latest Holding" check
            if not update_data(ts_token, force=False):
                st.error("数据更新失败，无法获取最新信号。")
                return
            load_history_data.clear()
            precalculate_all_scores.clear()
            
        # 2. Load & Calc
        with st.spinner("正在计算策略信号..."):
            history_data = load_history_data()
            scores_df = precalculate_all_scores(history_data, window=params['window'])
            rsrs_df = precalculate_all_rsrs(history_data)
            
            # 3. Run Backtest to determine state
            # We run from a reasonable past date to ensure state is correct
            # Start from 2024-01-01 or params['start_date']? 
            # Use params['start_date'] to be consistent with backtest settings
            backtest_params = params.copy()
            backtest_params['end_date'] = pd.Timestamp.now() # Ensure we go up to today
            
            df_res, df_trades, timeline, last_signal = run_backtest(history_data, scores_df, backtest_params)
            
            if not timeline:
                st.error("无法计算信号，请检查数据或日期范围。")
                return

            # 4. Display Result
            last_dt = timeline[-1]
            latest_date = last_dt.strftime('%Y-%m-%d')
            next_holding = last_signal.get('next_holding', '现金')
            next_score = last_signal.get('score', 0)
            next_holdings = last_signal.get('next_holdings', [])
            next_scores = last_signal.get('scores', {})
            
            st.divider()
            st.markdown(f"### 📅 数据日期: {latest_date}")
            
            # Big Display
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 建议持仓 (T+1)")
                if next_holding == '现金':
                    st.warning(f"### 💵 {next_holding}")
                else:
                    st.success(f"### 🚀 {next_holding}")
            
            with col2:
                if next_holdings:
                    st.markdown("#### 动量得分")
                    st.write(" | ".join([f"{a}: {float(next_scores.get(a, 0)):.1f}" for a in next_holdings]))
                else:
                    st.markdown("#### 动量得分")
                    st.metric("Score", f"{float(next_score):.1f}")
                
            st.divider()
            
            # Show details of candidates
            st.markdown("#### 📊 当日标的得分详情")
            
            if last_dt in scores_df.index:
                today_scores = scores_df.loc[last_dt].dropna().sort_values(ascending=False)
                
                # Format for display
                details = []
                for asset, score in today_scores.items():
                    name_to_code_disp = params.get("name_to_code", {}) or {}
                    asset_code = name_to_code_disp.get(asset)
                    cutoff_val = float(params.get("fallback_cutoff", 300))
                    if asset_code and 'user_cutoffs' in params and asset_code in params['user_cutoffs']:
                        cutoff_val = float(params['user_cutoffs'][asset_code])
                        
                    status = "✅ 候选"
                    if score > cutoff_val:
                        status = f"🚫 过热 (>{cutoff_val})"
                    elif score <= 0:
                        status = "📉 负动量"
                        
                    # Get RSRS
                    rsrs_val = np.nan
                    if last_dt in rsrs_df.index and asset in rsrs_df.columns:
                        rsrs_val = rsrs_df.loc[last_dt, asset]
                        
                    rsrs_str = f"{rsrs_val:.2f}" if not np.isnan(rsrs_val) else "-"
                    
                    # RSRS Status
                    rsrs_status = ""
                    if not np.isnan(rsrs_val):
                        if rsrs_val > 0.7:
                            rsrs_status = "🔥 强势"
                        elif rsrs_val < -0.7:
                            rsrs_status = "🧊 弱势"
                        else:
                            rsrs_status = "↔️ 震荡"
                        
                    details.append({
                        '标的': asset,
                        '动量得分': f"{score:.1f}",
                        'RSRS指标': f"{rsrs_str} {rsrs_status}",
                        '熔断阈值': cutoff_val,
                        '状态': status
                    })
                
                st.dataframe(pd.DataFrame(details))

def calculate_thermometer(df):
    """
    Calculate Strategy Thermometer Indicator.
    Input df must have 'high', 'low', 'value' (as close) columns.
    """
    # 1. 计算价格源 hlcc4 (Using 'value' as close)
    # Ensure columns exist, 'value' is close
    c = df['value']
    h = df['high']
    l = df['low']
    
    src = (h + l + c * 2) / 4 
    
    # 2. 计算 RSI (采用 RMA 平滑) 
    def rma(series, period): 
        return series.ewm(alpha=1/period, adjust=False).mean() 
    
    delta = src.diff() 
    up = rma(delta.clip(lower=0), 14) 
    down = rma(-delta.clip(upper=0), 14) 
    rsi = 100 - (100 / (1 + up / down)) 
    
    # 3. 计算 TSI (价格与时间的相关系数) 
    # Use integer index for correlation
    # We can create a temporary series
    tsi_window = 14
    
    # Rolling correlation requires Series
    # We correlate src with a rolling window of indices?
    # No, rolling correlation between two series.
    # We need a series that represents time.
    # If df has datetime index, we can't correlate directly with that easily in rolling.
    # Create a 0..N series
    time_idx = pd.Series(np.arange(len(df)), index=df.index)
    
    tsi = src.rolling(window=tsi_window).corr(time_idx)
    tsi_norm = (tsi + 1) / 2 * 100 
    
    # 4. 计算 BB%B (布林带百分比) 
    sma_bb = src.rolling(window=20).mean() 
    std_bb = src.rolling(window=20).std() 
    bb_percent = (src - (sma_bb - 2 * std_bb)) / (4 * std_bb) * 100 
    bb_percent = bb_percent.clip(0, 100) 

    # 5. 最终加权合成 (线性) 
    thermometer = (rsi * 0.45) + (tsi_norm * 0.26) + (bb_percent * 0.29) 
    
    # 6. 3日SMA平滑 
    plot_line = thermometer.rolling(window=3).mean() 
    
    return thermometer, plot_line

def render_backtest_page():
    # Sidebar Controls
    params = get_strategy_params()
    
    # Data Loading
    history_data = load_history_data()
    
    if st.sidebar.button("🚀 开始回测", type="primary"):
        with st.spinner("正在计算动量得分..."):
            # Pre-calculate scores based on window
            scores_df = precalculate_all_scores(history_data, window=params['window'])
            
        with st.spinner("正在执行回测..."):
            df_res, df_trades, timeline, last_signal = run_backtest(history_data, scores_df, params)
            
            # Store results in session state
            st.session_state['bt_results'] = {
                'df_res': df_res,
                'df_trades': df_trades,
                'timeline': timeline,
                'last_signal': last_signal,
                'scores_df': scores_df,
                'params': params
            }

    # Check if results exist
    if 'bt_results' in st.session_state:
        res = st.session_state['bt_results']
        df_res = res['df_res']
        df_trades = res['df_trades']
        timeline = res['timeline']
        last_signal = res['last_signal']
        scores_df = res['scores_df']
        run_params = res['params'] # Use the params that were used for this run

        if df_res is None or df_res.empty:
            st.error("该时间段内无数据或回测失败。")
        else:
            # Calculate daily cumulative return
            df_res['cum_return'] = df_res['value'] / run_params['initial_capital'] - 1

            # --- Metrics Calculation ---
            total_ret = df_res['value'].iloc[-1] / df_res['value'].iloc[0] - 1
            days = (df_res.index[-1] - df_res.index[0]).days
            if days > 0:
                ann_ret = (1 + total_ret) ** (365 / days) - 1
            else:
                ann_ret = 0
            
            # Volatility
            daily_ret = df_res['value'].pct_change().dropna()
            vol = daily_ret.std() * np.sqrt(252)
            
            # Sharpe
            risk_free = 0.02
            sharpe = (ann_ret - risk_free) / vol if vol != 0 else 0
            
            # Max Drawdown
            cum_max = df_res['value'].cummax()
            drawdown = (df_res['value'] - cum_max) / cum_max
            max_dd = drawdown.min()
            current_dd = drawdown.iloc[-1]
            
            # --- Display Metrics ---
            st.markdown("### 📊 回测绩效概览")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("总收益率", f"{total_ret:.2%}", delta_color="normal")
            col2.metric("年化收益率", f"{ann_ret:.2%}", delta_color="normal")
            col3.metric("夏普比率", f"{sharpe:.2f}")
            col4.metric("最大回撤", f"{max_dd:.2%}", delta_color="inverse")
            col5.metric("年化波动率", f"{vol:.2%}", delta_color="inverse")
            
            st.divider()

            # --- NEW: Asset Contribution Analysis ---
            st.markdown("### 🧬 各标的贡献分析")
            
            # 1. Holding Days
            # Group by 'holding' column in df_res
            if int(run_params.get("hold_top_n", 1) or 1) <= 1:
                holding_days = df_res.groupby('holding').size()
            else:
                holding_days_dict = {}
                if 'holding_assets' in df_res.columns:
                    for s in df_res['holding_assets'].fillna("").astype(str).tolist():
                        for a in [x for x in s.split("|") if x]:
                            holding_days_dict[a] = holding_days_dict.get(a, 0) + 1
                holding_days = pd.Series(holding_days_dict)
            
            # 2. PnL & Max Drawdown per Asset
            # PnL from trades
            asset_pnl = df_trades[df_trades['action'] == '卖出'].groupby('asset')['pnl_amount'].sum()
            
            # Max Drawdown Calculation per Asset (Strategy MDD during holding period)
            asset_mdd = {}
            
            # Identify continuous segments for each asset
            # Create a group id that changes when holding changes
            if int(run_params.get("hold_top_n", 1) or 1) <= 1:
                df_res['group'] = (df_res['holding'] != df_res['holding'].shift()).cumsum()
                
                for group_id, group_df in df_res.groupby('group'):
                    asset_name = group_df['holding'].iloc[0]
                    if asset_name == '现金':
                        continue
                        
                    vals = group_df['value']
                    cum_max_segment = vals.cummax()
                    dd_segment = (vals - cum_max_segment) / cum_max_segment
                    min_dd_segment = dd_segment.min()
                    
                    if asset_name not in asset_mdd:
                        asset_mdd[asset_name] = min_dd_segment
                    else:
                        asset_mdd[asset_name] = min(asset_mdd[asset_name], min_dd_segment)
            
            # Combine into DataFrame
            all_assets_involved = set(holding_days.index) | set(asset_pnl.index)
            # Remove Cash if present
            if '现金' in all_assets_involved:
                all_assets_involved.remove('现金')
                
            contribution_data = []
            for asset in all_assets_involved:
                days = holding_days.get(asset, 0)
                pnl = asset_pnl.get(asset, 0.0)
                mdd = asset_mdd.get(asset, 0.0)
                
                # Contribution to Total Return
                # Use initial capital as base
                contrib_pct = pnl / run_params['initial_capital']
                
                contribution_data.append({
                    '标的': asset,
                    '总持仓天数 (交易日)': days,
                    '持仓占比': days / len(df_res) if len(df_res) > 0 else 0,
                    '贡献收益率': contrib_pct,
                    '期间最大回撤': mdd
                })
                
            if contribution_data:
                df_contrib = pd.DataFrame(contribution_data).sort_values('贡献收益率', ascending=False)
                
                # Formatting
                st.dataframe(
                    df_contrib.style.format({
                        '持仓占比': '{:.1%}',
                        '贡献收益率': '{:.2%}',
                        '期间最大回撤': '{:.2%}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("暂无持仓数据。")
            
            st.divider()
            
            # --- NEW SECTION: Current Status Info ---
            # 1. Data Updated To
            last_dt = timeline[-1]
            latest_data_date = last_dt.strftime('%Y-%m-%d')
            
            # 2. T+1 Holding
            next_holding = last_signal.get('next_holding', '未知')
            next_score = last_signal.get('score', 0)
            next_holdings = last_signal.get('next_holdings', [])
            next_scores = last_signal.get('scores', {})
            
            name_to_code_all = {str(a.get("name", "")).strip(): str(a.get("code", "")).strip() for a in get_all_assets_config() if str(a.get("name", "")).strip() and str(a.get("code", "")).strip()}
            if next_holdings:
                holding_parts = []
                score_parts = []
                for a in next_holdings:
                    code = name_to_code_all.get(a, "")
                    holding_parts.append(f"{a} ({code})" if code else a)
                    score_parts.append(f"{a}: {float(next_scores.get(a, 0)):.1f}")
                holding_display = " | ".join(holding_parts)
                score_display = " | ".join(score_parts)
            else:
                holding_code = name_to_code_all.get(next_holding, '')
                holding_display = f"{next_holding} ({holding_code})" if holding_code else next_holding
                score_display = f"{float(next_score):.1f}"
            
            # 3. Current Drawdown
            # Calculated above as current_dd
            
            # 4. Other Scores
            # We need scores for the last date in timeline
            other_scores_display = ""
            if last_dt in scores_df.index:
                today_scores = scores_df.loc[last_dt].dropna().sort_values(ascending=False)
                # Filter out the winner to avoid duplication if desired, or show all
                # Let's show top 5 others
                score_strs = []
                holding_set = set(next_holdings) if next_holdings else {next_holding}
                for asset, score in today_scores.items():
                    if asset not in holding_set:
                        score_strs.append(f"{asset}: {score:.1f}")
                
                if score_strs:
                    other_scores_display = " | ".join(score_strs)
            
            st.info(f"""
            **📅 策略状态面板**  
            **数据更新至**: {latest_data_date}  
            **T+1 建议持仓**: **{holding_display}** (动量分: {score_display})  
            **当前回撤**: {current_dd:.2%}
            
            **其他标的得分**: {other_scores_display}
            """)
            
            st.divider()
            
            # --- Charts ---
            st.markdown("### 📈 收益与回撤曲线")
            
            # Plotly Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, 
                                subplot_titles=("策略净值曲线", "回撤曲线"),
                                row_heights=[0.7, 0.3])
            
            # Equity Curve
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['value'], 
                                     mode='lines', name='策略净值',
                                     line=dict(color='#00CC96', width=2)), row=1, col=1)
            
            # Drawdown Curve
            fig.add_trace(go.Scatter(x=df_res.index, y=drawdown, 
                                     mode='lines', name='回撤',
                                     fill='tozeroy',
                                     line=dict(color='#EF553B', width=1)), row=2, col=1)
            
            fig.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

            # --- Thermometer Chart ---
            st.markdown("### 🌡️ 策略温度计指标")
            
            # Calculate
            thermometer, plot_line = calculate_thermometer(df_res)
            
            fig_therm = go.Figure()
            
            fig_therm.add_trace(go.Scatter(
                x=thermometer.index, 
                y=thermometer,
                mode='lines',
                name='温度计 (Thermometer)',
                line=dict(color='#FFD700', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 215, 0, 0.1)'
            ))
            
            fig_therm.add_trace(go.Scatter(
                x=plot_line.index,
                y=plot_line,
                mode='lines',
                name='平滑线 (Signal)',
                line=dict(color='#FF4500', width=2)
            ))
            
            # Add horizontal lines
            fig_therm.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="超卖 (20)")
            fig_therm.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="超买 (80)")
            
            fig_therm.update_layout(
                height=300, 
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(range=[0, 100], title="温度"),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_therm, use_container_width=True)
            
            st.divider()

            # --- NEW: Periodic Return Analysis ---
            st.markdown("### 🗓️ 周期收益分析")
            
            # 1. Prepare Data
            # df_res has 'value' and 'cum_return'
            # We need to calculate periodic returns
            
            # Resample
            df_daily = df_res[['value']].copy()
            df_daily['return'] = df_daily['value'].pct_change()
            
            # Heatmap Data Construction
            # Year on Y-axis, Month on X-axis
            
            df_daily['year'] = df_daily.index.year
            df_daily['month'] = df_daily.index.month
            df_daily['quarter'] = df_daily.index.quarter
            
            # Calculate monthly returns by compounding daily returns
            monthly_rets = df_daily.groupby(['year', 'month'])['return'].apply(lambda x: (1 + x).prod() - 1)
            monthly_rets_df = monthly_rets.unstack(level='month') * 100 # In percent
            
            # Yearly returns
            yearly_rets = df_daily.groupby(['year'])['return'].apply(lambda x: (1 + x).prod() - 1) * 100
            
            # Quarterly returns
            quarterly_rets = df_daily.groupby(['year', 'quarter'])['return'].apply(lambda x: (1 + x).prod() - 1) * 100
            
            # UI Control
            period_view = st.radio("显示模式", ["月度热力图", "年度收益柱状图", "季度收益柱状图"], horizontal=True)
            
            if period_view == "月度热力图":
                # Heatmap
                # x: Month, y: Year, z: Return
                
                # Fill missing months with 0 or NaN
                # Reindex columns 1-12
                for m in range(1, 13):
                    if m not in monthly_rets_df.columns:
                        monthly_rets_df[m] = np.nan
                monthly_rets_df = monthly_rets_df[sorted(monthly_rets_df.columns)]
                
                # Add Year Total column?
                monthly_rets_df['Year Total'] = yearly_rets
                
                # Plotly Heatmap
                # We transpose for Y=Year, X=Month
                # But heatmap expects z as 2D array
                
                # Better to use text for values
                z_vals = monthly_rets_df.values
                x_labels = [f"{m}月" for m in range(1, 13)] + ['年度合计']
                y_labels = monthly_rets_df.index.astype(str)
                
                # Custom Color Scale: Green (Negative) -> White (Zero) -> Red (Positive)
                # Using specific colors for better visibility
                custom_colorscale = [
                    [0.0, '#008000'],   # Green for Loss
                    [0.5, '#ffffff'],   # White for Zero
                    [1.0, '#ff0000']    # Red for Profit
                ]
                
                # Determine symmetric range for color balance
                # Filter nans for calculation
                valid_vals = z_vals[~np.isnan(z_vals)]
                if len(valid_vals) > 0:
                    max_abs = np.max(np.abs(valid_vals))
                    # Ensure a minimum range to avoid solid colors for small returns
                    if max_abs < 1: max_abs = 1
                else:
                    max_abs = 10
                
                # Create annotations
                annotations = []
                for i, row in enumerate(z_vals):
                    for j, val in enumerate(row):
                        if pd.notnull(val):
                            # Contrast text color
                            # If background is dark (high absolute value), use white text
                            text_color = "white" if abs(val) > (max_abs * 0.5) else "black"
                            
                            annotations.append(dict(
                                x=x_labels[j], y=y_labels[i],
                                text=f"{val:.1f}%",
                                xref="x", yref="y",
                                showarrow=False,
                                font=dict(color=text_color, size=14)
                            ))
                
                fig_heat = go.Figure(data=go.Heatmap(
                    z=z_vals,
                    x=x_labels,
                    y=y_labels,
                    colorscale=custom_colorscale,
                    zmid=0,
                    zmin=-max_abs,
                    zmax=max_abs,
                    hoverongaps=False,
                    xgap=1, # Add gap between cells
                    ygap=1
                ))
                
                fig_heat.update_layout(
                    title="月度收益率热力图 (%)",
                    height=400 + len(y_labels) * 40, # Increase height per row
                    annotations=annotations,
                    xaxis_side="top",
                    margin=dict(l=0, r=0, t=50, b=0) # Adjust margins
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                
            elif period_view == "年度收益柱状图":
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=yearly_rets.index,
                    y=yearly_rets.values,
                    marker_color=['#EF553B' if x > 0 else '#00CC96' for x in yearly_rets.values],
                    text=[f"{x:.1f}%" for x in yearly_rets.values],
                    textposition='auto'
                ))
                fig_bar.update_layout(
                    title="年度收益率 (%)",
                    xaxis_title="年份",
                    yaxis_title="收益率 (%)",
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
            elif period_view == "季度收益柱状图":
                # Format index for display: "2023-Q1"
                q_labels = [f"{y}-Q{q}" for y, q in quarterly_rets.index]
                
                fig_q = go.Figure()
                fig_q.add_trace(go.Bar(
                    x=q_labels,
                    y=quarterly_rets.values,
                    marker_color=['#EF553B' if x > 0 else '#00CC96' for x in quarterly_rets.values],
                    text=[f"{x:.1f}%" for x in quarterly_rets.values],
                    textposition='auto'
                ))
                fig_q.update_layout(
                    title="季度收益率 (%)",
                    xaxis_title="季度",
                    yaxis_title="收益率 (%)",
                    showlegend=False
                )
                st.plotly_chart(fig_q, use_container_width=True)

            # --- Trade Log & Holdings ---
            st.markdown("### 📝 交易记录与持仓明细")
            
            tab1, tab2 = st.tabs(["调仓记录", "每日持仓"])
            
            with tab1:
                if df_trades is not None and not df_trades.empty:
                    # Sort descending
                    df_trades_sorted = df_trades.sort_values('date', ascending=False)
                    st.dataframe(df_trades_sorted.style.format({
                        'price': '{:.3f}', 
                        'shares': '{:.0f}', 
                        'amount': '{:.2f}',
                        'fee': '{:.2f}',
                        'return_pct': '{:.2%}',
                        'trade_return': '{:.2%}',
                        'pnl_amount': '{:,.2f}'
                    }, na_rep="-"), use_container_width=True)
                else:
                    st.info("该期间无交易记录。")
                    
            with tab2:
                # Sort descending
                df_res_sorted = df_res.sort_index(ascending=False)
                st.dataframe(df_res_sorted.style.format({
                    'value': '{:.2f}',
                    'cum_return': '{:.2%}'
                }), use_container_width=True)
            
            st.divider()

            # --- NEW: Asset Trade Visualization ---
            st.markdown("### 📈 标的交易详情可视化")
            st.info("点击下方展开查看各标的的收盘价曲线及买卖点标记。鼠标悬停在卖出点（绿色倒三角）可查看该笔交易的收益率。")

            # Get list of assets traded
            traded_assets = df_trades['asset'].unique() if df_trades is not None else []
            
            if len(traded_assets) > 0:
                for asset in traded_assets:
                    with st.expander(f"查看 {asset} 交易记录"):
                        # Get OHLC
                        if asset in history_data:
                            asset_ohlc = history_data[asset]
                            asset_trades = df_trades[df_trades['asset'] == asset]
                            
                            # Plot
                            fig_asset = plot_asset_trades(
                                asset, 
                                asset_ohlc, 
                                asset_trades, 
                                df_res.index[0], # Start Date of backtest
                                df_res.index[-1] # End Date of backtest
                            )
                            
                            if fig_asset:
                                st.plotly_chart(fig_asset, use_container_width=True)
                            else:
                                st.warning(f"无法获取 {asset} 的价格数据")
            else:
                st.write("无交易标的。")
                
    else:
        st.info("👈 请在左侧调整参数并点击“开始回测”")

# --- Main App Logic ---
st.sidebar.title("导航")
page = st.sidebar.radio("选择页面", ["回测系统", "最新持仓标的", "策略介绍"], index=0)

if page == "回测系统":
    render_backtest_page()
elif page == "最新持仓标的":
    render_latest_holding_page()
elif page == "策略介绍":
    render_intro_page()
