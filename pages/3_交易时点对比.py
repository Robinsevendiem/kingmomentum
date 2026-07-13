import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="交易时点对比", page_icon="⏱️", layout="wide")

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
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def load_custom_assets():
    data = _read_json(CUSTOM_ASSETS_PATH, [])
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict) and str(x.get("code", "")).strip() and str(x.get("name", "")).strip()]


def load_active_pool_codes():
    data = _read_json(ACTIVE_POOL_PATH, {})
    if isinstance(data, dict) and isinstance(data.get("codes"), list):
        codes = [str(x).strip() for x in data["codes"] if str(x).strip()]
        if codes:
            return codes
    return [x["code"] for x in BUILTIN_ASSETS]


def get_selected_assets_config():
    all_assets = BUILTIN_ASSETS + load_custom_assets()
    active = set(load_active_pool_codes())
    selected = [a for a in all_assets if str(a.get("code", "")).strip() in active]
    return selected if selected else BUILTIN_ASSETS[:]


def build_data_signature(selected_assets):
    signatures = []
    for asset in selected_assets:
        file_path = str(asset.get("file_path", "")).strip()
        code = str(asset.get("code", "")).strip()
        if file_path and os.path.exists(file_path):
            stat = os.stat(file_path)
            signatures.append((code, file_path, int(stat.st_mtime_ns), int(stat.st_size)))
        else:
            signatures.append((code, file_path, -1, -1))
    return tuple(signatures)


@st.cache_data
def load_history_data(selected_assets, data_signature):
    history_data = {}
    meta_rows = []
    for asset in selected_assets:
        file_path = str(asset.get("file_path", "")).strip()
        name = str(asset.get("name", "")).strip()
        code = str(asset.get("code", "")).strip()
        if not file_path or not name or not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path)
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str).str[:8], format="%Y%m%d")
        df = df.sort_values("trade_date").set_index("trade_date")
        history_data[name] = df
        meta_rows.append(
            {
                "代码": code,
                "名称": name,
                "起始日期": df.index.min().strftime("%Y-%m-%d"),
                "最新日期": df.index.max().strftime("%Y-%m-%d"),
                "数据行数": len(df),
            }
        )
    return history_data, pd.DataFrame(meta_rows)


@st.cache_data
def calculate_rolling_scores(series, window=25):
    scores = pd.Series(index=series.index, dtype=float)
    x = np.arange(window)
    weights = 1 + np.linspace(0, 1, window) ** 2
    values = np.log(series.astype(float)).values
    for i in range(window, len(values) + 1):
        window_data = values[i - window : i]
        if np.isnan(window_data).any():
            continue
        try:
            coeffs = np.polyfit(x, window_data, 1, w=weights)
            slope = coeffs[0]
            y_pred = np.polyval(coeffs, x)
            sse = np.sum(weights * (window_data - y_pred) ** 2)
            y_mean = np.average(window_data, weights=weights)
            sst = np.sum(weights * (window_data - y_mean) ** 2)
            r2 = 0 if sst == 0 else 1 - sse / sst
            scores.iloc[i - 1] = (np.exp(slope * 252) - 1) * r2 * 100
        except Exception:
            pass
    return scores


@st.cache_data
def precalculate_all_scores(history_data, window=25):
    all_scores = pd.DataFrame()
    for asset, df in history_data.items():
        series = df["adj_close"] if "adj_close" in df.columns else df["close"]
        scores = calculate_rolling_scores(series, window=window)
        scores.name = asset
        all_scores = pd.merge(all_scores, scores, left_index=True, right_index=True, how="outer")
    return all_scores.sort_index()


def build_signal(today_scores, current_set, name_to_code, user_cutoffs, fallback_cutoff, hold_top_n, buffer_score, exclude_overheated):
    pool_scores = today_scores.dropna()
    if pool_scores.empty:
        return []

    def get_cutoff(asset_name):
        code = name_to_code.get(asset_name)
        if code and code in user_cutoffs:
            return user_cutoffs[code]
        return fallback_cutoff

    current_cutoffs = pd.Series(pool_scores.index.map(get_cutoff), index=pool_scores.index)
    valid_candidates = pool_scores[(pool_scores <= current_cutoffs) & (pool_scores > 0)]
    if valid_candidates.empty:
        return []

    norm_basis = valid_candidates if exclude_overheated else pool_scores
    mn, mx = float(norm_basis.min()), float(norm_basis.max())
    if mx == mn:
        norm_scores = pd.Series(50.0, index=pool_scores.index)
    else:
        norm_scores = (pool_scores - mn) / (mx - mn) * 100

    best_assets = valid_candidates.sort_values(ascending=False).index.tolist()[:hold_top_n]
    valid_set = set(valid_candidates.index.tolist())
    if not best_assets:
        return []
    if not current_set:
        return best_assets
    if any(a not in valid_set for a in current_set):
        return best_assets
    if set(best_assets) == set(current_set):
        return best_assets

    curr_norms = norm_scores[list(current_set)] if current_set else pd.Series(dtype=float)
    min_curr_norm = float(curr_norms.min()) if not curr_norms.empty else -1e9
    trigger = any((a not in current_set) and (float(norm_scores.get(a, -1e9)) - min_curr_norm > buffer_score) for a in best_assets)
    return best_assets if trigger else list(current_set)


def summarize_performance(value_df, trade_count):
    daily_ret = value_df["value"].pct_change().fillna(0.0)
    total_return = float(value_df["value"].iloc[-1] / value_df["value"].iloc[0] - 1)
    annual_return = float(value_df["value"].iloc[-1] ** (252 / len(value_df)) - 1) if len(value_df) > 0 else 0.0
    vol = float(daily_ret.std())
    sharpe = float(daily_ret.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0
    drawdown = value_df["value"] / value_df["value"].cummax() - 1
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    return {
        "总收益": total_return,
        "年化收益": annual_return,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "交易次数": int(trade_count),
        "最终净值": float(value_df["value"].iloc[-1]),
    }


def enrich_value_df(value_df):
    if value_df.empty:
        return value_df
    df = value_df.copy()
    df["daily_return"] = df["value"].pct_change().fillna(0.0)
    df["cummax"] = df["value"].cummax()
    df["drawdown"] = df["value"] / df["cummax"] - 1
    return df


def plot_asset_trades(asset_name, df_ohlc, trades, start_date, end_date, execution_label):
    mask = (df_ohlc.index >= pd.Timestamp(start_date)) & (df_ohlc.index <= pd.Timestamp(end_date))
    chart_data = df_ohlc.loc[mask]
    if chart_data.empty:
        return None

    if "adj_close" in chart_data.columns:
        price_series = chart_data["adj_close"]
        price_type = "(后复权)"
    else:
        price_series = chart_data["close"]
        price_type = "(未复权)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=price_series,
            mode="lines",
            name=f"收盘价 {price_type}",
            line=dict(color="#1f77b4", width=2),
        )
    )

    buy_trades = trades[trades["action"] == "买入"]
    if not buy_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_trades["date"],
                y=buy_trades["price"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="red", line=dict(width=1, color="black")),
                name="买入点",
                hovertext=buy_trades.apply(
                    lambda row: f"买入价: {row['price']:.3f}<br>股数: {row['shares']:.2f}<br>金额: {row['amount']:.2f}",
                    axis=1,
                ),
                hoverinfo="text",
            )
        )

    sell_trades = trades[trades["action"] == "卖出"]
    if not sell_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_trades["date"],
                y=sell_trades["price"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="green", line=dict(width=1, color="black")),
                name="卖出点",
                hovertext=sell_trades.apply(
                    lambda row: f"卖出价: {row['price']:.3f}<br>股数: {row['shares']:.2f}<br>金额: {row['amount']:.2f}",
                    axis=1,
                ),
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=f"{asset_name} 交易复盘 - {execution_label} {price_type}",
        xaxis_title="日期",
        yaxis_title="价格",
        height=500,
        hovermode="closest",
    )
    return fig


def run_backtest_by_execution(history_data, raw_scores_df, params, execution_mode="open"):
    timeline = [d for d in raw_scores_df.index if params["start_date"] <= d <= params["end_date"]]
    timeline = sorted(timeline)
    if not timeline:
        return pd.DataFrame(), pd.DataFrame()

    price_open = {}
    price_close = {}
    for asset, df in history_data.items():
        if "adj_open" in df.columns and "adj_close" in df.columns:
            price_open[asset] = df["adj_open"].astype(float)
            price_close[asset] = df["adj_close"].astype(float)
        else:
            price_open[asset] = df["open"].astype(float)
            price_close[asset] = df["close"].astype(float)

    cash = params["initial_capital"]
    holdings = {}
    target_assets = []
    value_history = []
    trade_rows = []
    min_trade_shares = 1e-8
    min_trade_amount = 1e-6

    def _sell(asset, shares_to_sell, price, date):
        nonlocal cash
        if shares_to_sell <= min_trade_shares or not np.isfinite(price) or price <= 0:
            return
        gross_amount = shares_to_sell * price
        if gross_amount <= min_trade_amount:
            return
        proceeds = shares_to_sell * price * (1 - params["fee_rate"])
        cash += proceeds
        holdings[asset] = holdings.get(asset, 0.0) - shares_to_sell
        if holdings.get(asset, 0.0) <= min_trade_shares:
            holdings.pop(asset, None)
        trade_rows.append({"date": date, "action": "卖出", "asset": asset, "price": price, "shares": shares_to_sell, "amount": proceeds})

    def _buy(asset, shares_to_buy, price, date):
        nonlocal cash
        if shares_to_buy <= min_trade_shares or not np.isfinite(price) or price <= 0:
            return
        max_shares = cash / (price * (1 + params["fee_rate"]))
        shares_to_buy = min(shares_to_buy, max_shares)
        if shares_to_buy <= min_trade_shares:
            return
        cost = shares_to_buy * price * (1 + params["fee_rate"])
        if cost <= min_trade_amount:
            return
        cash -= cost
        holdings[asset] = holdings.get(asset, 0.0) + shares_to_buy
        trade_rows.append({"date": date, "action": "买入", "asset": asset, "price": price, "shares": shares_to_buy, "amount": cost})

    for date in timeline:
        if execution_mode == "open":
            target_list = [x for x in target_assets if x in history_data]
            nav_exec = cash
            for asset, shares in holdings.items():
                if date in price_open[asset].index:
                    nav_exec += shares * float(price_open[asset].loc[date])
                else:
                    px = price_close[asset].asof(date)
                    if pd.notna(px):
                        nav_exec += shares * float(px)

            if not target_list:
                for asset in list(holdings.keys()):
                    if date in price_open[asset].index:
                        _sell(asset, holdings.get(asset, 0.0), float(price_open[asset].loc[date]), date)
            else:
                desired_value = nav_exec / len(target_list)
                for asset in list(holdings.keys()):
                    if asset in target_list or date not in price_open[asset].index:
                        continue
                    _sell(asset, holdings.get(asset, 0.0), float(price_open[asset].loc[date]), date)
                for asset in list(holdings.keys()):
                    if asset not in target_list or date not in price_open[asset].index:
                        continue
                    p = float(price_open[asset].loc[date])
                    cur_val = holdings[asset] * p
                    if cur_val > desired_value:
                        _sell(asset, max(0.0, holdings[asset] - desired_value / p), p, date)
                for asset in target_list:
                    if date not in price_open[asset].index:
                        continue
                    p = float(price_open[asset].loc[date])
                    cur_val = holdings.get(asset, 0.0) * p
                    if cur_val < desired_value:
                        _buy(asset, max(0.0, desired_value / p - holdings.get(asset, 0.0)), p, date)

            day_value = cash
            for asset, shares in holdings.items():
                px = price_close[asset].loc[date] if date in price_close[asset].index else price_close[asset].asof(date)
                if pd.notna(px):
                    day_value += shares * float(px)
            holding_assets = "|".join(sorted(list(holdings.keys())))
            value_history.append(
                {
                    "date": date,
                    "value": day_value / params["initial_capital"],
                    "holding": holding_assets if holding_assets else "现金",
                    "holding_assets": holding_assets,
                    "cash_ratio": cash / day_value if day_value > 0 else 0.0,
                }
            )

            today_scores = raw_scores_df.loc[date].dropna()
            target_assets = build_signal(
                today_scores=today_scores,
                current_set=set(holdings.keys()),
                name_to_code=params["name_to_code"],
                user_cutoffs=params["user_cutoffs"],
                fallback_cutoff=params["fallback_cutoff"],
                hold_top_n=params["hold_top_n"],
                buffer_score=params["buffer_score"],
                exclude_overheated=params["exclude_overheated_from_norm"],
            )
        else:
            today_scores = raw_scores_df.loc[date].dropna()
            target_assets = build_signal(
                today_scores=today_scores,
                current_set=set(holdings.keys()),
                name_to_code=params["name_to_code"],
                user_cutoffs=params["user_cutoffs"],
                fallback_cutoff=params["fallback_cutoff"],
                hold_top_n=params["hold_top_n"],
                buffer_score=params["buffer_score"],
                exclude_overheated=params["exclude_overheated_from_norm"],
            )

            nav_exec = cash
            for asset, shares in holdings.items():
                px = price_close[asset].loc[date] if date in price_close[asset].index else price_close[asset].asof(date)
                if pd.notna(px):
                    nav_exec += shares * float(px)

            if not target_assets:
                for asset in list(holdings.keys()):
                    px = price_close[asset].loc[date] if date in price_close[asset].index else price_close[asset].asof(date)
                    if pd.notna(px):
                        _sell(asset, holdings.get(asset, 0.0), float(px), date)
            else:
                desired_value = nav_exec / len(target_assets)
                for asset in list(holdings.keys()):
                    if asset in target_assets:
                        continue
                    px = price_close[asset].loc[date] if date in price_close[asset].index else price_close[asset].asof(date)
                    if pd.notna(px):
                        _sell(asset, holdings.get(asset, 0.0), float(px), date)
                for asset in list(holdings.keys()):
                    if asset not in target_assets:
                        continue
                    px = price_close[asset].loc[date] if date in price_close[asset].index else price_close[asset].asof(date)
                    if pd.notna(px):
                        p = float(px)
                        cur_val = holdings[asset] * p
                        if cur_val > desired_value:
                            _sell(asset, max(0.0, holdings[asset] - desired_value / p), p, date)
                for asset in target_assets:
                    px = price_close[asset].loc[date] if date in price_close[asset].index else price_close[asset].asof(date)
                    if pd.notna(px):
                        p = float(px)
                        cur_val = holdings.get(asset, 0.0) * p
                        if cur_val < desired_value:
                            _buy(asset, max(0.0, desired_value / p - holdings.get(asset, 0.0)), p, date)

            day_value = cash
            for asset, shares in holdings.items():
                px = price_close[asset].loc[date] if date in price_close[asset].index else price_close[asset].asof(date)
                if pd.notna(px):
                    day_value += shares * float(px)
            holding_assets = "|".join(sorted(list(holdings.keys())))
            value_history.append(
                {
                    "date": date,
                    "value": day_value / params["initial_capital"],
                    "holding": holding_assets if holding_assets else "现金",
                    "holding_assets": holding_assets,
                    "cash_ratio": cash / day_value if day_value > 0 else 0.0,
                }
            )

    value_df = pd.DataFrame(value_history).set_index("date")
    trade_df = pd.DataFrame(trade_rows)
    if not trade_df.empty:
        trade_df["execution_mode"] = "信号日收盘成交" if execution_mode == "close" else "原始策略(T+1开盘成交)"
    return enrich_value_df(value_df), trade_df


st.title("⏱️ 交易时点对比")
st.caption("对比两种执行方式：信号日收盘成交 vs 原始策略 T+1 开盘成交。")

selected_assets = get_selected_assets_config()
data_signature = build_data_signature(selected_assets)
history_data, asset_meta_df = load_history_data(selected_assets, data_signature)
if not history_data:
    st.error("当前轮动池没有可用历史数据。")
    st.stop()

min_date = min(df.index.min() for df in history_data.values())
max_date = max(df.index.max() for df in history_data.values())
selected_assets = [a for a in selected_assets if a["name"] in history_data]
name_to_code = {str(a.get("name", "")).strip(): str(a.get("code", "")).strip() for a in selected_assets}
code_to_name = {v: k for k, v in name_to_code.items()}

st.sidebar.subheader("对比参数")
window = st.sidebar.number_input("动量窗口 (天)", min_value=5, max_value=60, value=25, step=1)
hold_top_n = st.sidebar.number_input("持有前 N 名", min_value=1, max_value=max(1, len(history_data)), value=1, step=1)

st.sidebar.subheader("📅 回测时间设置")
time_range_option = st.sidebar.radio(
    "选择回测时长",
    ("最近1年", "最近2年", "最近3年", "最近4年", "最近5年", "自定义时间范围"),
    index=5,
)

if time_range_option == "自定义时间范围":
    start_date = st.sidebar.date_input("开始日期", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date, min_value=min_date, max_value=max_date)
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
else:
    end_date = pd.Timestamp.now()
    years_map = {
        "最近1年": 1,
        "最近2年": 2,
        "最近3年": 3,
        "最近4年": 4,
        "最近5年": 5,
    }
    years_back = years_map[time_range_option]
    start_date = end_date - pd.DateOffset(years=years_back)
    if start_date < min_date:
        start_date = min_date
    if end_date > max_date:
        end_date = max_date
    st.sidebar.caption(f"范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")

global_cutoff = st.sidebar.number_input("全局熔断阈值", min_value=50, max_value=2000, value=500, step=50)
buffer_score = st.sidebar.number_input("换仓缓冲阈值", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
exclude_overheated = st.sidebar.checkbox("归一化时剔除过热标的", value=True)
fee_rate = st.sidebar.number_input("交易费率 (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01) / 100
initial_capital = st.sidebar.number_input("初始资金", min_value=10000, value=100000, step=10000)

user_cutoffs = {}
with st.sidebar.expander("分标的熔断阈值", expanded=False):
    for code, name in code_to_name.items():
        user_cutoffs[code] = st.number_input(f"{name} ({code})", min_value=50, max_value=2000, value=int(global_cutoff), step=50)

score_df = precalculate_all_scores(history_data, window=window)
params = {
    "start_date": pd.Timestamp(start_date),
    "end_date": pd.Timestamp(end_date),
    "hold_top_n": int(hold_top_n),
    "user_cutoffs": user_cutoffs,
    "name_to_code": name_to_code,
    "fallback_cutoff": float(global_cutoff),
    "buffer_score": float(buffer_score),
    "exclude_overheated_from_norm": bool(exclude_overheated),
    "fee_rate": float(fee_rate),
    "initial_capital": float(initial_capital),
}

open_value_df, open_trades = run_backtest_by_execution(history_data, score_df, params, execution_mode="open")
close_value_df, close_trades = run_backtest_by_execution(history_data, score_df, params, execution_mode="close")

if open_value_df.empty or close_value_df.empty:
    st.warning("当前参数下没有可比较的回测区间。")
    st.stop()

open_metrics = summarize_performance(open_value_df, len(open_trades))
close_metrics = summarize_performance(close_value_df, len(close_trades))
excess = float(close_value_df["value"].iloc[-1] / open_value_df["value"].iloc[-1] - 1)

metric_df = pd.DataFrame(
    {
        "指标": ["总收益", "年化收益", "夏普比率", "最大回撤", "交易次数", "最终净值"],
        "信号日收盘成交": [
            f'{close_metrics["总收益"]:.2%}',
            f'{close_metrics["年化收益"]:.2%}',
            f'{close_metrics["夏普比率"]:.2f}',
            f'{close_metrics["最大回撤"]:.2%}',
            close_metrics["交易次数"],
            f'{close_metrics["最终净值"]:.2f}',
        ],
        "原始策略(T+1开盘成交)": [
            f'{open_metrics["总收益"]:.2%}',
            f'{open_metrics["年化收益"]:.2%}',
            f'{open_metrics["夏普比率"]:.2f}',
            f'{open_metrics["最大回撤"]:.2%}',
            open_metrics["交易次数"],
            f'{open_metrics["最终净值"]:.2f}',
        ],
    }
)

col1, col2, col3 = st.columns(3)
col1.metric("轮动池标的数", len(history_data))
col2.metric("收盘成交相对净值优势", f"{excess:.2%}")
col3.metric("对比区间", f"{pd.Timestamp(start_date).strftime('%Y-%m-%d')} ~ {pd.Timestamp(end_date).strftime('%Y-%m-%d')}")

st.subheader("绩效对比")
st.dataframe(metric_df, use_container_width=True, hide_index=True)

merged = open_value_df.join(close_value_df, lsuffix="_open", rsuffix="_close", how="inner")
merged["excess_curve"] = merged["value_close"] / merged["value_open"]
merged["drawdown_open"] = merged["drawdown_open"] if "drawdown_open" in merged.columns else (merged["value_open"] / merged["value_open"].cummax() - 1)
merged["drawdown_close"] = merged["drawdown_close"] if "drawdown_close" in merged.columns else (merged["value_close"] / merged["value_close"].cummax() - 1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=merged.index, y=merged["value_close"], mode="lines", name="信号日收盘成交", line=dict(color="#d62728", width=2)))
fig.add_trace(go.Scatter(x=merged.index, y=merged["value_open"], mode="lines", name="原始策略(T+1开盘成交)", line=dict(color="#1f77b4", width=2)))
fig.update_layout(height=460, xaxis_title="日期", yaxis_title="净值", hovermode="x unified")
st.subheader("净值曲线对比")
st.plotly_chart(fig, use_container_width=True)

fig_ret = go.Figure()
fig_ret.add_trace(go.Scatter(x=merged.index, y=merged["daily_return_close"], mode="lines", name="收盘成交日收益", line=dict(color="#d62728", width=1.5)))
fig_ret.add_trace(go.Scatter(x=merged.index, y=merged["daily_return_open"], mode="lines", name="开盘成交日收益", line=dict(color="#1f77b4", width=1.5)))
fig_ret.update_layout(height=360, xaxis_title="日期", yaxis_title="单日收益", hovermode="x unified")
st.subheader("收益曲线")
st.plotly_chart(fig_ret, use_container_width=True)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=merged.index, y=merged["drawdown_close"], mode="lines", name="收盘成交回撤", line=dict(color="#d62728", width=1.8), fill="tozeroy"))
fig_dd.add_trace(go.Scatter(x=merged.index, y=merged["drawdown_open"], mode="lines", name="开盘成交回撤", line=dict(color="#1f77b4", width=1.8), fill="tozeroy"))
fig_dd.update_layout(height=360, xaxis_title="日期", yaxis_title="回撤", yaxis_tickformat=".0%", hovermode="x unified")
st.subheader("回撤曲线")
st.plotly_chart(fig_dd, use_container_width=True)

fig_excess = go.Figure()
fig_excess.add_trace(go.Scatter(x=merged.index, y=merged["excess_curve"], mode="lines", name="收盘成交 / 开盘成交", line=dict(color="#2ca02c", width=2)))
fig_excess.update_layout(height=320, xaxis_title="日期", yaxis_title="相对净值", hovermode="x unified")
st.subheader("相对净值曲线")
st.plotly_chart(fig_excess, use_container_width=True)

st.subheader("交易记录与持仓明细")
tab_close, tab_open = st.tabs(["信号日收盘成交", "原始策略(T+1开盘成交)"])

with tab_close:
    st.markdown("**持仓明细**")
    close_hold_df = close_value_df[["value", "daily_return", "drawdown", "holding", "holding_assets", "cash_ratio"]].copy()
    close_hold_df["daily_return"] = close_hold_df["daily_return"].map(lambda x: f"{x:.2%}")
    close_hold_df["drawdown"] = close_hold_df["drawdown"].map(lambda x: f"{x:.2%}")
    close_hold_df["cash_ratio"] = close_hold_df["cash_ratio"].map(lambda x: f"{x:.2%}")
    st.dataframe(close_hold_df.tail(250), use_container_width=True)

    st.markdown("**交易记录**")
    if close_trades.empty:
        st.info("当前参数下，收盘成交方式没有生成交易记录。")
    else:
        close_trade_show = close_trades.copy()
        close_trade_show["date"] = pd.to_datetime(close_trade_show["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(close_trade_show.tail(400), use_container_width=True, hide_index=True)

with tab_open:
    st.markdown("**持仓明细**")
    open_hold_df = open_value_df[["value", "daily_return", "drawdown", "holding", "holding_assets", "cash_ratio"]].copy()
    open_hold_df["daily_return"] = open_hold_df["daily_return"].map(lambda x: f"{x:.2%}")
    open_hold_df["drawdown"] = open_hold_df["drawdown"].map(lambda x: f"{x:.2%}")
    open_hold_df["cash_ratio"] = open_hold_df["cash_ratio"].map(lambda x: f"{x:.2%}")
    st.dataframe(open_hold_df.tail(250), use_container_width=True)

    st.markdown("**交易记录**")
    if open_trades.empty:
        st.info("当前参数下，开盘成交方式没有生成交易记录。")
    else:
        open_trade_show = open_trades.copy()
        open_trade_show["date"] = pd.to_datetime(open_trade_show["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(open_trade_show.tail(400), use_container_width=True, hide_index=True)

st.subheader("标的交易详情可视化")
viz_mode = st.radio("选择查看的执行方式", ["信号日收盘成交", "原始策略(T+1开盘成交)"], horizontal=True)
viz_trade_df = close_trades if viz_mode == "信号日收盘成交" else open_trades
asset_options = sorted(viz_trade_df["asset"].dropna().unique().tolist()) if not viz_trade_df.empty else []

if not asset_options:
    st.info("当前参数下暂无可视化交易记录。")
else:
    selected_asset = st.selectbox("选择标的", options=asset_options)
    asset_trades = viz_trade_df[viz_trade_df["asset"] == selected_asset].copy()
    fig_asset = plot_asset_trades(
        asset_name=selected_asset,
        df_ohlc=history_data[selected_asset],
        trades=asset_trades,
        start_date=params["start_date"],
        end_date=params["end_date"],
        execution_label=viz_mode,
    )
    if fig_asset is not None:
        st.plotly_chart(fig_asset, use_container_width=True)
    asset_trades_show = asset_trades.copy()
    asset_trades_show["date"] = pd.to_datetime(asset_trades_show["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(asset_trades_show, use_container_width=True, hide_index=True)

st.subheader("当前回测标的池")
st.dataframe(asset_meta_df, use_container_width=True, hide_index=True)

st.subheader("说明")
st.markdown(
    """
- `信号日收盘成交`：使用当日收盘动量分，直接按当日收盘价调仓。
- `原始策略(T+1开盘成交)`：使用当日收盘动量分，在下一交易日开盘价调仓。
- 两种方式使用完全相同的选股、熔断、归一化、换仓缓冲和费率参数，仅交易时点不同。
"""
)
