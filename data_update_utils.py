import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import tushare as ts


PRICE_COLUMNS = ["open", "high", "low", "close"]


def normalize_trade_date(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)


def normalize_trade_date_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = normalize_trade_date(out["trade_date"])
    return out


def get_open_trade_end_date(pro) -> str:
    sh_now = datetime.now(ZoneInfo("Asia/Shanghai"))
    cal_end = sh_now.strftime("%Y%m%d")
    cal_start = (sh_now - timedelta(days=60)).strftime("%Y%m%d")
    try:
        open_days = pro.trade_cal(exchange="", start_date=cal_start, end_date=cal_end, is_open="1")
        if open_days is not None and not open_days.empty:
            return str(open_days["cal_date"].max())
    except Exception:
        pass
    return cal_end


def read_history_csv(file_path: str) -> pd.DataFrame:
    if not file_path or not os.path.exists(file_path):
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    return normalize_trade_date_frame(df)


def fetch_pro_bar_chunked(ts_code: str, start_date: str, end_date: str, asset: str, adj=None, chunk_days: int = 700) -> pd.DataFrame:
    if not start_date or not end_date or start_date > end_date:
        return pd.DataFrame()

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    all_data = []

    current_start = start_dt
    while current_start <= end_dt:
        current_end = min(current_start + timedelta(days=chunk_days), end_dt)
        s_date = current_start.strftime("%Y%m%d")
        e_date = current_end.strftime("%Y%m%d")

        last_err = None
        for attempt in range(3):
            try:
                df = ts.pro_bar(ts_code=ts_code, start_date=s_date, end_date=e_date, adj=adj, asset=asset)
                if df is not None and not df.empty:
                    all_data.append(df)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(0.8)
        if last_err is not None:
            raise last_err

        current_start = current_end + timedelta(days=1)
        time.sleep(0.05)

    if not all_data:
        return pd.DataFrame()

    out = pd.concat(all_data, ignore_index=True)
    out = normalize_trade_date_frame(out)
    out.drop_duplicates(subset=["trade_date"], keep="last", inplace=True)
    out.sort_values("trade_date", inplace=True)
    return out


def fetch_adj_factors_chunked(pro, ts_code: str, start_date: str, end_date: str, asset_type: str, chunk_days: int = 1800) -> pd.DataFrame:
    if not start_date or not end_date or start_date > end_date:
        return pd.DataFrame()

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    all_data = []
    current_start = start_dt

    while current_start <= end_dt:
        current_end = min(current_start + timedelta(days=chunk_days), end_dt)
        s_date = current_start.strftime("%Y%m%d")
        e_date = current_end.strftime("%Y%m%d")

        last_err = None
        for attempt in range(3):
            try:
                if asset_type == "FD":
                    df = pro.fund_adj(ts_code=ts_code, start_date=s_date, end_date=e_date)
                elif asset_type == "E":
                    df = pro.adj_factor(ts_code=ts_code, start_date=s_date, end_date=e_date)
                else:
                    df = pd.DataFrame()
                if df is not None and not df.empty:
                    all_data.append(df)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(0.8)
        if last_err is not None:
            raise last_err

        current_start = current_end + timedelta(days=1)
        time.sleep(0.05)

    if not all_data:
        return pd.DataFrame()

    out = pd.concat(all_data, ignore_index=True)
    out = normalize_trade_date_frame(out)
    out.drop_duplicates(subset=["trade_date"], keep="last", inplace=True)
    out.sort_values("trade_date", inplace=True)
    return out


def build_forward_adjusted_from_factor(raw_df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = normalize_trade_date_frame(raw_df)
    factor_df = normalize_trade_date_frame(factor_df)
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    work = raw_df.copy()
    work = work.sort_values("trade_date")
    if factor_df is not None and not factor_df.empty and "adj_factor" in factor_df.columns:
        factor_map = factor_df[["trade_date", "adj_factor"]].copy()
        factor_map["adj_factor"] = factor_map["adj_factor"].astype(float)
        work = work.merge(factor_map, on="trade_date", how="left")
        work["adj_factor"] = work["adj_factor"].ffill().bfill()
    else:
        work["adj_factor"] = 1.0

    latest_factor = float(work["adj_factor"].dropna().iloc[-1]) if not work["adj_factor"].dropna().empty else 1.0
    if latest_factor == 0:
        latest_factor = 1.0

    ratio = work["adj_factor"].astype(float) / latest_factor
    adjusted_df = work[["trade_date"]].copy()
    for col in PRICE_COLUMNS:
        if col in work.columns:
            adjusted_df[col] = work[col].astype(float) * ratio
    return adjusted_df


def rebuild_history_with_full_adjusted(existing_df: pd.DataFrame, raw_incremental_df: pd.DataFrame, adjusted_full_df: pd.DataFrame) -> pd.DataFrame:
    existing_df = normalize_trade_date_frame(existing_df)
    raw_incremental_df = normalize_trade_date_frame(raw_incremental_df)
    adjusted_full_df = normalize_trade_date_frame(adjusted_full_df)

    raw_frames = []
    if existing_df is not None and not existing_df.empty:
        existing_raw_cols = [c for c in existing_df.columns if not c.startswith("adj_")]
        raw_frames.append(existing_df[existing_raw_cols].copy())
    if raw_incremental_df is not None and not raw_incremental_df.empty:
        raw_frames.append(raw_incremental_df.copy())

    if not raw_frames:
        return pd.DataFrame()

    raw_all = pd.concat(raw_frames, ignore_index=True)
    raw_all = normalize_trade_date_frame(raw_all)
    raw_all.drop_duplicates(subset=["trade_date"], keep="last", inplace=True)
    raw_all.sort_values("trade_date", inplace=True)

    raw_cols = raw_all.columns.tolist()
    raw_all = raw_all.set_index("trade_date").sort_index()

    if adjusted_full_df is not None and not adjusted_full_df.empty:
        adj_indexed = adjusted_full_df.set_index("trade_date").sort_index()
        adj_cols = [c for c in PRICE_COLUMNS if c in adj_indexed.columns]
        adj_subset = adj_indexed[adj_cols].rename(columns={c: f"adj_{c}" for c in adj_cols})
        final_df = raw_all.join(adj_subset, how="left")
    else:
        final_df = raw_all.copy()

    for col in PRICE_COLUMNS:
        adj_col = f"adj_{col}"
        if col in final_df.columns:
            if adj_col not in final_df.columns:
                final_df[adj_col] = final_df[col]
            else:
                final_df[adj_col] = final_df[adj_col].fillna(final_df[col])

    final_df = final_df.reset_index()
    adj_output_cols = [f"adj_{c}" for c in PRICE_COLUMNS if f"adj_{c}" in final_df.columns]
    final_df = final_df[raw_cols + [c for c in adj_output_cols if c not in raw_cols]]
    final_df.sort_values("trade_date", inplace=True)
    final_df["trade_date"] = normalize_trade_date(final_df["trade_date"])
    return final_df


def update_history_file(file_path: str, ts_code: str, asset_type: str, start_date: str, end_date: str, force: bool = False, pro=None) -> dict:
    existing_df = pd.DataFrame() if force else read_history_csv(file_path)
    old_max = existing_df["trade_date"].max() if not existing_df.empty and "trade_date" in existing_df.columns else ""

    raw_start = start_date
    if not force and old_max:
        raw_start = (datetime.strptime(old_max, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

    raw_incremental_df = pd.DataFrame()
    if force or not old_max or raw_start <= end_date:
        raw_incremental_df = fetch_pro_bar_chunked(
            ts_code=ts_code,
            start_date=raw_start,
            end_date=end_date,
            asset=asset_type,
            adj=None,
        )

    raw_full_df = fetch_pro_bar_chunked(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        asset=asset_type,
        adj=None,
    )

    factor_df = pd.DataFrame()
    if pro is not None and asset_type in {"FD", "E"}:
        factor_df = fetch_adj_factors_chunked(
            pro=pro,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            asset_type=asset_type,
        )

    if not raw_full_df.empty:
        adjusted_full_df = build_forward_adjusted_from_factor(raw_df=raw_full_df, factor_df=factor_df)
    else:
        adjusted_full_df = pd.DataFrame()

    final_df = rebuild_history_with_full_adjusted(
        existing_df=existing_df,
        raw_incremental_df=raw_incremental_df,
        adjusted_full_df=adjusted_full_df,
    )

    if final_df.empty:
        return {
            "file_path": file_path,
            "ts_code": ts_code,
            "old_max": old_max,
            "new_max": old_max,
            "new_rows": 0,
            "adj_rows": 0,
            "factor_rows": 0,
            "status": "no_data",
        }

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    final_df.to_csv(file_path, index=False, encoding="utf-8-sig")

    new_max = final_df["trade_date"].max()
    new_rows = int((final_df["trade_date"] > old_max).sum()) if old_max else len(final_df)
    status = "ok" if new_rows > 0 else "adj_refreshed"
    return {
        "file_path": file_path,
        "ts_code": ts_code,
        "old_max": old_max,
        "new_max": new_max,
        "new_rows": new_rows,
        "adj_rows": len(adjusted_full_df),
        "factor_rows": len(factor_df),
        "status": status,
    }
