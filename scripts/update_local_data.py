import glob
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import tushare as ts


def _normalize_trade_date(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)


def _get_end_date(pro) -> str:
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


def _fetch_pro_bar(ts_code: str, start_date: str, end_date: str, asset: str):
    last_err = None
    for _ in range(3):
        try:
            df_raw = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date, adj=None, asset=asset)
            df_adj = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date, adj="qfq", asset=asset)
            return df_raw, df_adj
        except Exception as e:
            last_err = e
            time.sleep(0.8)
    raise last_err


def _update_one(fp: str, pro, end_date: str):
    asset = "E" if fp.startswith("data/custom/") else "FD"
    base = os.path.basename(fp)
    ts_code = base.split("_", 1)[0]

    df_old = None
    old_max = ""
    start_date = None

    if os.path.exists(fp):
        df_old = pd.read_csv(fp)
        if df_old is not None and not df_old.empty and "trade_date" in df_old.columns:
            old_max = _normalize_trade_date(df_old["trade_date"]).max()
            start_date = (datetime.strptime(old_max, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

    if start_date and start_date > end_date:
        return {
            "file": fp,
            "ts_code": ts_code,
            "asset": asset,
            "added_rows": 0,
            "old_max": old_max,
            "new_max": old_max,
            "status": "latest",
        }

    fetch_start = start_date or "19900101"

    df_raw, df_adj = _fetch_pro_bar(ts_code=ts_code, start_date=fetch_start, end_date=end_date, asset=asset)
    if df_raw is None or df_raw.empty:
        return {
            "file": fp,
            "ts_code": ts_code,
            "asset": asset,
            "added_rows": 0,
            "old_max": old_max,
            "new_max": old_max,
            "status": "no_new",
        }

    if df_adj is None or df_adj.empty:
        df_adj = df_raw.copy()

    df_raw = df_raw.set_index("trade_date").sort_index()
    df_adj = df_adj.set_index("trade_date").sort_index()

    adj_cols = [c for c in ["open", "high", "low", "close"] if c in df_adj.columns]
    df_adj_subset = df_adj[adj_cols].rename(columns={c: f"adj_{c}" for c in adj_cols})

    df_new = df_raw.join(df_adj_subset, how="left").reset_index()
    df_new["trade_date"] = _normalize_trade_date(df_new["trade_date"])

    if df_old is not None and not df_old.empty:
        df_old = df_old.copy()
        df_old["trade_date"] = _normalize_trade_date(df_old["trade_date"])
        df_final = pd.concat([df_old, df_new], ignore_index=True)
        df_final.drop_duplicates(subset=["trade_date"], inplace=True)
        col_order = list(df_old.columns) + [c for c in df_final.columns if c not in df_old.columns]
        df_final = df_final[col_order]
    else:
        df_final = df_new

    df_final.sort_values("trade_date", inplace=True)
    new_max = df_final["trade_date"].max()

    added_rows = 0
    if start_date:
        added_rows = int((df_final["trade_date"] >= start_date).sum())

    df_final.to_csv(fp, index=False, encoding="utf-8-sig")

    return {
        "file": fp,
        "ts_code": ts_code,
        "asset": asset,
        "added_rows": added_rows,
        "old_max": old_max,
        "new_max": new_max,
        "status": "ok",
    }


def main():
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("缺少环境变量 TUSHARE_TOKEN")

    ts.set_token(token)
    pro = ts.pro_api(token)
    end_date = _get_end_date(pro)

    files = sorted(glob.glob("data/*_history.csv")) + sorted(glob.glob("data/custom/*_history.csv"))
    rows = []
    for fp in files:
        rows.append(_update_one(fp=fp, pro=pro, end_date=end_date))
        time.sleep(0.2)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    bad = out[out["status"].astype(str).str.startswith("error")]
    if not bad.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

