import glob
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_update_utils import get_open_trade_end_date, read_history_csv, update_history_file


def _load_custom_asset_meta():
    path = ROOT / "data" / "custom_assets.json"
    if not path.exists():
        return {}
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    meta = {}
    if not isinstance(items, list):
        return meta
    for item in items:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).strip()
        if not code:
            continue
        meta[code] = item
    return meta


def _infer_start_date(file_path: str, meta: dict) -> str:
    configured = str(meta.get("start_date", "")).strip()
    if configured:
        return configured
    df_existing = read_history_csv(file_path)
    if not df_existing.empty and "trade_date" in df_existing.columns:
        return str(df_existing["trade_date"].min())
    return "19900101"


def _update_one(file_path: str, end_date: str, custom_meta_map: dict, pro):
    base = os.path.basename(file_path)
    ts_code = base.split("_", 1)[0]
    custom_meta = custom_meta_map.get(ts_code, {})
    asset = str(custom_meta.get("asset_type", "")).strip()
    if not asset:
        asset = "E" if "data/custom/" in file_path else "FD"
    start_date = _infer_start_date(file_path, custom_meta)
    return update_history_file(
        file_path=file_path,
        ts_code=ts_code,
        asset_type=asset,
        start_date=start_date,
        end_date=end_date,
        force=False,
        pro=pro,
    )


def main():
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("缺少环境变量 TUSHARE_TOKEN")

    ts.set_token(token)
    pro = ts.pro_api(token)
    end_date = get_open_trade_end_date(pro)

    files = sorted(glob.glob("data/*_history.csv")) + sorted(glob.glob("data/custom/*_history.csv"))
    custom_meta_map = _load_custom_asset_meta()
    rows = []
    for file_path in files:
        rows.append(_update_one(file_path=file_path, end_date=end_date, custom_meta_map=custom_meta_map, pro=pro))
        time.sleep(0.2)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    bad = out[out["status"].astype(str).str.startswith("error")]
    if not bad.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
