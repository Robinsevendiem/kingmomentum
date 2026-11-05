"""免费行情源的盘中未复权报价获取工具。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
import re
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
EASTMONEY_UT = "fa5fd1943c7b386f172d6893dbbd1d0c"


@dataclass(frozen=True)
class IntradayQuote:
    code: str
    price: float
    source: str
    timestamp: datetime


class QuoteFetchError(RuntimeError):
    """报价源不可用或返回数据无法解析。"""


def _exchange_symbol(ts_code: str) -> tuple[str, str, str]:
    """返回行情源代码、东方财富市场代码和六码证券代码。"""
    parts = str(ts_code).strip().upper().split(".", 1)
    code = parts[0]
    suffix = parts[1] if len(parts) == 2 else ""
    exchange = suffix or ("SH" if code.startswith(("5", "6", "9")) else "SZ")
    if exchange not in {"SH", "SZ"} or not code.isdigit():
        raise ValueError(f"不支持的证券代码：{ts_code}")
    prefix = exchange.lower()
    market = "1" if exchange == "SH" else "0"
    return f"{prefix}{code}", market, code


def _parse_timestamp(value: object) -> datetime | None:
    """解析行情源常见的日期、时分或秒级更新时间。"""
    raw = str(value or "").strip()
    if not raw:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass

    digits = re.sub(r"\D", "", raw)
    for length, fmt in ((14, "%Y%m%d%H%M%S"), (12, "%Y%m%d%H%M")):
        if len(digits) >= length:
            try:
                return datetime.strptime(digits[:length], fmt)
            except ValueError:
                pass
    if len(digits) >= 8:
        try:
            return datetime.strptime(digits[:8] + "150000", "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None


def _valid_quote(code: str, raw_price: object, timestamp: object, source: str) -> IntradayQuote | None:
    try:
        price = float(str(raw_price).strip())
    except (TypeError, ValueError):
        return None
    parsed_time = _parse_timestamp(timestamp)
    # 只接受上海时区当天的报价，避免节假日或停牌时把旧价格伪装成盘中估值。
    if not math.isfinite(price) or price <= 0 or parsed_time is None or parsed_time.date() != datetime.now(SHANGHAI_TZ).date():
        return None
    return IntradayQuote(code=code, price=price, source=source, timestamp=parsed_time)


def _request_text(url: str, referer: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": referer,
            "Accept": "*/*",
            "Cache-Control": "no-cache",
        },
    )
    with urlopen(request, timeout=7) as response:
        if response.status != 200:
            raise QuoteFetchError(f"HTTP {response.status}")
        return response.read().decode("gb18030", errors="replace")


def _fetch_tencent(codes: list[str]) -> dict[str, IntradayQuote]:
    symbols = [_exchange_symbol(code)[0] for code in codes]
    url = "https://qt.gtimg.cn/q=" + ",".join(symbols)
    body = _request_text(url, "https://gu.qq.com/")
    parsed: dict[str, IntradayQuote] = {}
    for symbol, payload in re.findall(r'v_((?:sh|sz)\d{6})="([^"]*)"', body):
        fields = payload.split("~")
        if len(fields) < 6:
            continue
        timestamp = None
        for field in fields:
            if re.search(r"20\d{12}", field):
                timestamp = field
                break
        code = symbol[2:]
        quote = _valid_quote(code, fields[3], timestamp, "腾讯")
        if quote:
            parsed[code] = quote
    return parsed


def _fetch_sina(codes: list[str]) -> dict[str, IntradayQuote]:
    symbols = [_exchange_symbol(code)[0] for code in codes]
    url = "https://hq.sinajs.cn/list=" + ",".join(symbols)
    body = _request_text(url, "https://finance.sina.com.cn/")
    parsed: dict[str, IntradayQuote] = {}
    pattern = r'hq_str_((?:sh|sz)\d{6})="([^"]*)"'
    for symbol, payload in re.findall(pattern, body):
        fields = payload.split(",")
        if len(fields) < 4:
            continue
        timestamp = " ".join(fields[30:32]) if len(fields) > 31 else None
        code = symbol[2:]
        quote = _valid_quote(code, fields[3], timestamp, "新浪")
        if quote:
            parsed[code] = quote
    return parsed


def _fetch_eastmoney(codes: list[str]) -> dict[str, IntradayQuote]:
    secids = []
    for code in codes:
        _, market, numeric_code = _exchange_symbol(code)
        secids.append(f"{market}.{numeric_code}")
    query = urlencode(
        {
            "secids": ",".join(secids),
            "fields": "f2,f12,f14,f86",
            "fltt": "2",
            "invt": "2",
            "ut": EASTMONEY_UT,
        }
    )
    url = f"https://push2.eastmoney.com/api/qt/ulist.np/get?{query}"
    body = _request_text(url, "https://quote.eastmoney.com/")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise QuoteFetchError("返回内容不是有效 JSON") from exc

    data = payload.get("data") or {}
    diff = data.get("diff") or []
    rows = list(diff.values()) if isinstance(diff, dict) else diff
    parsed: dict[str, IntradayQuote] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = str(row.get("f12", "")).strip()
        quote = _valid_quote(code, row.get("f2"), row.get("f86"), "东方财富")
        if quote:
            parsed[code] = quote
    return parsed


def fetch_intraday_quotes(assets: list[dict]) -> tuple[dict[str, IntradayQuote], list[str]]:
    """按腾讯、新浪、东方财富顺序批量获取，并对缺失标的逐源回退。"""
    code_map: dict[str, str] = {}
    unsupported: list[str] = []
    for asset in assets:
        ts_code = str(asset.get("code", "")).strip()
        try:
            _, _, numeric_code = _exchange_symbol(ts_code)
            code_map[numeric_code] = ts_code
        except ValueError:
            unsupported.append(ts_code or str(asset.get("name", "未知标的")))

    quotes: dict[str, IntradayQuote] = {}
    errors: list[str] = [f"不支持的市场代码：{code}" for code in unsupported]
    providers: tuple[tuple[str, Callable[[list[str]], dict[str, IntradayQuote]], str], ...] = (
        ("腾讯", _fetch_tencent, "腾讯行情获取失败"),
        ("新浪", _fetch_sina, "新浪行情获取失败"),
        ("东方财富", _fetch_eastmoney, "东方财富行情获取失败"),
    )

    for _, fetcher, error_label in providers:
        missing = [code for code in code_map.values() if code not in quotes]
        if not missing:
            break
        try:
            provider_quotes = fetcher(missing)
        except Exception as exc:
            errors.append(f"{error_label}（{type(exc).__name__}）")
            continue

        for ts_code in missing:
            numeric_code = _exchange_symbol(ts_code)[2]
            quote = provider_quotes.get(numeric_code)
            if quote is not None:
                quotes[ts_code] = quote

    for ts_code in code_map.values():
        if ts_code not in quotes:
            errors.append(f"{ts_code} 未从可用行情源取得当天有效报价")
    return quotes, errors
