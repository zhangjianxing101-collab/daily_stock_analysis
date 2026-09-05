"""Bounded, read-only provider diagnostics. Never serialize raw provider text."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.collaborative_report.market_data import (  # noqa: E402
    MarketDataGateway, _canonical_code, _latest_completed_session_date, _normalize_bar_columns,
    _normalize_ths_snapshot, normalize_a_share_thscode,
)
from src.collaborative_report.snapshot_supplement import fetch_snapshot_supplement  # noqa: E402
from src.collaborative_report.settings import ThsSettings  # noqa: E402
from src.collaborative_report.ths_market_data import ThsMarketDataClient  # noqa: E402


def snapshot_summary(items):
    frame = pd.DataFrame([dict(item) for item in items])
    result = {"rows": len(frame)}
    for field in ("last_price", "volume", "turnover", "name", "volume_ratio", "turnover_ratio_pct"):
        result[field + "_present"] = int(frame[field].notna().sum()) if field in frame else 0
    if {"ticker", "last_price", "volume", "turnover"}.issubset(frame.columns):
        price = pd.to_numeric(frame["last_price"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        amount = pd.to_numeric(frame["turnover"], errors="coerce")
        invalid = ~np.isfinite(price) | (price <= 0)
        inactive = (volume == 0) & (amount == 0)
        result.update({
            "invalid_prices": int(invalid.sum()),
            "invalid_prices_zero_activity": int((invalid & inactive).sum()),
            "invalid_prices_unknown_activity": int((invalid & ~inactive).sum()),
            "invalid_codes": int(frame["ticker"].map(_canonical_code).isna().sum()),
        })
    return result


def bars_summary(frame):
    normalized = _normalize_bar_columns(frame)
    dates = normalized["date"].dropna().sort_values()
    return {
        "rows": len(normalized),
        "invalid_dates": int(normalized["date"].isna().sum()),
        "latest_dates": [value.date().isoformat() for value in dates.tail(5)],
    }


def main():
    now = datetime.now(timezone.utc)
    gateway = MarketDataGateway()
    output = {"observed_at": now.isoformat()}
    try:
        client = ThsMarketDataClient(ThsSettings.from_env())
        items = []
        for page in range(20):
            response = client.a_share_snapshot(limit=1000, offset=len(items))
            items.extend(response.items)
            total = response.data.get("total")
            if not response.items or (type(total) is int and len(items) >= total):
                break
        output["snapshot"] = snapshot_summary(items)
        quotes = _normalize_ths_snapshot(items)
        supported = []
        unsupported = {}
        for code in quotes["code"]:
            try:
                normalize_a_share_thscode(code)
                supported.append(code)
            except ValueError:
                prefix = code[:3] if code.isascii() and code.isdigit() else "invalid"
                unsupported[prefix] = unsupported.get(prefix, 0) + 1
        output["unsupported_prefix_counts"] = unsupported
        try:
            supplement = fetch_snapshot_supplement(supported)
            other = supplement.frame
            matches = quotes.merge(other, on="code", suffixes=("_ths", "_supplement"))
            output["supplement"] = {
                "requested": len(supported), "rows": len(other), "missing": supplement.missing_count,
                "price_agreements": int(((matches["price_ths"] - matches["price_supplement"]).abs() <= 0.010000001).sum()),
                "source_dates": sorted({pd.Timestamp(value).date().isoformat() for value in other["source_timestamp"]}),
            }
        except Exception:
            output["supplement"] = {"status": "probe_failed"}
    except Exception:
        output["snapshot"] = {"status": "probe_failed"}
    try:
        latest = _latest_completed_session_date("GC=F", now)
        end = latest + timedelta(days=1)
        output["gold_expected_date"] = latest.isoformat()
        windows = {
            "ten_year": {"period": "10y"},
            "recent": {"start": (end - timedelta(days=30)).isoformat()},
        }
        for name, window in windows.items():
            try:
                raw = gateway._download(
                    "GC=F", **window, end=end.isoformat(), interval="1d",
                    auto_adjust=False, progress=False, timeout=15,
                )
                output["gold_" + name] = bars_summary(raw)
            except Exception:
                output["gold_" + name] = {"status": "probe_failed"}
    except Exception:
        output["gold"] = {"status": "probe_failed"}
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
