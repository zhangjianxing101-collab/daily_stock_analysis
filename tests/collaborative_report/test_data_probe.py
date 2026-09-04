import json

import pandas as pd

from scripts.probe_collaborative_data import snapshot_summary, bars_summary


def test_snapshot_probe_outputs_counts_not_source_text():
    result = snapshot_summary([
        {"ticker": "600000", "last_price": 10, "volume": 2, "turnover": 20, "name": "private-token"},
        {"ticker": "600001", "last_price": None, "volume": 0, "turnover": 0},
        {"ticker": "secret", "last_price": None, "volume": 2, "turnover": 20},
    ])
    assert result["invalid_prices"] == 2
    assert result["invalid_prices_zero_activity"] == 1
    assert result["invalid_codes"] == 1
    assert "secret" not in json.dumps(result)
    assert "private-token" not in json.dumps(result)
    assert all(type(value) is int for value in result.values())


def test_bar_probe_only_exposes_parsed_dates():
    frame = pd.DataFrame({
        "date": ["secret", "2026-09-03"], "open": [1, 1], "high": [1, 1],
        "low": [1, 1], "close": [1, 1], "volume": [1, 1],
    })
    result = bars_summary(frame)
    assert result == {"rows": 2, "invalid_dates": 1, "latest_dates": ["2026-09-03"]}
    assert "secret" not in json.dumps(result)
