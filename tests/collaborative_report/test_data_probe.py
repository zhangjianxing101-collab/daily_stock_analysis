import json
from types import SimpleNamespace

import pandas as pd

import scripts.probe_collaborative_data as probe
from scripts.probe_collaborative_data import bars_summary, snapshot_summary, supplement_summary


def ths_item(code="600000", *, price=10.0, volume=2, turnover=20):
    return {
        "thscode": f"{code}.SH",
        "ticker": code,
        "last_price": price,
        "price_change_ratio_pct": 1.2,
        "volume": volume,
        "turnover": turnover,
    }


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


def test_supplement_probe_filters_unsupported_prefixes_and_unquoted_rows():
    requested = []

    def fetcher(codes):
        requested.extend(codes)
        return SimpleNamespace(
            frame=pd.DataFrame(columns=["code", "price", "source_timestamp"]),
            missing_count=len(codes),
        )

    result = supplement_summary([
        ths_item(),
        ths_item("900001"),
        ths_item("600001", price=None, volume=0, turnover=0),
    ], supplement_fetcher=fetcher)

    assert requested == ["600000"]
    assert result["unsupported_prefix_counts"] == {"900": 1}
    assert result["supplement"] == {
        "requested": 1, "rows": 0, "missing": 1, "price_agreements": 0, "source_dates": [],
    }


def test_supplement_probe_counts_matching_prices_without_source_text():
    def fetcher(codes):
        assert codes == ["600000", "600001"]
        return SimpleNamespace(
            frame=pd.DataFrame({
                "code": ["600000", "600001"],
                "price": [10.01, 11.02],
                "source_timestamp": ["2026-09-04T15:00:00+08:00", "2026-09-04T15:00:00+08:00"],
                "provider_secret": ["secret", "secret"],
            }),
            missing_count=0,
        )

    result = supplement_summary([ths_item(), ths_item("600001", price=11.0)], supplement_fetcher=fetcher)

    assert result["supplement"] == {
        "requested": 2, "rows": 2, "missing": 0, "price_agreements": 1, "source_dates": ["2026-09-04"],
    }
    assert "secret" not in json.dumps(result)


def test_supplement_probe_uses_fixed_stage_reasons_without_source_text():
    normalize_failure = supplement_summary([ths_item("600001", price=None)])
    fetch_failure = supplement_summary([ths_item()], supplement_fetcher=lambda _: (_ for _ in ()).throw(ValueError("secret")))

    assert normalize_failure == {
        "unsupported_prefix_counts": {},
        "supplement": {"status": "probe_failed", "reason": "normalize"},
    }
    assert fetch_failure == {
        "unsupported_prefix_counts": {},
        "supplement": {"status": "probe_failed", "reason": "supplement"},
    }
    assert "secret" not in json.dumps({"normalize": normalize_failure, "fetch": fetch_failure})


def test_main_keeps_snapshot_counts_when_supplement_fails(monkeypatch, capsys):
    client = SimpleNamespace(
        a_share_snapshot=lambda **_: SimpleNamespace(items=(ths_item(),), data={"total": 1}),
    )
    monkeypatch.setattr(probe, "ThsSettings", SimpleNamespace(from_env=lambda: object()))
    monkeypatch.setattr(probe, "ThsMarketDataClient", lambda _: client)
    monkeypatch.setattr(probe, "MarketDataGateway", lambda: SimpleNamespace(_download=lambda *_, **__: None))
    monkeypatch.setattr(probe, "fetch_snapshot_supplement", lambda _: (_ for _ in ()).throw(ValueError("secret")))

    probe.main()

    result = json.loads(capsys.readouterr().out)
    assert result["snapshot"]["rows"] == 1
    assert result["supplement"] == {"status": "probe_failed", "reason": "supplement"}
    assert "secret" not in json.dumps(result)
