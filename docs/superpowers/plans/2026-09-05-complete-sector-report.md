# Complete Sector Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete decision-first post-market report with validated industry and concept rankings, rotation evidence, persistence, candidate context and explicit partial-data handling.

**Architecture:** Introduce a pure sector-analysis module between the existing market-data gateway and report runner. Fetch industry and concept snapshots independently, normalize them into one contract, compare them with the previous completed post-market manifest, then render structured sector tables and a decision summary in both HTML and text.

**Tech Stack:** Python 3.12, pandas, AkShare/Eastmoney adapters, Jinja2, pytest, GitHub Actions artifacts.

---

## File Map

- Create `src/collaborative_report/sector_analysis.py`: normalized sector contracts, ranking, rotation, persistence and crowding rules.
- Create `tests/collaborative_report/test_sector_analysis.py`: deterministic unit tests for every classification boundary.
- Modify `src/collaborative_report/market_data.py`: independently fetch and validate industry and concept board snapshots.
- Modify `tests/collaborative_report/test_market_data.py`: provider normalization, timestamp and independent-failure tests.
- Modify `src/collaborative_report/models.py`: optional candidate sector context with backward-compatible defaults.
- Modify `src/collaborative_report/runner.py`: sector orchestration, prior-state validation, report modules, decision summary and manifest state.
- Modify `tests/collaborative_report/test_runner.py`: end-to-end module, degradation, prior-state and candidate-context tests.
- Modify `src/collaborative_report/report.py`: structured sector views and plain-text parity.
- Modify `templates/collaborative_report.html.j2`: decision summary and responsive industry/concept tables.
- Modify `tests/collaborative_report/test_report.py`: HTML/text parity and unavailable-field rendering.
- Modify `.github/workflows/01-collaborative-report.yml`: retrieve the latest earlier completed post-market state.
- Modify `tests/collaborative_report/test_workflow.py`: artifact selection, privacy and retention contracts.
- Modify `docs/collaborative-report.md` and `docs/CHANGELOG.md`: user-visible behavior and rollout notes.

### Task 1: Pure Sector Ranking And Classification

**Files:**
- Create: `src/collaborative_report/sector_analysis.py`
- Create: `tests/collaborative_report/test_sector_analysis.py`

- [ ] **Step 1: Write failing contract and ranking tests**

```python
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.collaborative_report.sector_analysis import analyze_sectors, classify_sector


NOW = datetime(2026, 9, 4, 16, 30, tzinfo=ZoneInfo("Asia/Shanghai"))


def sector_frame(count=24):
    return pd.DataFrame({
        "sector_type": ["industry"] * count,
        "name": [f"行业{i:02d}" for i in range(count)],
        "change_pct": [float(i) for i in range(count)],
        "advance_count": [60] * count,
        "decline_count": [40] * count,
        "turnover_rate": [float(i + 1) for i in range(count)],
        "leader_name": [f"龙头{i:02d}" for i in range(count)],
        "leader_code": [f"60{i:04d}" for i in range(count)],
        "leader_change_pct": [float(i) for i in range(count)],
    })


def test_rankings_are_deterministic_and_do_not_overlap():
    result = analyze_sectors(sector_frame(), previous=(), observed_at=NOW)
    assert [row.name for row in result.strongest] == [f"行业{i:02d}" for i in range(23, 13, -1)]
    assert [row.name for row in result.weakest] == [f"行业{i:02d}" for i in range(10)]
    assert not ({row.name for row in result.strongest} & {row.name for row in result.weakest})
    assert all(row.rotation == "first_observation" for row in result.strongest)
    assert all(row.persistence == "unavailable" for row in result.strongest)
```

- [ ] **Step 2: Run the ranking test and confirm the module is missing**

Run: `../../.venv/bin/python -m pytest tests/collaborative_report/test_sector_analysis.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.collaborative_report.sector_analysis'`.

- [ ] **Step 3: Add immutable contracts and deterministic ranking**

```python
@dataclass(frozen=True)
class SectorRow:
    sector_type: str
    name: str
    rank: int
    change_pct: float
    breadth_pct: float | None
    activity_percentile: float | None
    leader_name: str | None
    leader_code: str | None
    leader_change_pct: float | None
    rotation: str
    persistence: str
    crowding_risk: str


@dataclass(frozen=True)
class SectorAnalysis:
    strongest: tuple[SectorRow, ...]
    weakest: tuple[SectorRow, ...]
    watch: tuple[SectorRow, ...]
    valid_count: int
    warnings: tuple[str, ...] = ()


def analyze_sectors(frame, *, previous, observed_at, limit=10):
    validated = _validate_sector_frame(frame, observed_at)
    ranked = validated.sort_values(["change_pct", "name"], ascending=[False, True], kind="stable")
    rows = _classify_rows(ranked, previous)
    strongest = tuple(rows[:limit])
    used = {row.name for row in strongest}
    weakest = tuple(row for row in reversed(rows) if row.name not in used)[:limit]
    watch = tuple(row for row in strongest if row.persistence in {"high", "medium"})
    return SectorAnalysis(strongest, weakest, watch, len(rows))
```

- [ ] **Step 4: Add table-driven transition tests**

```python
@pytest.mark.parametrize(
    ("previous_rank", "current_rank", "change", "breadth", "activity_delta", "rotation"),
    [
        (None, 4, 2.0, 60.0, None, "new_start"),
        (8, 8, 2.0, 60.0, 0.0, "continuing"),
        (12, 5, 3.0, 60.0, 2.0, "accelerating"),
        (8, 5, 2.0, 45.0, 2.0, "diverging"),
        (8, 30, -1.0, 35.0, -2.0, "retreating"),
    ],
)
def test_rotation_boundaries(previous_rank, current_rank, change, breadth, activity_delta, rotation):
    current = {
        "rank": current_rank,
        "change_pct": change,
        "breadth_pct": breadth,
        "activity_percentile": 50.0 + (activity_delta or 0.0),
    }
    previous = None if previous_rank is None else {
        "rank": previous_rank,
        "change_pct": 1.0,
        "breadth_pct": 55.0,
        "activity_percentile": 50.0,
    }
    assert classify_sector(current, previous).rotation == rotation
```

- [ ] **Step 5: Implement rotation, persistence and crowding helpers, then run tests**

Implement public `classify_sector(current, previous) -> SectorClassification` plus private `_rotation`, `_persistence` and `_crowding_risk` helpers as direct translations of the approved design thresholds. `SectorClassification` contains `rotation`, `persistence` and `crowding_risk` strings. Return `unavailable` whenever required previous, breadth or activity evidence is absent. `analyze_sectors` calls this function for each ranked row.

Run: `../../.venv/bin/python -m pytest tests/collaborative_report/test_sector_analysis.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the pure analyzer**

```bash
git add src/collaborative_report/sector_analysis.py tests/collaborative_report/test_sector_analysis.py
git commit -m "Add deterministic sector analysis"
```

### Task 2: Industry And Concept Snapshot Gateway

**Files:**
- Modify: `src/collaborative_report/market_data.py`
- Modify: `tests/collaborative_report/test_market_data.py`

- [ ] **Step 1: Write failing normalization and independence tests**

```python
def test_sector_snapshot_normalizes_industry_fields():
    raw = pd.DataFrame({
        "板块名称": ["有色金属"], "涨跌幅": [2.5], "上涨家数": [40], "下跌家数": [20],
        "换手率": [3.2], "领涨股票": ["示例股份"], "领涨股票-涨跌幅": [7.1],
    })
    gateway = MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT)
    result = gateway.get_sector_snapshot("industry")
    assert result.frame.loc[0, "sector_type"] == "industry"
    assert result.frame.loc[0, "name"] == "有色金属"
    assert result.frame.loc[0, "change_pct"] == 2.5


def test_industry_failure_does_not_prevent_concept_fetch():
    gateway = MarketDataGateway(
        industry_sector_fetcher=Mock(side_effect=RuntimeError("private")),
        concept_sector_fetcher=lambda: valid_concept_frame(),
        clock=lambda: OBSERVED_AT,
    )
    with pytest.raises(ValueError, match="^industry sector provider unavailable$"):
        gateway.get_sector_snapshot("industry")
    assert gateway.get_sector_snapshot("concept").frame["sector_type"].eq("concept").all()
```

- [ ] **Step 2: Run focused tests and confirm constructor/API failures**

Run: `../../.venv/bin/python -m pytest tests/collaborative_report/test_market_data.py -q`

Expected: FAIL because `industry_sector_fetcher`, `concept_sector_fetcher` and `get_sector_snapshot` do not exist.

- [ ] **Step 3: Add independent fetchers and normalized contract**

```python
def get_sector_snapshot(self, sector_type: str) -> MarketDataset:
    if sector_type not in {"industry", "concept"}:
        raise ValueError("sector type invalid")
    observed_at = self._observed_at()
    try:
        raw = self._sector_fetcher(sector_type)()
    except Exception:
        raise ValueError(f"{sector_type} sector provider unavailable") from None
    frame, warnings, source_timestamp = _normalize_sector_snapshot(raw, sector_type, observed_at)
    return MarketDataset(
        frame,
        f"akshare.eastmoney_{sector_type}_boards",
        self._received_at(observed_at),
        warnings,
        source_timestamp,
    )
```

Use `akshare.stock_board_industry_name_em` and `akshare.stock_board_concept_name_em` as defaults. Match known Chinese/English aliases through `_matching_column`; keep optional values nullable and reject duplicate names, invalid changes and future timestamps.

- [ ] **Step 4: Add malformed, stale, duplicate and optional-field tests**

Ensure raw exception text never appears in raised messages or formatted tracebacks. Verify that missing breadth/activity fields remain `pd.NA`, while missing `change_pct` excludes the row and increments a fixed warning count.

- [ ] **Step 5: Run market-data tests and commit**

Run: `../../.venv/bin/python -m pytest tests/collaborative_report/test_market_data.py -q`

Expected: PASS.

```bash
git add src/collaborative_report/market_data.py tests/collaborative_report/test_market_data.py
git commit -m "Add validated industry and concept snapshots"
```

### Task 3: Prior Post-Market Sector State

**Files:**
- Modify: `src/collaborative_report/runner.py`
- Modify: `tests/collaborative_report/test_runner.py`

- [ ] **Step 1: Write failing state validation tests**

```python
def test_prior_sector_state_accepts_only_earlier_completed_postmarket(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "report_key": "2026-09-03-postmarket",
        "mode": "postmarket",
        "trading_date": "2026-09-03",
        "final_state": "sent",
        "test_email": False,
        "sector_state": [{
            "sector_type": "industry", "name": "有色金属", "rank": 3,
            "change_pct": 2.5, "breadth_pct": 60.0,
            "activity_percentile": 75.0,
            "source_timestamp": "2026-09-03T15:00:00+08:00",
        }],
    }), encoding="utf-8")
    session = SimpleNamespace(
        trading_date=date(2026, 9, 4),
        now_shanghai=datetime(2026, 9, 4, 16, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    state = _load_prior_sector_state(path, session)
    assert state[0]["name"] == "有色金属"


@pytest.mark.parametrize("field,value", [
    ("mode", "premarket"), ("final_state", "prepared"), ("test_email", True),
    ("trading_date", "2026-09-04"),
])
def test_prior_sector_state_rejects_wrong_artifact_identity(tmp_path, field, value):
    manifest = {
        "report_key": "2026-09-03-postmarket", "mode": "postmarket",
        "trading_date": "2026-09-03", "final_state": "sent", "test_email": False,
        "sector_state": [],
    }
    manifest[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    session = SimpleNamespace(
        trading_date=date(2026, 9, 4),
        now_shanghai=datetime(2026, 9, 4, 16, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)
```

- [ ] **Step 2: Run focused tests and confirm the loader is missing**

Run: `../../.venv/bin/python -m pytest tests/collaborative_report/test_runner.py -q`

Expected: FAIL because `_load_prior_sector_state` is undefined.

- [ ] **Step 3: Implement strict loading and sanitized serialization**

```python
def _load_prior_sector_state(path: Path | None, session: ReportSession):
    if path is None or not path.is_file():
        raise ValueError("prior sector state unavailable")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("prior sector state unavailable") from None
    if not isinstance(manifest, dict):
        raise ValueError("prior sector state unavailable")
    if (
        manifest.get("mode") != "postmarket"
        or manifest.get("final_state") != "sent"
        or manifest.get("test_email") is not False
        or date.fromisoformat(manifest["trading_date"]) >= session.trading_date
    ):
        raise ValueError("prior sector state unavailable")
    return validate_sector_state(manifest.get("sector_state"), session.now_shanghai)
```

Add `_sector_state` to serialize only normalized metrics, classifications, sector type/name, ranks and source timestamps. Reject unknown keys, non-finite values, duplicate identities and timestamps after the report checkpoint.

- [ ] **Step 4: Test manifest privacy and degraded missing state**

Assert that credentials, raw provider text and exception messages cannot enter `sector_state`. A missing/corrupt prior file must produce `first_observation`, not fail the report.

- [ ] **Step 5: Run runner tests and commit**

Run: `../../.venv/bin/python -m pytest tests/collaborative_report/test_runner.py -q`

Expected: PASS.

```bash
git add src/collaborative_report/runner.py tests/collaborative_report/test_runner.py
git commit -m "Persist validated postmarket sector state"
```

### Task 4: Runner Modules, Decision Summary And Candidate Context

**Files:**
- Modify: `src/collaborative_report/models.py`
- Modify: `src/collaborative_report/runner.py`
- Modify: `tests/collaborative_report/test_runner.py`

- [ ] **Step 1: Write failing integration tests**

```python
def valid_sector_frame(sector_type):
    return pd.DataFrame([{
        "sector_type": sector_type, "name": "有色金属", "change_pct": 2.5,
        "advance_count": 60, "decline_count": 40, "turnover_rate": 3.2,
        "leader_name": "示例股份", "leader_code": "600000", "leader_change_pct": 7.1,
    }])


def test_postmarket_builds_independent_sector_modules_and_decision_summary(tmp_path, deps, settings):
    deps.gateway.get_sector_snapshot.side_effect = [
        dataset(valid_sector_frame("industry")), dataset(valid_sector_frame("concept")),
    ]
    result = run_report(ReportMode.POSTMARKET, settings, deps=deps, output_root=tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["module_statuses"]["industry_sectors"] == "ok"
    assert manifest["module_statuses"]["concept_sectors"] == "ok"
    assert manifest["module_statuses"]["decision_summary"] == "ok"


def test_concept_failure_keeps_industry_and_report_available(tmp_path, deps, settings):
    deps.gateway.get_sector_snapshot.side_effect = [
        dataset(valid_sector_frame("industry")), ValueError("concept sector provider unavailable"),
    ]
    result = run_report(ReportMode.POSTMARKET, settings, deps=deps, output_root=tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["module_statuses"]["industry_sectors"] == "ok"
    assert manifest["module_statuses"]["concept_sectors"] == "unavailable"
    assert result.final_state is FinalState.PREVIEWED


def test_market_overview_contains_decision_fields(tmp_path, deps, settings):
    result = run_report(ReportMode.POSTMARKET, settings, deps=deps, output_root=tmp_path)
    payload = result.modules["market"].payload
    assert {"股票数量", "上涨家数", "下跌家数", "平盘家数", "上涨占比", "成交额", "市场温度"} <= payload.keys()
```

- [ ] **Step 2: Add backward-compatible candidate context**

```python
@dataclass(frozen=True)
class Candidate:
    # Existing fields remain unchanged and in their existing order.
    industry_sector: str = ""
    concept_sectors: tuple[str, ...] = ()
    sector_rotation: str = ""
    sector_persistence: str = ""
```

Use `dataclasses.replace` after technical screening to add sector context. Sector context may re-rank equal technical scores but must not create or remove candidates.

- [ ] **Step 3: Integrate two isolated fetch/analyze paths**

Add `_run_sector_module("industry", ...)` and `_run_sector_module("concept", ...)`. Convert failures into fixed warnings, store complete structured payloads, and build `decision_summary` from usable market breadth, existing risk results and sector agreement. Expand `_market_payload` with flat count, advance ratio, summed valid traded amount and a deterministic temperature label. Limit-up/down and market style render as `不可用` unless validated source fields exist. If market breadth is unavailable, direction and action stance are `数据不足，建议观望`.

- [ ] **Step 4: Add degradation and no-auto-trade assertions**

Cover both sources unavailable, partial rows, empty portfolio, missing holding prices and sector evidence that conflicts with technical screening. Assert that mail/trading fakes are never called in preview tests and that candidate count never increases because of sector output.

- [ ] **Step 5: Run runner/model tests and commit**

Run: `../../.venv/bin/python -m pytest tests/collaborative_report/test_runner.py tests/collaborative_report/test_screener.py -q`

Expected: PASS.

```bash
git add src/collaborative_report/models.py src/collaborative_report/runner.py tests/collaborative_report/test_runner.py
git commit -m "Integrate sector modules into postmarket reports"
```

### Task 5: Structured HTML And Text Rendering

**Files:**
- Modify: `src/collaborative_report/report.py`
- Modify: `templates/collaborative_report.html.j2`
- Modify: `tests/collaborative_report/test_report.py`

- [ ] **Step 1: Write failing parity and layout tests**

```python
def test_sector_tables_have_html_text_parity():
    sector_row = {
        "rank": 1, "name": "有色金属", "change_pct": 2.5,
        "breadth_pct": 60.0, "activity_percentile": 75.0,
        "leader_name": "示例股份", "leader_change_pct": 7.1,
        "rotation": "延续", "persistence": "高", "crowding_risk": "中",
    }
    modules = {
        "industry_sectors": ModuleResult(
            "industry_sectors", "ok", GENERATED_AT,
            {"valid_count": 1, "strongest": [sector_row], "weakest": [], "watch": []},
        ),
    }
    rendered = render_report(
        ReportMode.POSTMARKET,
        date(2026, 9, 4),
        modules=modules,
        short_term_candidates=(),
        swing_candidates=(),
        generated_at=GENERATED_AT,
    )
    for value in ("有色金属", "2.50%", "60.0%", "延续", "高", "中"):
        assert value in rendered.html
        assert value in rendered.text
    assert 'class="sector-table-wrap"' in rendered.html


def test_unavailable_sector_fields_are_not_rendered_as_zero():
    row = {
        "rank": 1, "name": "有色金属", "change_pct": 2.5,
        "breadth_pct": None, "activity_percentile": None,
        "leader_name": None, "leader_change_pct": None,
        "rotation": "首次观察", "persistence": "不可用", "crowding_risk": "不可用",
    }
    modules = {"industry_sectors": ModuleResult(
        "industry_sectors", "partial", GENERATED_AT,
        {"valid_count": 1, "strongest": [row], "weakest": [], "watch": []},
    )}
    rendered = render_report(
        ReportMode.POSTMARKET, date(2026, 9, 4), modules=modules,
        generated_at=GENERATED_AT,
    )
    sector_section = rendered.text.split("行业板块", 1)[1]
    assert "不可用" in sector_section
    assert "0.0%" not in sector_section
```

- [ ] **Step 2: Add sector view contracts**

```python
@dataclass(frozen=True)
class _SectorView:
    rank: str
    name: str
    change: str
    breadth: str
    activity: str
    leader: str
    leader_change: str
    rotation: str
    persistence: str
    crowding: str
```

Convert structured module payload rows with strict keys. Format nullable values as `不可用`, percentages consistently and warnings once.

- [ ] **Step 3: Add decision-first template sections and responsive tables**

Place `decision_summary` first. Render separate industry/concept strongest and weakest tables, coverage, source time, rotation summary and watch reasons. Wrap tables in `.sector-table-wrap { overflow-x: auto; }`; retain readable fixed font sizes and existing email-safe colors.

- [ ] **Step 4: Run renderer tests and inspect generated HTML**

Run: `../../.venv/bin/python -m pytest tests/collaborative_report/test_report.py -q`

Expected: PASS.

Generate a deterministic fixture report through the existing report test helper and open the HTML at desktop and narrow widths. Confirm no overlapping text, hidden columns or nested cards.

- [ ] **Step 5: Commit rendering**

```bash
git add src/collaborative_report/report.py templates/collaborative_report.html.j2 tests/collaborative_report/test_report.py
git commit -m "Render complete sector report sections"
```

### Task 6: Previous Post-Market Artifact Workflow

**Files:**
- Modify: `.github/workflows/01-collaborative-report.yml`
- Modify: `tests/collaborative_report/test_workflow.py`

- [ ] **Step 1: Write failing workflow contract tests**

```python
def test_workflow_selects_latest_earlier_completed_postmarket_artifact():
    workflow = _workflow()
    step = _step(workflow, "Extract prior sector state")
    assert "report-" in step["run"]
    assert "-postmarket" in step["run"]
    assert "PRIOR_SECTOR_STATE_PATH" in _step(workflow, "Run collaborative report")["env"]


def test_prior_sector_download_never_uses_preview_or_failed_artifacts():
    script = _step(_workflow(), "Extract prior sector state")["run"]
    assert 'final_state == "sent"' in script
    assert 'test_email is False' in script
    assert "trading_date < current_trading_date" in script
```

- [ ] **Step 2: Add bounded prior-artifact selection**

Query unexpired `report-YYYY-MM-DD-postmarket` artifacts, sort dates earlier than the current trading date descending, and inspect at most five candidates. Download only the selected candidate; accept a manifest only after report key, mode, trading date, final state and `test_email` validate.

- [ ] **Step 3: Pass a contained manifest path to the runner**

Set `PRIOR_SECTOR_STATE_PATH` only when extraction succeeds. On failure set the fixed warning `prior_postmarket_sector_state_unavailable`; do not write remote response text or stderr into the workflow summary.

- [ ] **Step 4: Run workflow tests and commit**

Run: `../../.venv/bin/python -m pytest tests/collaborative_report/test_workflow.py -q`

Expected: PASS.

```bash
git add .github/workflows/01-collaborative-report.yml tests/collaborative_report/test_workflow.py
git commit -m "Load prior postmarket sector state"
```

### Task 7: Documentation, Full Validation And No-Email Preview

**Files:**
- Modify: `docs/collaborative-report.md`
- Modify: `docs/CHANGELOG.md`

- [ ] **Step 1: Document the final behavior**

Add the industry/concept source contract, strongest/weakest limits, rotation thresholds, persistence/crowding definitions, first-observation behavior, partial-data policy, state provenance and candidate re-ranking boundary to `docs/collaborative-report.md`.

Add one flat `[新功能]` line under `[Unreleased]` in `docs/CHANGELOG.md`.

- [ ] **Step 2: Run focused and full validation**

Run:

```bash
../../.venv/bin/python -m pytest tests/collaborative_report -q
../../.venv/bin/python -m py_compile src/collaborative_report/sector_analysis.py src/collaborative_report/market_data.py src/collaborative_report/runner.py src/collaborative_report/report.py
git diff --check
```

Expected: all tests pass, compilation succeeds and `git diff --check` prints nothing.

- [ ] **Step 3: Commit documentation**

```bash
git add docs/collaborative-report.md docs/CHANGELOG.md
git commit -m "Document complete sector reports"
```

- [ ] **Step 4: Push and open a draft PR**

Push `codex/complete-sector-report` to `user-fork`. Create a draft PR targeting the fork's `main` with the root cause, behavior, tests, partial-data policy, rollback and a screenshot of the deterministic HTML fixture.

- [ ] **Step 5: Run a no-email post-market preview**

Dispatch `01-collaborative-report.yml` with `mode=postmarket`, `force=true`, `preview_only=true`, `test_email=false` and `reconcile_sent=false` on the feature branch.

Expected on a trading day: industry and concept modules contain valid coverage, report and diagnostic artifacts upload, no sent marker is created, and no email is sent. On a non-trading day: record `non_trading_day_skip` and retain deterministic tests as the available report evidence.

- [ ] **Step 6: Final review and merge gate**

Confirm all blocking CI checks pass, the PR body matches the final diff, source timestamps are current, HTML/text contain the same sector rows, and no raw provider or credential content appears in artifacts. Merge only after these checks and the already-authorized user review policy are satisfied.
