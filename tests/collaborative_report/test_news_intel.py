from datetime import datetime
from zoneinfo import ZoneInfo

from src.collaborative_report.news_intel import fetch_market_news


NOW = datetime(2026, 9, 9, 16, 30, tzinfo=ZoneInfo("Asia/Shanghai"))


class FakeService:
    def list_source_templates(self, **filters):
        assert filters == {"market": "cn", "source_type": "newsnow"}
        return {
            "items": [
                {"template_id": template_id, "name": name, "url": f"https://news.example/{template_id}"}
                for template_id, name in (
                    ("orz-cls", "财联社"),
                    ("orz-eastmoney", "东方财富"),
                    ("orz-sina-finance", "新浪财经"),
                    ("orz-xueqiu", "雪球"),
                    ("orz-baidu", "百度"),
                )
            ]
        }

    def test_source(self, template):
        template_id = template["template_id"]
        return {
            "sample_items": [
                {
                    "title": f"{template_id} 线索 {index}",
                    "source": template_id,
                    "url": "https://news.example/item",
                    "published_at": "2026-09-09T08:25:00",
                }
                for index in range(3)
            ]
        }


def test_market_news_keeps_financial_clues_separate_from_macro_attention() -> None:
    result = fetch_market_news(observed_at=NOW, service_factory=FakeService)

    assert result.status == "ok"
    assert result.payload["财经线索数"] == 8
    assert result.payload["宏观舆情线索数"] == 3
    assert len(result.payload["可用来源"]) == 5
    assert "orz-baidu" not in " ".join(
        item["标题"] for key, item in result.payload.items() if key.startswith("线索")
    )
    assert all(
        item["证据用途"] == "事件线索，未独立核验"
        for key, item in result.payload.items()
        if key.startswith("线索")
    )
    assert result.warnings == ("聚合新闻仅作事件线索，关键事实及发布时间需核验原始来源",)


def test_market_news_rejects_stale_future_and_undated_items() -> None:
    class InvalidItems(FakeService):
        def test_source(self, template):
            return {
                "sample_items": [
                    {"title": "stale", "source": "x", "published_at": "2026-09-01T08:00:00"},
                    {"title": "future", "source": "x", "published_at": "2026-09-10T08:00:00"},
                    {"title": "undated", "source": "x", "published_at": None},
                ]
            }

    result = fetch_market_news(observed_at=NOW, service_factory=InvalidItems)

    assert result.status == "unavailable"
    assert result.payload["财经线索数"] == 0
    assert result.payload["宏观舆情线索数"] == 0
    assert "未获取到可用财经新闻线索" in result.warnings


def test_market_news_isolates_failed_sources_without_exposing_errors() -> None:
    class PartialService(FakeService):
        def test_source(self, template):
            if template["template_id"] != "orz-cls":
                raise RuntimeError("secret token=https://provider.example")
            return super().test_source(template)

    result = fetch_market_news(observed_at=NOW, service_factory=PartialService)

    assert result.status == "partial"
    assert result.payload["财经线索数"] == 2
    assert "部分新闻源不可用（4个）" in result.warnings
    assert "secret" not in repr(result)
