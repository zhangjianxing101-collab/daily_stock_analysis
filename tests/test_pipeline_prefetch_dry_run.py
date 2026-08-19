# -*- coding: utf-8 -*-
"""
Regression tests for prefetch behavior in StockAnalysisPipeline.run().
"""

import os
import sys
import logging
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.core.pipeline import StockAnalysisPipeline


class TestPipelinePrefetchBehavior(unittest.TestCase):
    @staticmethod
    def _build_pipeline(process_result):
        pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
        pipeline.max_workers = 1
        pipeline.fetcher_manager = MagicMock()
        pipeline.db = MagicMock()
        pipeline.db.has_today_data.return_value = False
        pipeline.process_single_stock = MagicMock(return_value=process_result)
        pipeline.config = SimpleNamespace(
            stock_list=["000001"],
            refresh_stock_list=lambda: None,
            single_stock_notify=False,
            report_type="simple",
            analysis_delay=0,
        )
        return pipeline

    def test_run_dry_run_skips_stock_name_prefetch(self):
        pipeline = self._build_pipeline(process_result=None)

        pipeline.run(stock_codes=["000001"], dry_run=True, send_notification=False)

        pipeline.fetcher_manager.prefetch_stock_names.assert_not_called()

    def test_run_non_dry_run_prefetches_stock_names(self):
        pipeline = self._build_pipeline(process_result=SimpleNamespace(code="000001"))

        pipeline.run(stock_codes=["000001"], dry_run=False, send_notification=False)

        pipeline.fetcher_manager.prefetch_stock_names.assert_called_once_with(
            ["000001"], use_bulk=False
        )

    @patch("src.core.pipeline.logger")
    def test_run_logs_stock_count_without_stock_code_list(self, mock_logger):
        pipeline = self._build_pipeline(process_result=None)

        pipeline.run(stock_codes=["600519", "000001"], dry_run=True, send_notification=False)

        logged = " ".join(str(call) for call in mock_logger.info.call_args_list)
        self.assertIn("2", logged)
        self.assertNotIn("600519", logged)
        self.assertNotIn("000001", logged)

    def test_privacy_safe_run_redacts_all_pipeline_record_content(self):
        pipeline = self._build_pipeline(process_result=None)

        def emit_private_log(*args, **kwargs):
            logging.getLogger("src.core.pipeline").warning(
                "贵州茅台 600519 price=9876.54 user@example.com "
                "token=super-secret https://provider.example/query?code=600519"
            )
            logging.getLogger("provider.client").error(
                "provider query 贵州茅台 600519 token=other-secret"
            )
            return None

        pipeline.process_single_stock = emit_private_log
        with self.assertLogs(level="INFO") as captured:
            pipeline.run(
                stock_codes=["600519"],
                dry_run=True,
                send_notification=False,
                privacy_safe=True,
            )

        output = "\n".join(captured.output)
        for private_value in (
            "贵州茅台",
            "600519",
            "9876.54",
            "user@example.com",
            "super-secret",
            "other-secret",
            "provider.example",
        ):
            self.assertNotIn(private_value, output)

    def test_privacy_safe_context_is_concurrent_and_normal_runs_remain_compatible(self):
        barrier = threading.Barrier(2)

        def build(message):
            pipeline = self._build_pipeline(process_result=None)

            def emit(*args, **kwargs):
                barrier.wait(timeout=5)
                logging.getLogger("src.core.pipeline").warning(message)
                return None

            pipeline.process_single_stock = emit
            return pipeline

        private = build("PRIVATE 600519 token=secret-value")
        normal = build("NORMAL-COMPATIBLE")
        with self.assertLogs("src.core.pipeline", level="INFO") as captured:
            with ThreadPoolExecutor(max_workers=2) as executor:
                safe_future = executor.submit(
                    private.run,
                    stock_codes=["600519"],
                    dry_run=True,
                    send_notification=False,
                    privacy_safe=True,
                )
                normal_future = executor.submit(
                    normal.run,
                    stock_codes=["000001"],
                    dry_run=True,
                    send_notification=False,
                )
                safe_future.result(timeout=5)
                normal_future.result(timeout=5)

        output = "\n".join(captured.output)
        self.assertNotIn("PRIVATE", output)
        self.assertNotIn("600519", output)
        self.assertNotIn("secret-value", output)
        self.assertIn("NORMAL-COMPATIBLE", output)

    def test_run_dry_run_counts_existing_data_by_effective_trading_date(self):
        pipeline = self._build_pipeline(process_result=None)
        pipeline._resolve_resume_target_date = MagicMock(
            side_effect=[date(2026, 3, 27), date(2026, 3, 26)]
        )
        pipeline.db.has_today_data.side_effect = [True, False]

        pipeline.run(
            stock_codes=["600519", "AAPL"],
            dry_run=True,
            send_notification=False,
        )

        self.assertEqual(
            pipeline.db.has_today_data.call_args_list,
            [
                call("600519", date(2026, 3, 27)),
                call("AAPL", date(2026, 3, 26)),
            ],
        )

    def test_run_uses_one_frozen_reference_time_for_tasks_and_dry_run_stats(self):
        pipeline = self._build_pipeline(process_result=None)
        pipeline._resolve_resume_target_date = MagicMock(
            side_effect=[date(2026, 3, 27), date(2026, 3, 26)]
        )
        pipeline.db.has_today_data.side_effect = [True, False]

        pipeline.run(
            stock_codes=["600519", "AAPL"],
            dry_run=True,
            send_notification=False,
        )

        task_reference_times = [
            call.kwargs["current_time"]
            for call in pipeline.process_single_stock.call_args_list
        ]
        stats_reference_times = [
            call.kwargs["current_time"]
            for call in pipeline._resolve_resume_target_date.call_args_list
        ]

        self.assertEqual(len(task_reference_times), 2)
        self.assertEqual(len(stats_reference_times), 2)
        self.assertEqual(len({id(value) for value in task_reference_times}), 1)
        self.assertEqual(len({id(value) for value in stats_reference_times}), 1)
        self.assertIs(task_reference_times[0], stats_reference_times[0])

    def test_run_uses_supplied_reference_time_for_tasks_and_dry_run_stats(self):
        pipeline = self._build_pipeline(process_result=None)
        reference_time = datetime(2026, 3, 27, 1, 30, tzinfo=timezone.utc)
        pipeline._resolve_resume_target_date = MagicMock(
            side_effect=[date(2026, 3, 27), date(2026, 3, 26)]
        )
        pipeline.db.has_today_data.side_effect = [True, False]

        pipeline.run(
            stock_codes=["600519", "AAPL"],
            dry_run=True,
            send_notification=False,
            current_time=reference_time,
        )

        for process_call in pipeline.process_single_stock.call_args_list:
            self.assertIs(process_call.kwargs["current_time"], reference_time)
        for resolve_call in pipeline._resolve_resume_target_date.call_args_list:
            self.assertIs(resolve_call.kwargs["current_time"], reference_time)


if __name__ == "__main__":
    unittest.main()
