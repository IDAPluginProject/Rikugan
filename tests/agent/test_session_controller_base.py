"""Host-agnostic session orchestration in rikugan.ui.session_controller_base.

Covers how the controller builds providers from the active config (including
the per-provider TLS setting) and its multi-tab session bookkeeping.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.mocks.ida_mock import install_ida_mocks

install_ida_mocks()

from rikugan.core.config import RikuganConfig  # noqa: E402
from rikugan.core.types import Message, Role  # noqa: E402
from rikugan.tools.registry import ToolRegistry  # noqa: E402
from rikugan.ui.session_controller_base import SessionControllerBase  # noqa: E402


def _runner(running: bool) -> SimpleNamespace:
    return SimpleNamespace(agent_loop=SimpleNamespace(is_running=running))


class _ControllerTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg = RikuganConfig()
        self.cfg._config_dir = tempfile.mkdtemp()
        self.cfg.checkpoint_auto_save = False
        self.cfg.provider.name = "openai"
        self.cfg.provider.model = "gpt-test"
        self.cfg.provider.api_key = "sk-test"
        self.cfg.provider.api_base = "https://llm.internal/v1"
        self.ctrl = SessionControllerBase(
            self.cfg,
            tool_registry_factory=lambda _cfg: ToolRegistry(),
            database_path_getter=lambda: "",
            host_name="Test",
        )
        self.ctrl._runtime_init_done.wait(timeout=10.0)

    def tearDown(self):
        self.ctrl.shutdown()


class TestProviderConstruction(_ControllerTestCase):
    def _capture(self):
        provider = MagicMock()
        provider.context_window.return_value = 64000
        patcher = patch.object(self.ctrl.provider_registry, "get_or_create", return_value=provider)
        get_or_create = patcher.start()
        self.addCleanup(patcher.stop)
        return get_or_create, provider

    def test_provider_is_built_from_the_active_settings(self):
        get_or_create, provider = self._capture()

        self.assertIs(self.ctrl.get_provider(), provider)

        get_or_create.assert_called_once_with(
            "openai",
            api_key="sk-test",
            api_base="https://llm.internal/v1",
            model="gpt-test",
            ssl_verify=True,
        )
        provider.ensure_ready.assert_called_once_with()

    def test_ignore_certificate_setting_reaches_the_provider(self):
        get_or_create, _ = self._capture()
        self.cfg.provider.ssl_verify = False

        self.ctrl.get_provider()
        self.ctrl.get_context_window()

        for call in get_or_create.call_args_list:
            self.assertFalse(call.kwargs["ssl_verify"])

    def test_provider_extras_are_forwarded(self):
        get_or_create, _ = self._capture()
        self.cfg.provider.extra = {"custom_context_window": 32000}

        self.ctrl.get_provider()

        self.assertEqual(get_or_create.call_args.kwargs["custom_context_window"], 32000)

    def test_provider_failure_returns_none(self):
        with patch.object(self.ctrl.provider_registry, "get_or_create", side_effect=RuntimeError("bad key")):
            self.assertIsNone(self.ctrl.get_provider())

    def test_context_window_comes_from_the_provider(self):
        self._capture()
        self.assertEqual(self.ctrl.get_context_window(), 64000)

    def test_context_window_falls_back_to_the_config(self):
        self.cfg.provider.context_window = 99000
        with patch.object(self.ctrl.provider_registry, "get_or_create", side_effect=RuntimeError("offline")):
            self.assertEqual(self.ctrl.get_context_window(), 99000)

    def test_start_agent_reports_a_provider_error(self):
        with patch.object(self.ctrl.provider_registry, "get_or_create", side_effect=RuntimeError("bad key")):
            error = self.ctrl.start_agent("hello")
        self.assertEqual(error, "Provider error: bad key")
        self.assertFalse(self.ctrl.is_agent_running)


class TestTabs(_ControllerTestCase):
    def test_starts_with_one_active_tab(self):
        self.assertEqual(self.ctrl.tab_ids, [self.ctrl.active_tab_id])
        self.assertIs(self.ctrl.session, self.ctrl.get_session(self.ctrl.active_tab_id))
        self.assertEqual(self.ctrl.session.provider_name, "openai")
        self.assertEqual(self.ctrl.session.model_name, "gpt-test")

    def test_create_and_switch_tabs(self):
        first = self.ctrl.active_tab_id
        second = self.ctrl.create_tab()
        self.assertNotEqual(first, second)
        self.assertEqual(self.ctrl.active_tab_id, first, "creating a tab does not switch to it")

        self.ctrl.switch_tab(second)
        self.assertEqual(self.ctrl.active_tab_id, second)

    def test_switching_to_an_unknown_tab_is_ignored(self):
        before = self.ctrl.active_tab_id
        self.ctrl.switch_tab("missing")
        self.assertEqual(self.ctrl.active_tab_id, before)

    def test_tab_label_uses_the_first_user_message(self):
        tab = self.ctrl.active_tab_id
        self.assertEqual(self.ctrl.tab_label(tab), "New Chat")
        self.ctrl.session.add_message(Message(role=Role.USER, content="  which game engine does this use  "))
        self.assertEqual(self.ctrl.tab_label(tab), "which game engine do...")
        self.assertEqual(self.ctrl.tab_label("missing"), "New Chat")

    def test_fork_copies_history_without_sharing_it(self):
        source = self.ctrl.active_tab_id
        self.ctrl.session.add_message(Message(role=Role.USER, content="hello"))

        forked_id = self.ctrl.fork_session(source)

        forked = self.ctrl.get_session(forked_id)
        self.assertEqual([m.content for m in forked.messages], ["hello"])
        self.assertEqual(forked.metadata["forked_from"], self.ctrl.session.id)
        forked.add_message(Message(role=Role.USER, content="only in the fork"))
        self.assertEqual(len(self.ctrl.session.messages), 1)

    def test_fork_of_unknown_tab_returns_none(self):
        self.assertIsNone(self.ctrl.fork_session("missing"))

    def test_close_tab_removes_its_session_and_queue(self):
        extra = self.ctrl.create_tab()
        self.ctrl.queue_message("later", tab_id=extra)
        self.assertEqual(self.ctrl.tab_pending_count(extra), 1)

        self.ctrl.close_tab(extra)

        self.assertIsNone(self.ctrl.get_session(extra))
        self.assertEqual(self.ctrl.tab_pending_count(extra), 0)
        self.ctrl.close_tab(extra)  # closing twice is harmless


class TestRunnerState(_ControllerTestCase):
    def test_running_state_is_tracked_per_tab(self):
        idle_tab = self.ctrl.active_tab_id
        busy_tab = self.ctrl.create_tab()
        self.ctrl._runners[busy_tab] = _runner(True)
        self.ctrl._runners[idle_tab] = _runner(False)

        self.assertTrue(self.ctrl.is_tab_running(busy_tab))
        self.assertFalse(self.ctrl.is_tab_running(idle_tab))
        self.assertFalse(self.ctrl.is_agent_running, "the active tab is the idle one")
        self.assertTrue(self.ctrl.any_agent_running)
        self.assertTrue(self.ctrl.has_runners)
        self.assertEqual(self.ctrl.running_tab_ids(), [busy_tab])
        self.assertIs(self.ctrl.get_runner_for_tab(busy_tab), self.ctrl._runners[busy_tab])
        self.ctrl._runners.clear()

    def test_no_runners(self):
        self.assertFalse(self.ctrl.any_agent_running)
        self.assertFalse(self.ctrl.has_runners)
        self.assertIsNone(self.ctrl.get_runner())
        self.assertEqual(self.ctrl.running_tab_ids(), [])


if __name__ == "__main__":
    unittest.main()
