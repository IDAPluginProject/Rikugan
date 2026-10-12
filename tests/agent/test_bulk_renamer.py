"""Tests for the bulk function renaming engine (rikugan.agent.bulk_renamer)."""

from __future__ import annotations

import re
import threading
import unittest
from collections.abc import Callable, Iterator
from typing import Any
from unittest import mock

from rikugan.agent.bulk_renamer import (
    DEEP_ANALYSIS_PROMPT,
    QUICK_ANALYSIS_PROMPT,
    BulkRenamerEngine,
    RenameEvent,
    RenameEventType,
    RenameJob,
    RenameStatus,
)
from rikugan.agent.subagent_manager import SubagentStatus
from rikugan.agent.turn import TurnEvent
from rikugan.core.types import Message, Role

_FUNC_HEADER_RE = re.compile(r"// Function at 0x([0-9a-f]+)")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTools:
    """Minimal stand-in for ToolRegistry.execute()."""

    def __init__(
        self,
        decompile_fail: set[int] | None = None,
        disasm_fail: bool = False,
        rename_fail: set[int] | None = None,
        decompile_text: Callable[[int], str] | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._decompile_fail = decompile_fail or set()
        self._disasm_fail = disasm_fail
        self._rename_fail = rename_fail or set()
        self._decompile_text = decompile_text or (lambda ea: f"int f_{ea:x}(void) {{ return {ea}; }}")

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, dict(arguments)))
        ea = int(arguments["address"], 16)
        if name == "decompile_function":
            if ea in self._decompile_fail:
                raise RuntimeError(f"decompilation failed at 0x{ea:x}")
            return self._decompile_text(ea)
        if name == "read_disassembly":
            if self._disasm_fail:
                raise RuntimeError("no disassembly")
            return f"0x{ea:x}: push rbp"
        if name == "rename_function":
            if ea in self._rename_fail:
                raise RuntimeError(f"rename rejected at 0x{ea:x}")
            return f"Renamed 0x{ea:x}"
        raise AssertionError(f"unexpected tool {name}")

    def calls_named(self, name: str) -> list[dict[str, Any]]:
        return [args for tool, args in self.calls if tool == name]


def _echo_reply(prompt: str) -> str:
    """Default fake LLM: name every function present in the prompt."""
    return "\n".join(f"0x{addr} renamed_{addr}" for addr in _FUNC_HEADER_RE.findall(prompt))


class FakeProvider:
    """Stand-in for LLMProvider.chat() returning a scripted reply."""

    def __init__(self, reply: str | Callable[[str], str] = _echo_reply, error: Exception | None = None) -> None:
        self._reply = reply
        self._error = error
        self.prompts: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def chat(self, messages: list[Message], tools: Any = None, system: str = "") -> Message:
        with self._lock:
            self.calls.append({"messages": messages, "tools": tools})
            self.prompts.append(messages[0].content)
        if self._error is not None:
            raise self._error
        text = self._reply(messages[0].content) if callable(self._reply) else self._reply
        return Message(role=Role.ASSISTANT, content=text)


class FakeManager:
    """Records SubagentManager.register / update_external calls."""

    def __init__(self) -> None:
        self.registered: list[dict[str, Any]] = []
        self.updates: list[tuple[str, SubagentStatus, str, int]] = []
        self._lock = threading.Lock()

    def register(self, **kwargs: Any) -> str:
        with self._lock:
            self.registered.append(kwargs)
            return f"agent-{len(self.registered)}"

    def update_external(self, agent_id: str, status: SubagentStatus, summary: str = "", turn_count: int = 0) -> None:
        with self._lock:
            self.updates.append((agent_id, status, summary, turn_count))


def _make_runner_class(script: Callable[[str], Iterator[TurnEvent]]) -> tuple[type, list[str]]:
    """Build a fake SubagentRunner class whose run_task yields *script(task)*."""
    tasks: list[str] = []

    class _FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def run_task(self, task: str, max_turns: int = 20) -> Iterator[TurnEvent]:
            tasks.append(task)
            yield from script(task)

    return _FakeRunner, tasks


def _make_engine(
    tools: FakeTools | None = None,
    provider: FakeProvider | None = None,
    manager: FakeManager | None = None,
    batch_size: int = 10,
    max_workers: int = 1,
) -> BulkRenamerEngine:
    return BulkRenamerEngine(
        provider=provider or FakeProvider(),  # type: ignore[arg-type]
        tool_registry=tools or FakeTools(),  # type: ignore[arg-type]
        config=mock.MagicMock(),
        batch_size=batch_size,
        max_workers=max_workers,
        subagent_manager=manager,
    )


def _jobs(*addrs: int) -> list[RenameJob]:
    return [RenameJob(address=a, current_name=f"sub_{a:X}") for a in addrs]


def _drain(engine: BulkRenamerEngine) -> list[RenameEvent]:
    events: list[RenameEvent] = []
    while (ev := engine.poll_event()) is not None:
        events.append(ev)
    return events


def _of_type(events: list[RenameEvent], etype: RenameEventType) -> list[RenameEvent]:
    return [e for e in events if e.type == etype]


# ---------------------------------------------------------------------------
# should_skip (auto-generated name detection)
# ---------------------------------------------------------------------------


class TestShouldSkip(unittest.TestCase):
    def test_auto_generated_names_are_detected(self):
        for name in ("sub_401000", "FUN_00401000", "func_dead", "unnamed_1F", "loc_abc123", "sub_DEADBEEF"):
            with self.subTest(name=name):
                self.assertTrue(BulkRenamerEngine.should_skip(name))

    def test_human_or_partial_names_are_not_detected(self):
        for name in ("main", "sub_main", "sub_", "j_sub_401000", "sub_401000_0", "SUB_401000", "parse_header"):
            with self.subTest(name=name):
                self.assertFalse(BulkRenamerEngine.should_skip(name))


# ---------------------------------------------------------------------------
# Queue bookkeeping and control flags
# ---------------------------------------------------------------------------


class TestQueueAndControls(unittest.TestCase):
    def test_enqueue_records_original_names_and_pending_filter(self):
        engine = _make_engine()
        jobs = _jobs(0x1000, 0x2000, 0x3000)
        jobs[1].status = RenameStatus.COMPLETED
        engine.enqueue(jobs)
        self.assertEqual(
            engine._original_names,
            {0x1000: "sub_1000", 0x2000: "sub_2000", 0x3000: "sub_3000"},
        )
        self.assertEqual([j.address for j in engine._pending_jobs()], [0x1000, 0x3000])

    def test_pause_resume_cancel_flags(self):
        engine = _make_engine()
        self.assertTrue(engine._paused.is_set())  # not paused initially
        engine.pause()
        self.assertFalse(engine._paused.is_set())
        engine.resume()
        self.assertTrue(engine._paused.is_set())
        engine.pause()
        engine.cancel_renaming()
        self.assertTrue(engine._cancel.is_set())
        self.assertTrue(engine._paused.is_set(), "cancel must unblock a paused engine")

    def test_poll_event_empty_returns_none(self):
        self.assertIsNone(_make_engine().poll_event())

    def test_paused_worker_does_no_work_until_cancelled(self):
        tools = FakeTools()
        engine = _make_engine(tools=tools)
        engine.enqueue(_jobs(0x1000, 0x2000))
        engine.pause()
        worker = threading.Thread(target=engine._run_quick, daemon=True)
        worker.start()
        worker.join(timeout=0.2)
        self.assertTrue(worker.is_alive(), "worker should block while paused")
        self.assertEqual(tools.calls, [])

        engine.cancel_renaming()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(tools.calls, [])
        done = _of_type(_drain(engine), RenameEventType.ALL_DONE)
        self.assertEqual(len(done), 1)
        self.assertEqual((done[0].completed, done[0].total), (0, 2))

    def test_start_renaming_runs_quick_mode_in_background_thread(self):
        engine = _make_engine()
        engine.enqueue(_jobs(0x1000))
        engine.start_renaming(deep=False)
        assert engine._thread is not None
        self.assertEqual(engine._thread.name, "rikugan-bulk-renamer")
        engine._thread.join(timeout=5)
        self.assertFalse(engine._thread.is_alive())
        self.assertEqual(engine._jobs[0].status, RenameStatus.COMPLETED)
        self.assertEqual(engine._jobs[0].new_name, "renamed_1000")


# ---------------------------------------------------------------------------
# Quick-mode reply parsing
# ---------------------------------------------------------------------------


class TestQuickLlmCall(unittest.TestCase):
    def test_parses_only_well_formed_address_name_lines(self):
        reply = (
            "Here are the names:\n"
            "0x401000 parse_header\n"
            "   0x401080   send_packet   \n"
            "0xDEADBEEF Init_Crypto\n"
            "- 0x402000 bulleted_name\n"
            "0x403000 has trailing words\n"
            "401100 missing_prefix\n"
            "0x404000\n"
            "```\n"
        )
        engine = _make_engine(provider=FakeProvider(reply=reply))
        result = engine._quick_llm_call("prompt")
        self.assertEqual(
            result,
            {0x401000: "parse_header", 0x401080: "send_packet", 0xDEADBEEF: "Init_Crypto"},
        )

    def test_sends_single_user_message_without_tools(self):
        provider = FakeProvider(reply="")
        engine = _make_engine(provider=provider)
        self.assertEqual(engine._quick_llm_call("the prompt"), {})
        self.assertEqual(len(provider.calls), 1)
        call = provider.calls[0]
        self.assertIsNone(call["tools"])
        self.assertEqual(len(call["messages"]), 1)
        self.assertEqual(call["messages"][0].role, Role.USER)
        self.assertEqual(call["messages"][0].content, "the prompt")

    def test_duplicate_address_last_suggestion_wins(self):
        engine = _make_engine(provider=FakeProvider(reply="0x10 first\n0x10 second"))
        self.assertEqual(engine._quick_llm_call("p"), {0x10: "second"})


# ---------------------------------------------------------------------------
# Applying quick-mode results
# ---------------------------------------------------------------------------


class TestApplyQuickResults(unittest.TestCase):
    def test_successful_rename_marks_completed_and_emits_event(self):
        tools = FakeTools()
        engine = _make_engine(tools=tools)
        job = RenameJob(address=0x401000, current_name="sub_401000")
        engine._apply_quick_results([job], {0x401000: "parse_header"})

        self.assertEqual(job.status, RenameStatus.COMPLETED)
        self.assertEqual(job.new_name, "parse_header")
        self.assertEqual(
            tools.calls_named("rename_function"),
            [{"address": "0x401000", "new_name": "parse_header"}],
        )
        events = _drain(engine)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, RenameEventType.JOB_COMPLETED)
        self.assertEqual(
            (events[0].address, events[0].current_name, events[0].new_name),
            (0x401000, "sub_401000", "parse_header"),
        )

    def test_missing_suggestion_fails_without_calling_rename(self):
        tools = FakeTools()
        engine = _make_engine(tools=tools)
        job = RenameJob(address=0x401000, current_name="sub_401000")
        engine._apply_quick_results([job], {0x999: "other"})

        self.assertEqual(job.status, RenameStatus.FAILED)
        self.assertEqual(job.error, "No name suggested by LLM")
        self.assertEqual(tools.calls_named("rename_function"), [])
        events = _drain(engine)
        self.assertEqual([e.type for e in events], [RenameEventType.JOB_ERROR])
        self.assertEqual(events[0].error, "No name suggested by LLM")

    def test_rename_tool_exception_marks_failed(self):
        tools = FakeTools(rename_fail={0x2000})
        engine = _make_engine(tools=tools)
        ok, bad = _jobs(0x1000, 0x2000)
        engine._apply_quick_results([ok, bad], {0x1000: "good", 0x2000: "bad"})

        self.assertEqual(ok.status, RenameStatus.COMPLETED)
        self.assertEqual(bad.status, RenameStatus.FAILED)
        self.assertEqual(bad.new_name, "bad")
        self.assertIn("rename rejected at 0x2000", bad.error)
        types = [e.type for e in _drain(engine)]
        self.assertEqual(types, [RenameEventType.JOB_COMPLETED, RenameEventType.JOB_ERROR])

    def test_reports_incremental_progress_to_manager(self):
        manager = FakeManager()
        engine = _make_engine(manager=manager)
        engine._apply_quick_results(_jobs(0x1, 0x2), {0x1: "a", 0x2: "b"}, agent_id="agent-x")
        self.assertEqual(
            manager.updates,
            [
                ("agent-x", SubagentStatus.RUNNING, "1/2", 1),
                ("agent-x", SubagentStatus.RUNNING, "2/2", 2),
            ],
        )


# ---------------------------------------------------------------------------
# Quick mode end-to-end (run synchronously)
# ---------------------------------------------------------------------------


class TestRunQuick(unittest.TestCase):
    def test_happy_path_renames_all_and_builds_prompt(self):
        tools = FakeTools()
        provider = FakeProvider()
        engine = _make_engine(tools=tools, provider=provider)
        engine.enqueue(_jobs(0x1000, 0x2000, 0x3000))
        engine._run_quick()

        self.assertEqual([j.status for j in engine._jobs], [RenameStatus.COMPLETED] * 3)
        self.assertEqual([j.new_name for j in engine._jobs], ["renamed_1000", "renamed_2000", "renamed_3000"])
        self.assertEqual(engine._jobs[0].decompiled_code, "int f_1000(void) { return 4096; }")

        self.assertEqual(len(provider.prompts), 1)
        prompt = provider.prompts[0]
        self.assertTrue(prompt.startswith(QUICK_ANALYSIS_PROMPT))
        self.assertIn("// Function at 0x1000 (current name: sub_1000)", prompt)
        self.assertIn("// Decompiled:\nint f_2000", prompt)
        self.assertIn("// Disassembly:\n0x3000: push rbp", prompt)
        self.assertEqual(
            tools.calls_named("read_disassembly")[0],
            {"address": "0x1000", "count": 100},
        )

        events = _drain(engine)
        self.assertEqual(
            [e.type for e in events[:3]],
            [RenameEventType.JOB_STARTED] * 3,
            "all decompilation happens before any LLM call",
        )
        self.assertEqual(len(_of_type(events, RenameEventType.JOB_COMPLETED)), 3)
        self.assertEqual(events[-1].type, RenameEventType.ALL_DONE)
        self.assertEqual((events[-1].completed, events[-1].total), (3, 3))

    def test_decompile_failure_excludes_job_from_prompt(self):
        tools = FakeTools(decompile_fail={0x2000})
        provider = FakeProvider()
        engine = _make_engine(tools=tools, provider=provider)
        engine.enqueue(_jobs(0x1000, 0x2000, 0x3000))
        engine._run_quick()

        failed = engine._jobs[1]
        self.assertEqual(failed.status, RenameStatus.FAILED)
        self.assertIn("decompilation failed at 0x2000", failed.error)
        self.assertNotIn("0x2000", provider.prompts[0])
        self.assertEqual(engine._jobs[0].status, RenameStatus.COMPLETED)
        self.assertEqual(engine._jobs[2].status, RenameStatus.COMPLETED)
        renamed = [a["address"] for a in tools.calls_named("rename_function")]
        self.assertEqual(renamed, ["0x1000", "0x3000"])

        events = _drain(engine)
        errors = _of_type(events, RenameEventType.JOB_ERROR)
        self.assertEqual([e.address for e in errors], [0x2000])
        self.assertEqual((events[-1].completed, events[-1].total), (3, 3))

    def test_disassembly_failure_is_tolerated(self):
        provider = FakeProvider()
        engine = _make_engine(tools=FakeTools(disasm_fail=True), provider=provider)
        engine.enqueue(_jobs(0x1000))
        engine._run_quick()
        self.assertEqual(engine._jobs[0].status, RenameStatus.COMPLETED)
        self.assertNotIn("// Disassembly:", provider.prompts[0])

    def test_all_decompiles_fail_skips_llm(self):
        provider = FakeProvider()
        engine = _make_engine(tools=FakeTools(decompile_fail={0x1000, 0x2000}), provider=provider)
        engine.enqueue(_jobs(0x1000, 0x2000))
        engine._run_quick()
        self.assertEqual(provider.calls, [])
        done = _of_type(_drain(engine), RenameEventType.ALL_DONE)
        self.assertEqual(len(done), 1)
        self.assertEqual((done[0].completed, done[0].total), (0, 2))

    def test_batch_size_splits_into_sub_batches(self):
        provider = FakeProvider()
        engine = _make_engine(provider=provider, batch_size=2)
        engine.enqueue(_jobs(0x1000, 0x2000, 0x3000, 0x4000, 0x5000))
        engine._run_quick()

        per_prompt = [_FUNC_HEADER_RE.findall(p) for p in provider.prompts]
        self.assertEqual(per_prompt, [["1000", "2000"], ["3000", "4000"], ["5000"]])
        self.assertTrue(all(j.status == RenameStatus.COMPLETED for j in engine._jobs))
        progress = _of_type(_drain(engine), RenameEventType.BATCH_PROGRESS)
        self.assertEqual([p.completed for p in progress], [2, 4, 5])
        self.assertTrue(all(p.total == 5 for p in progress))

    def test_prompt_char_budget_splits_into_sub_batches(self):
        provider = FakeProvider()
        engine = _make_engine(provider=provider, batch_size=100)
        # Budget leaves room for exactly one function part per prompt.
        engine._MAX_PROMPT_CHARS = len(QUICK_ANALYSIS_PROMPT) + 150
        engine.enqueue(_jobs(0x1000, 0x2000, 0x3000))
        engine._run_quick()
        self.assertEqual(len(provider.prompts), 3)
        for prompt in provider.prompts:
            self.assertEqual(len(_FUNC_HEADER_RE.findall(prompt)), 1)

    def test_oversized_function_is_truncated(self):
        provider = FakeProvider()
        tools = FakeTools(decompile_text=lambda ea: "x" * 10_000)
        engine = _make_engine(tools=tools, provider=provider)
        engine.enqueue(_jobs(0x1000))
        engine._run_quick()
        prompt = provider.prompts[0]
        self.assertIn("// ... (truncated)", prompt)
        body = prompt[len(QUICK_ANALYSIS_PROMPT) :]
        self.assertEqual(len(body), engine._MAX_FUNC_CHARS + len("\n// ... (truncated)\n"))
        self.assertNotIn("// Disassembly:", body)

    def test_llm_error_fails_whole_sub_batch(self):
        manager = FakeManager()
        engine = _make_engine(provider=FakeProvider(error=RuntimeError("rate limited")), manager=manager)
        engine.enqueue(_jobs(0x1000, 0x2000))
        engine._run_quick()

        for job in engine._jobs:
            self.assertEqual(job.status, RenameStatus.FAILED)
            self.assertEqual(job.error, "rate limited")
        events = _drain(engine)
        self.assertEqual(len(_of_type(events, RenameEventType.JOB_ERROR)), 2)
        self.assertEqual((events[-1].completed, events[-1].total), (2, 2))
        self.assertEqual(manager.updates[-1], ("agent-1", SubagentStatus.FAILED, "rate limited", 1))

    def test_registers_sub_batch_with_manager(self):
        manager = FakeManager()
        engine = _make_engine(manager=manager)
        engine.enqueue(_jobs(0x1000, 0x2000, 0x3000, 0x4000, 0x5000))
        engine._run_quick()

        self.assertEqual(len(manager.registered), 1)
        reg = manager.registered[0]
        self.assertEqual(reg["name"], "quick_batch_1")
        self.assertEqual(reg["task"], "Quick batch 1 (5 funcs): 0x1000, 0x2000, 0x3000 +2 more")
        self.assertEqual(reg["agent_type"], "bulk_rename(quick)")
        self.assertEqual(reg["category"], "bulk_rename")
        self.assertEqual(manager.updates[0], ("agent-1", SubagentStatus.RUNNING, "", 0))
        self.assertEqual(
            manager.updates[-1],
            ("agent-1", SubagentStatus.COMPLETED, "Renamed 5/5 functions", 5),
        )

    def test_cancel_before_run_leaves_jobs_pending(self):
        tools = FakeTools()
        engine = _make_engine(tools=tools)
        engine.enqueue(_jobs(0x1000, 0x2000))
        engine._cancel.set()
        engine._run_quick()
        self.assertEqual(tools.calls, [])
        self.assertEqual([j.status for j in engine._jobs], [RenameStatus.PENDING] * 2)

    def test_only_pending_jobs_are_processed(self):
        tools = FakeTools()
        engine = _make_engine(tools=tools)
        jobs = _jobs(0x1000, 0x2000)
        jobs[0].status = RenameStatus.COMPLETED
        engine.enqueue(jobs)
        engine._run_quick()
        decompiled = [a["address"] for a in tools.calls_named("decompile_function")]
        self.assertEqual(decompiled, ["0x2000"])


# ---------------------------------------------------------------------------
# Deep mode (SubagentRunner replaced by a scripted fake)
# ---------------------------------------------------------------------------


def _deep_script(final_text: str, turns: int = 2) -> Callable[[str], Iterator[TurnEvent]]:
    def _script(task: str) -> Iterator[TurnEvent]:
        for i in range(turns):
            yield TurnEvent.turn_start(i + 1)
            yield TurnEvent.text_done(f"thinking {i}")
            yield TurnEvent.turn_end(i + 1)
        yield TurnEvent.text_done(final_text)

    return _script


class TestRunDeep(unittest.TestCase):
    def _run_deep(self, engine: BulkRenamerEngine, script: Callable[[str], Iterator[TurnEvent]]) -> list[str]:
        runner_cls, tasks = _make_runner_class(script)
        with mock.patch("rikugan.agent.subagent.SubagentRunner", runner_cls):
            engine._run_deep()
        return tasks

    def test_parses_rename_line_and_applies(self):
        tools = FakeTools()
        manager = FakeManager()
        engine = _make_engine(tools=tools, manager=manager)
        engine.enqueue(_jobs(0x401000))
        tasks = self._run_deep(engine, _deep_script("Analysis done.\nRENAME: 0x401000 decrypt_config", turns=3))

        job = engine._jobs[0]
        self.assertEqual(job.status, RenameStatus.COMPLETED)
        self.assertEqual(job.new_name, "decrypt_config")
        self.assertEqual(
            tools.calls_named("rename_function"),
            [{"address": "0x401000", "new_name": "decrypt_config"}],
        )
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0].startswith(DEEP_ANALYSIS_PROMPT))
        self.assertIn("// Function at 0x401000 (current name: sub_401000)", tasks[0])
        self.assertIn(job.decompiled_code, tasks[0])

        self.assertEqual(manager.registered[0]["name"], "agent_sub_401000")
        self.assertEqual(manager.registered[0]["agent_type"], "bulk_rename(deep)")
        self.assertEqual(
            manager.updates[-1],
            ("agent-1", SubagentStatus.COMPLETED, "Renamed to decrypt_config", 3),
        )

        events = _drain(engine)
        self.assertEqual(
            [e.type for e in events],
            [
                RenameEventType.JOB_STARTED,
                RenameEventType.JOB_COMPLETED,
                RenameEventType.BATCH_PROGRESS,
                RenameEventType.ALL_DONE,
            ],
        )
        self.assertEqual((events[-1].completed, events[-1].total), (1, 1))

    def test_missing_rename_line_fails(self):
        tools = FakeTools()
        engine = _make_engine(tools=tools)
        engine.enqueue(_jobs(0x401000))
        self._run_deep(engine, _deep_script("I think this is a parser."))

        job = engine._jobs[0]
        self.assertEqual(job.status, RenameStatus.FAILED)
        self.assertEqual(job.error, "No RENAME line found in subagent output")
        self.assertEqual(tools.calls_named("rename_function"), [])

    def test_only_last_text_done_is_parsed(self):
        def script(task: str) -> Iterator[TurnEvent]:
            yield TurnEvent.text_done("RENAME: 0x401000 early_guess")
            yield TurnEvent.text_done("RENAME: 0x401000 final_answer")

        engine = _make_engine()
        engine.enqueue(_jobs(0x401000))
        self._run_deep(engine, script)
        self.assertEqual(engine._jobs[0].new_name, "final_answer")

    def test_decompile_failure_skips_subagent(self):
        manager = FakeManager()
        engine = _make_engine(tools=FakeTools(decompile_fail={0x401000}), manager=manager)
        engine.enqueue(_jobs(0x401000))
        tasks = self._run_deep(engine, _deep_script("RENAME: 0x401000 never"))

        self.assertEqual(tasks, [])
        self.assertEqual(engine._jobs[0].status, RenameStatus.FAILED)
        self.assertEqual(manager.updates[-1][1], SubagentStatus.FAILED)
        done = _of_type(_drain(engine), RenameEventType.ALL_DONE)
        self.assertEqual((done[0].completed, done[0].total), (1, 1))

    def test_subagent_exception_marks_failed(self):
        def script(task: str) -> Iterator[TurnEvent]:
            yield TurnEvent.turn_end(1)
            raise RuntimeError("provider exploded")

        manager = FakeManager()
        engine = _make_engine(manager=manager)
        engine.enqueue(_jobs(0x401000))
        self._run_deep(engine, script)

        job = engine._jobs[0]
        self.assertEqual(job.status, RenameStatus.FAILED)
        self.assertEqual(job.error, "provider exploded")
        self.assertEqual(manager.updates[-1], ("agent-1", SubagentStatus.FAILED, "provider exploded", 1))

    def test_rename_tool_failure_marks_failed(self):
        engine = _make_engine(tools=FakeTools(rename_fail={0x401000}))
        engine.enqueue(_jobs(0x401000))
        self._run_deep(engine, _deep_script("RENAME: 0x401000 good_name"))
        job = engine._jobs[0]
        self.assertEqual(job.status, RenameStatus.FAILED)
        self.assertEqual(job.new_name, "good_name")
        self.assertIn("rename rejected", job.error)

    def test_cancel_during_subagent_reports_cancelled(self):
        manager = FakeManager()
        tools = FakeTools()
        engine = _make_engine(tools=tools, manager=manager)
        engine.enqueue(_jobs(0x401000))

        def script(task: str) -> Iterator[TurnEvent]:
            yield TurnEvent.turn_end(1)
            engine.cancel_renaming()
            yield TurnEvent.text_done("RENAME: 0x401000 too_late")

        self._run_deep(engine, script)
        self.assertEqual(manager.updates[-1], ("agent-1", SubagentStatus.CANCELLED, "Cancelled", 1))
        self.assertEqual(tools.calls_named("rename_function"), [])
        self.assertNotEqual(engine._jobs[0].status, RenameStatus.COMPLETED)


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


class TestUndoAll(unittest.TestCase):
    def test_reverts_completed_jobs_to_original_names(self):
        tools = FakeTools()
        engine = _make_engine(tools=tools)
        done, failed = _jobs(0x1000, 0x2000)
        engine.enqueue([done, failed])
        done.status, done.new_name = RenameStatus.COMPLETED, "parse_header"
        failed.status, failed.error = RenameStatus.FAILED, "boom"

        engine.undo_all()

        self.assertEqual(
            tools.calls_named("rename_function"),
            [{"address": "0x1000", "new_name": "sub_1000"}],
        )
        self.assertEqual(done.status, RenameStatus.PENDING)
        self.assertEqual(done.new_name, "")
        self.assertEqual(failed.status, RenameStatus.FAILED)

        events = _drain(engine)
        self.assertEqual([e.type for e in events], [RenameEventType.JOB_COMPLETED, RenameEventType.ALL_DONE])
        self.assertEqual((events[0].address, events[0].current_name, events[0].new_name), (0x1000, "sub_1000", ""))
        self.assertEqual((events[1].completed, events[1].total), (0, 1))

    def test_undo_failure_keeps_job_completed_and_reports_error(self):
        engine = _make_engine(tools=FakeTools(rename_fail={0x1000}))
        (job,) = _jobs(0x1000)
        engine.enqueue([job])
        job.status, job.new_name = RenameStatus.COMPLETED, "parse_header"

        engine.undo_all()

        self.assertEqual(job.status, RenameStatus.COMPLETED)
        self.assertEqual(job.new_name, "parse_header")
        events = _drain(engine)
        self.assertEqual(events[0].type, RenameEventType.JOB_ERROR)
        self.assertTrue(events[0].error.startswith("Undo failed: "))
        self.assertEqual(events[-1].type, RenameEventType.ALL_DONE)

    def test_quick_run_then_undo_round_trip(self):
        tools = FakeTools()
        engine = _make_engine(tools=tools)
        engine.enqueue(_jobs(0x1000, 0x2000))
        engine._run_quick()
        _drain(engine)
        engine.undo_all()

        renames = tools.calls_named("rename_function")
        self.assertEqual(
            renames[-2:],
            [
                {"address": "0x1000", "new_name": "sub_1000"},
                {"address": "0x2000", "new_name": "sub_2000"},
            ],
        )
        # Jobs are pending again, so a re-run would reprocess them.
        self.assertEqual(len(engine._pending_jobs()), 2)


# ---------------------------------------------------------------------------
# Manager status mapping
# ---------------------------------------------------------------------------


class TestUpdateMgrAgent(unittest.TestCase):
    def test_maps_known_statuses_and_ignores_unknown(self):
        manager = FakeManager()
        engine = _make_engine(manager=manager)
        engine._update_mgr_agent("a", "running", "s1", 1)
        engine._update_mgr_agent("a", "cancelled", "s2", 2)
        engine._update_mgr_agent("a", "bogus", "s3", 3)
        engine._update_mgr_agent(None, "completed", "s4", 4)
        self.assertEqual(
            manager.updates,
            [
                ("a", SubagentStatus.RUNNING, "s1", 1),
                ("a", SubagentStatus.CANCELLED, "s2", 2),
            ],
        )


if __name__ == "__main__":
    unittest.main()
