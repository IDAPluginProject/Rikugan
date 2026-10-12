"""Tests for rikugan.agent.subagent_manager.SubagentManager."""

from __future__ import annotations

import threading
import unittest
from collections.abc import Callable, Iterator
from typing import Any, ClassVar
from unittest.mock import patch

from rikugan.agent.agents.network_recon import NETWORK_RECON_MAX_TURNS, build_network_recon_addendum
from rikugan.agent.agents.perks import SUBAGENT_PERKS, build_perks_addendum
from rikugan.agent.agents.report_writer import REPORT_WRITER_MAX_TURNS, build_report_writer_addendum
from rikugan.agent.subagent_manager import SubagentInfo, SubagentManager, SubagentStatus
from rikugan.agent.turn import TurnEvent, TurnEventType
from rikugan.core.types import TokenUsage

_TIMEOUT = 5.0


class FakeRunner:
    """Stand-in for SubagentRunner: records construction and replays a script."""

    instances: ClassVar[list[FakeRunner]] = []
    script: ClassVar[Callable[[FakeRunner], Iterator[TurnEvent]] | None] = None

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.run_args: dict[str, Any] = {}
        FakeRunner.instances.append(self)

    def run_task(self, task: str, max_turns: int = 20, system_addendum: str = "") -> Iterator[TurnEvent]:
        self.run_args = {"task": task, "max_turns": max_turns, "system_addendum": system_addendum}
        script = FakeRunner.script
        if script is None:
            return iter(())
        return script(self)


class _ManagerTestBase(unittest.TestCase):
    def setUp(self) -> None:
        FakeRunner.instances = []
        FakeRunner.script = None
        self.provider = object()
        self.tools = object()
        self.config = object()
        self.skills = object()
        self.mgr = SubagentManager(
            provider=self.provider,  # type: ignore[arg-type]
            tool_registry=self.tools,  # type: ignore[arg-type]
            config=self.config,  # type: ignore[arg-type]
            host_name="TestHost",
            skill_registry=self.skills,  # type: ignore[arg-type]
        )
        patcher = patch("rikugan.agent.subagent.SubagentRunner", FakeRunner)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._join_all_subagent_threads)

    @staticmethod
    def _subagent_threads(agent_id: str | None = None) -> list[threading.Thread]:
        prefix = "rikugan-subagent-" + (agent_id[:6] if agent_id else "")
        return [t for t in threading.enumerate() if t.name.startswith(prefix)]

    def _join(self, agent_id: str) -> None:
        for t in self._subagent_threads(agent_id):
            t.join(_TIMEOUT)
            self.assertFalse(t.is_alive(), "subagent thread did not finish")

    def _join_all_subagent_threads(self) -> None:
        for t in self._subagent_threads():
            t.join(_TIMEOUT)

    def _drain(self) -> list[TurnEvent]:
        events: list[TurnEvent] = []
        while (ev := self.mgr.poll_event()) is not None:
            events.append(ev)
        return events

    def _spawn_and_wait(self, **kwargs: Any) -> str:
        kwargs.setdefault("name", "agent")
        kwargs.setdefault("task", "do things")
        agent_id = self.mgr.spawn(**kwargs)
        self._join(agent_id)
        return agent_id


class TestSpawnLifecycle(_ManagerTestBase):
    def test_successful_run_completes_with_summary_turns_and_usage(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        def script(_runner: FakeRunner) -> Iterator[TurnEvent]:
            yield TurnEvent.turn_start(1)
            yield TurnEvent.text_done("draft")
            yield TurnEvent.turn_end(1)
            yield TurnEvent.usage_update(usage)
            yield TurnEvent.turn_start(2)
            yield TurnEvent.text_done("final summary")
            yield TurnEvent.turn_end(2)

        FakeRunner.script = script
        agent_id = self._spawn_and_wait(name="Recon", task="find c2")

        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertEqual(info.status, SubagentStatus.COMPLETED)
        self.assertEqual(info.summary, "final summary")
        self.assertEqual(info.turn_count, 2)
        self.assertIs(info.token_usage, usage)
        self.assertIsNotNone(info.completed_at)
        self.assertGreaterEqual(info.completed_at, info.created_at)  # type: ignore[arg-type]

        events = self._drain()
        types = [e.type for e in events]
        self.assertEqual(
            types,
            [
                TurnEventType.SUBAGENT_SPAWNED,
                TurnEventType.SUBAGENT_PROGRESS,
                TurnEventType.SUBAGENT_PROGRESS,
                TurnEventType.SUBAGENT_COMPLETED,
            ],
        )
        spawned, prog1, prog2, done = events
        self.assertEqual(spawned.text, "Recon")
        self.assertEqual(spawned.metadata["agent_id"], agent_id)
        self.assertEqual(spawned.metadata["task"], "find c2")
        self.assertEqual(spawned.metadata["agent_type"], "custom")
        self.assertEqual([prog1.metadata["turn_count"], prog2.metadata["turn_count"]], [1, 2])
        self.assertEqual(done.text, "final summary")
        self.assertEqual(done.metadata["name"], "Recon")
        self.assertEqual(done.metadata["turn_count"], 2)
        self.assertGreaterEqual(done.metadata["elapsed"], 0.0)

    def test_empty_text_done_does_not_clobber_previous_text(self) -> None:
        def script(_runner: FakeRunner) -> Iterator[TurnEvent]:
            yield TurnEvent.text_done("kept")
            yield TurnEvent.text_done("")

        FakeRunner.script = script
        agent_id = self._spawn_and_wait()
        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertEqual(info.summary, "kept")
        self.assertEqual(info.turn_count, 0)

    def test_runner_constructed_with_manager_dependencies(self) -> None:
        agent_id = self._spawn_and_wait(task="analyze main", max_turns=7)

        self.assertEqual(len(FakeRunner.instances), 1)
        runner = FakeRunner.instances[0]
        kw = runner.init_kwargs
        self.assertIs(kw["provider"], self.provider)
        self.assertIs(kw["tool_registry"], self.tools)
        self.assertIs(kw["config"], self.config)
        self.assertEqual(kw["host_name"], "TestHost")
        self.assertIs(kw["skill_registry"], self.skills)
        self.assertIsInstance(kw["cancel_event"], threading.Event)
        self.assertFalse(kw["cancel_event"].is_set())
        self.assertEqual(runner.run_args["task"], "analyze main")
        self.assertEqual(runner.run_args["max_turns"], 7)
        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertEqual(info.status, SubagentStatus.COMPLETED)
        self.assertEqual(info.summary, "")

    def test_exception_in_runner_marks_failed(self) -> None:
        def script(_runner: FakeRunner) -> Iterator[TurnEvent]:
            yield TurnEvent.turn_end(1)
            raise RuntimeError("provider exploded")

        FakeRunner.script = script
        agent_id = self._spawn_and_wait(name="Crashy")

        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertEqual(info.status, SubagentStatus.FAILED)
        self.assertEqual(info.summary, "Error: provider exploded")
        self.assertEqual(info.turn_count, 1)
        self.assertIsNotNone(info.completed_at)

        events = self._drain()
        self.assertEqual(events[-1].type, TurnEventType.SUBAGENT_FAILED)
        self.assertEqual(events[-1].error, "provider exploded")
        self.assertEqual(events[-1].metadata["name"], "Crashy")
        self.assertNotIn(TurnEventType.SUBAGENT_COMPLETED, [e.type for e in events])

    def test_status_is_running_while_runner_is_active(self) -> None:
        started = threading.Event()
        gate = threading.Event()

        def script(_runner: FakeRunner) -> Iterator[TurnEvent]:
            started.set()
            gate.wait(_TIMEOUT)
            yield TurnEvent.text_done("done")

        FakeRunner.script = script
        agent_id = self.mgr.spawn(name="slow", task="t")
        try:
            self.assertTrue(started.wait(_TIMEOUT))
            info = self.mgr.get(agent_id)
            assert info is not None
            self.assertEqual(info.status, SubagentStatus.RUNNING)
            self.assertEqual(self.mgr.running_count(), 1)
            self.assertEqual(self.mgr.completed_count(), 0)
        finally:
            gate.set()
            self._join(agent_id)

        self.assertEqual(self.mgr.running_count(), 0)
        self.assertEqual(self.mgr.completed_count(), 1)

    def test_cancel_running_agent_stops_and_marks_cancelled(self) -> None:
        started = threading.Event()
        gate = threading.Event()

        def script(_runner: FakeRunner) -> Iterator[TurnEvent]:
            started.set()
            gate.wait(_TIMEOUT)
            yield TurnEvent.turn_end(1)
            yield TurnEvent.text_done("should never be summarized")

        FakeRunner.script = script
        agent_id = self.mgr.spawn(name="victim", task="t")
        try:
            self.assertTrue(started.wait(_TIMEOUT))
            self.mgr.cancel(agent_id)
            # The cancel event shared with the runner is set.
            self.assertTrue(FakeRunner.instances[0].init_kwargs["cancel_event"].is_set())
        finally:
            gate.set()
            self._join(agent_id)

        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertEqual(info.status, SubagentStatus.CANCELLED)
        self.assertEqual(info.summary, "")
        self.assertEqual(info.turn_count, 0)  # loop bailed before counting the turn
        self.assertIsNotNone(info.completed_at)

        events = self._drain()
        types = [e.type for e in events]
        self.assertNotIn(TurnEventType.SUBAGENT_COMPLETED, types)
        self.assertNotIn(TurnEventType.SUBAGENT_PROGRESS, types)
        failed = [e for e in events if e.type == TurnEventType.SUBAGENT_FAILED]
        self.assertGreaterEqual(len(failed), 1)
        self.assertTrue(all(e.error == "Cancelled by user" for e in failed))


class TestAgentTypesAndPerks(_ManagerTestBase):
    def test_custom_agent_receives_perks_addendum_and_unknown_perks_are_ignored(self) -> None:
        perks = ["string_harvesting", "not_a_real_perk", "import_mapping"]
        agent_id = self._spawn_and_wait(perks=perks)

        addendum = FakeRunner.instances[0].run_args["system_addendum"]
        self.assertEqual(addendum, build_perks_addendum(perks))
        self.assertIn(SUBAGENT_PERKS["string_harvesting"], addendum)
        self.assertIn(SUBAGENT_PERKS["import_mapping"], addendum)
        self.assertNotIn("not_a_real_perk", addendum)
        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertEqual(info.perks, perks)

    def test_custom_agent_without_perks_gets_empty_addendum(self) -> None:
        agent_id = self._spawn_and_wait()
        self.assertEqual(FakeRunner.instances[0].run_args["system_addendum"], "")
        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertEqual(info.perks, [])

    def test_network_recon_uses_its_addendum_and_ignores_perks(self) -> None:
        self._spawn_and_wait(agent_type="network_recon", perks=["hypothesis_mode"])
        addendum = FakeRunner.instances[0].run_args["system_addendum"]
        self.assertEqual(addendum, build_network_recon_addendum())
        self.assertNotIn(SUBAGENT_PERKS["hypothesis_mode"], addendum)

    def test_report_writer_uses_its_addendum(self) -> None:
        self._spawn_and_wait(agent_type="report_writer")
        self.assertEqual(FakeRunner.instances[0].run_args["system_addendum"], build_report_writer_addendum())

    def test_known_agent_types_fall_back_to_their_default_max_turns_when_zero(self) -> None:
        self._spawn_and_wait(agent_type="network_recon", max_turns=0)
        self._spawn_and_wait(agent_type="report_writer", max_turns=0)
        self.assertEqual(FakeRunner.instances[0].run_args["max_turns"], NETWORK_RECON_MAX_TURNS)
        self.assertEqual(FakeRunner.instances[1].run_args["max_turns"], REPORT_WRITER_MAX_TURNS)

    def test_explicit_max_turns_is_respected_for_known_agent_types(self) -> None:
        self._spawn_and_wait(agent_type="network_recon", max_turns=3)
        self.assertEqual(FakeRunner.instances[0].run_args["max_turns"], 3)

    def test_spawn_records_metadata(self) -> None:
        agent_id = self._spawn_and_wait(name="N", task="T", agent_type="report_writer", category="bulk_rename")
        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertIsInstance(info, SubagentInfo)
        self.assertEqual((info.id, info.name, info.task), (agent_id, "N", "T"))
        self.assertEqual(info.agent_type, "report_writer")
        self.assertEqual(info.category, "bulk_rename")
        self.assertIsNone(info.parent_id)
        self.assertEqual(len(agent_id), 12)


class TestRegisterAndExternalUpdates(_ManagerTestBase):
    def test_register_does_not_start_a_runner(self) -> None:
        agent_id = self.mgr.register(name="ext", task="rename 0x401000", category="bulk_rename")

        self.assertEqual(FakeRunner.instances, [])
        self.assertEqual(self._subagent_threads(agent_id), [])
        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertEqual(info.status, SubagentStatus.PENDING)
        self.assertEqual(info.category, "bulk_rename")
        events = self._drain()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, TurnEventType.SUBAGENT_SPAWNED)
        self.assertEqual(events[0].metadata["agent_id"], agent_id)

    def test_update_external_running_emits_progress(self) -> None:
        agent_id = self.mgr.register(name="ext", task="t")
        self._drain()

        self.mgr.update_external(agent_id, SubagentStatus.RUNNING, turn_count=3)

        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertEqual(info.status, SubagentStatus.RUNNING)
        self.assertEqual(info.turn_count, 3)
        self.assertIsNone(info.completed_at)
        self.assertEqual(self.mgr.running_count(), 1)
        (event,) = self._drain()
        self.assertEqual(event.type, TurnEventType.SUBAGENT_PROGRESS)
        self.assertEqual(event.metadata, {"agent_id": agent_id, "turn_count": 3})

    def test_update_external_completed_emits_completed_with_summary(self) -> None:
        agent_id = self.mgr.register(name="ext", task="t")
        self._drain()

        self.mgr.update_external(agent_id, SubagentStatus.COMPLETED, summary="renamed to parse_hdr", turn_count=4)

        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertEqual(info.status, SubagentStatus.COMPLETED)
        self.assertEqual(info.summary, "renamed to parse_hdr")
        self.assertIsNotNone(info.completed_at)
        self.assertEqual(self.mgr.completed_count(), 1)
        (event,) = self._drain()
        self.assertEqual(event.type, TurnEventType.SUBAGENT_COMPLETED)
        self.assertEqual(event.text, "renamed to parse_hdr")
        self.assertEqual(event.metadata["turn_count"], 4)
        self.assertEqual(event.metadata["name"], "ext")
        self.assertGreaterEqual(event.metadata["elapsed"], 0.0)

    def test_update_external_failed_and_cancelled_emit_failed_events(self) -> None:
        failed_id = self.mgr.register(name="f", task="t")
        cancelled_id = self.mgr.register(name="c", task="t")
        self._drain()

        self.mgr.update_external(failed_id, SubagentStatus.FAILED, summary="timeout")
        self.mgr.update_external(cancelled_id, SubagentStatus.CANCELLED, summary="stopped")

        events = self._drain()
        self.assertEqual([e.type for e in events], [TurnEventType.SUBAGENT_FAILED] * 2)
        self.assertEqual(
            [(e.metadata["agent_id"], e.error) for e in events], [(failed_id, "timeout"), (cancelled_id, "stopped")]
        )
        for agent_id, status in ((failed_id, SubagentStatus.FAILED), (cancelled_id, SubagentStatus.CANCELLED)):
            info = self.mgr.get(agent_id)
            assert info is not None
            self.assertEqual(info.status, status)
            self.assertIsNotNone(info.completed_at)

    def test_update_external_unknown_id_is_noop(self) -> None:
        self.mgr.update_external("missing", SubagentStatus.COMPLETED, summary="x")
        self.assertIsNone(self.mgr.poll_event())
        self.assertEqual(self.mgr.list_all(), [])

    def test_update_external_pending_emits_nothing(self) -> None:
        agent_id = self.mgr.register(name="ext", task="t")
        self._drain()
        self.mgr.update_external(agent_id, SubagentStatus.PENDING, summary="queued")
        self.assertIsNone(self.mgr.poll_event())
        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertEqual(info.summary, "queued")


class TestCancel(_ManagerTestBase):
    def test_cancel_pending_registered_agent(self) -> None:
        agent_id = self.mgr.register(name="ext", task="t")
        self._drain()

        self.mgr.cancel(agent_id)

        info = self.mgr.get(agent_id)
        assert info is not None
        self.assertEqual(info.status, SubagentStatus.CANCELLED)
        self.assertIsNotNone(info.completed_at)
        (event,) = self._drain()
        self.assertEqual(event.type, TurnEventType.SUBAGENT_FAILED)
        self.assertEqual(event.error, "Cancelled by user")
        self.assertEqual(event.metadata["name"], "ext")

    def test_cancel_finished_agent_is_noop(self) -> None:
        agent_id = self.mgr.register(name="ext", task="t")
        self.mgr.update_external(agent_id, SubagentStatus.COMPLETED, summary="ok")
        info = self.mgr.get(agent_id)
        assert info is not None
        completed_at = info.completed_at
        self._drain()

        self.mgr.cancel(agent_id)

        self.assertEqual(info.status, SubagentStatus.COMPLETED)
        self.assertEqual(info.completed_at, completed_at)
        self.assertIsNone(self.mgr.poll_event())

    def test_cancel_unknown_id_is_noop(self) -> None:
        self.mgr.cancel("does-not-exist")
        self.assertIsNone(self.mgr.poll_event())


class TestQueries(_ManagerTestBase):
    def test_get_unknown_and_empty_manager(self) -> None:
        self.assertIsNone(self.mgr.get("nope"))
        self.assertEqual(self.mgr.list_all(), [])
        self.assertEqual(self.mgr.tree(), [])
        self.assertIsNone(self.mgr.poll_event())
        self.assertEqual(self.mgr.running_count(), 0)
        self.assertEqual(self.mgr.completed_count(), 0)

    def test_tree_links_parents_and_children(self) -> None:
        root_a = self.mgr.register(name="root_a", task="t")
        root_b = self.mgr.register(name="root_b", task="t")
        child_1 = self.mgr.register(name="child_1", task="t", parent_id=root_a)
        child_2 = self.mgr.register(name="child_2", task="t", parent_id=root_a)
        grandchild = self.mgr.register(name="gc", task="t", parent_id=child_1)

        roots = self.mgr.tree()
        self.assertEqual([r.id for r in roots], [root_a, root_b])

        a = self.mgr.get(root_a)
        c1 = self.mgr.get(child_1)
        b = self.mgr.get(root_b)
        assert a is not None and c1 is not None and b is not None
        self.assertEqual(a.children, [child_1, child_2])
        self.assertEqual(c1.children, [grandchild])
        self.assertEqual(b.children, [])
        gc = self.mgr.get(grandchild)
        assert gc is not None
        self.assertEqual(gc.parent_id, child_1)
        self.assertEqual(len(self.mgr.list_all()), 5)

    def test_spawned_child_is_linked_to_registered_parent(self) -> None:
        parent = self.mgr.register(name="parent", task="t")
        child = self._spawn_and_wait(name="child", parent_id=parent)

        p = self.mgr.get(parent)
        assert p is not None
        self.assertEqual(p.children, [child])
        self.assertEqual([r.id for r in self.mgr.tree()], [parent])

    def test_child_with_unknown_parent_is_not_linked_and_not_a_root(self) -> None:
        orphan = self.mgr.register(name="orphan", task="t", parent_id="ghost")

        info = self.mgr.get(orphan)
        assert info is not None
        self.assertEqual(info.parent_id, "ghost")
        self.assertEqual(self.mgr.tree(), [])
        self.assertEqual([a.id for a in self.mgr.list_all()], [orphan])

    def test_ids_are_unique(self) -> None:
        ids = {self.mgr.register(name=f"a{i}", task="t") for i in range(50)}
        self.assertEqual(len(ids), 50)


if __name__ == "__main__":
    unittest.main()
