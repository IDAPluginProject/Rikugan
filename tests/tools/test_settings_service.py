"""Tests for rikugan.ui.settings_service (settings discovery and MCP persistence)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rikugan.mcp.config import MCPServerConfig, load_mcp_config
from rikugan.skills.loader import SkillDefinition
from rikugan.ui.settings_service import DiscoveredMCP, DiscoveredSkills, SettingsService

_EXT_SKILLS = "rikugan.core.external_sources.discover_all_external_skills"
_EXT_MCP = "rikugan.core.external_sources.discover_all_external_mcp"
_LOG_ERROR = "rikugan.ui.settings_service.log_error"


class _FakeRegistry:
    def __init__(self, defs):
        self._defs = defs

    def list_tools(self):
        return list(self._defs)


def _tool(name: str, category):
    return SimpleNamespace(name=name, category=category)


class _FakeConfig:
    """The two paths SettingsService reads from RikuganConfig, rooted in a temp dir.

    A fake rather than RikuganConfig itself: other test modules replace
    ``rikugan.core.config`` in sys.modules with MagicMock stubs, and the real
    config would point at the user's config directory anyway.
    """

    def __init__(self, root: str):
        self.skills_dir = os.path.join(root, "skills")
        self.mcp_config_path = os.path.join(root, "mcp.json")


class _ServiceTestBase(unittest.TestCase):
    """Builds services against a throwaway config dir with external discovery stubbed."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rikugan_settings_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.config = _FakeConfig(self.tmp)
        # Keep Rikugan's logger off the real user log file during these tests.
        quiet = logging.getLogger("rikugan.tests.settings_service")
        quiet.addHandler(logging.NullHandler())
        quiet.propagate = False
        p0 = patch("rikugan.core.logging._logger", quiet, create=True)
        p0.start()
        self.addCleanup(p0.stop)
        self.ext_skills = {}
        self.ext_mcp = {}
        p1 = patch(_EXT_SKILLS, side_effect=lambda: self.ext_skills)
        p2 = patch(_EXT_MCP, side_effect=lambda: self.ext_mcp)
        self.mock_ext_skills = p1.start()
        self.mock_ext_mcp = p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def _write_mcp_json(self, servers: dict) -> None:
        with open(self.config.mcp_config_path, "w", encoding="utf-8") as f:
            json.dump({"mcpServers": servers}, f)

    def _write_user_skill(self, slug: str, name: str) -> None:
        skill_dir = os.path.join(self.config.skills_dir, slug)
        os.makedirs(skill_dir)
        with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(f"---\nname: {name}\ndescription: test skill\n---\nTask: do the thing\n")


class TestDiscoveredDefaults(unittest.TestCase):
    def test_defaults_are_empty_and_independent(self):
        a, b = DiscoveredSkills(), DiscoveredSkills()
        a.rikugan.append("x")
        a.external["claude"] = []
        self.assertEqual(b.rikugan, [])
        self.assertEqual(b.external, {})
        m = DiscoveredMCP()
        self.assertEqual(m.rikugan, [])
        self.assertEqual(m.external, {})


class TestToolsByCategory(unittest.TestCase):
    def test_none_registry_gives_empty_mapping(self):
        self.assertEqual(SettingsService._build_tools_by_category(None), {})

    def test_groups_capitalizes_and_sorts(self):
        registry = _FakeRegistry(
            [
                _tool("xrefs_to", "xrefs"),
                _tool("decompile_function", "decompiler"),
                _tool("get_xrefs_from", "xrefs"),
                _tool("misc_tool", None),
                _tool("other_tool", ""),
            ]
        )
        result = SettingsService._build_tools_by_category(registry)
        self.assertEqual(
            result,
            {
                "Xrefs": ["get_xrefs_from", "xrefs_to"],
                "Decompiler": ["decompile_function"],
                "General": ["misc_tool", "other_tool"],
            },
        )

    def test_capitalize_lowercases_remaining_letters(self):
        result = SettingsService._build_tools_by_category(_FakeRegistry([_tool("get_il", "IL")]))
        self.assertEqual(result, {"Il": ["get_il"]})


class TestMCPDiscovery(_ServiceTestBase):
    def test_missing_mcp_file_yields_no_servers(self):
        service = SettingsService(self.config)
        self.assertEqual(service.mcp.rikugan, [])
        self.assertEqual(service.mcp.external, {})

    def test_loads_servers_from_config_dir(self):
        self._write_mcp_json(
            {
                "local": {"command": "node", "args": ["server.js"], "timeout": 5},
                "remote": {"url": "http://127.0.0.1:9000/mcp", "enabled": False},
                "broken": {"args": ["no-command-or-url"]},
            }
        )
        service = SettingsService(self.config)
        by_name = {s.name: s for s in service.mcp.rikugan}
        self.assertEqual(set(by_name), {"local", "remote"})
        self.assertEqual(by_name["local"].command, "node")
        self.assertEqual(by_name["local"].args, ["server.js"])
        self.assertEqual(by_name["local"].timeout, 5.0)
        self.assertTrue(by_name["remote"].is_remote)
        self.assertFalse(by_name["remote"].enabled)

    def test_external_mcp_is_passed_through(self):
        ext = MCPServerConfig(name="ext", command="ext-cmd")
        self.ext_mcp = {"claude": [ext]}
        service = SettingsService(self.config)
        self.assertEqual(service.mcp.external, {"claude": [ext]})

    def test_external_mcp_failure_is_logged_and_local_still_loads(self):
        self._write_mcp_json({"local": {"command": "node"}})
        self.mock_ext_mcp.side_effect = RuntimeError("boom")
        with patch(_LOG_ERROR) as log_error:
            service = SettingsService(self.config)
        self.assertEqual([s.name for s in service.mcp.rikugan], ["local"])
        self.assertEqual(service.mcp.external, {})
        messages = [c.args[0] for c in log_error.call_args_list]
        self.assertTrue(any("external MCP" in m and "boom" in m for m in messages), messages)

    def test_local_mcp_failure_is_logged(self):
        with (
            patch("rikugan.ui.settings_service.load_mcp_config", side_effect=ValueError("bad json")),
            patch(_LOG_ERROR) as log_error,
        ):
            service = SettingsService(self.config)
        self.assertEqual(service.mcp.rikugan, [])
        messages = [c.args[0] for c in log_error.call_args_list]
        self.assertTrue(any("Rikugan MCP config" in m and "bad json" in m for m in messages), messages)


class TestSkillDiscovery(_ServiceTestBase):
    def test_user_skill_is_discovered_alongside_builtins(self):
        self._write_user_skill("my-custom-skill", "My Custom Skill")
        service = SettingsService(self.config)
        by_slug = {s.slug: s for s in service.skills.rikugan}
        self.assertIn("my-custom-skill", by_slug)
        self.assertEqual(by_slug["my-custom-skill"].name, "My Custom Skill")
        # Built-in skills ship with the plugin and are always present.
        self.assertIn("generic-re", by_slug)

    def test_external_skills_are_passed_through(self):
        ext = SkillDefinition(name="Ext", description="d", directory=os.path.join(self.tmp, "ext"))
        self.ext_skills = {"codex": [ext]}
        service = SettingsService(self.config)
        self.assertEqual(service.skills.external, {"codex": [ext]})

    def test_registry_failure_is_logged_and_external_still_loads(self):
        ext = SkillDefinition(name="Ext", description="d", directory=os.path.join(self.tmp, "ext"))
        self.ext_skills = {"claude": [ext]}
        with (
            patch("rikugan.skills.registry.SkillRegistry", side_effect=OSError("disk gone")),
            patch(_LOG_ERROR) as log_error,
        ):
            service = SettingsService(self.config)
        self.assertEqual(service.skills.rikugan, [])
        self.assertEqual(service.skills.external, {"claude": [ext]})
        messages = [c.args[0] for c in log_error.call_args_list]
        self.assertTrue(any("Rikugan skills" in m and "disk gone" in m for m in messages), messages)

    def test_discovery_runs_once_at_construction(self):
        service = SettingsService(self.config)
        _ = service.skills, service.skills, service.mcp, service.mcp
        self.assertEqual(self.mock_ext_skills.call_count, 1)
        self.assertEqual(self.mock_ext_mcp.call_count, 1)
        self.assertIs(service.skills, service.skills)


class TestSaveMCPServers(_ServiceTestBase):
    def test_save_round_trips_through_config_dir(self):
        service = SettingsService(self.config)
        servers = [
            MCPServerConfig(name="local", command="python", args=["-m", "srv"], env={"K": "V"}, timeout=12.5),
            MCPServerConfig(name="remote", url="http://localhost:1234/mcp", enabled=False, max_tools=3),
        ]
        service.save_mcp_servers(servers)

        self.assertTrue(os.path.isfile(self.config.mcp_config_path))
        with open(self.config.mcp_config_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.assertEqual(set(raw["mcpServers"]), {"local", "remote"})
        self.assertEqual(raw["mcpServers"]["local"]["timeout"], 12.5)
        self.assertNotIn("timeout", raw["mcpServers"]["remote"])

        self.assertEqual(load_mcp_config(self.config.mcp_config_path), servers)
        # A freshly constructed service sees the persisted servers.
        self.assertEqual(SettingsService(self.config).mcp.rikugan, servers)

    def test_save_empty_list_clears_servers(self):
        self._write_mcp_json({"local": {"command": "node"}})
        service = SettingsService(self.config)
        service.save_mcp_servers([])
        with open(self.config.mcp_config_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"mcpServers": {}})

    def test_save_failure_is_logged_not_raised(self):
        service = SettingsService(self.config)
        with (
            patch("rikugan.ui.settings_service.save_mcp_config", side_effect=OSError("read-only")),
            patch(_LOG_ERROR) as log_error,
        ):
            service.save_mcp_servers([MCPServerConfig(name="x", command="y")])
        log_error.assert_called_once()
        self.assertIn("Failed to save MCP config: read-only", log_error.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
