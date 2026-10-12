"""Tests for the MiniMax provider (Anthropic SDK against MiniMax's API)."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.mocks.ida_mock import install_ida_mocks

install_ida_mocks()

from rikugan.core.errors import AuthenticationError, ProviderError  # noqa: E402
from rikugan.core.types import Message, Role  # noqa: E402
from rikugan.providers.minimax_provider import MiniMaxProvider  # noqa: E402

_TOOL = {
    "type": "function",
    "function": {"name": "decompile_function", "description": "d", "parameters": {"type": "object"}},
}


def _fake_anthropic() -> SimpleNamespace:
    """Stand-in for the anthropic SDK module, recording client construction."""
    return SimpleNamespace(
        Anthropic=MagicMock(name="Anthropic"),
        DefaultHttpxClient=MagicMock(name="DefaultHttpxClient", return_value="unverified-client"),
    )


class TestConstruction(unittest.TestCase):
    def test_defaults_to_minimax_endpoint(self):
        provider = MiniMaxProvider(api_key="k")
        self.assertEqual(provider.api_base, MiniMaxProvider.DEFAULT_API_BASE)
        self.assertEqual(provider.model, "MiniMax-M2.5")
        self.assertEqual(provider.name, "minimax")

    def test_custom_endpoint_wins(self):
        provider = MiniMaxProvider(api_key="k", api_base="https://llm.internal/anthropic")
        self.assertEqual(provider.api_base, "https://llm.internal/anthropic")

    def test_key_is_used_as_is_with_no_oauth_lookup(self):
        with patch("rikugan.providers.auth_cache.resolve_auth_cached") as resolve:
            provider = MiniMaxProvider(api_key="")
        resolve.assert_not_called()
        self.assertEqual(provider.api_key, "")
        self.assertEqual(provider._auth_type, "api_key")

    def test_auth_status(self):
        self.assertEqual(MiniMaxProvider(api_key="k").auth_status(), ("API Key", "ok"))
        self.assertEqual(MiniMaxProvider().auth_status(), ("", "none"))

    def test_capabilities_disable_cache_control(self):
        caps = MiniMaxProvider(api_key="k").capabilities
        self.assertFalse(caps.supports_cache_control)
        self.assertTrue(caps.tool_use)
        self.assertEqual(caps.max_context_window, 204800)


class TestClient(unittest.TestCase):
    def _client_for(self, provider: MiniMaxProvider, sdk: SimpleNamespace):
        with patch("rikugan.providers.minimax_provider.importlib.import_module", return_value=sdk):
            return provider._get_client()

    def test_client_targets_the_configured_endpoint(self):
        sdk = _fake_anthropic()
        provider = MiniMaxProvider(api_key="k")
        client = self._client_for(provider, sdk)

        sdk.Anthropic.assert_called_once_with(api_key="k", base_url=MiniMaxProvider.DEFAULT_API_BASE)
        self.assertIs(client, sdk.Anthropic.return_value)

    def test_client_is_built_once(self):
        sdk = _fake_anthropic()
        provider = MiniMaxProvider(api_key="k")
        first = self._client_for(provider, sdk)
        second = self._client_for(provider, sdk)
        self.assertIs(first, second)
        sdk.Anthropic.assert_called_once()

    def test_ignoring_certificates_passes_an_unverified_http_client(self):
        sdk = _fake_anthropic()
        provider = MiniMaxProvider(api_key="k")
        provider.ssl_verify = False
        with patch("rikugan.providers.base.log_warning") as warn:
            self._client_for(provider, sdk)

        warn.assert_called_once()
        self.assertIn("verification is disabled", warn.call_args.args[0])
        sdk.DefaultHttpxClient.assert_called_once_with(verify=False)
        self.assertEqual(sdk.Anthropic.call_args.kwargs["http_client"], "unverified-client")

    def test_verified_client_keeps_the_sdk_transport(self):
        sdk = _fake_anthropic()
        self._client_for(MiniMaxProvider(api_key="k"), sdk)
        sdk.DefaultHttpxClient.assert_not_called()
        self.assertNotIn("http_client", sdk.Anthropic.call_args.kwargs)

    def test_missing_key_is_an_authentication_error(self):
        with self.assertRaises(AuthenticationError):
            self._client_for(MiniMaxProvider(), _fake_anthropic())

    def test_missing_sdk_is_a_provider_error(self):
        with (
            patch(
                "rikugan.providers.minimax_provider.importlib.import_module",
                side_effect=ImportError("no anthropic"),
            ),
            self.assertRaises(ProviderError),
        ):
            MiniMaxProvider(api_key="k")._get_client()


class TestModels(unittest.TestCase):
    def test_live_list_keeps_only_minimax_models(self):
        provider = MiniMaxProvider(api_key="k")
        listing = SimpleNamespace(
            data=[
                SimpleNamespace(id="MiniMax-M2.7", display_name="MiniMax M2.7"),
                SimpleNamespace(id="claude-sonnet", display_name="Claude"),
                SimpleNamespace(id="minimax-m2", display_name=None),
            ]
        )
        provider._client = SimpleNamespace(models=SimpleNamespace(list=MagicMock(return_value=listing)))

        models = provider._fetch_models_live()

        self.assertEqual([m.id for m in models], ["MiniMax-M2.7", "minimax-m2"])
        self.assertEqual(models[0].name, "MiniMax M2.7")
        self.assertEqual(models[1].name, "minimax-m2", "falls back to the id without a display name")

    def test_empty_or_failed_listing_falls_back_to_builtins(self):
        provider = MiniMaxProvider(api_key="k")
        builtin_ids = [m.id for m in provider._builtin_models()]

        provider._client = SimpleNamespace(
            models=SimpleNamespace(list=MagicMock(return_value=SimpleNamespace(data=[])))
        )
        self.assertEqual([m.id for m in provider._fetch_models_live()], builtin_ids)

        provider._client = SimpleNamespace(models=SimpleNamespace(list=MagicMock(side_effect=RuntimeError("503"))))
        self.assertEqual([m.id for m in provider._fetch_models_live()], builtin_ids)

    def test_builtins_are_all_minimax_tool_models(self):
        models = MiniMaxProvider._builtin_models()
        self.assertIn("MiniMax-M2.5", [m.id for m in models])
        self.assertTrue(all(m.provider == "minimax" and m.supports_tools for m in models))


class TestRequestKwargs(unittest.TestCase):
    def test_cache_control_is_stripped_everywhere(self):
        provider = MiniMaxProvider(api_key="k")
        messages = [
            Message(role=Role.USER, content="a"),
            Message(role=Role.ASSISTANT, content="b"),
            Message(role=Role.USER, content="c"),
            Message(role=Role.ASSISTANT, content="d"),
            Message(role=Role.USER, content="e"),
        ]

        kwargs = provider._build_request_kwargs(messages, [_TOOL], "You are a reverse engineer.")

        self.assertEqual(kwargs["system"], "You are a reverse engineer.", "a lone text block collapses to a string")
        for tool in kwargs["tools"]:
            self.assertNotIn("cache_control", tool)
        for msg in kwargs["messages"]:
            if isinstance(msg["content"], list):
                for block in msg["content"]:
                    self.assertNotIn("cache_control", block)

    def test_no_system_prompt_and_no_tools(self):
        provider = MiniMaxProvider(api_key="k")
        kwargs = provider._build_request_kwargs([Message(role=Role.USER, content="hi")], None, "")
        self.assertNotIn("system", kwargs)
        self.assertNotIn("tools", kwargs)
        self.assertEqual(kwargs["model"], "MiniMax-M2.5")


if __name__ == "__main__":
    unittest.main()
