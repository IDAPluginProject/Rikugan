"""Per-provider "Ignore SSL certificate errors" setting."""

from __future__ import annotations

import os
import ssl
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.mocks.ida_mock import install_ida_mocks

install_ida_mocks()

from rikugan.core.config import ProviderConfig, RikuganConfig  # noqa: E402
from rikugan.providers.registry import ProviderRegistry  # noqa: E402


class TestConfigKeepsTheSettingPerProvider(unittest.TestCase):
    def test_verification_is_on_by_default(self):
        self.assertTrue(ProviderConfig().ssl_verify)

    def test_switching_restores_each_providers_own_choice(self):
        cfg = RikuganConfig()
        cfg.provider.name = "openai_compat"
        cfg.provider.ssl_verify = False

        cfg.switch_provider("anthropic")
        self.assertTrue(cfg.provider.ssl_verify, "a fresh provider must not inherit the opt-out")

        cfg.switch_provider("openai_compat")
        self.assertFalse(cfg.provider.ssl_verify)

    def test_custom_provider_starts_verified(self):
        cfg = RikuganConfig()
        cfg.provider.ssl_verify = False
        cfg.add_custom_provider("internal-llm", {})
        cfg.switch_provider("internal-llm")
        self.assertTrue(cfg.provider.ssl_verify)


class TestRegistry(unittest.TestCase):
    def test_create_sets_the_flag(self):
        reg = ProviderRegistry()
        provider = reg.create("openai", api_key="k", ssl_verify=False)
        self.assertFalse(provider.ssl_verify)
        self.assertTrue(reg.create("openai", api_key="k").ssl_verify)

    def test_toggling_rebuilds_the_instance(self):
        reg = ProviderRegistry()
        first = reg.get_or_create("openai", api_key="k")
        self.assertIs(reg.get_or_create("openai", api_key="k"), first)
        second = reg.get_or_create("openai", api_key="k", ssl_verify=False)
        self.assertIsNot(second, first)
        self.assertFalse(second.ssl_verify)


class TestProviderTransport(unittest.TestCase):
    def _provider(self, ssl_verify: bool):
        provider = ProviderRegistry().create("openai", api_key="k", ssl_verify=ssl_verify)
        return provider

    def test_verified_provider_leaves_the_sdk_default(self):
        provider = self._provider(True)
        self.assertIsNone(provider._unverified_http_client(SimpleNamespace()))
        self.assertIsNone(provider._urlopen_ssl_context())

    def test_unverified_provider_builds_a_non_verifying_client(self):
        provider = self._provider(False)
        calls = []
        sdk = SimpleNamespace(DefaultHttpxClient=lambda **kw: calls.append(kw) or "client")
        with patch("rikugan.providers.base.log_warning") as warn:
            self.assertEqual(provider._unverified_http_client(sdk), "client")
        self.assertEqual(calls, [{"verify": False}])
        warn.assert_called_once()

    def test_unverified_urllib_context_skips_checks(self):
        context = self._provider(False)._urlopen_ssl_context()
        self.assertIsInstance(context, ssl.SSLContext)
        self.assertFalse(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_NONE)


if __name__ == "__main__":
    unittest.main()
