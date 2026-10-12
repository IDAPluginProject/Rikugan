"""Theme-aware colors of the tool cards in rikugan.ui.tool_widgets.

The IDA arm of these stylesheets used to hard-code dark-theme greys, so a tool
group header was pale grey on pale grey under IDA's light theme. Every color
is now resolved against the card fill it is drawn on.
"""

from __future__ import annotations

import re
import unittest

from tests.qt_stubs import ensure_pyside6_stubs

ensure_pyside6_stubs()

import rikugan.ui.styles as styles  # noqa: E402
from rikugan.ui.styles import contrast_ratio  # noqa: E402
from rikugan.ui.tool_widgets import (  # noqa: E402
    _TOOL_INK_BASE,
    _muted_text,
    _toggle_css,
    _tool_card_css,
    _tool_ink,
)

_LIGHT = {
    "panel": "#f0f0f0",
    "chat_canvas": "#ececec",
    "tool_bg": "#e5e5e5",
    "text": "#000000",
    "muted": "#c8c8c8",
    "border": "#c0c0c0",
    "accent": "#0078d4",
    "accent_text": "#ffffff",
}
_DARK = {
    "panel": "#363636",
    "chat_canvas": "#3e3e3e",
    "tool_bg": "#4a4a4a",
    "text": "#dddddd",
    "muted": "#555555",
    "border": "#5a5a5a",
    "accent": "#4ec9b0",
    "accent_text": "#1e1e1e",
}


def _color(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r" \{[^}]*?color: (#[0-9a-f]{6})", css)
    assert match, f"no color for {selector} in {css!r}"
    return match.group(1)


class TestToolInk(unittest.TestCase):
    def test_every_role_reads_on_a_light_card(self):
        for role in ("text", "muted", *_TOOL_INK_BASE):
            with self.subTest(role=role):
                ink = _tool_ink(role, _LIGHT)
                floor = 7.0 if role == "text" else 4.5
                self.assertGreaterEqual(contrast_ratio(ink, _LIGHT["tool_bg"]), floor)

    def test_every_role_reads_on_a_dark_card(self):
        for role in ("text", "muted", *_TOOL_INK_BASE):
            with self.subTest(role=role):
                ink = _tool_ink(role, _DARK)
                self.assertGreaterEqual(contrast_ratio(ink, _DARK["tool_bg"]), 4.5)

    def test_status_hues_keep_their_identity_when_already_readable(self):
        # Teal on a near-black card needs no adjustment.
        dark = dict(_DARK, tool_bg="#1e1e1e")
        self.assertEqual(_tool_ink("ok", dark), _TOOL_INK_BASE["ok"])

    def test_unknown_palette_falls_back_to_the_fixed_hue(self):
        original = styles.get_chat_color_tokens

        def broken(source=None):
            raise RuntimeError("no palette")

        import rikugan.ui.tool_widgets as tool_widgets

        tool_widgets.get_chat_color_tokens = broken
        self.addCleanup(setattr, tool_widgets, "get_chat_color_tokens", original)
        self.assertEqual(_tool_ink("error", None), _TOOL_INK_BASE["error"])


class TestToggleCss(unittest.TestCase):
    def _with_native(self, native: bool) -> None:
        import rikugan.ui.tool_widgets as tool_widgets

        original = tool_widgets.host_stylesheet
        tool_widgets.host_stylesheet = lambda custom, native_css="": native_css if native else custom
        self.addCleanup(setattr, tool_widgets, "host_stylesheet", original)

    def test_ida_arrow_is_readable_and_brightens_on_hover(self):
        self._with_native(True)
        css = _toggle_css(_DARK)
        idle = _color(css, "QToolButton#collapse_button")
        hover = _color(css, "QToolButton#collapse_button:hover")
        self.assertGreaterEqual(contrast_ratio(idle, _DARK["tool_bg"]), 4.5)
        self.assertGreater(contrast_ratio(hover, _DARK["tool_bg"]), contrast_ratio(idle, _DARK["tool_bg"]))

    def test_other_hosts_keep_the_shared_theme(self):
        self._with_native(False)
        self.assertEqual(_toggle_css(_DARK), "")


class TestCardCss(unittest.TestCase):
    def test_card_uses_the_palette_fill_and_border(self):
        css = _tool_card_css(_LIGHT)
        self.assertIn(f"background-color: {_LIGHT['tool_bg']}", css)
        self.assertIn(f"border: 1px solid {_LIGHT['border']}", css)

    def test_accent_and_background_override_the_palette(self):
        css = _tool_card_css(_LIGHT, accent="#ff0000", background="#00ff00", object_name="x")
        self.assertTrue(css.startswith("QFrame#x {"))
        self.assertIn("border: 1px solid #ff0000", css)
        self.assertIn("background-color: #00ff00", css)

    def test_muted_text_comes_from_the_palette(self):
        self.assertEqual(_muted_text(_LIGHT), _LIGHT["muted"])


if __name__ == "__main__":
    unittest.main()
