"""Behavior of the chat composer (rikugan.ui.composer).

PySide6 is not available in the test environment, so the composer is built
without running its Qt constructor and given mock child widgets; the logic
under test is the composer's own: Send/Stop switching, input gating, theme
refresh and text handling.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from tests.qt_stubs import ensure_pyside6_stubs

ensure_pyside6_stubs()

import rikugan.ui.composer as composer_module  # noqa: E402
from rikugan.ui.composer import _SEND_GLYPH, _STOP_GLYPH, Composer  # noqa: E402


def _make_composer() -> Composer:
    composer = Composer.__new__(Composer)
    composer.setStyleSheet = MagicMock()
    composer._send_callback = None
    composer._cancel_callback = None
    composer._running = False
    composer._frame = MagicMock()
    composer.input_area = MagicMock()
    composer._skills_btn = MagicMock()
    composer._modify_btn = MagicMock()
    composer._model_label = MagicMock()
    composer._send_btn = MagicMock()
    return composer


class TestSendButton(unittest.TestCase):
    def test_idle_click_sends(self):
        composer = _make_composer()
        send, cancel = MagicMock(), MagicMock()
        composer.set_send_callback(send)
        composer.set_cancel_callback(cancel)

        composer._on_send()

        send.assert_called_once_with()
        cancel.assert_not_called()

    def test_running_click_cancels_instead_of_sending(self):
        composer = _make_composer()
        send, cancel = MagicMock(), MagicMock()
        composer.set_send_callback(send)
        composer.set_cancel_callback(cancel)
        composer.set_running(True)

        composer._on_send()

        cancel.assert_called_once_with()
        send.assert_not_called()

    def test_click_without_callbacks_is_harmless(self):
        composer = _make_composer()
        composer._on_send()
        composer.set_running(True)
        composer._on_send()

    def test_cancel_callback_also_reaches_the_editor_escape_key(self):
        composer = _make_composer()
        cancel = MagicMock()
        composer.set_cancel_callback(cancel)
        composer.input_area.set_cancel_callback.assert_called_once_with(cancel)

    def test_submit_callback_is_owned_by_the_editor(self):
        composer = _make_composer()
        submit = MagicMock()
        composer.set_submit_callback(submit)
        composer.input_area.set_submit_callback.assert_called_once_with(submit)


class TestRunningState(unittest.TestCase):
    def test_stop_replaces_send_in_place(self):
        composer = _make_composer()
        composer.set_running(True)

        btn = composer._send_btn
        btn.setObjectName.assert_called_with("composer_stop")
        btn.setText.assert_called_with(_STOP_GLYPH)
        btn.setToolTip.assert_called_with("Stop (Esc)")
        # The object name drives the stylesheet, so the style must re-polish.
        btn.style.return_value.polish.assert_called_with(btn)

    def test_send_returns_when_the_turn_ends(self):
        composer = _make_composer()
        composer.set_running(True)
        composer.set_running(False)

        btn = composer._send_btn
        btn.setObjectName.assert_called_with("composer_send")
        btn.setText.assert_called_with(_SEND_GLYPH)
        btn.setToolTip.assert_called_with("Send (Enter)")


class TestInputGating(unittest.TestCase):
    def test_disabling_input_disables_text_and_chips(self):
        composer = _make_composer()
        composer.set_input_enabled(False)

        composer.input_area.set_enabled.assert_called_once_with(False)
        composer._skills_btn.setEnabled.assert_called_once_with(False)
        composer._modify_btn.setEnabled.assert_called_once_with(False)
        composer._send_btn.setEnabled.assert_called_once_with(False)

    def test_stop_stays_clickable_while_input_is_gated(self):
        composer = _make_composer()
        composer.set_running(True)
        composer.set_input_enabled(False)
        composer._send_btn.setEnabled.assert_called_with(True)


class TestTheme(unittest.TestCase):
    def test_refresh_restyles_the_frame_and_the_editor(self):
        composer = _make_composer()
        with patch.object(composer_module, "build_composer_stylesheet", return_value="CSS") as build:
            composer.refresh_theme()

        build.assert_called_once_with(composer)
        composer.setStyleSheet.assert_called_once_with("CSS")
        composer.input_area.refresh_theme.assert_called_once_with()

    def test_refresh_before_the_editor_exists(self):
        # The constructor styles the composer before it builds the editor.
        composer = _make_composer()
        del composer.input_area
        with patch.object(composer_module, "build_composer_stylesheet", return_value="CSS"):
            composer.refresh_theme()
        composer.setStyleSheet.assert_called_once_with("CSS")

    def test_focus_toggles_the_frame_property_and_repolishes(self):
        composer = _make_composer()
        composer._on_focus_changed(True)
        composer._frame.setProperty.assert_called_with("focused", "true")
        composer._on_focus_changed(False)
        composer._frame.setProperty.assert_called_with("focused", "false")
        composer._frame.style.return_value.polish.assert_called_with(composer._frame)


class TestText(unittest.TestCase):
    def test_text_is_stripped(self):
        composer = _make_composer()
        composer.input_area.toPlainText.return_value = "  /modify patch the check \n"
        self.assertEqual(composer.text(), "/modify patch the check")

    def test_set_text_moves_the_cursor_to_the_end_and_focuses(self):
        composer = _make_composer()
        cursor = MagicMock()
        composer.input_area.textCursor.return_value = cursor

        composer.set_text("/modify ")

        composer.input_area.setPlainText.assert_called_once_with("/modify ")
        cursor.movePosition.assert_called_once_with(cursor.MoveOperation.End)
        composer.input_area.setTextCursor.assert_called_once_with(cursor)
        composer.input_area.setFocus.assert_called_once_with()

    def test_model_pill_shows_a_short_name_with_the_full_id_as_tooltip(self):
        composer = _make_composer()
        composer.set_model("claude-opus-5-5")
        composer._model_label.setText.assert_called_once_with("opus-5.5")
        composer._model_label.setToolTip.assert_called_once_with("claude-opus-5-5")

    def test_placeholder_and_skills_pass_through_to_the_editor(self):
        composer = _make_composer()
        composer.set_placeholder("Ask about this binary")
        composer.set_skill_slugs(["ctf"])
        composer.clear()
        composer.focus_input()

        composer.input_area.setPlaceholderText.assert_called_once_with("Ask about this binary")
        composer.input_area.set_skill_slugs.assert_called_once_with(["ctf"])
        composer.input_area.clear.assert_called_once_with()
        composer.input_area.setFocus.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
