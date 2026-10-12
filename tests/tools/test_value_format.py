"""Tests for rikugan.tools.value_format (global/data value formatting)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rikugan.tools.value_format import (
    bytes_needed_for_type,
    format_global_value,
    normalize_type_hint,
)


def _fmt(data: bytes, type_hint: str = "auto", pointer_size: int = 8, **kwargs) -> list[str]:
    text = format_global_value(address=0x401000, data=data, pointer_size=pointer_size, type_hint=type_hint, **kwargs)
    return text.split("\n")


class TestNormalizeTypeHint(unittest.TestCase):
    def test_none_and_empty_default_to_auto(self):
        self.assertEqual(normalize_type_hint(None), "auto")
        self.assertEqual(normalize_type_hint(""), "auto")

    def test_strips_and_lowercases(self):
        self.assertEqual(normalize_type_hint("  DWORD  "), "dword")

    def test_c_style_int_names_are_shortened(self):
        self.assertEqual(normalize_type_hint("uint32"), "u32")
        self.assertEqual(normalize_type_hint("UInt16"), "u16")
        self.assertEqual(normalize_type_hint("int64"), "i64")
        self.assertEqual(normalize_type_hint("int8"), "i8")


class TestBytesNeededForType(unittest.TestCase):
    def test_fixed_width_types(self):
        self.assertEqual(bytes_needed_for_type("u8", 8), 1)
        self.assertEqual(bytes_needed_for_type("word", 8), 2)
        self.assertEqual(bytes_needed_for_type("int32", 8), 4)
        self.assertEqual(bytes_needed_for_type("float", 8), 4)
        self.assertEqual(bytes_needed_for_type("qword", 4), 8)
        self.assertEqual(bytes_needed_for_type("double", 4), 8)

    def test_fixed_width_ignores_requested_size(self):
        self.assertEqual(bytes_needed_for_type("u16", 8, requested_size=100), 2)

    def test_pointer_uses_pointer_size(self):
        self.assertEqual(bytes_needed_for_type("ptr", 4), 4)
        self.assertEqual(bytes_needed_for_type("ptr", 8), 8)

    def test_pointer_with_zero_pointer_size_reads_at_least_one_byte(self):
        self.assertEqual(bytes_needed_for_type("ptr", 0), 1)

    def test_string_like_hints_default_to_64(self):
        for hint in ("string", "str", "utf8", "utf-16", "bytes", "auto", None):
            with self.subTest(hint=hint):
                self.assertEqual(bytes_needed_for_type(hint, 8), 64)

    def test_string_requested_size_is_clamped(self):
        self.assertEqual(bytes_needed_for_type("string", 8, requested_size=10), 10)
        self.assertEqual(bytes_needed_for_type("string", 8, requested_size=1000), 256)
        self.assertEqual(bytes_needed_for_type("bytes", 8, requested_size=-5), 1)

    def test_unknown_hint_falls_back_to_requested_or_pointer_size(self):
        self.assertEqual(bytes_needed_for_type("my_struct", 8), 8)
        self.assertEqual(bytes_needed_for_type("my_struct", 4), 4)
        self.assertEqual(bytes_needed_for_type("my_struct", 8, requested_size=24), 24)
        self.assertEqual(bytes_needed_for_type("my_struct", 8, requested_size=4096), 256)


class TestFormatHeader(unittest.TestCase):
    def test_header_with_name(self):
        lines = _fmt(b"\x01\x02", type_hint="bytes", name="g_counter")
        self.assertEqual(lines[0], "Global value at 0x401000 (g_counter)")
        self.assertEqual(lines[1], "Type hint: bytes")
        self.assertEqual(lines[2], "Bytes: 01 02")

    def test_header_without_name_has_no_parens(self):
        lines = _fmt(b"\x01", type_hint="bytes")
        self.assertEqual(lines[0], "Global value at 0x401000")

    def test_header_shows_normalized_hint(self):
        lines = _fmt(b"\x78\x56\x34\x12", type_hint="UINT32")
        self.assertEqual(lines[1], "Type hint: u32")


class TestFormatIntegers(unittest.TestCase):
    def test_u32_little_endian(self):
        lines = _fmt(b"\x78\x56\x34\x12", type_hint="u32")
        self.assertEqual(
            lines,
            [
                "Global value at 0x401000",
                "Type hint: u32",
                "Bytes: 78 56 34 12",
                "u32: 305419896 (0x12345678)",
            ],
        )

    def test_signed_hint_reports_negative_and_no_unsigned_line(self):
        lines = _fmt(b"\xfe\xff", type_hint="i16")
        self.assertIn("i16: -2", lines)
        self.assertFalse(any(line.startswith("u16") for line in lines))

    def test_signed_boundaries(self):
        self.assertIn("i8: 127", _fmt(b"\x7f", type_hint="i8"))
        self.assertIn("i8: -128", _fmt(b"\x80", type_hint="i8"))
        self.assertIn("i32: -1", _fmt(b"\xff\xff\xff\xff", type_hint="int32"))

    def test_named_unsigned_widths(self):
        self.assertIn("u8: 255 (0xff)", _fmt(b"\xff", type_hint="byte"))
        self.assertIn("u16: 4660 (0x1234)", _fmt(b"\x34\x12", type_hint="word"))
        lines = _fmt(b"\x01\x00\x00\x00\x00\x00\x00\x80", type_hint="qword")
        self.assertIn("u64: 9223372036854775809 (0x8000000000000001)", lines)
        self.assertFalse(any(line.startswith(("i64", "s64")) for line in lines))

    def test_only_leading_bytes_are_interpreted(self):
        lines = _fmt(b"\x01\x00\xaa\xbb", type_hint="u16", pointer_size=4)
        self.assertIn("u16: 1 (0x1)", lines)

    def test_data_shorter_than_width_falls_back_to_ascii(self):
        lines = _fmt(b"AB\x00\x01", type_hint="u64")
        self.assertFalse(any(line.startswith("u64") for line in lines))
        self.assertEqual(lines[-1], "ASCII: AB..")

    def test_auto_reports_every_width_signed_and_unsigned(self):
        lines = _fmt(b"\xff" * 8, type_hint="auto")
        for expected in (
            "s8: -1",
            "u8: 255 (0xff)",
            "s16: -1",
            "u16: 65535 (0xffff)",
            "s32: -1",
            "u32: 4294967295 (0xffffffff)",
            "s64: -1",
            "u64: 18446744073709551615 (0xffffffffffffffff)",
            "ptr64: 0xffffffffffffffff",
        ):
            self.assertIn(expected, lines)

    def test_auto_skips_widths_larger_than_data(self):
        lines = _fmt(b"\x05\x00", type_hint="auto")
        self.assertIn("u16: 5 (0x5)", lines)
        self.assertFalse(any(line.startswith(("u32", "s32", "u64", "s64", "ptr")) for line in lines))


class TestFormatPointers(unittest.TestCase):
    def test_ptr_hint_resolves_target(self):
        calls: list[int] = []

        def resolve(ea: int) -> str:
            calls.append(ea)
            return "sub_401000"

        data = (0x401000).to_bytes(8, "little")
        lines = _fmt(data, type_hint="ptr", resolve_pointer=resolve)
        self.assertEqual(calls, [0x401000])
        self.assertIn("ptr64: 0x401000 -> sub_401000", lines)
        # A pointer hint does not print integer interpretations.
        self.assertFalse(any(line.startswith(("u64", "i64", "s64")) for line in lines))

    def test_empty_resolution_has_no_arrow(self):
        data = (0x1000).to_bytes(4, "little")
        lines = _fmt(data, type_hint="ptr", pointer_size=4, resolve_pointer=lambda ea: "")
        self.assertIn("ptr32: 0x1000", lines)

    def test_pointer_line_requires_enough_data(self):
        lines = _fmt(b"\x00\x10\x40\x00", type_hint="u32", pointer_size=8)
        self.assertFalse(any(line.startswith("ptr") for line in lines))

    def test_unsupported_pointer_size_has_no_pointer_line(self):
        lines = _fmt(b"\x00\x10", type_hint="u16", pointer_size=2)
        self.assertFalse(any(line.startswith("ptr") for line in lines))


class TestFormatStrings(unittest.TestCase):
    def test_utf8_string_trimmed_at_nul(self):
        lines = _fmt(b"hello\x00junk", type_hint="string")
        self.assertIn("UTF-8 string: 'hello'", lines)
        self.assertFalse(any(line.startswith("UTF-16LE") for line in lines))

    def test_utf16_string_trimmed_at_wide_nul(self):
        data = "AB".encode("utf-16-le") + b"\x00\x00" + "zz".encode("utf-16-le")
        lines = _fmt(data, type_hint="utf16")
        self.assertIn("UTF-16LE string: 'AB'", lines)
        self.assertFalse(any(line.startswith("UTF-8") for line in lines))

    def test_auto_tries_both_encodings(self):
        data = "Hi".encode("utf-16-le") + b"\x00\x00\x00\x00"
        lines = _fmt(data, type_hint="auto")
        self.assertIn("UTF-16LE string: 'Hi'", lines)

    def test_non_printable_string_falls_back_to_ascii(self):
        lines = _fmt(b"\x01\x02\x03\x00", type_hint="string")
        self.assertFalse(any(line.startswith("UTF-8") for line in lines))
        self.assertEqual(lines[-1], "ASCII: ....")

    def test_leading_nul_is_not_a_string(self):
        lines = _fmt(b"\x00abc", type_hint="string")
        self.assertFalse(any(line.startswith("UTF-8") for line in lines))
        self.assertEqual(lines[-1], "ASCII: .abc")

    def test_bytes_hint_returns_ascii_only(self):
        data = b"Hi\x00\x7f" + b"\x00" * 4
        lines = _fmt(data, type_hint="bytes")
        self.assertEqual(
            lines,
            [
                "Global value at 0x401000",
                "Type hint: bytes",
                "Bytes: 48 69 00 7f 00 00 00 00",
                "ASCII: Hi......",
            ],
        )

    def test_empty_data(self):
        lines = _fmt(b"", type_hint="auto")
        self.assertEqual(lines, ["Global value at 0x401000", "Type hint: auto", "Bytes: ", "ASCII: "])


if __name__ == "__main__":
    unittest.main()
