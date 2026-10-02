#!/usr/bin/env python3
"""Focused tests for relocation-aware unit verification."""

import struct
import unittest

from check_unit import (
    common_symbols,
    owned_bss_size,
    owns_exact_bss,
    relocate_sdata_item,
    trim_sdata_item,
)


class SdataRelocationTests(unittest.TestCase):
    def test_resolves_local_text_and_external_pointers(self):
        item = struct.pack("<II", 0x24, 4)
        relocs = {
            0x10: ("R_MIPS_32", ".text"),
            0x14: ("R_MIPS_32", "externalCallback"),
        }
        funcs = [(0x20, 0x10, "localCallback")]
        syms = {"localCallback": 0x00101000, "externalCallback": 0x00202000}

        linked, problems = relocate_sdata_item(item, 0x10, relocs, funcs, syms)

        self.assertEqual(problems, [])
        self.assertEqual(struct.unpack("<II", linked), (0x00101004, 0x00202004))

    def test_rejects_unresolved_local_text_target(self):
        linked, problems = relocate_sdata_item(
            struct.pack("<I", 0x40), 0, {0: ("R_MIPS_32", ".text")},
            [(0x20, 0x10, "localCallback")], {"localCallback": 0x00101000})

        self.assertEqual(linked, struct.pack("<I", 0x40))
        self.assertEqual(problems, ["unresolved R_MIPS_32 .text at +0x0"])

    def test_rejects_unsupported_relocation(self):
        linked, problems = relocate_sdata_item(
            struct.pack("<I", 0), 0, {0: ("R_MIPS_GPREL16", "global")}, [],
            {"global": 0x00303000})

        self.assertEqual(linked, struct.pack("<I", 0))
        self.assertEqual(problems, ["unsupported R_MIPS_GPREL16 at +0x0"])

    def test_relocation_words_are_not_trimmed_as_zero_padding(self):
        item = b"\0" * 12
        relocs = {0x10: ("R_MIPS_32", ".text"), 0x14: ("R_MIPS_32", ".text")}

        self.assertEqual(trim_sdata_item(item, 0x10, relocs), b"\0" * 8)


class OwnedBssTests(unittest.TestCase):
    YAML = """\
segments:
  - name: main
    subsegments:
      - { start: 0x1000, type: .bss, vram: 0x003DC658, name: file/fileManager }
      - { start: 0x1000, type: bss, vram: 0x003DC698, name: bss }
"""

    def test_dict_form_unit_owns_exact_span(self):
        self.assertEqual(owned_bss_size(self.YAML, "file/fileManager"), 0x40)
        self.assertTrue(owns_exact_bss(self.YAML, "file/fileManager", 0x40))
        self.assertFalse(owns_exact_bss(self.YAML, "file/fileManager", 0x3C))
        self.assertFalse(owns_exact_bss(self.YAML, "file/fileManager", 0x44))

    def test_list_form_or_wrong_unit_does_not_claim_bss(self):
        list_form = """\
segments:
  - [0x1000, .bss, file/fileManager]
  - { start: 0x1000, type: bss, vram: 0x003DC698, name: bss }
"""
        self.assertIsNone(owned_bss_size(list_form, "file/fileManager"))
        self.assertIsNone(owned_bss_size(self.YAML, "file/anotherUnit"))
        self.assertIsNone(owned_bss_size(
            self.YAML.replace("type: .bss", "type: bss", 1),
            "file/fileManager",
        ))

    def test_requires_immediate_explicit_next_vram_boundary(self):
        missing_boundary = """\
segments:
  - name: main
    subsegments:
      - { start: 0x1000, type: .bss, vram: 0x003DC658, name: file/fileManager }
      - [0x1000, bin, trailer]
      - { start: 0x1000, type: bss, vram: 0x003DC698, name: bss }
"""
        self.assertIsNone(
            owned_bss_size(missing_boundary, "file/fileManager")
        )

        crossed_parent = """\
segments:
  - name: main
    subsegments:
      - { start: 0x1000, type: .bss, vram: 0x003DC658, name: file/fileManager }
  - name: another
    subsegments:
      - { start: 0x2000, type: bss, vram: 0x003DC698, name: bss }
"""
        self.assertIsNone(owned_bss_size(crossed_parent, "file/fileManager"))

    def test_rejects_duplicate_or_nonadvancing_claims(self):
        duplicate = self.YAML + """\
      - { start: 0x1000, type: .bss, vram: 0x003DC700, name: file/fileManager }
      - { start: 0x1000, type: bss, vram: 0x003DC740, name: bss }
"""
        backwards = self.YAML.replace("0x003DC698", "0x003DC650")
        self.assertIsNone(owned_bss_size(duplicate, "file/fileManager"))
        self.assertIsNone(owned_bss_size(backwards, "file/fileManager"))

    def test_common_symbols_are_kept_out_of_owned_bss(self):
        nm = """\
00000004 00000040 c fileManagerWork
00000000 00000040 B explicitFileManagerWork
00000080 00000008 C ordinaryCommon
"""
        self.assertEqual(
            common_symbols(nm),
            [("fileManagerWork", 0x40), ("ordinaryCommon", 8)],
        )


if __name__ == "__main__":
    unittest.main()
