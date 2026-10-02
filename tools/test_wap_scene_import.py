from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import field_world  # noqa: E402
import fld  # noqa: E402
import fld_model  # noqa: E402
import fld_scene  # noqa: E402
import wap  # noqa: E402
import wap_scene_import  # noqa: E402
from test_fld_scene import SOURCE as FIELD_SOURCE  # noqa: E402


def transition_table() -> wap.WapFile:
    table = wap.default_file(wap.PROFILES["dds1"])
    entries = list(table.entries)
    entries[3] = replace(
        entries[3],
        kind=1,
        area=1,
        name=wap.FixedString("01heal_01"),
        warp_args=(24, 3, 0),
        position=wap.FixedString("03pos_02"),
        camera=wap.FixedString("03cam_01"),
    )
    entries[4] = replace(
        entries[4],
        kind=1,
        area=1,
        name=wap.FixedString("01heal_01"),
        flag_mode=2,
        flag=77,
        warp_type=3,
        warp_args=(606, 0, 12),
        bgm=4,
        footstep=2,
        after_flag=8,
        after_script=wap.FixedString("after_warp"),
        tail=(3, 1, 2, 3, 4, 5, 6, 7),
    )
    entries[5] = replace(
        entries[5],
        kind=1,
        area=1,
        name=wap.FixedString("missing_01"),
        warp_args=(0, 2, 0),
    )
    return replace(table, entries=tuple(entries))


def scene(table: wap.WapFile) -> dict:
    builder = fld_model.GltfBuilder.create()
    document, _ = fld_scene.append_field_scene(
        builder.document,
        bytes(builder.binary),
        fld.encode(fld.parse_source(FIELD_SOURCE)),
        meters_per_unit=0.01,
        transitions=field_world.area_transitions(table, 11, 1),
    )
    return document


def linked_rows(document: dict) -> list[dict]:
    node = next(
        node for node in document["nodes"] if node.get("name") == "01heal_01"
    )
    return node["extras"]["ddsTransitions"]


class WapSceneImportTests(unittest.TestCase):
    def test_unchanged_transition_metadata_preserves_every_wap_byte(self) -> None:
        table = transition_table()
        rebuilt, summary = wap_scene_import.import_transitions(
            table, scene(table), 11, 1
        )
        self.assertEqual(wap.encode(rebuilt), wap.encode(table))
        self.assertEqual(summary, wap_scene_import.ImportSummary(2, 0))

    def test_imports_semantic_destination_and_transition_state(self) -> None:
        table = transition_table()
        document = scene(table)
        row = linked_rows(document)[0]
        row["destination"]["field"] = 25
        row["destination"]["area"] = 4
        row["destination"]["position"] = "04pos_03"
        row["destination"]["camera"] = {
            "name": "04cam_02",
            "mode": 2,
            "table": 9,
        }
        row["gate"] = {"mode": 3, "flag": 123}
        row["attributes"] = 5
        row["after"] = {
            "bgm": 7,
            "footstep": 6,
            "flags": 4,
            "script": "after_edit",
        }
        row["tail"] = {"control": 1, "arguments": [7, 6, 5, 4, 3, 2, 1]}

        rebuilt, summary = wap_scene_import.import_transitions(
            table, document, 11, 1
        )

        self.assertEqual(summary, wap_scene_import.ImportSummary(2, 1))
        entry = rebuilt.entries[3]
        self.assertEqual(entry.warp_args, (25, 4, 0))
        self.assertEqual(entry.position.value, "04pos_03")
        self.assertEqual(
            (entry.camera.value, entry.camera_mode, entry.camera_table),
            ("04cam_02", 2, 9),
        )
        self.assertEqual((entry.flag_mode, entry.flag, entry.attributes), (3, 123, 5))
        self.assertEqual(
            (entry.bgm, entry.footstep, entry.after_flag, entry.after_script.value),
            (7, 6, 4, "after_edit"),
        )
        self.assertEqual(entry.tail, (1, 7, 6, 5, 4, 3, 2, 1))
        self.assertEqual(wap.decode(wap.encode(rebuilt)), rebuilt)

    def test_accepts_raw_argument_edits_when_typed_view_is_unchanged(self) -> None:
        table = transition_table()
        document = scene(table)
        row = linked_rows(document)[1]
        row["destination"]["arguments"] = [700, 9, 13]

        rebuilt, _ = wap_scene_import.import_transitions(table, document, 11, 1)
        self.assertEqual(rebuilt.entries[4].warp_args, (700, 9, 13))

    def test_rejects_conflicts_and_identity_changes(self) -> None:
        table = transition_table()
        document = scene(table)
        row = linked_rows(document)[0]
        row["destination"]["arguments"][0] = 26
        row["destination"]["field"] = 25
        with self.assertRaisesRegex(
            wap_scene_import.TransitionImportError,
            "raw and semantic values conflict",
        ):
            wap_scene_import.import_transitions(table, document, 11, 1)

        document = scene(table)
        linked_rows(document)[0]["entry"] = 9
        with self.assertRaisesRegex(
            wap_scene_import.TransitionImportError,
            "changes its transition rows",
        ):
            wap_scene_import.import_transitions(table, document, 11, 1)

    def test_rejects_unlinked_summary_changes(self) -> None:
        table = transition_table()
        document = scene(table)
        wrapper = next(
            node
            for node in document["nodes"]
            if node.get("name") == "FLD2 field data"
        )
        wrapper["extras"]["ddsUnlinkedTransitionActors"][0]["entries"] = [9]
        with self.assertRaisesRegex(
            wap_scene_import.TransitionImportError,
            "unlinked-transition summary differs",
        ):
            wap_scene_import.import_transitions(table, document, 11, 1)


if __name__ == "__main__":
    unittest.main()
