from __future__ import annotations

import math
import struct
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fld  # noqa: E402
import fld_model  # noqa: E402
import fld_scene  # noqa: E402
import fld_scene_import  # noqa: E402
from test_fld_scene import SOURCE  # noqa: E402


class FieldSceneImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.field_data = fld.encode(fld.parse_source(SOURCE))
        builder = fld_model.GltfBuilder.create()
        self.document, self.binary = fld_scene.append_field_scene(
            builder.document,
            bytes(builder.binary),
            self.field_data,
            meters_per_unit=0.01,
        )
        self.document["asset"]["extras"] = {
            "ddsMetersPerUnit": 0.01,
            "ddsNativeAxesPreserved": True,
        }

    def test_unchanged_glb_preserves_every_field_byte(self) -> None:
        glb = fld_model.encode_glb(self.document, self.binary)
        decoded = fld_scene_import.decode_glb(glb)
        rebuilt, summary = fld_scene_import.import_transforms(
            self.field_data, decoded
        )
        self.assertEqual(rebuilt, self.field_data)
        self.assertEqual(
            summary,
            fld_scene_import.ImportSummary(3, 0, 0, 0, 0),
        )

    def test_imports_changed_translation_rotation_and_scale(self) -> None:
        by_name = {node["name"]: node for node in self.document["nodes"]}
        by_name["01heal_01"]["translation"] = [4.0, 5.0, 6.0]
        half = math.sqrt(0.5)
        by_name["01cam_01"]["rotation"] = [0.0, 0.0, half, half]
        by_name["01all"]["scale"] = [2.0, 3.0, 4.0]

        rebuilt, summary = fld_scene_import.import_transforms(
            self.field_data, self.document
        )
        self.assertEqual(
            summary,
            fld_scene_import.ImportSummary(3, 3, 1, 1, 1),
        )
        words, data_end, _ = fld._read_header(rebuilt)
        resources = fld._read_resources(
            rebuilt, fld._read_types(rebuilt, words, data_end)
        )
        transforms = {
            fld._fixed_string(rebuilt, resource.name, "name"): struct.unpack_from(
                "<12f", rebuilt, resource.transform
            )
            for resource in resources
        }
        self.assertEqual(transforms["01heal_01"][:4], (400.0, 500.0, 600.0, 1.0))
        for actual, expected in zip(
            transforms["01cam_01"][4:8], (0.0, 0.0, half, half), strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(transforms["01all"][8:12], (2.0, 3.0, 4.0, 1.0))
        fld.parse_source(fld.render_source(rebuilt))

    def test_rejects_identity_mismatch_and_matrix_nodes(self) -> None:
        placement = next(
            node for node in self.document["nodes"] if node["name"] == "01heal_01"
        )
        placement["name"] = "wrong"
        with self.assertRaisesRegex(
            fld_scene_import.FieldSceneImportError, "expected '01heal_01'"
        ):
            fld_scene_import.import_transforms(self.field_data, self.document)

        placement["name"] = "01heal_01"
        placement["matrix"] = [1.0] * 16
        with self.assertRaisesRegex(
            fld_scene_import.FieldSceneImportError, "preserve editable TRS"
        ):
            fld_scene_import.import_transforms(self.field_data, self.document)

    def test_accepts_an_empty_field_wrapper_and_requires_native_axes(self) -> None:
        source = """\
fld2 1
header version=23 magic=FLD2 type_count=1 type_table=@types word_1c=0 word_20=0 word_24=0 word_28=0 word_2c=0 word_30=0 word_34=0 word_38=0 word_3c=0
label types
type id=99 count=0 resources=@data_end
label data_end
end_data
"""
        data = fld.encode(fld.parse_source(source))
        builder = fld_model.GltfBuilder.create()
        document, _ = fld_scene.append_field_scene(
            builder.document, bytes(builder.binary), data, meters_per_unit=1.0
        )
        document["asset"]["extras"] = {
            "ddsMetersPerUnit": 1.0,
            "ddsNativeAxesPreserved": True,
        }
        rebuilt, summary = fld_scene_import.import_transforms(data, document)
        self.assertEqual(rebuilt, data)
        self.assertEqual(summary.resources, 0)

        document["asset"]["extras"]["ddsNativeAxesPreserved"] = False
        with self.assertRaisesRegex(
            fld_scene_import.FieldSceneImportError, "native DDS axes"
        ):
            fld_scene_import.import_transforms(data, document)


if __name__ == "__main__":
    unittest.main()
