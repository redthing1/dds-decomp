from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import amb  # noqa: E402
import field_world_import  # noqa: E402
import fld  # noqa: E402
import fld_model  # noqa: E402
import fld_scene  # noqa: E402
import lb  # noqa: E402
import tmx  # noqa: E402
import wap  # noqa: E402
from test_amb_scene import SOURCE as AMB_SOURCE  # noqa: E402
from test_fld_model_import import (  # noqa: E402
    EDIT_SOURCE,
    accessor_offset,
    first_motion_track,
)
from test_fld_scene import SOURCE as FIELD_SOURCE  # noqa: E402
from test_lb import _archive  # noqa: E402
from test_tmx import texture_bundle  # noqa: E402


MODEL_SOURCE = EDIT_SOURCE.replace(
    " scalar=1", " primary_texture=0 scalar=1"
)


def fixture() -> tuple[bytes, bytes, bytes, tmx.BundleSource, dict, bytearray]:
    model = fld.encode(fld.parse_source(MODEL_SOURCE))
    field = fld.encode(fld.parse_source(FIELD_SOURCE))
    automap = amb.encode(amb.parse_source(AMB_SOURCE))
    bundle = texture_bundle(8, 8, 0x00, bytes((1, 2, 3, 0x40)) * 64)
    textures = tmx.BundleSource(tmx._parse_bundle_records(bundle))
    document, binary = fld_scene.build_scene(
        model,
        field,
        textures=tmx.parse_bundle(bundle),
        meters_per_unit=0.01,
        frames_per_second=30.0,
        placement_marker_size=0.0,
        automap_data=automap,
        icon_marker_size=0.0,
    )
    return model, field, automap, textures, document, bytearray(binary)


def replace_view(
    document: dict,
    binary: bytearray,
    view_index: int,
    data: bytes,
) -> None:
    view = document["bufferViews"][view_index]
    start = view.get("byteOffset", 0)
    old_size = view["byteLength"]
    binary[start : start + old_size] = data
    delta = len(data) - old_size
    view["byteLength"] = len(data)
    if delta:
        for other_index, other in enumerate(document["bufferViews"]):
            if other_index != view_index and other.get("byteOffset", 0) > start:
                other["byteOffset"] = other.get("byteOffset", 0) + delta
        document["buffers"][0]["byteLength"] += delta


class FieldWorldImportTests(unittest.TestCase):
    def test_unchanged_composed_scene_preserves_every_resource(self) -> None:
        model, field, automap, textures, document, binary = fixture()
        result = field_world_import.import_world(
            model,
            field,
            document,
            bytes(binary),
            textures=textures,
            automap_data=automap,
        )

        self.assertEqual(result.model, model)
        self.assertEqual(result.field, field)
        self.assertEqual(result.automap, automap)
        self.assertEqual(tmx.encode(result.textures), tmx.encode(textures))
        self.assertEqual(result.model_summary.changed_tracks, 0)
        self.assertEqual(result.field_summary.changed_resources, 0)
        self.assertEqual(result.texture_summary.changed_images, 0)
        self.assertEqual(result.automap_summary.changed_meshes, 0)

    def test_one_scene_edits_model_field_and_texture_resources(self) -> None:
        model, field, automap, textures, document, binary = fixture()

        image = document["images"][0]
        image_view = image["bufferView"]
        view = document["bufferViews"][image_view]
        start = view["byteOffset"]
        end = start + view["byteLength"]
        width, height, rgba = tmx.decode_png(bytes(binary[start:end]))
        edited = bytearray(rgba)
        edited[:4] = bytes((9, 8, 7, 0xFF))
        replace_view(
            document,
            binary,
            image_view,
            tmx.encode_png(tmx.Texture(0, width, height, 0, bytes(edited))),
        )

        animation = document["animations"][0]
        sampler = animation["samplers"][0]
        animation_offset = accessor_offset(document, sampler["output"])
        struct.pack_into("<f", binary, animation_offset + 12, 2.5)

        placement = next(
            node for node in document["nodes"] if node.get("name") == "01heal_01"
        )
        placement["translation"] = [4.0, 5.0, 6.0]
        collision = next(
            node for node in document["nodes"] if node.get("name") == "01all"
        )
        primitive = document["meshes"][collision["mesh"]]["primitives"][0]
        collision_offset = accessor_offset(
            document, primitive["attributes"]["POSITION"]
        )
        struct.pack_into("<f", binary, collision_offset, 1.25)

        result = field_world_import.import_world(
            model,
            field,
            document,
            bytes(binary),
            textures=textures,
            automap_data=automap,
        )

        self.assertEqual(result.model_summary.changed_tracks, 1)
        self.assertEqual(result.field_summary.translations, 1)
        self.assertEqual(result.field_summary.collision_vertices, 1)
        self.assertEqual(result.texture_summary.changed_images, 1)
        self.assertEqual(
            fld_model._track_values(
                result.model, first_motion_track(result.model)
            )[1],
            (250.0, 0.0, 0.0),
        )
        self.assertEqual(
            tmx.parse_bundle(tmx.encode(result.textures))[0].rgba[:4],
            bytes((9, 8, 7, 0xFF)),
        )
        self.assertEqual(result.automap, automap)

    def test_archive_import_preserves_an_unchanged_lb_exactly(self) -> None:
        model, field, _, textures, document, binary = fixture()
        bundle = tmx.encode(textures)
        archive = _archive(
            [(1, 0, "TBN", bundle), (1, 0, "F2", field), (1, 0, "F1", model)]
        )

        rebuilt, result = field_world_import.import_archive(
            archive, document, bytes(binary)
        )

        self.assertEqual(rebuilt, archive)
        self.assertEqual(result.model, model)
        self.assertEqual(result.field, field)

    def test_tracked_field_worlds_round_trip_together(self) -> None:
        for version in ("dds1", "dds2"):
            base = ROOT / "src" / version / "data" / "field"
            stem = base / "f011_001"
            model = fld.encode(
                fld.parse_source(
                    stem.with_suffix(".f1asm").read_text(encoding="utf-8")
                )
            )
            field = fld.encode(
                fld.parse_source(
                    stem.with_suffix(".fldasm").read_text(encoding="utf-8")
                )
            )
            automap = amb.encode(
                amb.parse_source(
                    (base / "f011.ambasm").read_text(encoding="utf-8")
                )
            )
            textures = tmx.parse_source(
                stem.with_suffix(".tbnasm").read_text(encoding="utf-8")
            )
            warp_path = base / "f011.wapasm"
            references = wap.load_references(
                ROOT / "src" / version / "scripts" / "field" / "f011.bfasm",
                base / "f011.infasm",
            )
            warps = wap.parse_source(
                warp_path.read_text(encoding="utf-8"), references
            )
            bundle = tmx.encode(textures)
            document, binary = fld_scene.build_scene(
                model,
                field,
                textures=tmx.parse_bundle(bundle),
                meters_per_unit=0.01,
                frames_per_second=30.0,
                placement_marker_size=0.0,
                automap_data=automap,
                icon_marker_size=0.0,
                warp_data=wap.encode(warps),
                field_number=11,
                area_number=1,
            )

            result = field_world_import.import_world(
                model,
                field,
                document,
                binary,
                textures=textures,
                automap_data=automap,
                warps=warps,
                current_field=11,
                current_area=1,
            )
            self.assertEqual(result.model, model, version)
            self.assertEqual(result.field, field, version)
            self.assertEqual(result.automap, automap, version)
            self.assertEqual(tmx.encode(result.textures), bundle, version)
            self.assertEqual(wap.encode(result.warps), wap.encode(warps), version)


if __name__ == "__main__":
    unittest.main()
