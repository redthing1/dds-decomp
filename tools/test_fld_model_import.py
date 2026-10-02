from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fld  # noqa: E402
import fld_model  # noqa: E402
import fld_model_import  # noqa: E402
import tmx  # noqa: E402
from test_fld_model import SOURCE  # noqa: E402


EDIT_SOURCE = (
    SOURCE.replace(
        "model_material index=0",
        "model_material index=0 color_0=128,128,128,128 "
        "primary_uv_transform=0,0,1,1,0 scalar=1",
    )
    .replace("model_draw material=0 qwords=4", "model_draw material=0 qwords=10")
    .replace(
        "mesh_program address=12",
        "mesh_normals\n"
        "normal 0,0,1 0,0,1 0,0,1\n"
        "mesh_texcoords\n"
        "texcoord 0,0 1,0 0,1\n"
        "mesh_colors\n"
        "color 128,64,0,64 128,64,0,128 128,64,0,128\n"
        "mesh_program address=12\n"
        "vif_nops count=3",
    )
)

ATTRIBUTE_SOURCE = (
    SOURCE.replace("model_draw material=0 qwords=4", "model_draw material=0 qwords=8")
    .replace(
        "mesh_program address=12",
        "mesh_attributes\n"
        "attribute 1,2,3,4 5,6,7,8 9,10,11,12\n"
        "mesh_program address=12\n"
        "vif_nops count=3",
    )
)

HIERARCHY_SOURCE = SOURCE.replace("model_items count=1", "model_items count=2").replace(
    "label bounds",
    "model_item node_id=1 parent=0 rotation=0,0,0 position=0,0,0,1 "
    "scale=1,1,1,0 bounds=null commands=null\n"
    "label bounds",
)

EULER_MOTION_SOURCE = EDIT_SOURCE.replace(
    "selector=translation", "selector=euler_rotation"
).replace(
    "motion_vector3 0,0,0 100,0,0",
    "motion_vector3 0,0,0 0.1,0.2,0.3",
)

QUATERNION_MOTION_SOURCE = EDIT_SOURCE.replace(
    "selector=translation", "selector=quaternion"
).replace(
    "format=vector3", "format=quaternion_s16"
).replace(
    "motion_vector3 0,0,0 100,0,0",
    "motion_quaternion_s16 0,0,0,4096 0,0,0,4096",
)


def accessor_offset(document: dict, accessor_index: int) -> int:
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    return view.get("byteOffset", 0) + accessor.get("byteOffset", 0)


def first_mesh(data: bytes) -> fld.ModelMesh:
    words, data_end, relocations = fld._read_header(data)
    resources = fld._read_resources(data, fld._read_types(data, words, data_end))
    resource = next(resource for resource in resources if resource.type_id == 2)
    model, items = fld._read_model_resource(data, resource.data, data_end, "test")
    materials = fld._read_model_materials(
        data, model.materials, data_end, "test materials"
    )
    roots, lists, draws = fld._read_model_draw_graph(
        data, items, len(materials), data_end, set(relocations), "test"
    )
    draw_list = lists[roots[items[0].commands][0]]
    draw = draws[draw_list.draws[0]]
    meshes, _ = fld._read_model_mesh_packet(
        data, draw.packet, draw.quadwords * 0x10, "test packet"
    )
    return meshes[0]


def first_material(data: bytes) -> fld.ModelMaterial:
    words, data_end, _ = fld._read_header(data)
    resources = fld._read_resources(data, fld._read_types(data, words, data_end))
    resource = next(resource for resource in resources if resource.type_id == 2)
    model, _ = fld._read_model_resource(data, resource.data, data_end, "test")
    return fld._read_model_materials(
        data, model.materials, data_end, "test materials"
    )[0]


def first_motion_track(data: bytes) -> fld.ModelMotionTrack:
    words, data_end, relocations = fld._read_header(data)
    resources = fld._read_resources(data, fld._read_types(data, words, data_end))
    resource = next(resource for resource in resources if resource.type_id == 2)
    model, items = fld._read_model_resource(data, resource.data, data_end, "test")
    materials = fld._read_model_materials(
        data, model.materials, data_end, "test materials"
    )
    motion = fld._read_model_motion(
        data,
        model.motion,
        data_end,
        set(relocations),
        len(items),
        len(materials),
        "test motion",
    )
    clip = motion.clips[0]
    if clip is None:
        raise AssertionError("test fixture has no first motion clip")
    return clip.tracks[0]


class FldModelImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = fld.encode(fld.parse_source(EDIT_SOURCE))
        self.document, binary = fld_model.build_gltf(
            self.data, meters_per_unit=0.01
        )
        self.binary = bytearray(binary)
        self.primitive = self.document["meshes"][0]["primitives"][0]

    def test_unchanged_glb_preserves_every_fld1_byte(self) -> None:
        glb = fld_model.encode_glb(self.document, bytes(self.binary))
        document, binary = fld_model_import.decode_glb(glb)
        rebuilt, summary = fld_model_import.import_geometry(
            self.data, document, binary
        )
        self.assertEqual(rebuilt, self.data)
        self.assertEqual(
            summary,
            fld_model_import.ImportSummary(
                1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0
            ),
        )

    def test_imports_vertex_attributes_without_changing_packet_shape(self) -> None:
        attributes = self.primitive["attributes"]
        struct.pack_into(
            "<f",
            self.binary,
            accessor_offset(self.document, attributes["POSITION"]),
            1.25,
        )
        struct.pack_into(
            "<f",
            self.binary,
            accessor_offset(self.document, attributes["NORMAL"]),
            0.5,
        )
        struct.pack_into(
            "<f",
            self.binary,
            accessor_offset(self.document, attributes["TEXCOORD_0"]),
            0.25,
        )
        self.binary[
            accessor_offset(self.document, attributes["COLOR_0"])
        ] = 0x80

        rebuilt, summary = fld_model_import.import_geometry(
            self.data, self.document, bytes(self.binary)
        )
        self.assertEqual(
            summary,
            fld_model_import.ImportSummary(
                1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0
            ),
        )
        mesh = first_mesh(rebuilt)
        self.assertEqual(mesh.positions[0][0], 125.0)
        self.assertEqual(mesh.normals[0][0], 0.5)
        self.assertEqual(mesh.texcoords[0][0], 0.25)
        self.assertEqual(mesh.colors[0][0], 64)
        self.assertEqual(mesh.triangles, first_mesh(self.data).triangles)
        self.assertEqual(mesh.controls, first_mesh(self.data).controls)
        words, data_end, _ = fld._read_header(rebuilt)
        resource = next(
            resource
            for resource in fld._read_resources(
                rebuilt, fld._read_types(rebuilt, words, data_end)
            )
            if resource.type_id == 2
        )
        _, items = fld._read_model_resource(rebuilt, resource.data, data_end, "test")
        self.assertEqual(
            struct.unpack_from("<6f", rebuilt, items[0].bounds),
            (0.0, 0.0, 0.0, 125.0, 100.0, 0.0),
        )
        self.assertEqual(len(rebuilt), len(self.data))
        fld.parse_source(fld.render_source(rebuilt))

    def test_imports_unclassified_four_float_attribute(self) -> None:
        data = fld.encode(fld.parse_source(ATTRIBUTE_SOURCE))
        document, binary = fld_model.build_gltf(data)
        binary = bytearray(binary)
        accessor = document["meshes"][0]["primitives"][0]["attributes"][
            "_DDS_ATTRIBUTE"
        ]
        struct.pack_into("<f", binary, accessor_offset(document, accessor), 2.5)

        rebuilt, summary = fld_model_import.import_geometry(
            data, document, bytes(binary)
        )
        self.assertEqual(
            summary,
            fld_model_import.ImportSummary(
                1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0
            ),
        )
        self.assertEqual(first_mesh(rebuilt).attributes[0][0], 2.5)

    def test_imports_model_item_and_resource_transforms(self) -> None:
        item_node = next(
            node
            for node in self.document["nodes"]
            if node["name"] == "test_model/node_0"
        )
        wrapper = next(
            node for node in self.document["nodes"] if node["name"] == "test_model"
        )
        item_rotation = fld_model._euler_quaternion(0.25, -0.5, 0.75)
        half = 2**-0.5
        item_node["translation"] = [4.0, 5.0, 6.0]
        item_node["rotation"] = item_rotation
        item_node["scale"] = [2.0, 3.0, 4.0]
        wrapper["translation"] = [7.0, 8.0, 9.0]
        wrapper["rotation"] = [0.0, 0.0, half, half]

        rebuilt, summary = fld_model_import.import_geometry(
            self.data, self.document, bytes(self.binary)
        )
        self.assertEqual(summary.changed_nodes, 2)
        self.assertEqual(summary.translations, 2)
        self.assertEqual(summary.rotations, 2)
        self.assertEqual(summary.scales, 1)
        words, data_end, _ = fld._read_header(rebuilt)
        resource = next(
            resource
            for resource in fld._read_resources(
                rebuilt, fld._read_types(rebuilt, words, data_end)
            )
            if resource.type_id == 2
        )
        _, items = fld._read_model_resource(rebuilt, resource.data, data_end, "test")
        self.assertEqual(items[0].position, (400.0, 500.0, 600.0, 1.0))
        for actual, expected in zip(
            items[0].rotation, (0.25, -0.5, 0.75), strict=True
        ):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(items[0].scale, (2.0, 3.0, 4.0, 0.0))
        native = struct.unpack_from("<12f", rebuilt, resource.transform)
        self.assertEqual(native[:4], (700.0, 800.0, 900.0, 1.0))
        for actual, expected in zip(native[4:8], (0.0, 0.0, half, half), strict=True):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(len(rebuilt), len(self.data))
        self.assertEqual(
            fld.encode(fld.parse_source(fld.render_source(rebuilt))), rebuilt
        )

    def test_imports_model_hierarchy_changes(self) -> None:
        data = fld.encode(fld.parse_source(HIERARCHY_SOURCE))
        document, binary = fld_model.build_gltf(data)
        by_name = {node["name"]: node for node in document["nodes"]}
        first = by_name["test_model/node_0"]
        second_index = document["nodes"].index(by_name["test_model/node_1"])
        wrapper = by_name["test_model"]
        first["children"].remove(second_index)
        wrapper["children"] = [*wrapper["children"], second_index]

        rebuilt, summary = fld_model_import.import_geometry(data, document, binary)
        self.assertEqual(summary.changed_nodes, 1)
        self.assertEqual(summary.parents, 1)
        words, data_end, _ = fld._read_header(rebuilt)
        resource = next(
            resource
            for resource in fld._read_resources(
                rebuilt, fld._read_types(rebuilt, words, data_end)
            )
            if resource.type_id == 2
        )
        _, items = fld._read_model_resource(rebuilt, resource.data, data_end, "test")
        self.assertEqual([item.parent for item in items], [-1, -1])

        first_index = document["nodes"].index(first)
        wrapper["children"].remove(first_index)
        wrapper["children"].remove(second_index)
        by_name["test_model/node_1"]["children"] = [first_index]
        first["children"] = [second_index]
        with self.assertRaisesRegex(
            fld_model_import.ModelImportError, "hierarchy contains a cycle"
        ):
            fld_model_import.import_geometry(data, document, binary)

    def test_imports_motion_values_and_preserves_frame_keys(self) -> None:
        animation = self.document["animations"][0]
        sampler = animation["samplers"][0]
        output_offset = accessor_offset(self.document, sampler["output"])
        struct.pack_into("<f", self.binary, output_offset + 12, 2.5)

        rebuilt, summary = fld_model_import.import_geometry(
            self.data, self.document, bytes(self.binary)
        )
        self.assertEqual(summary.animations, 1)
        self.assertEqual(summary.tracks, 1)
        self.assertEqual(summary.changed_tracks, 1)
        track = first_motion_track(rebuilt)
        self.assertEqual(track.frames, (0, 30))
        self.assertEqual(
            fld_model._track_values(rebuilt, track)[1], (250.0, 0.0, 0.0)
        )

        time_offset = accessor_offset(self.document, sampler["input"])
        struct.pack_into("<f", self.binary, time_offset + 4, 0.5)
        with self.assertRaisesRegex(
            fld_model_import.ModelImportError, "changes its frame keys"
        ):
            fld_model_import.import_geometry(
                self.data, self.document, bytes(self.binary)
            )

    def test_imports_scale_motion(self) -> None:
        data = fld.encode(
            fld.parse_source(
                EDIT_SOURCE.replace("selector=translation", "selector=scale")
            )
        )
        document, binary = fld_model.build_gltf(data)
        binary = bytearray(binary)
        sampler = document["animations"][0]["samplers"][0]
        output_offset = accessor_offset(document, sampler["output"])
        struct.pack_into("<f", binary, output_offset + 16, 2.5)

        rebuilt, summary = fld_model_import.import_geometry(
            data, document, bytes(binary)
        )
        self.assertEqual(summary.changed_tracks, 1)
        track = first_motion_track(rebuilt)
        self.assertEqual(
            fld_model._track_values(rebuilt, track)[1], (100.0, 2.5, 0.0)
        )

    def test_rejects_added_motion_clip(self) -> None:
        duplicate = dict(self.document["animations"][0])
        duplicate["name"] = "unexpected/clip_0"
        self.document["animations"].append(duplicate)
        with self.assertRaisesRegex(
            fld_model_import.ModelImportError,
            "changes the selected models' animation set",
        ):
            fld_model_import.import_geometry(
                self.data, self.document, bytes(self.binary)
            )

    def test_imports_euler_and_packed_quaternion_motion(self) -> None:
        for source, expected_format in (
            (EULER_MOTION_SOURCE, "vector3"),
            (QUATERNION_MOTION_SOURCE, "quaternion_s16"),
        ):
            with self.subTest(format=expected_format):
                data = fld.encode(fld.parse_source(source))
                document, binary = fld_model.build_gltf(data)
                binary = bytearray(binary)
                sampler = document["animations"][0]["samplers"][0]
                output_offset = accessor_offset(document, sampler["output"])
                target = fld_model._euler_quaternion(0.4, -0.2, 0.7)
                struct.pack_into("<4f", binary, output_offset + 16, *target)

                rebuilt, summary = fld_model_import.import_geometry(
                    data, document, bytes(binary)
                )
                self.assertEqual(summary.changed_tracks, 1)
                track = first_motion_track(rebuilt)
                values = fld_model._track_values(rebuilt, track)[1]
                if expected_format == "vector3":
                    for actual, expected in zip(
                        values, (0.4, -0.2, 0.7), strict=True
                    ):
                        self.assertAlmostEqual(actual, expected, places=6)
                else:
                    rebuilt_rotation = fld_model._normalized_quaternion(
                        tuple(float(value) / 4096.0 for value in values)
                    )
                    self.assertAlmostEqual(
                        abs(sum(a * b for a, b in zip(rebuilt_rotation, target))),
                        1.0,
                        places=6,
                    )

    def test_tracked_fld1_sources_round_trip_through_gltf(self) -> None:
        for version in ("dds1", "dds2"):
            path = ROOT / "src" / version / "data" / "field" / "f011_001.f1asm"
            data = fld.encode(fld.parse_source(path.read_text(encoding="utf-8")))
            document, binary = fld_model.build_gltf(data)
            rebuilt, _ = fld_model_import.import_geometry(data, document, binary)
            self.assertEqual(rebuilt, data, path.name)

    def test_imports_semantic_material_fields(self) -> None:
        gltf_material = self.document["materials"][self.primitive["material"]]
        fields = gltf_material["extras"]["ddsMaterialFields"]
        fields["color_0"] = [12, 34, 56, 78]
        fields["primary_uv_transform"][0] = 0.25
        fields["scalar"][0] = 0.75

        rebuilt, summary = fld_model_import.import_geometry(
            self.data, self.document, bytes(self.binary)
        )
        self.assertEqual(summary.changed_materials, 1)
        values = dict(first_material(rebuilt).fields)
        self.assertEqual(values["color_0"], (12, 34, 56, 78))
        self.assertEqual(values["primary_uv_transform"][0], 0.25)
        self.assertEqual(values["scalar"], (0.75,))
        self.assertEqual(first_mesh(rebuilt), first_mesh(self.data))

    def test_texture_reference_edits_require_and_obey_bundle_metadata(self) -> None:
        source = EDIT_SOURCE.replace(" scalar=1", " primary_texture=0 scalar=1")
        data = fld.encode(fld.parse_source(source))
        document, binary = fld_model.build_gltf(data)
        fields = document["materials"][0]["extras"]["ddsMaterialFields"]
        fields["primary_texture"] = [1]
        with self.assertRaisesRegex(
            fld_model_import.ModelImportError, "without bundle metadata"
        ):
            fld_model_import.import_geometry(data, document, binary)

        texture = tmx.Texture(0, 8, 8, 0x01, bytes((0, 0, 0, 0xFF)) * 64)
        document, binary = fld_model.build_gltf(
            data, textures=(texture, texture)
        )
        material_index = document["meshes"][0]["primitives"][0]["material"]
        fields = document["materials"][material_index]["extras"][
            "ddsMaterialFields"
        ]
        fields["primary_texture"] = [2]
        with self.assertRaisesRegex(
            fld_model_import.ModelImportError, "bundle has 2 textures"
        ):
            fld_model_import.import_geometry(data, document, binary)
        fields["primary_texture"] = [1]
        rebuilt, _ = fld_model_import.import_geometry(data, document, binary)
        self.assertEqual(dict(first_material(rebuilt).fields)["primary_texture"], (1,))

    def test_rejects_topology_and_identity_changes(self) -> None:
        indices = self.primitive["indices"]
        self.binary[accessor_offset(self.document, indices)] = 2
        with self.assertRaisesRegex(
            fld_model_import.ModelImportError, "triangle topology"
        ):
            fld_model_import.import_geometry(
                self.data, self.document, bytes(self.binary)
            )

        self.binary[accessor_offset(self.document, indices)] = 0
        self.primitive["extras"]["ddsPacketMeshIndex"] = 9
        with self.assertRaisesRegex(
            fld_model_import.ModelImportError, "identity metadata"
        ):
            fld_model_import.import_geometry(
                self.data, self.document, bytes(self.binary)
            )

    def test_requires_native_axes_and_matching_channels(self) -> None:
        self.document["asset"]["extras"]["ddsNativeAxesPreserved"] = False
        with self.assertRaisesRegex(
            fld_model_import.ModelImportError, "native DDS axes"
        ):
            fld_model_import.import_geometry(
                self.data, self.document, bytes(self.binary)
            )
        self.document["asset"]["extras"]["ddsNativeAxesPreserved"] = True
        del self.primitive["attributes"]["NORMAL"]
        with self.assertRaisesRegex(
            fld_model_import.ModelImportError, "vertex-channel layout"
        ):
            fld_model_import.import_geometry(
                self.data, self.document, bytes(self.binary)
            )


if __name__ == "__main__":
    unittest.main()
