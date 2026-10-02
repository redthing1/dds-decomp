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
            fld_model_import.ImportSummary(1, 1, 0, 0, 0, 0, 0, 0, 1, 0),
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
            fld_model_import.ImportSummary(1, 1, 1, 1, 1, 1, 0, 1, 1, 0),
        )
        mesh = first_mesh(rebuilt)
        self.assertEqual(mesh.positions[0][0], 125.0)
        self.assertEqual(mesh.normals[0][0], 0.5)
        self.assertEqual(mesh.texcoords[0][0], 0.25)
        self.assertEqual(mesh.colors[0][0], 64)
        self.assertEqual(mesh.triangles, first_mesh(self.data).triangles)
        self.assertEqual(mesh.controls, first_mesh(self.data).controls)
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
            fld_model_import.ImportSummary(1, 1, 1, 0, 0, 0, 1, 0, 1, 0),
        )
        self.assertEqual(first_mesh(rebuilt).attributes[0][0], 2.5)

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
