from __future__ import annotations

import struct
import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fld  # noqa: E402
import fld_model  # noqa: E402
import fld_scene  # noqa: E402
import field_world  # noqa: E402
import wap  # noqa: E402


SOURCE = """\
fld2 1
header version=23 magic=FLD2 type_count=3 type_table=@types word_1c=0 word_20=0 word_24=0 word_28=0 word_2c=0 word_30=0 word_34=0 word_38=0 word_3c=0
label types
type id=3 count=1 resources=@collision_resources
type id=4 count=1 resources=@camera_resources
type id=10 count=1 resources=@placement_resources
label collision_resources
resource serial=2 flags=0 type=3 name=@collision_name reserved=0 transform=@collision_transform area=null link=null sblock=null data=@collision
label camera_resources
resource serial=3 flags=1 type=4 name=@camera_name reserved=0 transform=@camera_transform area=null link=null sblock=null data=@camera
label placement_resources
resource serial=4 flags=1 type=10 name=@placement_name reserved=0 transform=@placement_transform area=null link=null sblock=null data=@placement
label collision_name
string16 01all
label camera_name
string16 01cam_01
label placement_name
string16 01heal_01
label collision_transform
transform position=0,0,0,1 rotation=0,0,0,1 scale=1,1,1,1
label camera_transform
transform position=400,500,600,1 rotation=0,0,0,1 scale=1,1,1,1
label placement_transform
transform position=100,200,300,1 rotation=0,0,0,1 scale=1,1,1,1
label collision
collision vertex_count=4 face_count=1 extra_count=0 vertices=@vertices faces=@faces stop=null reserved=0,0
label vertices
vertex 0 0 0 1
vertex 100 0 0 1
vertex 100 0 100 1
vertex 0 0 100 1
label faces
face attributes=0x00002800 move_floor=0 sound=0 stop=0 place=0 automap_block=1 automap_upper_name=2 vertices=0,1,2,3 encounter_zone=7 special=0,0
label camera
camera fovy=0.7853981852531433
label placement
placement kind=8 event=-1 visible=1 payload=@special
label special
special_point kind=heal id=3
label data_end
end_data
"""


MOTION_SOURCE = """\
fld2 1
header version=23 magic=FLD2 type_count=1 type_table=@types word_1c=0 word_20=0 word_24=0 word_28=0 word_2c=0 word_30=0 word_34=0 word_38=0 word_3c=0
label types
type id=9 count=1 resources=@motion_resources
label motion_resources
resource serial=7 flags=2 type=9 name=@motion_name reserved=0 transform=null area=null link=null sblock=null data=@motion_data
label motion_name
string16 01cam_01_MOTION
label motion_data
motion tracks=vector3:@vector_curve,quaternion:@rotation_curve,scalar:@scalar_curve,light:@light_curve
label vector_curve
motion_curve count=2 values=@vector_values keys=@vector_keys word_0c=1
label vector_values
vector3 100 200 300
vector3 400 500 600
label vector_keys
keys 0 30
label rotation_curve
motion_curve count=2 values=@rotation_values keys=@rotation_keys word_0c=0
label rotation_values
quaternion 0 0 0 1
quaternion 0 0 0.70710677 0.70710677
label rotation_keys
keys 0 30
label scalar_curve
motion_curve count=2 values=@scalar_values keys=@scalar_keys word_0c=1
label scalar_values
scalar 45
scalar 55
label scalar_keys
keys 0 30
label light_curve
motion_curve count=2 values=@light_values keys=@light_keys word_0c=0
label light_values
motion_light 1 2 3 4 5 6 7 8 9 10
motion_light 11 12 13 14 15 16 17 18 19 20
label light_keys
keys 0 30
label data_end
end_data
"""


class FldSceneTests(unittest.TestCase):
    def test_appends_collision_camera_and_placement(self) -> None:
        builder = fld_model.GltfBuilder.create()
        document, binary = fld_scene.append_field_scene(
            builder.document,
            bytes(builder.binary),
            fld.encode(fld.parse_source(SOURCE)),
            meters_per_unit=0.01,
            placement_marker_size=50.0,
        )
        wrapper = document["nodes"][-1]
        self.assertEqual(wrapper["name"], "FLD2 field data")
        self.assertEqual(
            wrapper["extras"],
            {
                "ddsCollisionResources": 1,
                "ddsCameraResources": 1,
                "ddsMotionResources": 0,
                "ddsPlacementResources": 1,
            },
        )
        children = [document["nodes"][index] for index in wrapper["children"]]
        self.assertEqual([node["name"] for node in children], ["01all", "01cam_01", "01heal_01"])
        self.assertEqual(children[1]["extras"]["ddsCameraYFov"], 0.7853981852531433)
        self.assertEqual(children[2]["translation"], [1.0, 2.0, 3.0])
        self.assertEqual(
            children[2]["extras"]["ddsSpecialPoint"], {"kind": "heal", "id": 3}
        )

        collision = document["meshes"][children[0]["mesh"]]
        primitive = collision["primitives"][0]
        accessor = document["accessors"][primitive["indices"]]
        view = document["bufferViews"][accessor["bufferView"]]
        indices = struct.unpack_from("<6H", binary, view["byteOffset"])
        self.assertEqual(indices, (0, 1, 2, 0, 2, 3))
        self.assertEqual(collision["extras"]["ddsTriangleCount"], 2)
        self.assertEqual(collision["extras"]["ddsVertexCount"], 4)
        self.assertEqual(
            collision["extras"]["ddsCollisionFaces"],
            [
                {
                    "index": 0,
                    "firstTriangle": 0,
                    "triangleCount": 2,
                    "flags": 0,
                    "moveFloor": 0,
                    "sound": 0,
                    "stop": 0,
                    "place": 0,
                    "automap": {"block": 1, "upperName": 2},
                    "vertices": [0, 1, 2, 3],
                    "encounterZone": 7,
                    "special": [0, 0],
                }
            ],
        )
        self.assertIn("KHR_materials_unlit", document["extensionsUsed"])
        fld_model.encode_glb(document, binary)

    def test_exports_path_motion_as_animation_and_typed_metadata(self) -> None:
        builder = fld_model.GltfBuilder.create()
        document, binary = fld_scene.append_field_scene(
            builder.document,
            bytes(builder.binary),
            fld.encode(fld.parse_source(MOTION_SOURCE)),
            meters_per_unit=0.01,
            frames_per_second=30.0,
        )

        node = next(
            node
            for node in document["nodes"]
            if node.get("name") == "01cam_01_MOTION"
        )
        tracks = node["extras"]["ddsMotionTracks"]
        self.assertEqual(
            [track["representation"] for track in tracks],
            ["translation", "rotation", "extras", "extras"],
        )
        self.assertEqual(tracks[2]["values"], [[45.0], [55.0]])
        self.assertEqual(tracks[3]["values"][1][-1], 20.0)
        animation = next(
            animation
            for animation in document["animations"]
            if animation["name"] == "01cam_01_MOTION/motion"
        )
        self.assertEqual(
            [channel["target"]["path"] for channel in animation["channels"]],
            ["translation", "rotation"],
        )
        self.assertEqual(animation["extras"]["ddsTrackIndices"], [0, 1])
        self.assertEqual(document["nodes"][-1]["extras"]["ddsMotionResources"], 1)
        fld_model.encode_glb(document, binary)

    def test_rejects_negative_marker_size(self) -> None:
        builder = fld_model.GltfBuilder.create()
        with self.assertRaisesRegex(fld.FldError, "marker size"):
            fld_scene.append_field_scene(
                builder.document,
                bytes(builder.binary),
                fld.encode(fld.parse_source(SOURCE)),
                meters_per_unit=0.01,
                placement_marker_size=-1.0,
            )

    def test_empty_field_omits_empty_children_array(self) -> None:
        source = """\
fld2 1
header version=23 magic=FLD2 type_count=1 type_table=@types word_1c=0 word_20=0 word_24=0 word_28=0 word_2c=0 word_30=0 word_34=0 word_38=0 word_3c=0
label types
type id=99 count=0 resources=@data_end
label data_end
end_data
"""
        builder = fld_model.GltfBuilder.create()
        document, binary = fld_scene.append_field_scene(
            builder.document,
            bytes(builder.binary),
            fld.encode(fld.parse_source(source)),
            meters_per_unit=1.0,
        )
        self.assertNotIn("children", document["nodes"][-1])
        fld_model.encode_glb(document, binary)

    def test_placement_retains_all_owned_wap_transitions(self) -> None:
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
            name=wap.FixedString("missing_actor"),
            warp_args=(0, 2, 0),
        )
        table = replace(table, entries=tuple(entries))
        transitions = field_world.area_transitions(table, 11, 1)

        builder = fld_model.GltfBuilder.create()
        document, binary = fld_scene.append_field_scene(
            builder.document,
            bytes(builder.binary),
            fld.encode(fld.parse_source(SOURCE)),
            meters_per_unit=0.01,
            transitions=transitions,
        )

        wrapper = document["nodes"][-1]
        children = [document["nodes"][index] for index in wrapper["children"]]
        placement = next(node for node in children if node["name"] == "01heal_01")
        linked = placement["extras"]["ddsTransitions"]
        self.assertEqual([row["entry"] for row in linked], [3, 4])
        self.assertEqual(
            linked[0]["destination"],
            {
                "typeId": 0,
                "type": "field",
                "arguments": [24, 3, 0],
                "position": "03pos_02",
                "camera": {"name": "03cam_01", "mode": 0, "table": 0},
                "field": 24,
                "area": 3,
            },
        )
        self.assertEqual(linked[1]["gate"], {"mode": 2, "flag": 77})
        self.assertEqual(linked[1]["destination"]["event"], 606)
        self.assertEqual(linked[1]["destination"]["alternateField"], 12)
        self.assertEqual(
            linked[1]["after"],
            {"bgm": 4, "footstep": 2, "flags": 8, "script": "after_warp"},
        )
        self.assertEqual(
            linked[1]["tail"],
            {"control": 3, "arguments": [1, 2, 3, 4, 5, 6, 7]},
        )
        self.assertEqual(wrapper["extras"]["ddsTransitionRows"], 3)
        self.assertEqual(wrapper["extras"]["ddsLinkedTransitionRows"], 2)
        self.assertEqual(
            wrapper["extras"]["ddsUnlinkedTransitionActors"],
            [{"actor": "missing_actor", "entries": [5]}],
        )
        fld_model.encode_glb(document, binary)


if __name__ == "__main__":
    unittest.main()
