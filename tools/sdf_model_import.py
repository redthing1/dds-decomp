#!/usr/bin/env python3
"""Apply glTF vertex-stream edits to DDS SDF model graphs."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import fld
import fld_model
from gltf_import import (
    GltfImportError as ModelImportError,
    decode_glb,
    f32 as _f32,
    records as _records,
)


@dataclass(frozen=True)
class ImportSummary:
    models: int
    meshes: int
    changed_meshes: int
    positions: int
    normals: int
    texcoords: int
    attributes: int
    colors: int
    materials: int
    changed_materials: int


@dataclass(frozen=True)
class ModelGraph:
    name: str
    items: tuple[fld.ModelItem, ...]
    materials: tuple[fld.ModelMaterial, ...]
    draw_roots: dict[int, tuple[int, ...]]
    draw_lists: dict[int, fld.ModelDrawList]
    draws: dict[int, fld.ModelDraw]


def _asset_metadata(document: dict) -> tuple[float, int | None]:
    asset = document.get("asset")
    extras = asset.get("extras") if isinstance(asset, dict) else None
    if not isinstance(extras, dict):
        raise ModelImportError("GLB has no DDS asset metadata")
    scale = extras.get("ddsMetersPerUnit")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ModelImportError("GLB has no positive finite DDS unit scale")
    if extras.get("ddsNativeAxesPreserved") is not True:
        raise ModelImportError("GLB does not preserve native DDS axes")
    texture_count = extras.get("ddsTextureCount")
    if texture_count is not None and (
        not isinstance(texture_count, int)
        or isinstance(texture_count, bool)
        or texture_count < 0
    ):
        raise ModelImportError("GLB has an invalid DDS texture count")
    return float(scale), texture_count


def import_model_graphs(
    source_data: bytes,
    document: dict,
    binary: bytes,
    graphs: tuple[ModelGraph, ...],
) -> tuple[bytes, ImportSummary]:
    """Apply changed material and vertex data without rebuilding SDF packets."""

    meters_per_unit, texture_count = _asset_metadata(document)
    meshes = document.get("meshes")
    if not isinstance(meshes, list):
        raise ModelImportError("GLB has no mesh array")
    if not graphs:
        raise ModelImportError("no SDF model graphs were selected")
    graph_names = tuple(graph.name for graph in graphs)
    if len(set(graph_names)) != len(graph_names):
        raise ModelImportError("selected SDF model graphs repeat a name")
    selected_prefixes = tuple(f"{name}/node_" for name in graph_names)

    meshes_by_name: dict[str, dict] = {}
    for mesh in meshes:
        name = mesh.get("name") if isinstance(mesh, dict) else None
        if isinstance(name, str) and name.startswith(selected_prefixes):
            if name in meshes_by_name:
                raise ModelImportError(f"GLB repeats mesh name {name!r}")
            meshes_by_name[name] = mesh

    output = bytearray(source_data)
    gltf_materials = document.get("materials")
    if not isinstance(gltf_materials, list):
        raise ModelImportError("GLB has no material array")
    material_specs = {
        name: (kind, width)
        for _, name, kind, width in fld.MODEL_MATERIAL_FIELDS
    }
    material_indices: dict[tuple[str, int], set[int]] = {}
    changed_materials: set[tuple[str, int]] = set()
    material_count = 0

    def material_values(
        value: object,
        kind: str,
        width: int,
        context: str,
    ) -> tuple[int | float, ...]:
        if not isinstance(value, list) or len(value) != width:
            raise ModelImportError(f"{context} has an incompatible value")
        if kind in {"rgba8", "resource", "resource_pair"}:
            limit = 0xFF if kind == "rgba8" else 0xFFFF
            if any(
                not isinstance(item, int) or not 0 <= item <= limit
                for item in value
            ):
                raise ModelImportError(f"{context} has an out-of-range integer")
            return tuple(value)
        if kind == "float":
            if any(
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(item)
                for item in value
            ):
                raise ModelImportError(f"{context} has a non-finite value")
            return tuple(_f32(float(item)) for item in value)
        raise AssertionError(kind)

    for graph in graphs:
        used = set()
        for draw in graph.draws.values():
            packet_meshes, _ = fld._read_model_mesh_packet(
                source_data,
                draw.packet,
                draw.quadwords * 0x10,
                f"model {graph.name} packet",
            )
            if packet_meshes:
                used.add(draw.material)
        represented: dict[int, tuple[tuple[str, tuple[int | float, ...]], ...]] = {}
        represented_indices: dict[int, set[int]] = {}
        for gltf_index, material in enumerate(gltf_materials):
            if not isinstance(material, dict):
                continue
            extras = material.get("extras")
            if (
                not isinstance(extras, dict)
                or extras.get("ddsModelGraph") != graph.name
            ):
                continue
            native_index = extras.get("ddsMaterialIndex")
            if not isinstance(native_index, int) or not 0 <= native_index < len(
                graph.materials
            ):
                raise ModelImportError(
                    f"model {graph.name} has a GLB material with an invalid DDS index"
                )
            if native_index not in used:
                raise ModelImportError(
                    f"model {graph.name} exposes unused material {native_index}"
                )
            source_material = graph.materials[native_index]
            if extras.get("ddsMaterialFlags") != source_material.flags:
                raise ModelImportError(
                    f"model {graph.name} material {native_index} "
                    "changes its field layout"
                )
            fields = extras.get("ddsMaterialFields")
            source_fields = dict(source_material.fields)
            if not isinstance(fields, dict) or set(fields) != set(source_fields):
                raise ModelImportError(
                    f"model {graph.name} material {native_index} changes its fields"
                )
            values = tuple(
                (
                    field_name,
                    material_values(
                        fields[field_name],
                        *material_specs[field_name],
                        f"model {graph.name} material {native_index} {field_name}",
                    ),
                )
                for field_name, _ in source_material.fields
            )
            old = represented.get(native_index)
            if old is not None and old != values:
                raise ModelImportError(
                    f"model {graph.name} material {native_index} variants disagree"
                )
            represented[native_index] = values
            represented_indices.setdefault(native_index, set()).add(gltf_index)

        missing = used - set(represented)
        if missing:
            raise ModelImportError(
                f"model {graph.name} is missing DDS materials "
                + ", ".join(str(index) for index in sorted(missing))
            )
        material_count += len(used)
        for native_index in used:
            source_material = graph.materials[native_index]
            old_fields = dict(source_material.fields)
            offsets = dict(source_material.field_offsets)
            changed = False
            for field_name, values in represented[native_index]:
                old_values = old_fields[field_name]
                if values == old_values:
                    continue
                if field_name in {"primary_texture", "secondary_texture"}:
                    old_index = old_values[0]
                    new_index = values[0]
                    if new_index != old_index:
                        if texture_count is None:
                            raise ModelImportError(
                                f"model {graph.name} material {native_index} "
                                "changes a texture reference without bundle metadata"
                            )
                        if new_index >= texture_count:
                            raise ModelImportError(
                                f"model {graph.name} material {native_index} "
                                f"references texture {new_index}, but the bundle has "
                                f"{texture_count} textures"
                            )
                kind, width = material_specs[field_name]
                offset = offsets[field_name]
                if kind == "rgba8":
                    struct.pack_into("<4B", output, offset, *values)
                elif kind == "float":
                    for index, (value, old_value) in enumerate(
                        zip(values, old_values, strict=True)
                    ):
                        if value != old_value:
                            struct.pack_into("<f", output, offset + index * 4, value)
                elif kind == "resource":
                    struct.pack_into("<H", output, offset, values[0])
                elif kind == "resource_pair":
                    struct.pack_into("<2H", output, offset, *values)
                else:
                    raise AssertionError(kind)
                changed = True
            key = graph.name, native_index
            material_indices[key] = represented_indices[native_index]
            if changed:
                changed_materials.add(key)

    stream_values: dict[tuple[int, str], tuple[tuple[int | float, ...], ...]] = {}
    mesh_keys: set[tuple[int, int, int]] = set()
    changed_meshes: set[tuple[int, int, int]] = set()
    changes = {
        name: 0
        for name in ("positions", "normals", "texcoords", "attributes", "colors")
    }

    def patch_floats(
        key: tuple[int, str],
        offset: int,
        actual: tuple[tuple[int | float, ...], ...],
        expected: tuple[tuple[float, ...], ...],
        scale: float,
        context: str,
    ) -> bool:
        previous = stream_values.get(key)
        if previous is not None:
            if previous != actual:
                raise ModelImportError(f"{context} conflicts with another shared draw")
            return False
        stream_values[key] = actual
        if len(actual) != len(expected):
            raise ModelImportError(
                f"{context} has {len(actual)} records, expected {len(expected)}"
            )
        changed = False
        width = len(expected[0]) if expected else 0
        for row, (new_record, old_record) in enumerate(
            zip(actual, expected, strict=True)
        ):
            if len(new_record) != width or not all(
                isinstance(value, float) and math.isfinite(value)
                for value in new_record
            ):
                raise ModelImportError(f"{context} has non-finite or malformed values")
            for column, (value, old) in enumerate(
                zip(new_record, old_record, strict=True)
            ):
                exported = _f32(old * scale)
                if value == exported:
                    continue
                native = _f32(value / scale)
                struct.pack_into(
                    "<f", output, offset + (row * width + column) * 4, native
                )
                changed = True
        return changed

    def patch_colors(
        key: tuple[int, str],
        offset: int,
        actual: tuple[tuple[int | float, ...], ...],
        expected: tuple[tuple[int, int, int, int], ...],
        context: str,
    ) -> bool:
        previous = stream_values.get(key)
        if previous is not None:
            if previous != actual:
                raise ModelImportError(f"{context} conflicts with another shared draw")
            return False
        stream_values[key] = actual
        if len(actual) != len(expected):
            raise ModelImportError(
                f"{context} has {len(actual)} records, expected {len(expected)}"
            )
        changed = False
        for row, (new_record, old_record) in enumerate(
            zip(actual, expected, strict=True)
        ):
            if len(new_record) != 4 or any(
                not isinstance(value, int) or not 0 <= value <= 0xFF
                for value in new_record
            ):
                raise ModelImportError(f"{context} has malformed color values")
            for column, (value, old) in enumerate(
                zip(new_record, old_record, strict=True)
            ):
                if value == min(old * 2, 0xFF):
                    continue
                output[offset + row * 4 + column] = min((value + 1) // 2, 0x80)
                changed = True
        return changed

    consumed_mesh_names: set[str] = set()
    for graph in graphs:
        source_name = graph.name
        command_meshes: dict[int, str] = {}
        for item in graph.items:
            if not item.commands or item.commands in command_meshes:
                continue
            mesh_name = f"{source_name}/node_{item.node_id}"
            command_meshes[item.commands] = mesh_name
            try:
                gltf_mesh = meshes_by_name[mesh_name]
            except KeyError as exc:
                raise ModelImportError(f"GLB is missing mesh {mesh_name!r}") from exc
            consumed_mesh_names.add(mesh_name)
            primitives = gltf_mesh.get("primitives")
            if not isinstance(primitives, list):
                raise ModelImportError(f"mesh {mesh_name!r} has no primitives")
            expected_primitives = []
            for list_index, list_offset in enumerate(
                graph.draw_roots[item.commands]
            ):
                draw_list = graph.draw_lists[list_offset]
                for draw_index, draw_offset in enumerate(draw_list.draws):
                    draw = graph.draws[draw_offset]
                    packet_size = draw.quadwords * 0x10
                    packet_meshes, _ = fld._read_model_mesh_packet(
                        source_data,
                        draw.packet,
                        packet_size,
                        f"model {source_name} packet",
                    )
                    for packet_mesh_index, source_mesh in enumerate(packet_meshes):
                        expected_primitives.append(
                            (
                                list_index,
                                draw_index,
                                draw,
                                draw_list,
                                packet_size,
                                packet_mesh_index,
                                source_mesh,
                            )
                        )
            if len(primitives) != len(expected_primitives):
                raise ModelImportError(
                    f"mesh {mesh_name!r} has {len(primitives)} primitives, "
                    f"expected {len(expected_primitives)}"
                )
            for primitive, expected in zip(
                primitives, expected_primitives, strict=True
            ):
                if not isinstance(primitive, dict) or primitive.get("mode", 4) != 4:
                    raise ModelImportError(
                        f"mesh {mesh_name!r} has a non-triangle primitive"
                    )
                (
                    list_index,
                    draw_index,
                    draw,
                    draw_list,
                    packet_size,
                    packet_mesh_index,
                    source_mesh,
                ) = expected
                context = f"mesh {mesh_name!r} primitive {len(mesh_keys)}"
                if primitive.get("material") not in material_indices[
                    graph.name, draw.material
                ]:
                    raise ModelImportError(
                        f"{context} changes its DDS material assignment"
                    )
                extras = primitive.get("extras")
                metadata = {
                    "ddsMaterial": draw.material,
                    "ddsDrawSelector": draw_list.selector,
                    "ddsDrawListIndex": list_index,
                    "ddsDrawIndex": draw_index,
                    "ddsPacketMeshIndex": packet_mesh_index,
                    "ddsMeshControls": list(source_mesh.controls),
                    "ddsProgramAddress": source_mesh.program,
                }
                if not isinstance(extras, dict) or any(
                    extras.get(name) != value for name, value in metadata.items()
                ):
                    raise ModelImportError(f"{context} DDS identity metadata differs")
                indices = _records(
                    document,
                    binary,
                    primitive.get("indices"),
                    "SCALAR",
                    {
                        fld_model.UNSIGNED_BYTE,
                        fld_model.UNSIGNED_SHORT,
                        fld_model.UNSIGNED_INT,
                    },
                    context + " indices",
                    normalized=False,
                )
                expected_indices = tuple(
                    (value,)
                    for triangle in source_mesh.triangles
                    for value in triangle[:3]
                )
                if indices != expected_indices:
                    raise ModelImportError(f"{context} changes triangle topology")
                controls = _records(
                    document,
                    binary,
                    extras.get("ddsTriangleControlAccessor"),
                    "SCALAR",
                    {fld_model.UNSIGNED_BYTE},
                    context + " triangle controls",
                    normalized=False,
                )
                if controls != tuple(
                    (triangle[3],) for triangle in source_mesh.triangles
                ):
                    raise ModelImportError(f"{context} changes triangle controls")

                source_attributes = {
                    "POSITION": (
                        source_mesh.positions,
                        source_mesh.positions_offset,
                        3,
                    ),
                }
                if source_mesh.normals is not None:
                    source_attributes["NORMAL"] = (
                        source_mesh.normals,
                        source_mesh.normals_offset,
                        3,
                    )
                if source_mesh.texcoords is not None:
                    source_attributes["TEXCOORD_0"] = (
                        source_mesh.texcoords,
                        source_mesh.texcoords_offset,
                        2,
                    )
                if source_mesh.attributes is not None:
                    source_attributes["_DDS_ATTRIBUTE"] = (
                        source_mesh.attributes,
                        source_mesh.attributes_offset,
                        4,
                    )
                if source_mesh.colors is not None:
                    source_attributes["COLOR_0"] = (
                        source_mesh.colors,
                        source_mesh.colors_offset,
                        4,
                    )
                attributes = primitive.get("attributes")
                if not isinstance(attributes, dict) or set(attributes) != set(
                    source_attributes
                ):
                    raise ModelImportError(
                        f"{context} changes the vertex-channel layout"
                    )

                mesh_key = draw.packet, packet_size, packet_mesh_index
                mesh_keys.add(mesh_key)
                mesh_changed = False
                for attribute_name, (
                    old_records,
                    native_offset,
                    width,
                ) in source_attributes.items():
                    if native_offset is None:
                        raise AssertionError(attribute_name)
                    value_type = f"VEC{width}"
                    stream_key = native_offset, attribute_name
                    if attribute_name == "COLOR_0":
                        actual = _records(
                            document,
                            binary,
                            attributes[attribute_name],
                            value_type,
                            {fld_model.UNSIGNED_BYTE},
                            context + " colors",
                            normalized=True,
                        )
                        changed = patch_colors(
                            stream_key,
                            native_offset,
                            actual,
                            old_records,
                            context + " colors",
                        )
                        label = "colors"
                    else:
                        actual = _records(
                            document,
                            binary,
                            attributes[attribute_name],
                            value_type,
                            {fld_model.FLOAT},
                            context + f" {attribute_name}",
                            normalized=False,
                        )
                        label = {
                            "POSITION": "positions",
                            "NORMAL": "normals",
                            "TEXCOORD_0": "texcoords",
                            "_DDS_ATTRIBUTE": "attributes",
                        }[attribute_name]
                        changed = patch_floats(
                            stream_key,
                            native_offset,
                            actual,
                            old_records,
                            meters_per_unit if attribute_name == "POSITION" else 1.0,
                            context + f" {label}",
                        )
                    if changed:
                        changes[label] += 1
                        mesh_changed = True
                if mesh_changed:
                    changed_meshes.add(mesh_key)
    extra_meshes = set(meshes_by_name) - consumed_mesh_names
    if extra_meshes:
        raise ModelImportError("GLB contains an unrecognized selected-model mesh")
    rebuilt = bytes(output)
    return rebuilt, ImportSummary(
        len(graphs),
        len(mesh_keys),
        len(changed_meshes),
        changes["positions"],
        changes["normals"],
        changes["texcoords"],
        changes["attributes"],
        changes["colors"],
        material_count,
        len(changed_materials),
    )
