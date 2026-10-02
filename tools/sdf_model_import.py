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
    nodes: int = 0
    changed_nodes: int = 0
    translations: int = 0
    rotations: int = 0
    scales: int = 0
    parents: int = 0
    bounds: int = 0
    animations: int = 0
    tracks: int = 0
    changed_tracks: int = 0


@dataclass(frozen=True)
class ModelGraph:
    name: str
    items: tuple[fld.ModelItem, ...]
    materials: tuple[fld.ModelMaterial, ...]
    draw_roots: dict[int, tuple[int, ...]]
    draw_lists: dict[int, fld.ModelDrawList]
    draws: dict[int, fld.ModelDraw]
    root_transform: int = 0
    motion: fld.ModelMotionPlaybook | None = None


def _asset_metadata(document: dict) -> tuple[float, float | None, int | None]:
    asset = document.get("asset")
    extras = asset.get("extras") if isinstance(asset, dict) else None
    if not isinstance(extras, dict):
        raise ModelImportError("GLB has no DDS asset metadata")
    scale = extras.get("ddsMetersPerUnit")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ModelImportError("GLB has no positive finite DDS unit scale")
    if extras.get("ddsNativeAxesPreserved") is not True:
        raise ModelImportError("GLB does not preserve native DDS axes")
    frames_per_second = extras.get("ddsFramesPerSecond")
    if frames_per_second is not None and (
        not isinstance(frames_per_second, (int, float))
        or isinstance(frames_per_second, bool)
        or not math.isfinite(frames_per_second)
        or frames_per_second <= 0
    ):
        raise ModelImportError("GLB has an invalid DDS frame rate")
    texture_count = extras.get("ddsTextureCount")
    if texture_count is not None and (
        not isinstance(texture_count, int)
        or isinstance(texture_count, bool)
        or texture_count < 0
    ):
        raise ModelImportError("GLB has an invalid DDS texture count")
    return (
        float(scale),
        float(frames_per_second) if frames_per_second is not None else None,
        texture_count,
    )


def _node_vector(
    node: dict,
    key: str,
    width: int,
    default: tuple[float, ...],
    context: str,
) -> tuple[float, ...]:
    value = node.get(key, list(default))
    if (
        not isinstance(value, list)
        or len(value) != width
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(item)
            for item in value
        )
    ):
        raise ModelImportError(f"{context} has an invalid {key}")
    return tuple(float(item) for item in value)


def _write_floats(
    output: bytearray,
    offset: int,
    values: tuple[float, ...],
    context: str,
) -> None:
    try:
        struct.pack_into(f"<{len(values)}f", output, offset, *values)
    except (OverflowError, struct.error) as exc:
        raise ModelImportError(
            f"{context} cannot be represented as native float32 values"
        ) from exc


def _quaternion_euler(rotation: tuple[float, ...], context: str) -> tuple[float, ...]:
    """Invert the DDS Rx(-x) * Ry(-y) * Rz(-z) Euler conversion."""

    length = math.sqrt(sum(value * value for value in rotation))
    if not math.isfinite(length) or length == 0.0:
        raise ModelImportError(f"{context} is not a finite quaternion")
    x, y, z, w = (value / length for value in rotation)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = max(-1.0, min(1.0, 2.0 * (x * z + y * w)))
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    converted_y = math.asin(r02)
    if abs(math.cos(converted_y)) > 1e-7:
        converted_x = math.atan2(-r12, r22)
        converted_z = math.atan2(-r01, r00)
    else:
        converted_x = math.copysign(math.atan2(r10, r11), converted_y)
        converted_z = 0.0
    return -converted_x, -converted_y, -converted_z


def _same_rotation(actual: tuple[float, ...], expected: list[float]) -> bool:
    return actual == tuple(expected) or actual == tuple(-value for value in expected)


def _represented_motion_tracks(
    motion: fld.ModelMotionPlaybook,
    clip: fld.ModelMotionClip,
) -> tuple[
    tuple[tuple[int, fld.ModelMotionBinding, fld.ModelMotionTrack], ...],
    tuple[int, ...],
]:
    represented = []
    skipped = []
    for binding_index, (binding, track) in enumerate(
        zip(motion.bindings, clip.tracks, strict=True)
    ):
        if binding.family != 0 or binding.selector == 4:
            continue
        if any(
            left >= right for left, right in zip(track.frames, track.frames[1:])
        ):
            skipped.append(binding_index)
            continue
        represented.append((binding_index, binding, track))
    return tuple(represented), tuple(skipped)


def _patch_graph_nodes(
    source_data: bytes,
    output: bytearray,
    document: dict,
    graph: ModelGraph,
    meters_per_unit: float,
) -> tuple[dict[int, dict], dict[str, int]]:
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise ModelImportError("GLB has no node array")
    wrapper_indices = [
        index
        for index, node in enumerate(nodes)
        if isinstance(node, dict) and node.get("name") == graph.name
    ]
    if len(wrapper_indices) != 1:
        raise ModelImportError(
            f"model {graph.name} has {len(wrapper_indices)} GLB wrappers, expected 1"
        )
    wrapper_index = wrapper_indices[0]
    wrapper = nodes[wrapper_index]

    item_nodes: dict[int, dict] = {}
    item_indices: dict[int, int] = {}
    for item in graph.items:
        name = f"{graph.name}/node_{item.node_id}"
        matches = [
            (index, node)
            for index, node in enumerate(nodes)
            if isinstance(node, dict) and node.get("name") == name
        ]
        if len(matches) != 1:
            raise ModelImportError(
                f"model {graph.name} node {item.node_id} has {len(matches)} "
                "GLB representations, expected 1"
            )
        node_index, node = matches[0]
        extras = node.get("extras")
        if not isinstance(extras, dict) or extras.get("ddsNodeId") != item.node_id:
            raise ModelImportError(f"model node {name!r} changes its DDS identity")
        item_nodes[item.node_id] = node
        item_indices[item.node_id] = node_index

    by_gltf_index = {index: node_id for node_id, index in item_indices.items()}
    parents: dict[int, list[int]] = {node_id: [] for node_id in item_nodes}
    for parent_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        children = node.get("children", [])
        if not isinstance(children, (list, tuple)) or any(
            not isinstance(child, int) or isinstance(child, bool) for child in children
        ):
            raise ModelImportError(
                f"node {node.get('name', '<unnamed>')!r} has invalid children"
            )
        for child in children:
            child_id = by_gltf_index.get(child)
            if child_id is not None:
                parents[child_id].append(parent_index)

    native_parents: dict[int, int] = {}
    for node_id, parent_indices in parents.items():
        if len(parent_indices) != 1:
            raise ModelImportError(
                f"model {graph.name} node {node_id} has {len(parent_indices)} parents"
            )
        parent_index = parent_indices[0]
        if parent_index == wrapper_index:
            native_parents[node_id] = -1
        elif parent_index in by_gltf_index:
            native_parents[node_id] = by_gltf_index[parent_index]
        else:
            raise ModelImportError(
                f"model {graph.name} node {node_id} has a parent outside its model"
            )
    for node_id in native_parents:
        visited = set()
        cursor = node_id
        while cursor >= 0:
            if cursor in visited:
                raise ModelImportError(f"model {graph.name} hierarchy contains a cycle")
            visited.add(cursor)
            cursor = native_parents[cursor]

    counts = {
        "nodes": len(graph.items) + int(bool(graph.root_transform)),
        "changed_nodes": 0,
        "translations": 0,
        "rotations": 0,
        "scales": 0,
        "parents": 0,
    }

    def patch_trs(
        node: dict,
        position_offset: int,
        rotation_offset: int,
        scale_offset: int,
        old_position: tuple[float, ...],
        old_rotation: tuple[float, ...],
        old_scale: tuple[float, ...],
        context: str,
        *,
        euler_rotation: bool,
    ) -> bool:
        changed = False
        translation = (
            None
            if "translation" not in node
            and not all(math.isfinite(value) for value in old_position[:3])
            else _node_vector(node, "translation", 3, (0.0, 0.0, 0.0), context)
        )
        expected_translation = tuple(
            value * meters_per_unit for value in old_position[:3]
        )
        if translation is not None and translation != expected_translation:
            native = tuple(_f32(value / meters_per_unit) for value in translation)
            _write_floats(output, position_offset, native, context + " translation")
            counts["translations"] += 1
            changed = True

        expected_rotation = (
            fld_model._euler_quaternion(*old_rotation)
            if euler_rotation
            else fld_model._normalized_quaternion(old_rotation)
        )
        rotation = (
            None
            if "rotation" not in node
            and not all(math.isfinite(value) for value in old_rotation)
            else _node_vector(node, "rotation", 4, (0.0, 0.0, 0.0, 1.0), context)
        )
        if rotation is not None and not _same_rotation(rotation, expected_rotation):
            length = math.sqrt(sum(value * value for value in rotation))
            if not math.isfinite(length) or length == 0.0:
                raise ModelImportError(f"{context} rotation is not a finite quaternion")
            normalized = tuple(value / length for value in rotation)
            native_rotation = (
                _quaternion_euler(normalized, context + " rotation")
                if euler_rotation
                else tuple(_f32(value) for value in normalized)
            )
            _write_floats(
                output, rotation_offset, native_rotation, context + " rotation"
            )
            counts["rotations"] += 1
            changed = True

        scale = (
            None
            if "scale" not in node
            and not all(math.isfinite(value) for value in old_scale[:3])
            else _node_vector(node, "scale", 3, (1.0, 1.0, 1.0), context)
        )
        if scale is not None and scale != old_scale[:3]:
            native_scale = tuple(_f32(value) for value in scale)
            _write_floats(output, scale_offset, native_scale, context + " scale")
            counts["scales"] += 1
            changed = True
        return changed

    for item in graph.items:
        context = f"model {graph.name} node {item.node_id}"
        changed = patch_trs(
            item_nodes[item.node_id],
            item.offset + 0x20,
            item.offset + 0x10,
            item.offset + 0x30,
            item.position,
            item.rotation,
            item.scale,
            context,
            euler_rotation=True,
        )
        parent = native_parents[item.node_id]
        if parent != item.parent:
            struct.pack_into("<i", output, item.offset + 0x0C, parent)
            counts["parents"] += 1
            changed = True
        counts["changed_nodes"] += changed

    if graph.root_transform:
        native = struct.unpack_from("<12f", source_data, graph.root_transform)
        if patch_trs(
            wrapper,
            graph.root_transform,
            graph.root_transform + 0x10,
            graph.root_transform + 0x20,
            native[:4],
            native[4:8],
            native[8:12],
            f"model {graph.name} resource",
            euler_rotation=False,
        ):
            counts["changed_nodes"] += 1

    return item_nodes, counts


def _patch_graph_animations(
    source_data: bytes,
    output: bytearray,
    document: dict,
    binary: bytes,
    graph: ModelGraph,
    item_nodes: dict[int, dict],
    meters_per_unit: float,
    frames_per_second: float | None,
) -> tuple[int, int, int]:
    motion = graph.motion
    if motion is None:
        return 0, 0, 0
    if frames_per_second is None:
        raise ModelImportError("GLB has no positive finite DDS frame rate")
    animations = document.get("animations", [])
    if not isinstance(animations, list):
        raise ModelImportError("GLB has an invalid animation array")
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise ModelImportError("GLB has no node array")
    node_indices = {
        node_id: next(
            index for index, candidate in enumerate(nodes) if candidate is node
        )
        for node_id, node in item_nodes.items()
    }

    imported_animations = imported_tracks = changed_tracks = 0
    for clip_index, clip in enumerate(motion.clips):
        if clip is None:
            continue
        expected, skipped = _represented_motion_tracks(motion, clip)
        if not expected:
            continue

        name = f"{graph.name}/clip_{clip_index}"
        matches = [
            animation
            for animation in animations
            if isinstance(animation, dict) and animation.get("name") == name
        ]
        if len(matches) != 1:
            raise ModelImportError(
                f"model {graph.name} clip {clip_index} has {len(matches)} "
                "GLB animations, expected 1"
            )
        animation = matches[0]
        extras = animation.get("extras")
        metadata = {
            "ddsDurationFrames": clip.duration,
            "ddsReserved": clip.reserved,
            "ddsIgnoredMaterialBindings": sum(
                binding.family == 1 for binding in motion.bindings
            ),
        }
        if skipped:
            metadata["ddsSkippedUnorderedBindings"] = skipped
        if not isinstance(extras, dict) or any(
            extras.get(key) != value for key, value in metadata.items()
        ):
            raise ModelImportError(f"animation {name!r} DDS metadata differs")
        samplers = animation.get("samplers")
        channels = animation.get("channels")
        if (
            not isinstance(samplers, list)
            or not isinstance(channels, list)
            or len(samplers) != len(expected)
            or len(channels) != len(expected)
        ):
            raise ModelImportError(f"animation {name!r} changes its channel layout")

        for channel_index, (binding_index, binding, track) in enumerate(expected):
            context = f"animation {name!r} binding {binding_index}"
            channel = channels[channel_index]
            if not isinstance(channel, dict):
                raise ModelImportError(f"{context} has an invalid channel")
            sampler_index = channel.get("sampler")
            if sampler_index != channel_index:
                raise ModelImportError(f"{context} changes its sampler identity")
            sampler = samplers[channel_index]
            if (
                not isinstance(sampler, dict)
                or sampler.get("interpolation", "LINEAR") != "LINEAR"
            ):
                raise ModelImportError(f"{context} changes its interpolation")
            if binding.selector in {1, 3}:
                path = "rotation"
            elif binding.selector == 0:
                path = "translation"
            else:
                path = "scale"
            target = channel.get("target")
            if not isinstance(target, dict) or target != {
                "node": node_indices[binding.target],
                "path": path,
            }:
                raise ModelImportError(f"{context} changes its target")
            times = _records(
                document,
                binary,
                sampler.get("input"),
                "SCALAR",
                {fld_model.FLOAT},
                context + " times",
                normalized=False,
            )
            expected_times = tuple(
                (_f32(float(frame) / frames_per_second),)
                for frame in track.frames
            )
            if times != expected_times:
                raise ModelImportError(f"{context} changes its frame keys")

            width = 4 if path == "rotation" else 3
            values = _records(
                document,
                binary,
                sampler.get("output"),
                f"VEC{width}",
                {fld_model.FLOAT},
                context + " values",
                normalized=False,
            )
            source_values = fld_model._track_values(source_data, track)
            if len(values) != len(source_values):
                raise ModelImportError(f"{context} changes its key count")
            track_changed = False
            for key_index, (actual, native) in enumerate(
                zip(values, source_values, strict=True)
            ):
                if not all(
                    isinstance(value, float) and math.isfinite(value)
                    for value in actual
                ):
                    raise ModelImportError(f"{context} has non-finite values")
                value_offset = track.values + key_index * track.stride
                if binding.selector == 0:
                    exported = tuple(
                        _f32(float(value) * meters_per_unit) for value in native
                    )
                    edited = tuple(
                        old_native if value == old else value / meters_per_unit
                        for value, old, old_native in zip(
                            actual, exported, native, strict=True
                        )
                    )
                    if actual != exported:
                        _write_floats(output, value_offset, edited, context)
                        track_changed = True
                elif binding.selector == 2:
                    exported = tuple(_f32(float(value)) for value in native)
                    edited = tuple(
                        old_native if value == old else value
                        for value, old, old_native in zip(
                            actual, exported, native, strict=True
                        )
                    )
                    if actual != exported:
                        _write_floats(output, value_offset, edited, context)
                        track_changed = True
                else:
                    exported = (
                        fld_model._euler_quaternion(*native)
                        if binding.selector == 1
                        else fld_model._normalized_quaternion(
                            tuple(float(value) / 4096.0 for value in native)
                        )
                    )
                    expected_rotation = [_f32(value) for value in exported]
                    if _same_rotation(actual, expected_rotation):
                        continue
                    length = math.sqrt(sum(value * value for value in actual))
                    if not math.isfinite(length) or length == 0.0:
                        raise ModelImportError(
                            f"{context} key {key_index} is not a finite quaternion"
                        )
                    normalized = tuple(value / length for value in actual)
                    if binding.selector == 1:
                        _write_floats(
                            output,
                            value_offset,
                            _quaternion_euler(normalized, context),
                            context,
                        )
                    else:
                        packed = tuple(round(value * 4096.0) for value in normalized)
                        if any(not -32768 <= value <= 32767 for value in packed):
                            raise ModelImportError(
                                f"{context} key {key_index} exceeds packed "
                                "quaternion range"
                            )
                        struct.pack_into("<4h", output, value_offset, *packed)
                    track_changed = True
            imported_tracks += 1
            changed_tracks += track_changed
        imported_animations += 1
    return imported_animations, imported_tracks, changed_tracks


def import_model_graphs(
    source_data: bytes,
    document: dict,
    binary: bytes,
    graphs: tuple[ModelGraph, ...],
) -> tuple[bytes, ImportSummary]:
    """Apply changed model nodes, animation, materials, and vertices in place."""

    meters_per_unit, frames_per_second, texture_count = _asset_metadata(document)
    meshes = document.get("meshes")
    if not isinstance(meshes, list):
        raise ModelImportError("GLB has no mesh array")
    if not graphs:
        raise ModelImportError("no SDF model graphs were selected")
    graph_names = tuple(graph.name for graph in graphs)
    if len(set(graph_names)) != len(graph_names):
        raise ModelImportError("selected SDF model graphs repeat a name")
    selected_prefixes = tuple(f"{name}/node_" for name in graph_names)

    animations = document.get("animations", [])
    if not isinstance(animations, list) or any(
        not isinstance(animation, dict)
        or not isinstance(animation.get("name"), str)
        for animation in animations
    ):
        raise ModelImportError("GLB has an invalid animation array")
    animation_names = tuple(animation["name"] for animation in animations)
    expected_animation_names = tuple(
        f"{graph.name}/clip_{clip_index}"
        for graph in graphs
        if graph.motion is not None
        for clip_index, clip in enumerate(graph.motion.clips)
        if clip is not None
        and _represented_motion_tracks(graph.motion, clip)[0]
    )
    if (
        len(set(animation_names)) != len(animation_names)
        or set(animation_names) != set(expected_animation_names)
    ):
        raise ModelImportError("GLB changes the selected models' animation set")

    graph_nodes: dict[str, dict[int, dict]] = {}
    node_changes = {
        name: 0
        for name in (
            "nodes",
            "changed_nodes",
            "translations",
            "rotations",
            "scales",
            "parents",
        )
    }
    output = bytearray(source_data)
    animation_changes = {"animations": 0, "tracks": 0, "changed_tracks": 0}
    for graph in graphs:
        item_nodes, counts = _patch_graph_nodes(
            source_data, output, document, graph, meters_per_unit
        )
        graph_nodes[graph.name] = item_nodes
        for name, count in counts.items():
            node_changes[name] += count
        animations, tracks, changed_tracks = _patch_graph_animations(
            source_data,
            output,
            document,
            binary,
            graph,
            item_nodes,
            meters_per_unit,
            frames_per_second,
        )
        animation_changes["animations"] += animations
        animation_changes["tracks"] += tracks
        animation_changes["changed_tracks"] += changed_tracks

    meshes_by_name: dict[str, dict] = {}
    mesh_indices_by_name: dict[str, int] = {}
    for mesh_index, mesh in enumerate(meshes):
        name = mesh.get("name") if isinstance(mesh, dict) else None
        if isinstance(name, str) and name.startswith(selected_prefixes):
            if name in meshes_by_name:
                raise ModelImportError(f"GLB repeats mesh name {name!r}")
            meshes_by_name[name] = mesh
            mesh_indices_by_name[name] = mesh_index

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
    changed_position_streams: set[int] = set()
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
            node_mesh = graph_nodes[graph.name][item.node_id].get("mesh")
            if node_mesh != mesh_indices_by_name[mesh_name]:
                raise ModelImportError(
                    f"model node {mesh_name!r} changes its mesh assignment"
                )
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
                        if attribute_name == "POSITION":
                            changed_position_streams.add(native_offset)
                if mesh_changed:
                    changed_meshes.add(mesh_key)
    extra_meshes = set(meshes_by_name) - consumed_mesh_names
    if extra_meshes:
        raise ModelImportError("GLB contains an unrecognized selected-model mesh")

    changed_bounds: set[int] = set()
    for graph in graphs:
        for item in graph.items:
            if not item.bounds or not item.commands:
                continue
            edited_positions = []
            for list_offset in graph.draw_roots[item.commands]:
                draw_list = graph.draw_lists[list_offset]
                for draw_offset in draw_list.draws:
                    draw = graph.draws[draw_offset]
                    packet_meshes, _ = fld._read_model_mesh_packet(
                        output,
                        draw.packet,
                        draw.quadwords * 0x10,
                        f"model {graph.name} edited packet",
                    )
                    edited_positions.extend(
                        position
                        for mesh in packet_meshes
                        if mesh.positions_offset in changed_position_streams
                        for position in mesh.positions
                    )
            if not edited_positions:
                continue
            bounds = struct.unpack_from("<6f", output, item.bounds)
            if not all(math.isfinite(value) for value in bounds):
                raise ModelImportError(
                    f"model {graph.name} node {item.node_id} has non-finite bounds"
                )
            expanded = tuple(
                min(
                    bounds[axis],
                    *(position[axis] for position in edited_positions),
                )
                for axis in range(3)
            ) + tuple(
                max(
                    bounds[axis + 3],
                    *(position[axis] for position in edited_positions),
                )
                for axis in range(3)
            )
            if expanded != bounds:
                _write_floats(
                    output,
                    item.bounds,
                    expanded,
                    f"model {graph.name} node {item.node_id} bounds",
                )
                changed_bounds.add(item.bounds)

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
        node_changes["nodes"],
        node_changes["changed_nodes"],
        node_changes["translations"],
        node_changes["rotations"],
        node_changes["scales"],
        node_changes["parents"],
        len(changed_bounds),
        animation_changes["animations"],
        animation_changes["tracks"],
        animation_changes["changed_tracks"],
    )
