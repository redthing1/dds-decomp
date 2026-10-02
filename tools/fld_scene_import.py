#!/usr/bin/env python3
"""Import edited static FLD2 resource transforms from a composed GLB."""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import fld
import fld_model


class FieldSceneImportError(ValueError):
    """Raised when a GLB cannot be mapped safely to one FLD2 source."""


SUPPORTED_TYPES = frozenset({3, 4, 10})
GLB_HEADER = struct.Struct("<4sII")
GLB_CHUNK = struct.Struct("<II")


@dataclass(frozen=True)
class ImportSummary:
    resources: int
    changed_resources: int
    translations: int
    rotations: int
    scales: int


def decode_glb(data: bytes) -> dict:
    """Read and validate the JSON document from a glTF 2.0 binary."""

    if len(data) < GLB_HEADER.size + GLB_CHUNK.size:
        raise FieldSceneImportError("GLB is truncated")
    magic, version, total_size = GLB_HEADER.unpack_from(data)
    if magic != b"glTF" or version != 2:
        raise FieldSceneImportError("input is not a glTF 2.0 binary")
    if total_size != len(data):
        raise FieldSceneImportError(
            f"GLB header size is {total_size}, but file size is {len(data)}"
        )
    json_size, chunk_type = GLB_CHUNK.unpack_from(data, GLB_HEADER.size)
    if chunk_type != fld_model.GLB_JSON_CHUNK:
        raise FieldSceneImportError("GLB first chunk is not JSON")
    json_start = GLB_HEADER.size + GLB_CHUNK.size
    json_end = json_start + json_size
    if json_end > len(data):
        raise FieldSceneImportError("GLB JSON chunk is truncated")
    try:
        document = json.loads(data[json_start:json_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FieldSceneImportError(f"invalid GLB JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise FieldSceneImportError("GLB JSON root is not an object")
    return document


def _resource_name(data: bytes, resource: fld.Resource) -> str:
    if resource.name:
        return fld._fixed_string(data, resource.name, "field resource name")
    return f"type_{resource.type_id}_resource_{resource.serial:04d}"


def _vec(node: dict, key: str, default: tuple[float, ...]) -> list[float]:
    value = node.get(key, default)
    if (
        not isinstance(value, list)
        or len(value) != len(default)
        or any(not isinstance(item, (int, float)) for item in value)
    ):
        raise FieldSceneImportError(
            f"node {node.get('name', '<unnamed>')!r} has an invalid {key}"
        )
    converted = [float(item) for item in value]
    if not all(math.isfinite(item) for item in converted):
        raise FieldSceneImportError(
            f"node {node.get('name', '<unnamed>')!r} has a non-finite {key}"
        )
    return converted


def _same(actual: list[float], expected: list[float]) -> bool:
    return len(actual) == len(expected) and all(
        left == right for left, right in zip(actual, expected, strict=True)
    )


def _pack_component(
    output: bytearray,
    offset: int,
    values: list[float],
    context: str,
) -> None:
    try:
        struct.pack_into(f"<{len(values)}f", output, offset, *values)
    except (OverflowError, struct.error) as exc:
        raise FieldSceneImportError(
            f"{context} cannot be represented as FLD2 float32 values"
        ) from exc


def import_transforms(
    field_data: bytes,
    document: dict,
) -> tuple[bytes, ImportSummary]:
    """Apply changed FLD2 collision, camera, and placement TRS components."""

    fld.validate(field_data)
    if field_data[4:8] != b"FLD2":
        raise FieldSceneImportError("transform import requires an FLD2 file")
    asset = document.get("asset")
    extras = asset.get("extras") if isinstance(asset, dict) else None
    if not isinstance(extras, dict):
        raise FieldSceneImportError("GLB has no DDS asset metadata")
    meters_per_unit = extras.get("ddsMetersPerUnit")
    if (
        not isinstance(meters_per_unit, (int, float))
        or not math.isfinite(meters_per_unit)
        or meters_per_unit <= 0.0
    ):
        raise FieldSceneImportError("GLB has no positive finite DDS unit scale")
    if extras.get("ddsNativeAxesPreserved") is not True:
        raise FieldSceneImportError("GLB does not preserve native DDS axes")

    words, data_end, _ = fld._read_header(field_data)
    resources = fld._read_resources(
        field_data, fld._read_types(field_data, words, data_end)
    )
    by_key: dict[tuple[int, int], fld.Resource] = {}
    for resource in resources:
        if resource.type_id not in SUPPORTED_TYPES:
            continue
        key = resource.type_id, resource.serial
        if key in by_key:
            raise FieldSceneImportError(
                f"FLD2 has duplicate type {key[0]} serial {key[1]} resources"
            )
        if not resource.transform:
            raise FieldSceneImportError(
                f"FLD2 type {key[0]} serial {key[1]} has no transform"
            )
        by_key[key] = resource

    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise FieldSceneImportError("GLB has no node array")
    output = bytearray(field_data)
    seen: set[tuple[int, int]] = set()
    changed_resources: set[tuple[int, int]] = set()
    translations = rotations = scales = 0
    imported = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_extras = node.get("extras")
        if not isinstance(node_extras, dict):
            continue
        type_id = node_extras.get("ddsResourceType")
        serial = node_extras.get("ddsResourceSerial")
        if type_id not in SUPPORTED_TYPES or not isinstance(serial, int):
            continue
        key = int(type_id), serial
        if key in seen:
            raise FieldSceneImportError(
                f"GLB repeats type {key[0]} serial {key[1]}"
            )
        seen.add(key)
        try:
            resource = by_key[key]
        except KeyError as exc:
            raise FieldSceneImportError(
                f"GLB type {key[0]} serial {key[1]} is absent from the FLD2"
            ) from exc
        expected_name = _resource_name(field_data, resource)
        if node.get("name") != expected_name:
            raise FieldSceneImportError(
                f"GLB type {key[0]} serial {key[1]} is named "
                f"{node.get('name')!r}, expected {expected_name!r}"
            )
        if "matrix" in node:
            raise FieldSceneImportError(
                f"node {expected_name!r} uses a matrix; preserve editable TRS fields"
            )

        native = list(struct.unpack_from("<12f", field_data, resource.transform))
        translation_values = native[:3]
        translation = (
            None
            if "translation" not in node
            and not all(math.isfinite(value) for value in translation_values)
            else _vec(node, "translation", (0.0, 0.0, 0.0))
        )
        old_translation = [
            value * meters_per_unit for value in translation_values
        ]
        if translation is not None and (
            not all(math.isfinite(value) for value in old_translation)
            or not _same(translation, old_translation)
        ):
            _pack_component(
                output,
                resource.transform,
                [value / meters_per_unit for value in translation],
                f"node {expected_name!r} translation",
            )
            translations += 1
            changed_resources.add(key)

        rotation_values = native[4:8]
        rotation = (
            None
            if "rotation" not in node
            and not all(math.isfinite(value) for value in rotation_values)
            else _vec(node, "rotation", (0.0, 0.0, 0.0, 1.0))
        )
        old_rotation = (
            fld_model._normalized_quaternion(tuple(rotation_values))
            if all(math.isfinite(value) for value in rotation_values)
            else []
        )
        if rotation is not None and not _same(rotation, old_rotation):
            length = math.sqrt(sum(value * value for value in rotation))
            if not math.isfinite(length) or length == 0.0:
                raise FieldSceneImportError(
                    f"node {expected_name!r} rotation is not a finite quaternion"
                )
            normalized = [value / length for value in rotation]
            _pack_component(
                output,
                resource.transform + 0x10,
                normalized,
                f"node {expected_name!r} rotation",
            )
            rotations += 1
            changed_resources.add(key)

        scale_values = native[8:11]
        scale = (
            None
            if "scale" not in node
            and not all(math.isfinite(value) for value in scale_values)
            else _vec(node, "scale", (1.0, 1.0, 1.0))
        )
        if scale is not None and (
            not all(math.isfinite(value) for value in scale_values)
            or not _same(scale, scale_values)
        ):
            _pack_component(
                output,
                resource.transform + 0x20,
                scale,
                f"node {expected_name!r} scale",
            )
            scales += 1
            changed_resources.add(key)
        imported += 1

    if not imported:
        has_field_wrapper = any(
            isinstance(node, dict)
            and node.get("name") == "FLD2 field data"
            and isinstance(node.get("extras"), dict)
            for node in nodes
        )
        if by_key or not has_field_wrapper:
            raise FieldSceneImportError(
                "GLB has no FLD2 collision, camera, or placement nodes"
            )
    rebuilt = bytes(output)
    fld.validate(rebuilt)
    return rebuilt, ImportSummary(
        imported,
        len(changed_resources),
        translations,
        rotations,
        scales,
    )


def _source_or_binary(path: Path) -> bytes:
    if path.suffix.lower() == ".fldasm":
        return fld.encode(fld.parse_source(path.read_text(encoding="utf-8")))
    return path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("input", type=Path, help="FLD2 binary or .fldasm source")
    parser.add_argument("output", type=Path, help="output FLD2 binary or .fldasm source")
    args = parser.parse_args()
    try:
        document = decode_glb(args.scene.read_bytes())
        data, summary = import_transforms(_source_or_binary(args.input), document)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".fldasm":
            args.output.write_text(fld.render_source(data), encoding="utf-8")
        else:
            args.output.write_bytes(data)
    except (FieldSceneImportError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"imported {summary.resources} resources; changed "
        f"{summary.changed_resources} resources "
        f"({summary.translations} translations, {summary.rotations} rotations, "
        f"{summary.scales} scales)"
    )


if __name__ == "__main__":
    main()
