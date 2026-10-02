#!/usr/bin/env python3
"""Import one edited field-world GLB into its exact DDS resources."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import amb
import amb_scene_import
import fld
import fld_model_import
import fld_scene_import
import lb
import tmx
import tmx_gltf_import
from gltf_import import GltfImportError, decode_glb


@dataclass(frozen=True)
class WorldImportResult:
    model: bytes
    field: bytes
    textures: tmx.BundleSource | None
    automap: bytes | None
    model_summary: fld_model_import.ImportSummary
    field_summary: fld_scene_import.ImportSummary
    texture_summary: tmx_gltf_import.ImportSummary | None
    automap_summary: amb_scene_import.ImportSummary | None


def import_world(
    model_data: bytes,
    field_data: bytes,
    document: dict,
    binary: bytes,
    *,
    textures: tmx.BundleSource | None = None,
    automap_data: bytes | None = None,
) -> WorldImportResult:
    """Validate and import every supplied resource before returning any output."""

    model, model_summary = fld_model_import.import_geometry(
        model_data, document, binary
    )
    field, field_summary = fld_scene_import.import_scene(
        field_data, document, binary
    )
    rebuilt_textures = None
    texture_summary = None
    if textures is not None:
        rebuilt_textures, texture_summary = tmx_gltf_import.import_images(
            textures, document, binary
        )
    automap = None
    automap_summary = None
    if automap_data is not None:
        automap, automap_summary = amb_scene_import.import_geometry(
            automap_data, document, binary
        )
    return WorldImportResult(
        model,
        field,
        rebuilt_textures,
        automap,
        model_summary,
        field_summary,
        texture_summary,
        automap_summary,
    )


def import_archive(
    archive_data: bytes,
    document: dict,
    binary: bytes,
) -> tuple[bytes, WorldImportResult]:
    """Import a field scene and rebuild its F1, F2, and TBN archive entries."""

    archive = lb.parse_archive(archive_data)

    def one(extension: str) -> lb.Entry:
        matches = [
            entry
            for entry in archive.entries
            if entry.extension.upper() == extension
        ]
        if len(matches) != 1:
            raise lb.LbError(
                f"field-world import requires exactly one {extension} entry"
            )
        return matches[0]

    model_entry = one("F1")
    field_entry = one("F2")
    texture_entry = one("TBN")
    texture_data = lb.entry_data(texture_entry)
    textures = tmx.BundleSource(tmx._parse_bundle_records(texture_data))
    result = import_world(
        lb.entry_data(model_entry),
        lb.entry_data(field_entry),
        document,
        binary,
        textures=textures,
    )
    if result.textures is None:
        raise AssertionError("archive texture import did not return a bundle")
    rebuilt = lb.replace_entries(
        archive_data,
        {
            model_entry.index: result.model,
            field_entry.index: result.field,
            texture_entry.index: tmx.encode(result.textures),
        },
    )
    return rebuilt, result


def _fld_data(path: Path, source_suffix: str) -> bytes:
    if path.suffix.lower() == source_suffix:
        return fld.encode(fld.parse_source(path.read_text(encoding="utf-8")))
    return path.read_bytes()


def _amb_data(path: Path) -> bytes:
    if path.suffix.lower() == ".ambasm":
        return amb.encode(amb.parse_source(path.read_text(encoding="utf-8")))
    return path.read_bytes()


def _texture_source(path: Path) -> tmx.BundleSource:
    if path.suffix.lower() == ".tbnasm":
        return tmx.parse_source(path.read_text(encoding="utf-8"))
    data = path.read_bytes()
    if data[8:12] != b"TXP0":
        raise tmx.TmxError("field-world import requires a complete TBN packet")
    return tmx.BundleSource(tmx._parse_bundle_records(data))


def _write_fld(path: Path, data: bytes, source_suffix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == source_suffix:
        path.write_text(fld.render_source(data), encoding="utf-8")
    else:
        path.write_bytes(data)


def _write_automap(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".ambasm":
        path.write_text(amb.render_source(data), encoding="utf-8")
    else:
        path.write_bytes(data)


def _write_textures(path: Path, source: tmx.BundleSource) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".tbnasm":
        path.write_text(tmx.render_bundle_source(source), encoding="utf-8")
    else:
        path.write_bytes(tmx.encode(source))


def _print_summary(result: WorldImportResult) -> None:
    parts = [
        f"{result.model_summary.changed_meshes} model meshes",
        f"{result.model_summary.changed_materials} model materials",
        f"{result.model_summary.changed_nodes} model nodes",
        f"{result.model_summary.changed_tracks} animation tracks",
        f"{result.field_summary.changed_resources} field resources",
        f"{result.field_summary.changed_collision_meshes} collision meshes",
    ]
    if result.texture_summary is not None:
        parts.append(f"{result.texture_summary.changed_images} textures")
    if result.automap_summary is not None:
        parts.extend(
            (
                f"{result.automap_summary.changed_meshes} automap meshes",
                f"{result.automap_summary.changed_nodes} automap nodes",
            )
        )
    print("changed " + ", ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    archive_parser = commands.add_parser(
        "archive", help="rebuild an edited F1/F2/TBN LB archive"
    )
    archive_parser.add_argument("scene", type=Path)
    archive_parser.add_argument("input", type=Path)
    archive_parser.add_argument("output", type=Path)

    sources = commands.add_parser(
        "sources", help="write edited loose binaries or exact sources"
    )
    sources.add_argument("scene", type=Path)
    sources.add_argument("model_input", type=Path)
    sources.add_argument("model_output", type=Path)
    sources.add_argument(
        "--field", nargs=2, required=True, type=Path, metavar=("INPUT", "OUTPUT")
    )
    sources.add_argument(
        "--textures", nargs=2, type=Path, metavar=("INPUT", "OUTPUT")
    )
    sources.add_argument(
        "--automap", nargs=2, type=Path, metavar=("INPUT", "OUTPUT")
    )

    args = parser.parse_args()
    try:
        document, binary = decode_glb(args.scene.read_bytes())
        if args.command == "archive":
            rebuilt, result = import_archive(
                args.input.read_bytes(), document, binary
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(rebuilt)
        else:
            textures = (
                _texture_source(args.textures[0])
                if args.textures is not None
                else None
            )
            automap_data = (
                _amb_data(args.automap[0])
                if args.automap is not None
                else None
            )
            result = import_world(
                _fld_data(args.model_input, ".f1asm"),
                _fld_data(args.field[0], ".fldasm"),
                document,
                binary,
                textures=textures,
                automap_data=automap_data,
            )
            _write_fld(args.model_output, result.model, ".f1asm")
            _write_fld(args.field[1], result.field, ".fldasm")
            if args.textures is not None:
                if result.textures is None:
                    raise AssertionError("texture import did not return a bundle")
                _write_textures(args.textures[1], result.textures)
            if args.automap is not None:
                if result.automap is None:
                    raise AssertionError("automap import did not return data")
                _write_automap(args.automap[1], result.automap)
        _print_summary(result)
    except (
        GltfImportError,
        OSError,
        amb.AmbError,
        fld.FldError,
        lb.LbError,
        tmx.TmxError,
        ValueError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
