#!/usr/bin/env python3
"""Import edited field-transition metadata from a composed DDS GLB."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import field_world
import wap
from gltf_import import GltfImportError, decode_glb


class TransitionImportError(GltfImportError):
    """Raised when scene metadata cannot map safely to a WAP table."""


@dataclass(frozen=True)
class ImportSummary:
    rows: int
    changed_rows: int


def _integer(
    value: object,
    minimum: int,
    maximum: int,
    context: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise TransitionImportError(
            f"{context} must be an integer in {minimum}..{maximum}"
        )
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TransitionImportError(f"{context} must be a string")
    return value


def _fixed_string(original: wap.FixedString, value: str) -> wap.FixedString:
    return original if value == original.value else wap.FixedString(value)


def _arguments(value: object, context: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise TransitionImportError(f"{context} must contain three integers")
    return tuple(
        _integer(item, -0x8000, 0x7FFF, context) for item in value
    )  # type: ignore[return-value]


def _reconcile_argument(
    original: int,
    raw: int,
    semantic: int,
    context: str,
) -> int:
    if raw != original and semantic != original and raw != semantic:
        raise TransitionImportError(
            f"{context} raw and semantic values conflict"
        )
    return semantic if semantic != original else raw


def _destination(
    entry: wap.Entry,
    value: object,
    current_field: int,
    context: str,
) -> tuple[tuple[int, int, int], wap.FixedString, int, int, wap.FixedString]:
    expected = field_world._transition_destination(entry, current_field)
    if not isinstance(value, dict) or set(value) != set(expected):
        raise TransitionImportError(f"{context} changes its destination layout")
    if (
        value.get("typeId") != entry.warp_type
        or value.get("type") != expected["type"]
    ):
        raise TransitionImportError(f"{context} changes its transition type")

    raw = list(_arguments(value.get("arguments"), context + " arguments"))
    original = entry.warp_args
    if entry.warp_type == 0:
        field = _integer(value.get("field"), -0x8000, 0x7FFF, context + " field")
        area = _integer(value.get("area"), -0x8000, 0x7FFF, context + " area")
        physical_field = (
            0 if field == current_field and original[0] == 0 else field
        )
        raw[0] = _reconcile_argument(
            original[0], raw[0], physical_field, context + " field"
        )
        raw[1] = _reconcile_argument(
            original[1], raw[1], area, context + " area"
        )
    elif entry.warp_type == 1:
        table = _integer(value.get("table"), -0x8000, 0x7FFF, context + " table")
        floor = _integer(value.get("floor"), -0x8000, 0x7FFF, context + " floor")
        raw[0] = _reconcile_argument(
            original[0], raw[0], table, context + " table"
        )
        raw[1] = _reconcile_argument(
            original[1], raw[1], floor, context + " floor"
        )
    elif entry.warp_type == 2:
        if (
            value.get("actionId") != original[0]
            or value.get("action") != expected["action"]
        ):
            raise TransitionImportError(f"{context} changes its facility action")
        names = wap.FACILITY_ARGUMENT_NAMES.get(original[0])
        if names is not None:
            second = _integer(
                value.get(names[0]), -0x8000, 0x7FFF, context + " " + names[0]
            )
            third = _integer(
                value.get("floorFlag"),
                -0x8000,
                0x7FFF,
                context + " floor flag",
            )
            raw[1] = _reconcile_argument(
                original[1], raw[1], second, context + " " + names[0]
            )
            raw[2] = _reconcile_argument(
                original[2], raw[2], third, context + " floor flag"
            )
    elif entry.warp_type == 3:
        event = _integer(value.get("event"), -0x8000, 0x7FFF, context + " event")
        alternate = _integer(
            value.get("alternateField"),
            -0x8000,
            0x7FFF,
            context + " alternate field",
        )
        raw[0] = _reconcile_argument(
            original[0], raw[0], event, context + " event"
        )
        raw[2] = _reconcile_argument(
            original[2], raw[2], alternate, context + " alternate field"
        )

    position = _text(value.get("position"), context + " position")
    camera = value.get("camera")
    if not isinstance(camera, dict) or set(camera) != {"name", "mode", "table"}:
        raise TransitionImportError(f"{context} has invalid camera metadata")
    camera_name = _text(camera.get("name"), context + " camera name")
    camera_mode = _integer(
        camera.get("mode"), 0, 0xFF, context + " camera mode"
    )
    camera_table = _integer(
        camera.get("table"), 0, 0xFF, context + " camera table"
    )
    return (
        tuple(raw),  # type: ignore[arg-type]
        _fixed_string(entry.position, position),
        camera_mode,
        camera_table,
        _fixed_string(entry.camera, camera_name),
    )


def _row(
    entry: wap.Entry,
    value: object,
    entry_index: int,
    current_field: int,
    default: wap.Entry,
) -> wap.Entry:
    context = f"transition entry {entry_index}"
    allowed = {
        "entry",
        "kindId",
        "kind",
        "destination",
        "gate",
        "attributes",
        "after",
        "tail",
    }
    if not isinstance(value, dict) or not {
        "entry",
        "kindId",
        "kind",
        "destination",
    } <= set(value) or set(value) - allowed:
        raise TransitionImportError(f"{context} has invalid fields")
    if value.get("entry") != entry_index:
        raise TransitionImportError(f"{context} changes its row identity")
    if (
        value.get("kindId") != entry.kind
        or value.get("kind") != wap.ENTRY_KIND_NAMES.get(entry.kind, "unknown")
    ):
        raise TransitionImportError(f"{context} changes its actor kind")

    gate = value.get("gate")
    if gate is None:
        flag_mode = flag = 0
    elif isinstance(gate, dict) and set(gate) == {"mode", "flag"}:
        flag_mode = _integer(gate.get("mode"), 0, 0xFF, context + " gate mode")
        flag = _integer(gate.get("flag"), -0x8000, 0x7FFF, context + " gate flag")
    else:
        raise TransitionImportError(f"{context} has invalid gate metadata")

    attributes = _integer(
        value.get("attributes", 0), 0, 0xFF, context + " attributes"
    )
    after = value.get("after")
    if after is None:
        after = {}
    if not isinstance(after, dict) or set(after) - {
        "bgm",
        "footstep",
        "flags",
        "script",
    }:
        raise TransitionImportError(f"{context} has invalid after metadata")
    bgm = _integer(after.get("bgm", default.bgm), 0, 0xFF, context + " BGM")
    footstep = _integer(
        after.get("footstep", default.footstep),
        0,
        0xFF,
        context + " footstep",
    )
    after_flag = _integer(
        after.get("flags", 0), 0, 0xFF, context + " after flags"
    )
    after_script = _text(after.get("script", ""), context + " after script")

    tail_value = value.get("tail")
    if tail_value is None:
        tail = default.tail
    elif (
        isinstance(tail_value, dict)
        and set(tail_value) == {"control", "arguments"}
        and isinstance(tail_value.get("arguments"), list)
        and len(tail_value["arguments"]) == len(default.tail) - 1
    ):
        tail = (
            _integer(
                tail_value.get("control"), 0, 0xFF, context + " tail control"
            ),
            *(
                _integer(item, 0, 0xFF, context + " tail argument")
                for item in tail_value["arguments"]
            ),
        )
    else:
        raise TransitionImportError(f"{context} has invalid tail metadata")

    warp_args, position, camera_mode, camera_table, camera = _destination(
        entry, value["destination"], current_field, context
    )
    return replace(
        entry,
        flag_mode=flag_mode,
        flag=flag,
        warp_args=warp_args,
        position=position,
        camera_mode=camera_mode,
        camera_table=camera_table,
        camera=camera,
        attributes=attributes,
        bgm=bgm,
        footstep=footstep,
        after_flag=after_flag,
        after_script=_fixed_string(entry.after_script, after_script),
        tail=tail,
    )


def import_transitions(
    table: wap.WapFile,
    document: dict,
    current_field: int,
    current_area: int,
) -> tuple[wap.WapFile, ImportSummary]:
    """Apply transition-value edits while preserving row ownership and layout."""

    if not 0 <= current_field <= 999 or not 0 <= current_area <= 999:
        raise TransitionImportError("field and area numbers must be in 0..999")
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise TransitionImportError("GLB has no node array")

    wrappers = [
        node
        for node in nodes
        if isinstance(node, dict) and node.get("name") == "FLD2 field data"
    ]
    if len(wrappers) != 1 or not isinstance(wrappers[0].get("extras"), dict):
        raise TransitionImportError("GLB has no unique FLD2 field-data wrapper")
    wrapper_extras = wrappers[0]["extras"]

    transitions = field_world.area_transitions(
        table, current_field, current_area
    )
    placements: dict[str, list[dict]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("extras"), dict):
            continue
        extras = node["extras"]
        if extras.get("ddsResourceType") != 10:
            continue
        name = node.get("name")
        if not isinstance(name, str):
            raise TransitionImportError("FLD2 placement has no valid name")
        placements.setdefault(name, []).append(node)

    linked = {name for name in transitions if name in placements}
    expected_rows = sum(len(rows) for rows in transitions.values())
    expected_linked = sum(len(transitions[name]) for name in linked)
    if (
        wrapper_extras.get("ddsTransitionRows") != expected_rows
        or wrapper_extras.get("ddsLinkedTransitionRows") != expected_linked
    ):
        raise TransitionImportError("FLD2 transition summary differs")
    expected_unlinked = [
        {"actor": name, "entries": [row["entry"] for row in rows]}
        for name, rows in sorted(transitions.items())
        if name not in linked
    ]
    if wrapper_extras.get("ddsUnlinkedTransitionActors", []) != expected_unlinked:
        raise TransitionImportError("FLD2 unlinked-transition summary differs")

    for name, actor_nodes in placements.items():
        expected = transitions.get(name)
        values = [node["extras"].get("ddsTransitions") for node in actor_nodes]
        if expected is None:
            if any(value is not None for value in values):
                raise TransitionImportError(
                    f"placement {name!r} has unexpected transition metadata"
                )
            continue
        if any(not isinstance(value, list) for value in values):
            raise TransitionImportError(
                f"placement {name!r} is missing transition metadata"
            )
        if any(value != values[0] for value in values[1:]):
            raise TransitionImportError(
                f"placement {name!r} transition copies disagree"
            )
        actual = values[0]
        if [row.get("entry") if isinstance(row, dict) else None for row in actual] != [
            row["entry"] for row in expected
        ]:
            raise TransitionImportError(
                f"placement {name!r} changes its transition rows"
            )

    entries = list(table.entries)
    default = wap.default_entry(table.profile)
    changed = 0
    for name in sorted(linked):
        actual = placements[name][0]["extras"]["ddsTransitions"]
        for value in actual:
            entry_index = value["entry"]
            edited = _row(
                entries[entry_index],
                value,
                entry_index,
                current_field,
                default,
            )
            changed += edited != entries[entry_index]
            entries[entry_index] = edited

    rebuilt = replace(table, entries=tuple(entries))
    wap.encode(rebuilt)
    return rebuilt, ImportSummary(expected_linked, changed)


def _source(path: Path, references: wap.References) -> wap.WapFile:
    if path.suffix.lower() == ".wapasm":
        return wap.parse_source(path.read_text(encoding="utf-8"), references)
    return wap.decode(path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("input", type=Path, help="WAP binary or .wapasm source")
    parser.add_argument("output", type=Path, help="output WAP binary or source")
    parser.add_argument("--field", type=int, required=True)
    parser.add_argument("--area", type=int, required=True)
    parser.add_argument("--scripts", type=Path, help="paired BF source")
    parser.add_argument("--interactions", type=Path, help="paired INF source")
    args = parser.parse_args()
    try:
        document, _ = decode_glb(args.scene.read_bytes())
        references = wap.load_references(args.scripts, args.interactions)
        rebuilt, summary = import_transitions(
            _source(args.input, references), document, args.field, args.area
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".wapasm":
            args.output.write_text(
                wap.render_source(rebuilt, references), encoding="utf-8"
            )
        else:
            args.output.write_bytes(wap.encode(rebuilt))
    except (GltfImportError, OSError, ValueError, wap.WapError) as exc:
        parser.error(str(exc))
    print(
        f"imported {summary.rows} transition rows; "
        f"changed {summary.changed_rows}"
    )


if __name__ == "__main__":
    main()
