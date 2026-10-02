#!/usr/bin/env python3
"""Decode DDS TBN/TXP0 texture bundles and their TMX0 images."""

from __future__ import annotations

import argparse
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path


PSM_NAMES = {
    0x00: "PSMCT32",
    0x01: "PSMCT24",
    0x02: "PSMCT16",
    0x13: "PSMT8",
    0x14: "PSMT4",
}


class TmxError(ValueError):
    """Raised when a DDS texture bundle is malformed or unsupported."""


@dataclass(frozen=True)
class Texture:
    index: int
    width: int
    height: int
    psm: int
    rgba: bytes
    mipmap_count: int = 0
    clut_psm: int = 0
    texture_flags: int = 0
    payload: bytes = b""


@dataclass(frozen=True)
class EncodedTexture:
    index: int
    width: int
    height: int
    psm: int
    mipmap_count: int
    clut_psm: int
    texture_flags: int
    payload: bytes


@dataclass(frozen=True)
class BundleSource:
    textures: tuple[EncodedTexture, ...]


def _u32(data: bytes, offset: int, context: str) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise TmxError(f"{context} lies outside the texture bundle")
    return struct.unpack_from("<I", data, offset)[0]


def _expand_alpha(value: int) -> int:
    return min(value * 2, 0xFF)


def _unswizzle_psmt8_palette(palette: bytes) -> bytes:
    """Convert the GS CSM1 order of a 256-color palette to index order."""

    result = bytearray(palette)
    for block in range(0, 256, 32):
        for index in range(8):
            left = (block + 8 + index) * 4
            right = (block + 16 + index) * 4
            result[left : left + 4], result[right : right + 4] = (
                result[right : right + 4],
                result[left : left + 4],
            )
    return bytes(result)


def _palette(data: bytes, count: int) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (
            data[index * 4],
            data[index * 4 + 1],
            data[index * 4 + 2],
            _expand_alpha(data[index * 4 + 3]),
        )
        for index in range(count)
    )


def _palette_16(data: bytes, count: int) -> tuple[tuple[int, int, int, int], ...]:
    result = []
    for index in range(count):
        value = struct.unpack_from("<H", data, index * 2)[0]
        result.append(
            (
                (value & 0x1F) * 0xFF // 0x1F,
                ((value >> 5) & 0x1F) * 0xFF // 0x1F,
                ((value >> 10) & 0x1F) * 0xFF // 0x1F,
                0xFF if value & 0x8000 else 0,
            )
        )
    return tuple(result)


def _decode_pixels(
    data: bytes,
    width: int,
    height: int,
    psm: int,
    clut_psm: int,
) -> bytes:
    count = width * height
    rgba = bytearray(count * 4)
    if psm == 0x13:
        if clut_psm != 0:
            raise TmxError(f"unsupported PSMT8 CLUT mode 0x{clut_psm:x}")
        palette = _palette(_unswizzle_psmt8_palette(data[:0x400]), 256)
        indices = data[0x400:]
        for index, palette_index in enumerate(indices):
            rgba[index * 4 : index * 4 + 4] = bytes(palette[palette_index])
    elif psm == 0x14:
        if clut_psm == 0:
            palette_size = 0x40
            palette = _palette(data[:palette_size], 16)
        elif clut_psm == 2:
            palette_size = 0x20
            palette = _palette_16(data[:palette_size], 16)
        else:
            raise TmxError(f"unsupported PSMT4 CLUT mode 0x{clut_psm:x}")
        indices = data[palette_size:]
        for index in range(count):
            palette_index = (indices[index // 2] >> ((index & 1) * 4)) & 0xF
            rgba[index * 4 : index * 4 + 4] = bytes(palette[palette_index])
    elif psm == 0x00:
        for index in range(count):
            source = index * 4
            rgba[source : source + 3] = data[source : source + 3]
            rgba[source + 3] = _expand_alpha(data[source + 3])
    elif psm == 0x01:
        for index in range(count):
            source = index * 3
            rgba[index * 4 : index * 4 + 4] = data[source : source + 3] + b"\xff"
    elif psm == 0x02:
        for index in range(count):
            value = struct.unpack_from("<H", data, index * 2)[0]
            rgba[index * 4 : index * 4 + 4] = bytes(
                (
                    (value & 0x1F) * 0xFF // 0x1F,
                    ((value >> 5) & 0x1F) * 0xFF // 0x1F,
                    ((value >> 10) & 0x1F) * 0xFF // 0x1F,
                    0xFF if value & 0x8000 else 0,
                )
            )
    else:
        raise TmxError(f"unsupported GS pixel storage mode 0x{psm:x}")
    return bytes(rgba)


def _pixel_data_size(
    width: int,
    height: int,
    psm: int,
    mipmap_count: int,
    clut_psm: int,
) -> int:
    dimensions = (
        (max(1, width >> level), max(1, height >> level))
        for level in range(mipmap_count + 1)
    )
    pixels = sum(level_width * level_height for level_width, level_height in dimensions)
    try:
        return {
            0x00: pixels * 4,
            0x01: pixels * 3,
            0x02: pixels * 2,
            0x13: (0x400 if clut_psm == 0 else 0) + pixels,
            0x14: ({0: 0x40, 2: 0x20}.get(clut_psm, 0)) + (pixels + 1) // 2,
        }[psm]
    except KeyError as exc:
        raise TmxError(f"unsupported GS pixel storage mode 0x{psm:x}") from exc


def _parse_bundle_records(data: bytes) -> tuple[EncodedTexture, ...]:
    """Parse exact texture records without expanding their pixels."""

    if data[:4] == b"TXP0":
        base = 0
    elif data[8:12] == b"TXP0":
        base = 8
        if _u32(data, 0, "TBN type") != 9 or _u32(data, 4, "TBN size") != len(data):
            raise TmxError("invalid TBN packet header")
    else:
        raise TmxError("texture bundle has no TXP0 header")

    if _u32(data, base + 4, "TXP0 reserved word") != 0:
        raise TmxError("TXP0 reserved word is nonzero")
    count = _u32(data, base + 8, "TXP0 texture count")
    table_end = base + 0xC + count * 4
    if table_end > len(data):
        raise TmxError("TXP0 offset table exceeds the bundle")
    offsets = struct.unpack_from(f"<{count}I", data, base + 0xC)
    if tuple(sorted(offsets)) != offsets or len(set(offsets)) != count:
        raise TmxError("TXP0 texture offsets are not strictly increasing")
    if offsets and offsets[0] < 0xC + count * 4:
        raise TmxError("first TMX0 overlaps the TXP0 offset table")

    textures = []
    for index, relative in enumerate(offsets):
        magic = base + relative
        packet = magic - 8
        if packet < base or data[magic : magic + 4] != b"TMX0":
            raise TmxError(f"texture {index} has no TMX0 header")
        packet_type = _u32(data, packet, f"texture {index} packet type")
        packet_size = _u32(data, packet + 4, f"texture {index} packet size")
        end = packet + packet_size
        expected_end = (
            base + offsets[index + 1] - 8 if index + 1 < count else len(data)
        )
        if packet_type != 2 or packet_size < 0x40 or end != expected_end:
            raise TmxError(f"texture {index} has an invalid packet boundary")
        if _u32(data, magic + 4, f"texture {index} reserved word") != 0:
            raise TmxError(f"texture {index} has a nonzero reserved word")
        kind, clut_psm, width, height, psm, mipmap_count, texture_flags, word_12 = (
            struct.unpack_from("<BBHHBBHH", data, magic + 8)
        )
        if not width or not height:
            raise TmxError(f"texture {index} has zero dimensions")
        expected_kind = 1 if psm in {0x13, 0x14} else 0
        valid_clut = clut_psm == 0 or (psm == 0x14 and clut_psm == 2)
        if (
            kind != expected_kind
            or not valid_clut
            or word_12 != 0xFF00
        ):
            raise TmxError(f"texture {index} has an unsupported TMX0 profile")
        if any(data[magic + 0x14 : magic + 0x38]):
            raise TmxError(f"texture {index} has nonzero TMX0 padding")
        pixel_size = _pixel_data_size(width, height, psm, mipmap_count, clut_psm)
        pixel_data = data[magic + 0x38 : end]
        if len(pixel_data) != pixel_size:
            raise TmxError(
                f"texture {index} has {len(pixel_data)} pixel bytes, expected {pixel_size}"
            )
        textures.append(
            EncodedTexture(
                index,
                width,
                height,
                psm,
                mipmap_count,
                clut_psm,
                texture_flags,
                pixel_data,
            )
        )
    return tuple(textures)


def parse_bundle(data: bytes) -> tuple[Texture, ...]:
    """Parse a DDS TBN payload or its embedded TXP0 packet."""

    return tuple(
        Texture(
            texture.index,
            texture.width,
            texture.height,
            texture.psm,
            _decode_pixels(
                texture.payload,
                texture.width,
                texture.height,
                texture.psm,
                texture.clut_psm,
            ),
            texture.mipmap_count,
            texture.clut_psm,
            texture.texture_flags,
            texture.payload,
        )
        for texture in _parse_bundle_records(data)
    )


def render_source(data: bytes) -> str:
    """Render a complete TBN packet as exact, editable text source."""

    if data[8:12] != b"TXP0":
        raise TmxError("exact source requires a complete TBN packet")
    lines = ["tbn 1"]
    for texture in _parse_bundle_records(data):
        clut = (
            "PSMCT16"
            if texture.clut_psm == 2
            else "PSMCT32"
            if texture.psm in {0x13, 0x14}
            else "none"
        )
        lines.append(
            "texture "
            f"index={texture.index} width={texture.width} height={texture.height} "
            f"psm={PSM_NAMES[texture.psm]} mipmaps={texture.mipmap_count} "
            f"clut={clut} flags=0x{texture.texture_flags:04x}"
        )
        for offset in range(0, len(texture.payload), 32):
            lines.append(f"data {texture.payload[offset : offset + 32].hex()}")
        lines.append("end_texture")
    return "\n".join(lines) + "\n"


def _source_fields(parts: list[str], line: int) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            raise TmxError(f"line {line}: expected name=value, got {part!r}")
        name, value = part.split("=", 1)
        if not name or name in fields:
            raise TmxError(f"line {line}: duplicate or empty field {name!r}")
        fields[name] = value
    return fields


def parse_source(text: str) -> BundleSource:
    """Parse exact TBN text source."""

    operations = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if line:
            operations.append((line_number, line.split()))
    if not operations or operations[0][1] != ["tbn", "1"]:
        raise TmxError("source must begin with 'tbn 1'")

    names = {name: value for value, name in PSM_NAMES.items()}
    textures = []
    current: dict | None = None
    payload = bytearray()
    for line_number, parts in operations[1:]:
        directive = parts[0]
        if directive == "texture":
            if current is not None:
                raise TmxError(f"line {line_number}: nested texture")
            fields = _source_fields(parts[1:], line_number)
            expected = {"index", "width", "height", "psm", "mipmaps", "clut", "flags"}
            if set(fields) != expected:
                raise TmxError(
                    f"line {line_number}: texture fields are {sorted(fields)}, "
                    f"expected {sorted(expected)}"
                )
            try:
                psm = names[fields["psm"]]
                clut_psm = {"none": 0, "PSMCT32": 0, "PSMCT16": 2}[
                    fields["clut"]
                ]
                current = {
                    "index": int(fields["index"], 0),
                    "width": int(fields["width"], 0),
                    "height": int(fields["height"], 0),
                    "psm": psm,
                    "mipmap_count": int(fields["mipmaps"], 0),
                    "clut_psm": clut_psm,
                    "texture_flags": int(fields["flags"], 0),
                }
            except (KeyError, ValueError) as exc:
                raise TmxError(f"line {line_number}: invalid texture field") from exc
            indexed = psm in {0x13, 0x14}
            if indexed != (fields["clut"] != "none"):
                raise TmxError(
                    f"line {line_number}: {fields['psm']} requires "
                    f"{'a CLUT' if indexed else 'clut=none'}"
                )
            payload.clear()
        elif directive == "data":
            if current is None or len(parts) != 2:
                raise TmxError(f"line {line_number}: data must be inside a texture")
            try:
                payload.extend(bytes.fromhex(parts[1]))
            except ValueError as exc:
                raise TmxError(f"line {line_number}: invalid hexadecimal data") from exc
        elif directive == "end_texture":
            if current is None or len(parts) != 1:
                raise TmxError(f"line {line_number}: unexpected end_texture")
            index = current["index"]
            if index != len(textures):
                raise TmxError(
                    f"line {line_number}: texture index {index}, expected {len(textures)}"
                )
            expected_size = _pixel_data_size(
                current["width"],
                current["height"],
                current["psm"],
                current["mipmap_count"],
                current["clut_psm"],
            )
            if len(payload) != expected_size:
                raise TmxError(
                    f"line {line_number}: texture {index} has {len(payload)} pixel bytes, "
                    f"expected {expected_size}"
                )
            textures.append(EncodedTexture(**current, payload=bytes(payload)))
            current = None
            payload.clear()
        else:
            raise TmxError(f"line {line_number}: unknown directive {directive!r}")
    if current is not None:
        raise TmxError("source ends inside a texture")
    return BundleSource(tuple(textures))


def encode(source: BundleSource) -> bytes:
    """Assemble exact TBN source into its packet representation."""

    count = len(source.textures)
    first_magic = (0x14 + count * 4 + 0x3F) & ~0x3F
    offsets = []
    children = []
    cursor = first_magic
    for index, texture in enumerate(source.textures):
        if texture.index != index:
            raise TmxError(f"texture index {texture.index}, expected {index}")
        if not 1 <= texture.width <= 0xFFFF or not 1 <= texture.height <= 0xFFFF:
            raise TmxError(
                f"texture {index} dimensions {texture.width}x{texture.height} "
                "are outside the TMX0 range"
            )
        if not 0 <= texture.mipmap_count <= 0xFF:
            raise TmxError(f"texture {index} mipmap count is outside the TMX0 range")
        if not 0 <= texture.texture_flags <= 0xFFFF:
            raise TmxError(f"texture {index} flags are outside the TMX0 range")
        if texture.psm not in PSM_NAMES:
            raise TmxError(
                f"texture {index} uses unsupported GS pixel storage mode "
                f"0x{texture.psm:x}"
            )
        valid_clut = texture.clut_psm == 0 or (
            texture.psm == 0x14 and texture.clut_psm == 2
        )
        if not valid_clut:
            raise TmxError(f"texture {index} has an unsupported TMX0 profile")
        expected_size = _pixel_data_size(
            texture.width,
            texture.height,
            texture.psm,
            texture.mipmap_count,
            texture.clut_psm,
        )
        if len(texture.payload) != expected_size:
            raise TmxError(
                f"texture {index} has {len(texture.payload)} pixel bytes, "
                f"expected {expected_size}"
            )
        kind = 1 if texture.psm in {0x13, 0x14} else 0
        header = b"".join(
            (
                b"TMX0",
                bytes(4),
                struct.pack(
                    "<BBHHBBHH",
                    kind,
                    texture.clut_psm,
                    texture.width,
                    texture.height,
                    texture.psm,
                    texture.mipmap_count,
                    texture.texture_flags,
                    0xFF00,
                ),
                bytes(0x24),
            )
        )
        packet_size = 8 + len(header) + len(texture.payload)
        offsets.append(cursor)
        children.append(struct.pack("<II", 2, packet_size) + header + texture.payload)
        cursor += packet_size

    txp = b"TXP0" + bytes(4) + struct.pack("<I", count)
    if offsets:
        txp += struct.pack(f"<{count}I", *offsets)
        txp += bytes(first_magic - 8 - len(txp))
    else:
        txp += bytes(0x38 - len(txp))
    body = txp + b"".join(children)
    return struct.pack("<II", 9, 8 + len(body)) + body


def encode_png(texture: Texture) -> bytes:
    """Encode a decoded texture as an RGBA8 PNG using only the standard library."""

    raw = b"".join(
        b"\0"
        + texture.rgba[row * texture.width * 4 : (row + 1) * texture.width * 4]
        for row in range(texture.height)
    )

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return b"".join(
            (
                struct.pack(">I", len(payload)),
                kind,
                payload,
                struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF),
            )
        )

    header = struct.pack(
        ">IIBBBBB", texture.width, texture.height, 8, 6, 0, 0, 0
    )
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", header),
            chunk(b"IDAT", zlib.compress(raw, 9)),
            chunk(b"IEND", b""),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("extract", "disassemble", "assemble"):
        child = subparsers.add_parser(command)
        child.add_argument("input", type=Path)
        child.add_argument("output", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("input", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "extract":
            textures = parse_bundle(args.input.read_bytes())
            args.output.mkdir(parents=True, exist_ok=True)
            for texture in textures:
                name = (
                    f"{texture.index:03d}_{PSM_NAMES[texture.psm]}_"
                    f"{texture.width}x{texture.height}.png"
                )
                (args.output / name).write_bytes(encode_png(texture))
        elif args.command == "disassemble":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                render_source(args.input.read_bytes()), encoding="utf-8"
            )
        elif args.command == "assemble":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(
                encode(parse_source(args.input.read_text(encoding="utf-8")))
            )
        elif args.command == "verify":
            data = args.input.read_bytes()
            rebuilt = encode(parse_source(render_source(data)))
            if rebuilt != data:
                raise TmxError("TBN source round trip differs from its input")
        else:
            raise AssertionError(args.command)
    except (OSError, TmxError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
