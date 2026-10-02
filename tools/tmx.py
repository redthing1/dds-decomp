#!/usr/bin/env python3
"""Decode, assemble, and edit DDS TBN/TXP0 texture bundles."""

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


def render_bundle_source(source: BundleSource) -> str:
    """Render parsed texture records as canonical, editable text source."""

    lines = ["tbn 1"]
    for texture in source.textures:
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


def render_source(data: bytes) -> str:
    """Render a complete TBN packet as exact, editable text source."""

    if data[8:12] != b"TXP0":
        raise TmxError("exact source requires a complete TBN packet")
    return render_bundle_source(BundleSource(_parse_bundle_records(data)))


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


def decode_png(data: bytes) -> tuple[int, int, bytes]:
    """Decode a non-interlaced 8-bit PNG to RGBA pixels."""

    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise TmxError("replacement image is not a PNG")
    offset = 8
    header: tuple[int, ...] | None = None
    palette: bytes | None = None
    transparency: bytes | None = None
    compressed = bytearray()
    saw_end = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise TmxError("truncated PNG chunk")
        size = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + size
        if end > len(data):
            raise TmxError("PNG chunk exceeds the file")
        payload = data[offset + 8 : offset + 8 + size]
        checksum = struct.unpack_from(">I", data, offset + 8 + size)[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != checksum:
            raise TmxError(f"PNG {kind.decode('ascii', 'replace')} CRC mismatch")
        if kind == b"IHDR":
            if header is not None or size != 13:
                raise TmxError("invalid PNG IHDR")
            header = struct.unpack(">IIBBBBB", payload)
        elif kind == b"PLTE":
            palette = payload
        elif kind == b"tRNS":
            transparency = payload
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            if payload:
                raise TmxError("invalid PNG IEND")
            saw_end = True
            if end != len(data):
                raise TmxError("PNG has data after IEND")
            break
        offset = end
    if header is None or not saw_end or not compressed:
        raise TmxError("PNG is missing IHDR, IDAT, or IEND")

    width, height, depth, color_type, compression, filtering, interlace = header
    if not width or not height:
        raise TmxError("PNG has zero dimensions")
    if depth != 8 or compression or filtering or interlace:
        raise TmxError("replacement PNG must be non-interlaced 8-bit data")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise TmxError(f"unsupported PNG color type {color_type}")
    if color_type == 3 and (
        palette is None or len(palette) % 3 or not 1 <= len(palette) // 3 <= 256
    ):
        raise TmxError("indexed PNG has an invalid palette")
    if transparency is not None and color_type not in {3}:
        raise TmxError("PNG transparency is supported only for indexed images")

    stride = width * channels
    try:
        filtered = zlib.decompress(bytes(compressed))
    except zlib.error as exc:
        raise TmxError("invalid PNG compressed data") from exc
    expected = height * (stride + 1)
    if len(filtered) != expected:
        raise TmxError(
            f"PNG has {len(filtered)} decompressed bytes, expected {expected}"
        )

    rows: list[bytes] = []
    source = 0
    for row_index in range(height):
        filter_kind = filtered[source]
        source += 1
        encoded = filtered[source : source + stride]
        source += stride
        previous = rows[-1] if rows else bytes(stride)
        row = bytearray(stride)
        for index, value in enumerate(encoded):
            left = row[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_kind == 0:
                predictor = 0
            elif filter_kind == 1:
                predictor = left
            elif filter_kind == 2:
                predictor = above
            elif filter_kind == 3:
                predictor = (left + above) // 2
            elif filter_kind == 4:
                estimate = left + above - upper_left
                distances = (
                    abs(estimate - left),
                    abs(estimate - above),
                    abs(estimate - upper_left),
                )
                predictor = (left, above, upper_left)[distances.index(min(distances))]
            else:
                raise TmxError(
                    f"PNG row {row_index} uses invalid filter {filter_kind}"
                )
            row[index] = (value + predictor) & 0xFF
        rows.append(bytes(row))

    pixels = bytearray(width * height * 4)
    destination = 0
    for row in rows:
        for offset in range(0, len(row), channels):
            sample = row[offset : offset + channels]
            if color_type == 0:
                rgba = bytes((sample[0], sample[0], sample[0], 0xFF))
            elif color_type == 2:
                rgba = sample + b"\xff"
            elif color_type == 3:
                palette_index = sample[0]
                palette_offset = palette_index * 3
                if palette is None or palette_offset + 3 > len(palette):
                    raise TmxError("indexed PNG pixel exceeds its palette")
                alpha = (
                    transparency[palette_index]
                    if transparency is not None and palette_index < len(transparency)
                    else 0xFF
                )
                rgba = palette[palette_offset : palette_offset + 3] + bytes((alpha,))
            elif color_type == 4:
                rgba = bytes((sample[0], sample[0], sample[0], sample[1]))
            else:
                rgba = sample
            pixels[destination : destination + 4] = rgba
            destination += 4
    return width, height, bytes(pixels)


def _base_pixel_size(width: int, height: int, psm: int) -> int:
    pixels = width * height
    return {
        0x00: pixels * 4,
        0x01: pixels * 3,
        0x02: pixels * 2,
        0x13: pixels,
        0x14: (pixels + 1) // 2,
    }[psm]


def _palette_size(psm: int, clut_psm: int) -> int:
    if psm == 0x13:
        return 0x400
    if psm == 0x14:
        return {0: 0x40, 2: 0x20}[clut_psm]
    return 0


def _contract_alpha(value: int) -> int:
    return min((value + 1) // 2, 0x80)


def _encode_base_pixels(texture: EncodedTexture, rgba: bytes) -> bytes:
    count = texture.width * texture.height
    if len(rgba) != count * 4:
        raise TmxError("replacement image has an invalid RGBA payload")
    if texture.psm == 0x00:
        return bytes(
            channel
            for offset in range(0, len(rgba), 4)
            for channel in (
                rgba[offset],
                rgba[offset + 1],
                rgba[offset + 2],
                _contract_alpha(rgba[offset + 3]),
            )
        )
    if texture.psm == 0x01:
        if any(rgba[offset + 3] != 0xFF for offset in range(0, len(rgba), 4)):
            raise TmxError("PSMCT24 replacement PNG must be fully opaque")
        return bytes(
            channel
            for offset in range(0, len(rgba), 4)
            for channel in rgba[offset : offset + 3]
        )
    if texture.psm == 0x02:
        result = bytearray()
        for offset in range(0, len(rgba), 4):
            red, green, blue, alpha = rgba[offset : offset + 4]
            value = (
                (red * 31 + 127) // 255
                | ((green * 31 + 127) // 255) << 5
                | ((blue * 31 + 127) // 255) << 10
                | (0x8000 if alpha >= 0x80 else 0)
            )
            result.extend(struct.pack("<H", value))
        return bytes(result)

    palette_size = _palette_size(texture.psm, texture.clut_psm)
    stored_palette = texture.payload[:palette_size]
    if texture.psm == 0x13:
        colors = _palette(_unswizzle_psmt8_palette(stored_palette), 256)
    elif texture.clut_psm == 0:
        colors = _palette(stored_palette, 16)
    else:
        colors = _palette_16(stored_palette, 16)
    indices_by_color: dict[tuple[int, int, int, int], int] = {}
    for index, color in enumerate(colors):
        indices_by_color.setdefault(color, index)
    original_data = texture.payload[palette_size:]
    if texture.psm == 0x13:
        original_indices = original_data[:count]
    else:
        original_indices = bytes(
            (original_data[index // 2] >> ((index & 1) * 4)) & 0xF
            for index in range(count)
        )
    indices = []
    for pixel, offset in enumerate(range(0, len(rgba), 4)):
        color = tuple(rgba[offset : offset + 4])
        original_index = original_indices[pixel]
        if color == colors[original_index]:
            indices.append(original_index)
            continue
        try:
            indices.append(indices_by_color[color])
        except KeyError as exc:
            raise TmxError(
                f"replacement color {color} is absent from texture "
                f"{texture.index}'s existing palette"
            ) from exc
    if texture.psm == 0x13:
        return bytes(indices)
    packed = bytearray((len(indices) + 1) // 2)
    for index, value in enumerate(indices):
        packed[index // 2] |= value << ((index & 1) * 4)
    return bytes(packed)


def replace_base_image(
    source: BundleSource,
    texture_index: int,
    width: int,
    height: int,
    rgba: bytes,
) -> BundleSource:
    """Replace one base image while preserving its native TMX0 profile and mips."""

    if not 0 <= texture_index < len(source.textures):
        raise TmxError(f"texture index {texture_index} is outside the bundle")
    texture = source.textures[texture_index]
    if (width, height) != (texture.width, texture.height):
        raise TmxError(
            f"replacement image is {width}x{height}, expected "
            f"{texture.width}x{texture.height}"
        )
    palette_size = _palette_size(texture.psm, texture.clut_psm)
    base_size = _base_pixel_size(texture.width, texture.height, texture.psm)
    base = _encode_base_pixels(texture, rgba)
    payload = (
        texture.payload[:palette_size]
        + base
        + texture.payload[palette_size + base_size :]
    )
    textures = list(source.textures)
    textures[texture_index] = EncodedTexture(
        texture.index,
        texture.width,
        texture.height,
        texture.psm,
        texture.mipmap_count,
        texture.clut_psm,
        texture.texture_flags,
        payload,
    )
    return BundleSource(tuple(textures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("extract", "disassemble", "assemble"):
        child = subparsers.add_parser(command)
        child.add_argument("input", type=Path)
        child.add_argument("output", type=Path)
    importer = subparsers.add_parser("import")
    importer.add_argument("input", type=Path)
    importer.add_argument("image", type=Path)
    importer.add_argument("output", type=Path)
    importer.add_argument("--texture", type=int, required=True)
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
        elif args.command == "import":
            if args.input.suffix.lower() == ".tbnasm":
                source = parse_source(args.input.read_text(encoding="utf-8"))
            else:
                data = args.input.read_bytes()
                if data[8:12] != b"TXP0":
                    raise TmxError("texture import requires a complete TBN packet")
                source = BundleSource(_parse_bundle_records(data))
            width, height, rgba = decode_png(args.image.read_bytes())
            replaced = replace_base_image(
                source, args.texture, width, height, rgba
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.suffix.lower() == ".tbnasm":
                args.output.write_text(
                    render_bundle_source(replaced), encoding="utf-8"
                )
            else:
                args.output.write_bytes(encode(replaced))
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
