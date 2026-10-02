from __future__ import annotations

import hashlib
import struct
import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import tmx  # noqa: E402


def texture_bundle(
    width: int,
    height: int,
    psm: int,
    pixels: bytes,
    *,
    mipmap_count: int = 0,
    clut_psm: int = 0,
    texture_flags: int = 0,
) -> bytes:
    kind = 1 if psm in {0x13, 0x14} else 0
    tmx_header = b"".join(
        (
            b"TMX0",
            bytes(4),
            struct.pack(
                "<BBHHBBHH",
                kind,
                clut_psm,
                width,
                height,
                psm,
                mipmap_count,
                texture_flags,
                0xFF00,
            ),
            bytes(0x24),
        )
    )
    packet_size = 8 + len(tmx_header) + len(pixels)
    child = struct.pack("<II", 2, packet_size) + tmx_header + pixels
    txp = b"TXP0" + bytes(4) + struct.pack("<II", 1, 0x40)
    body = txp + bytes(0x38 - len(txp)) + child
    return struct.pack("<II", 9, 8 + len(body)) + body


class TmxTests(unittest.TestCase):
    def test_accepts_empty_texture_bundle(self) -> None:
        body = b"TXP0" + bytes(4) + struct.pack("<I", 0) + bytes(0x2C)
        data = struct.pack("<II", 9, 8 + len(body)) + body
        self.assertEqual(tmx.parse_bundle(data), ())
        self.assertEqual(tmx.encode(tmx.parse_source(tmx.render_source(data))), data)

    def test_decodes_low_nibble_first_psmt4_and_expands_alpha(self) -> None:
        palette = bytearray(0x40)
        palette[4:8] = bytes((10, 20, 30, 0x40))
        palette[8:12] = bytes((40, 50, 60, 0x80))
        indices = bytes((0x21,)) * (32 * 32 // 2)
        texture = tmx.parse_bundle(texture_bundle(32, 32, 0x14, palette + indices))[0]
        self.assertEqual(texture.rgba[:8], bytes((10, 20, 30, 0x80, 40, 50, 60, 0xFF)))

    def test_unswizzles_psmt8_palette(self) -> None:
        palette = bytearray(0x400)
        palette[8 * 4 : 8 * 4 + 4] = bytes((1, 2, 3, 0x80))
        palette[16 * 4 : 16 * 4 + 4] = bytes((4, 5, 6, 0x80))
        stored = tmx._unswizzle_psmt8_palette(palette)
        indices = bytes((8, 16)) * (16 * 16 // 2)
        texture = tmx.parse_bundle(texture_bundle(16, 16, 0x13, stored + indices))[0]
        self.assertEqual(texture.rgba[:8], bytes((1, 2, 3, 0xFF, 4, 5, 6, 0xFF)))

    def test_encodes_self_contained_png(self) -> None:
        pixels = bytes((11, 22, 33, 0x80)) * (8 * 8)
        texture = tmx.parse_bundle(texture_bundle(8, 8, 0x00, pixels))[0]
        png = tmx.encode_png(texture)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack_from(">II", png, 16), (8, 8))
        self.assertEqual(texture.rgba[:4], bytes((11, 22, 33, 0xFF)))

    def test_decodes_psmct24(self) -> None:
        pixels = bytes((11, 22, 33)) * (8 * 8)
        texture = tmx.parse_bundle(texture_bundle(8, 8, 0x01, pixels))[0]
        self.assertEqual(texture.rgba[:4], bytes((11, 22, 33, 0xFF)))

    def test_ignores_mip_pixels_after_the_base_image(self) -> None:
        palette = bytearray(0x20)
        struct.pack_into("<H", palette, 2, 0xFFFF)
        base = bytes((0x11,)) * (32 * 32 // 2)
        mip = bytes((0,)) * (16 * 16 // 2)
        texture = tmx.parse_bundle(
            texture_bundle(
                32,
                32,
                0x14,
                palette + base + mip,
                mipmap_count=1,
                clut_psm=2,
                texture_flags=0x0F65,
            )
        )[0]
        self.assertEqual(texture.rgba[:4], bytes((0xFF, 0xFF, 0xFF, 0xFF)))
        self.assertEqual(texture.mipmap_count, 1)
        self.assertEqual(texture.clut_psm, 2)

    def test_exact_source_round_trip(self) -> None:
        palette = bytes(range(0x20))
        base = bytes((0x21,)) * (32 * 32 // 2)
        mip = bytes((0x43,)) * (16 * 16 // 2)
        data = texture_bundle(
            32,
            32,
            0x14,
            palette + base + mip,
            mipmap_count=1,
            clut_psm=2,
            texture_flags=0x0F65,
        )
        source = tmx.render_source(data)
        self.assertIn("psm=PSMT4 mipmaps=1 clut=PSMCT16 flags=0x0f65", source)
        self.assertEqual(tmx.encode(tmx.parse_source(source)), data)

    def test_offset_table_leaves_room_for_child_header(self) -> None:
        data = texture_bundle(8, 8, 0x01, bytes((1, 2, 3)) * 64)
        texture = tmx._parse_bundle_records(data)[0]
        source = tmx.BundleSource(
            tuple(replace(texture, index=index) for index in range(13))
        )
        rebuilt = tmx.encode(source)
        self.assertEqual(struct.unpack_from("<I", rebuilt, 0x14)[0], 0x80)
        self.assertEqual(len(tmx.parse_bundle(rebuilt)), 13)

    def test_rejects_bad_texture_boundaries(self) -> None:
        pixels = bytes((0, 0, 0, 0x80)) * (8 * 8)
        data = bytearray(texture_bundle(8, 8, 0x00, pixels))
        struct.pack_into("<I", data, 0x44, 0x40)
        with self.assertRaisesRegex(tmx.TmxError, "packet boundary"):
            tmx.parse_bundle(bytes(data))

    def test_tracked_sources_rebuild_exactly(self) -> None:
        expected = {
            "dds1": "9cd61610692ef06b49476f3568f4608db3d23866",
            "dds2": "2db8c2ad7cdae9e4bc0113a1fc9f0e1cc7193052",
        }
        for game, digest in expected.items():
            path = ROOT / "src" / game / "data" / "field" / "f011_001.tbnasm"
            with self.subTest(game=game):
                rebuilt = tmx.encode(
                    tmx.parse_source(path.read_text(encoding="utf-8"))
                )
                self.assertEqual(hashlib.sha1(rebuilt).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
