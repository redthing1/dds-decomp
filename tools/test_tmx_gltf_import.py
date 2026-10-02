from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fld  # noqa: E402
import fld_model  # noqa: E402
import tmx  # noqa: E402
import tmx_gltf_import  # noqa: E402
from test_tmx import texture_bundle  # noqa: E402


def texture_source(count: int = 1) -> tuple[bytes, tmx.BundleSource]:
    data = texture_bundle(8, 8, 0x00, bytes((1, 2, 3, 0x40)) * 64)
    texture = tmx._parse_bundle_records(data)[0]
    source = tmx.BundleSource(
        tuple(replace(texture, index=index) for index in range(count))
    )
    return tmx.encode(source), source


def texture_glb(
    source: tmx.BundleSource,
    texture_index: int = 0,
) -> tuple[dict, bytearray]:
    textures = tmx.parse_bundle(tmx.encode(source))
    builder = fld_model.GltfBuilder.create()
    builder.texture(textures[texture_index])
    builder.document["asset"]["extras"] = {
        "ddsTextureCount": len(source.textures)
    }
    builder.document["buffers"] = [{"byteLength": len(builder.binary)}]
    return builder.document, builder.binary


class TmxGltfImportTests(unittest.TestCase):
    def test_unchanged_embedded_png_preserves_every_bundle_byte(self) -> None:
        data, source = texture_source()
        document, binary = texture_glb(source)
        glb = fld_model.encode_glb(document, bytes(binary))
        document, binary = tmx_gltf_import.decode_glb(glb)

        rebuilt, summary = tmx_gltf_import.import_images(
            source, document, binary
        )

        self.assertEqual(tmx.encode(rebuilt), data)
        self.assertEqual(summary, tmx_gltf_import.ImportSummary(1, 0))

    def test_imports_changed_png_and_preserves_native_profile(self) -> None:
        _, source = texture_source()
        document, binary = texture_glb(source)
        image = document["images"][0]
        view = document["bufferViews"][image["bufferView"]]
        start = view["byteOffset"]
        end = start + view["byteLength"]
        width, height, rgba = tmx.decode_png(bytes(binary[start:end]))
        edited = bytearray(rgba)
        edited[:4] = bytes((9, 8, 7, 0xFF))
        png = tmx.encode_png(tmx.Texture(0, width, height, 0, bytes(edited)))
        binary[start:end] = png
        view["byteLength"] = len(png)
        document["buffers"][0]["byteLength"] = len(binary)

        rebuilt, summary = tmx_gltf_import.import_images(
            source, document, bytes(binary)
        )

        self.assertEqual(summary, tmx_gltf_import.ImportSummary(1, 1))
        texture = tmx.parse_bundle(tmx.encode(rebuilt))[0]
        self.assertEqual(texture.rgba[:4], bytes((9, 8, 7, 0xFF)))
        self.assertEqual(rebuilt.textures[0].psm, source.textures[0].psm)

    def test_imports_only_images_exported_from_a_larger_bundle(self) -> None:
        data, source = texture_source(2)
        document, binary = texture_glb(source, texture_index=1)
        rebuilt, summary = tmx_gltf_import.import_images(
            source, document, bytes(binary)
        )
        self.assertEqual(tmx.encode(rebuilt), data)
        self.assertEqual(summary, tmx_gltf_import.ImportSummary(1, 0))

    def test_rejects_native_metadata_and_texture_set_changes(self) -> None:
        _, source = texture_source()
        document, binary = texture_glb(source)
        document["images"][0]["extras"]["ddsTextureFlags"] ^= 1
        with self.assertRaisesRegex(
            tmx_gltf_import.TextureImportError, "native metadata differs"
        ):
            tmx_gltf_import.import_images(source, document, bytes(binary))

        document, binary = texture_glb(source)
        document["textures"].clear()
        with self.assertRaisesRegex(
            tmx_gltf_import.TextureImportError, "exported texture set"
        ):
            tmx_gltf_import.import_images(source, document, bytes(binary))

    def test_rejects_out_of_bounds_png_view(self) -> None:
        _, source = texture_source()
        document, binary = texture_glb(source)
        image = document["images"][0]
        document["bufferViews"][image["bufferView"]]["byteLength"] = len(binary) + 1
        with self.assertRaisesRegex(
            tmx_gltf_import.TextureImportError, "exceeds the GLB"
        ):
            tmx_gltf_import.import_images(source, document, bytes(binary))

    def test_tracked_field_bundles_round_trip_through_gltf(self) -> None:
        for version in ("dds1", "dds2"):
            base = ROOT / "src" / version / "data" / "field" / "f011_001"
            model_data = fld.encode(
                fld.parse_source(
                    base.with_suffix(".f1asm").read_text(encoding="utf-8")
                )
            )
            source = tmx.parse_source(
                base.with_suffix(".tbnasm").read_text(encoding="utf-8")
            )
            bundle_data = tmx.encode(source)
            document, binary = fld_model.build_gltf(
                model_data, textures=tmx.parse_bundle(bundle_data)
            )

            rebuilt, _ = tmx_gltf_import.import_images(source, document, binary)
            self.assertEqual(tmx.encode(rebuilt), bundle_data, version)


if __name__ == "__main__":
    unittest.main()
