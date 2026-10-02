#!/usr/bin/env python3
"""Import edited DDS texture images from a self-contained field GLB."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import tmx
from gltf_import import GltfImportError, buffer_view_data, decode_glb


class TextureImportError(GltfImportError):
    """Raised when GLB images cannot map safely to a DDS texture bundle."""


@dataclass(frozen=True)
class ImportSummary:
    images: int
    changed_images: int


def import_images(
    source: tmx.BundleSource,
    document: dict,
    binary: bytes,
) -> tuple[tmx.BundleSource, ImportSummary]:
    """Apply embedded PNG edits while preserving each native texture profile."""

    asset = document.get("asset")
    extras = asset.get("extras") if isinstance(asset, dict) else None
    texture_count = (
        extras.get("ddsTextureCount") if isinstance(extras, dict) else None
    )
    if (
        not isinstance(texture_count, int)
        or isinstance(texture_count, bool)
        or texture_count != len(source.textures)
    ):
        raise TextureImportError("GLB texture count does not match the bundle")

    images = document.get("images", [])
    textures = document.get("textures", [])
    if not isinstance(images, list) or not isinstance(textures, list):
        raise TextureImportError("GLB has an invalid image or texture array")
    if len(images) != len(textures):
        raise TextureImportError("GLB changes the exported texture set")

    textures_by_image: dict[int, dict] = {}
    for texture in textures:
        if not isinstance(texture, dict):
            raise TextureImportError("GLB has an invalid texture")
        image_index = texture.get("source")
        if (
            not isinstance(image_index, int)
            or isinstance(image_index, bool)
            or not 0 <= image_index < len(images)
        ):
            raise TextureImportError("GLB texture has an invalid image source")
        if image_index in textures_by_image:
            raise TextureImportError("GLB repeats an exported texture image")
        textures_by_image[image_index] = texture
    if len(textures_by_image) != len(images):
        raise TextureImportError("GLB leaves an exported image unreferenced")

    updated = source
    seen: set[int] = set()
    changed = 0
    for image_index, image in enumerate(images):
        if not isinstance(image, dict):
            raise TextureImportError("GLB has an invalid image")
        image_extras = image.get("extras")
        texture_index = (
            image_extras.get("ddsTextureIndex")
            if isinstance(image_extras, dict)
            else None
        )
        if (
            not isinstance(texture_index, int)
            or isinstance(texture_index, bool)
            or not 0 <= texture_index < len(source.textures)
        ):
            raise TextureImportError("GLB image has an invalid DDS texture index")
        if texture_index in seen:
            raise TextureImportError(f"GLB repeats DDS texture {texture_index}")
        seen.add(texture_index)

        native = source.textures[texture_index]
        name = f"texture_{texture_index:03d}"
        expected_metadata = {
            "ddsTextureIndex": texture_index,
            "ddsPixelStorageMode": tmx.PSM_NAMES[native.psm],
            "ddsMipmapCount": native.mipmap_count,
            "ddsClutStorageMode": native.clut_psm,
            "ddsTextureFlags": native.texture_flags,
        }
        texture = textures_by_image[image_index]
        if image.get("name") != name or texture.get("name") != name:
            raise TextureImportError(f"DDS texture {texture_index} changes its name")
        if not isinstance(image_extras, dict) or any(
            image_extras.get(key) != value
            for key, value in expected_metadata.items()
        ):
            raise TextureImportError(
                f"DDS texture {texture_index} native metadata differs"
            )
        if image.get("mimeType") != "image/png" or "uri" in image:
            raise TextureImportError(
                f"DDS texture {texture_index} is not an embedded PNG"
            )
        try:
            payload = buffer_view_data(
                document,
                binary,
                image.get("bufferView"),
                f"DDS texture {texture_index}",
            )
            width, height, rgba = tmx.decode_png(payload)
            replaced = tmx.replace_base_image(
                updated, texture_index, width, height, rgba
            )
        except (GltfImportError, tmx.TmxError) as exc:
            raise TextureImportError(f"DDS texture {texture_index}: {exc}") from exc
        changed += replaced.textures[texture_index] != updated.textures[texture_index]
        updated = replaced

    return updated, ImportSummary(len(images), changed)


def _source(path: Path) -> tmx.BundleSource:
    if path.suffix.lower() == ".tbnasm":
        return tmx.parse_source(path.read_text(encoding="utf-8"))
    data = path.read_bytes()
    if data[8:12] != b"TXP0":
        raise tmx.TmxError("GLB texture import requires a complete TBN packet")
    return tmx.BundleSource(tmx._parse_bundle_records(data))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("input", type=Path, help="TBN binary or .tbnasm source")
    parser.add_argument("output", type=Path, help="output TBN binary or source")
    args = parser.parse_args()
    try:
        document, binary = decode_glb(args.scene.read_bytes())
        source, summary = import_images(_source(args.input), document, binary)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".tbnasm":
            args.output.write_text(tmx.render_bundle_source(source), encoding="utf-8")
        else:
            args.output.write_bytes(tmx.encode(source))
    except (GltfImportError, OSError, tmx.TmxError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"imported {summary.images} texture images; "
        f"changed {summary.changed_images}"
    )


if __name__ == "__main__":
    main()
