# Field texture bundles (`TBN`, `TXP0`, and `TMX0`)

`tools/tmx.py` assembles exact texture source and decodes the texture bundle
paired with a field's FLD1 models. The extractor accepts either the complete
type-9 TBN packet or its embedded TXP0 payload and writes one RGBA PNG per
texture:

```sh
python3 tools/tmx.py extract extracted/f011_001.tbn textures/f011_001
```

The decoder uses only the Python standard library. It checks every packet
boundary, texture offset, header profile, palette size, mip level, and pixel
payload before writing output.

## Exact source

A complete TBN packet can be converted to text and rebuilt byte for byte:

```sh
python3 tools/tmx.py disassemble extracted/f011_001.tbn f011_001.tbnasm
python3 tools/tmx.py assemble f011_001.tbnasm rebuilt/f011_001.tbn
python3 tools/tmx.py verify extracted/f011_001.tbn
```

The source records the image dimensions, GS pixel mode, palette format, mip
count, texture flags, and exact packed image payload. Packet sizes, offsets,
type tags, fixed words, and alignment are derived by the assembler:

```text
tbn 1
texture index=0 width=128 height=128 psm=PSMT8 mipmaps=0 clut=PSMCT32 flags=0x0f65
data 0000000000000000000000000000000000000000000000000000000000000000
...
end_texture
```

The packed data stays hexadecimal so that it preserves palettes, indexed
pixels, mip levels, and console-native channel values exactly. PNG extraction
provides the convenient visual representation. The paired DDS1 and DDS2
sources under `src/<v>/data/field/` participate in the normal field-data build
and replace the TBN entry when the exact LB archive is rebuilt.

Every archived bundle has also passed the same source round trip: 551 DDS1
bundles containing 6,395 textures and 566 DDS2 bundles containing 9,111
textures. Tracking all of that packed image data as hexadecimal would expand
about 631 MB of binaries to about 1.38 GB of text, so the repository carries
paired source examples while the complete corpus remains the format-coverage
gate.

## Container layout

A field TBN is a normal eight-byte SDF resource packet followed by TXP0. TXP0
contains a texture count and file-relative offsets to ordered type-2 TMX0
packets. Model material fields `primary_texture` and `secondary_texture`
select entries in that order. The TMX0 image header is `0x38` bytes from its
magic through the start of image data; including the child packet header,
pixel data begins at offset `0x40`.

The complete DDS1 and DDS2 field corpora contain 1,117 bundles and 15,506
textures. One DDS2 bundle is intentionally empty. All 154,642 texture
references in archived FLD1 materials resolve within the corresponding
bundle.

## Pixel profiles

| GS mode | Payload |
|---|---|
| `PSMCT32` (`0x00`) | Four-byte RGBA pixels |
| `PSMCT24` (`0x01`) | Three-byte RGB pixels; decoded alpha is opaque |
| `PSMCT16` (`0x02`) | Packed RGB5A1 pixels |
| `PSMT8` (`0x13`) | 256-entry RGBA32 palette followed by byte indices |
| `PSMT4` (`0x14`) | 16-entry RGBA32 or RGB5A1 palette followed by low-nibble-first indices |

PSMT8 palettes use the GS CSM1 order, so entries 8–15 and 16–23 in every
32-entry group are exchanged when converted to linear index order. PS2 alpha
runs from 0 through 128; PNG output expands that range to 0 through 255.

Some field textures contain one additional mip level. The TMX0 header records
the count and the payload stores successively halved images after the base
level, sharing the indexed texture's palette. PNG output represents the base
level, while the decoder retains the mip count and texture control word for
model-export metadata.

## Texture editing

One decoded PNG can be imported into an exact bundle source or binary while
retaining that texture's dimensions, GS pixel format, CLUT format, flags, and
existing mip levels:

```sh
python3 tools/tmx.py import src/dds1/data/field/f011_001.tbnasm \
  edited-000.png edited-f011_001.tbnasm --texture 0
python3 tools/tmx.py assemble edited-f011_001.tbnasm edited-f011_001.tbn
```

The replacement must be a non-interlaced 8-bit PNG with the original width
and height. Direct-color textures convert into their existing `PSMCT32`,
`PSMCT24`, or `PSMCT16` profile; 24-bit input must remain opaque, and PS2
alpha or RGB5A1 channels use their native quantization. Indexed textures keep
their original palette and accept only colors already present in it. This
fails visibly instead of silently choosing a new palette or changing material
indices. Mip payloads remain byte-identical because a single replacement image
edits only the base level.

Writing an output whose name ends in `.tbnasm` produces canonical source;
other output names produce an assembled TBN. Importing an unmodified PNG from
`extract` recreates every supported native pixel profile byte-for-byte.

An edited model or composed-scene GLB can return all of its embedded PNGs to
the exact bundle in one pass:

```sh
python3 tools/tmx_gltf_import.py edited-field.glb \
  src/dds1/data/field/f011_001.tbnasm edited-f011_001.tbnasm
```

Only textures referenced by the exported models appear in the GLB; all other
bundle entries stay byte-exact. Each embedded image retains its native texture
index, pixel mode, CLUT mode, mip count, and flags as identity metadata. The
importer requires that identity and the exported image set to remain intact,
then applies the same lossless profile checks as single-PNG import. This lets a
scene editor change several textures without extracting or addressing them
separately.

## Model export

`tools/fld_model.py` accepts an LB archive directly, or a loose F1/F1 source
plus `--texture-bundle`. It follows each model draw through its material asset
to the ordered TBN entry and embeds referenced PNGs in the self-contained GLB.
Secondary texture identities are retained as material metadata until their
blend operation is recovered. See [`fld.md`](fld.md#model-export) for the
complete command and model conversion rules.
