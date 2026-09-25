"""Put a smooth render onto the key art's pixel grid and palette.

A pose workflow keeps the character but not the art: Qwen-Image-Edit-2511 with
the AnyPose LoRAs re-posed Belter faithfully on 2026-09-25 and drew every frame
as a smooth cartoon, off the pixel grid, with its inked lines thin and its
backdrop drifted to violet. Nothing in the prompt brought the style back, so
the frame is put back on the key art's terms after it is drawn, deterministically
and without a model:

- **grid**: averaged down to the key art's pixel size, then scaled back up with
  hard nearest-neighbour edges;
- **ink**: a block that is mostly inked line takes the ink's colour rather than
  its average, so the seams, lapels and features a plain average dissolves
  stay readable;
- **palette**: every pixel snapped to the key art's own colours, so a frame
  cannot drift from the picture it copies;
- **outline and backdrop**: a one-pixel black silhouette outline, as the style
  asks for, on exact #FF00FF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

#: The key art's pixel, in canvas pixels. Measured on Belter's Qwen-Image-2.1
#: key art at 1024x1536 by averaging down and back up: the error stayed near 3
#: to a block of 4 and jumped to 4.8 at 5.
BLOCK = 4

#: How many colours the key art's figure is reduced to for the palette.
COLOURS = 48

#: A block at least this much inked line is drawn as ink.
INK_SHARE = 0.25

MAGENTA = (255, 0, 255)
OUTLINE = (0, 0, 0)


def backdrop(a: np.ndarray) -> np.ndarray:
    """Magenta to violet: red and blue both well clear of green.

    Wide enough for the violet a pose render drifts to; the crimson, denim and
    skin of a costume all fail it on one channel or another.
    """
    r, g, b = (a[..., i].astype(int) for i in range(3))
    return (r - g > 90) & (b - g > 90) & (g < 90) & (b > 120)


def _luminance(a: np.ndarray) -> np.ndarray:
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def ink(image: Image.Image) -> np.ndarray:
    """Inked lines: dark, and clearly darker than what surrounds them.

    The second test is what keeps a dark top or a boot from reading as one
    solid block of line.
    """
    lum = _luminance(np.asarray(image).astype(float))
    around = _luminance(np.asarray(image.filter(ImageFilter.GaussianBlur(6))).astype(float))
    return (lum < 80) & (lum < around - 30)


def palette(key_art: Path | str) -> Image.Image:
    """The key art's figure, reduced to `COLOURS` colours, as a palette image."""
    with Image.open(key_art) as image:
        a = np.asarray(image.convert("RGB"))
    figure = a[~backdrop(a)]
    return Image.fromarray(figure.reshape(1, -1, 3)).quantize(
        colors=COLOURS, method=Image.Quantize.MEDIANCUT
    )


def _blocks(x: np.ndarray, h: int, w: int) -> np.ndarray:
    """(H, W, ...) canvas pixels -> (h, w, BLOCK*BLOCK, ...) per grid pixel."""
    x = x[: h * BLOCK, : w * BLOCK]
    x = x.reshape(h, BLOCK, w, BLOCK, *x.shape[2:]).swapaxes(1, 2)
    return x.reshape(h, w, BLOCK * BLOCK, *x.shape[4:])


def _mask(mask: np.ndarray, kind: ImageFilter.Filter) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(kind)) > 127


def snap(image: Image.Image, key_palette: Image.Image) -> Image.Image:
    """`image` on the key art's grid and palette, outlined, on exact magenta."""
    image = image.convert("RGB")
    a = np.asarray(image).astype(float)
    w, h = image.width // BLOCK, image.height // BLOCK

    pixels, inked = _blocks(a, h, w), _blocks(ink(image), h, w)
    ink_colour = (pixels * inked[..., None]).sum(axis=2) / np.maximum(
        inked.sum(axis=2), 1
    )[..., None]
    colour = np.where(
        (inked.mean(axis=2) >= INK_SHARE)[..., None], ink_colour, pixels.mean(axis=2)
    )
    grid = Image.fromarray(colour.clip(0, 255).astype(np.uint8))
    out = np.asarray(
        grid.quantize(palette=key_palette, dither=Image.Dither.NONE).convert("RGB")
    ).copy()

    # The figure is what most of a block is not backdrop. A majority filter then
    # drops the stray single pixels that leave a hair edge ragged.
    figure = _blocks(~backdrop(a.astype(np.uint8)), h, w).mean(axis=2) > 0.5
    figure = _mask(figure, ImageFilter.ModeFilter(3))
    grown = _mask(figure, ImageFilter.MaxFilter(3))
    out[~grown] = MAGENTA
    out[grown & ~figure] = OUTLINE
    return Image.fromarray(out).resize((w * BLOCK, h * BLOCK), Image.Resampling.NEAREST)


def snap_file(path: Path, key_art: Path | str) -> Path:
    """Snap the render at `path` in place, keeping the render as drawn beside it.

    The render moves to `<name>.raw.png` first; a frame that already has one was
    snapped on an earlier run and is left alone, so a resumed set is not
    snapped twice.
    """
    raw = path.with_name(f"{path.stem}.raw.png")
    if raw.exists():
        return path
    path.replace(raw)
    with Image.open(raw) as image:
        snap(image, palette(key_art)).save(path, format="PNG")
    return path
