"""The cell finish: the video path's cut-outs made into clean pixel art.

A SCAIL frame is a soft illustration, and cutting it off the magenta leaves
magenta in it: a fringe on the silhouette, pockets where the backdrop shows
through hair. The cell finish runs on each set's cells — after the shrink,
because the render is drawn about 3.6x the cell's size and anything done to it
there is averaged away — in four steps, each for something measured on Belter's
`club-01` dance on `local16` (2026-09-26):

1. **Defringe.** Pixels tinted toward the backdrop (`tinted`) within two pixels
   of the transparency leave the figure; those deeper in take the average of
   their untinted neighbours. The fringe sat in the hair and on the silhouette.
2. **One palette per set,** 64 colours by median cut over the set reference's
   figure and every defringed cell of the set. From the key art alone (48
   colours) the denim had no near colour and came out as grey patches on the
   thigh; the set's own pixels give it one.
3. **Quantize, no dither,** and keep the defringed alpha as the silhouette,
   exactly. A majority filter over the silhouette, at 1 px, refilled the tip of
   the gap between the legs and painted it the palette's colour nearest the
   magenta, a pale pixel at the crotch.
4. **A 1 px black outline inside the silhouette,** so no figure grows by a
   pixel against the others or against the pose-edit path's cells.

The result is opaque or transparent, nothing between.

The cut-outs themselves are kept under `cells-cut/<set>/` and the finish writes
`cells/<set>/`, so it never reads its own output. `finish.sha` beside the cells
is a digest of the cut-outs, the set reference and `FINISH_VERSION`: a rebuild
that re-cuts the same cut-outs finishes nothing.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageFilter

from .snap import backdrop

#: Bumped whenever the finish changes, so every finished set is redone.
FINISH_VERSION = "finish/1"
#: Colours in a set's palette.
PALETTE_COLOURS = 64
#: Beside the finished cells: the digest they were finished under.
STAMP = "finish.sha"
#: A tinted pixel this close to the transparency is fringe, not figure: the
#: window of `MaxFilter`, so two pixels either way.
FRINGE_WINDOW = 5
#: The window an inner pocket takes its colour from (`BoxBlur` radius).
FILL_RADIUS = 3


def tinted(rgb: np.ndarray) -> np.ndarray:
    """Pixels leaning toward the magenta backdrop: red and blue both clear of green.

    Looser than `cag.snap.backdrop`, which finds the backdrop itself: this finds
    figure pixels the backdrop has bled into.
    """
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    return (np.minimum(r, b) - g > 22) & (b - g > 35)


def _mask_filter(mask: np.ndarray, image_filter: ImageFilter.Filter) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(image_filter)).astype(bool)


def erode(mask: np.ndarray, size: int = 3) -> np.ndarray:
    return _mask_filter(mask, ImageFilter.MinFilter(size))


def dilate(mask: np.ndarray, size: int = 3) -> np.ndarray:
    return _mask_filter(mask, ImageFilter.MaxFilter(size))


def defringe(cell: np.ndarray) -> np.ndarray:
    """An RGBA cell with the backdrop's tint taken out, and alpha made 0 or 255.

    Tinted pixels by the transparency leave the figure; tinted pixels inside it
    take the colour of their untinted neighbours.
    """
    a = cell.copy()
    fig = a[..., 3] > 128
    t = tinted(a[..., :3]) & fig
    fig &= ~(t & dilate(~fig, FRINGE_WINDOW))
    inner = t & fig
    if inner.any():
        ok = fig & ~t
        blur = ImageFilter.BoxBlur(FILL_RADIUS)
        num = np.stack(
            [
                np.asarray(Image.fromarray((a[..., c] * ok).astype(np.uint8)).filter(blur)).astype(float)
                for c in range(3)
            ],
            -1,
        )
        den = np.asarray(Image.fromarray((ok * 255).astype(np.uint8)).filter(blur)).astype(float)[..., None] / 255
        a[inner, :3] = (num / np.maximum(den, 1e-3)).clip(0, 255).astype(np.uint8)[inner]
    a[..., 3] = fig * 255
    return a


def set_palette(reference: Path, cells: Sequence[np.ndarray]) -> Image.Image | None:
    """The set's palette: the set reference's figure and every defringed cell's opaque pixels.

    None when there is not one figure pixel to build it from.
    """
    with Image.open(reference) as image:
        rgb = np.asarray(image.convert("RGB"))
    pool = np.concatenate([rgb[~backdrop(rgb)]] + [cell[..., :3][cell[..., 3] > 0] for cell in cells])
    if not len(pool):
        return None
    return Image.fromarray(pool.reshape(1, -1, 3)).quantize(
        colors=PALETTE_COLOURS, method=Image.Quantize.MEDIANCUT
    )


def finish_cell(cell: np.ndarray, palette: Image.Image | None) -> np.ndarray:
    """A defringed cell quantized to `palette`, outlined inside its own silhouette."""
    fig = cell[..., 3] > 0
    if palette is None:
        q = cell[..., :3].copy()
    else:
        q = np.asarray(
            Image.fromarray(np.ascontiguousarray(cell[..., :3]))
            .quantize(palette=palette, dither=Image.Dither.NONE)
            .convert("RGB")
        ).copy()
    q[fig & ~erode(fig)] = 0
    return np.dstack([q, fig.astype(np.uint8) * 255])


def finish_key(cuts: dict[int, Path], reference: Path) -> str:
    """Everything a set's finished cells are a function of."""
    digest = hashlib.sha256(FINISH_VERSION.encode() + b"\0" + Path(reference).read_bytes())
    for index, path in sorted(cuts.items()):
        digest.update(f"\0{index}\0".encode())
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def finish_set(
    cuts: dict[int, Path],
    dst_for: Callable[[int], Path],
    reference: Path,
    label: str = "",
) -> dict[int, Path]:
    """Finish one set's cut-outs into its cells, unless they already are.

    `cuts` are the cut-outs by frame index, `dst_for` a frame's finished cell,
    and `reference` the set reference, on its magenta backdrop.
    """
    cells = {index: dst_for(index) for index in sorted(cuts)}
    if not cells:
        return cells
    stamp = next(iter(cells.values())).parent / STAMP
    key = finish_key(cuts, reference)
    if (
        stamp.exists()
        and stamp.read_text().strip() == key
        and all(path.exists() for path in cells.values())
    ):
        print(f"[{label}] cell finish cached", file=sys.stderr, flush=True)
        return cells
    stamp.unlink(missing_ok=True)
    defringed = {}
    for index, path in sorted(cuts.items()):
        with Image.open(path) as image:
            defringed[index] = defringe(np.asarray(image.convert("RGBA")))
    palette = set_palette(reference, list(defringed.values()))
    for index, cell in defringed.items():
        cells[index].parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(finish_cell(cell, palette)).save(cells[index], format="PNG")
    stamp.write_text(key + "\n")
    print(
        f"[{label}] cell finish: {len(cells)} cells on one {PALETTE_COLOURS}-colour palette",
        file=sys.stderr,
        flush=True,
    )
    return cells
