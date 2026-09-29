"""The detail ladder: detail levels 1-10 derived from one render, in equal steps.

Qwen-Image-2.1 ignores the detail level it is asked for: the sample brief's
key art drawn at levels 1-10 on one seed came back as ten copies of one
picture, level 1 as rich as level 10 (2026-09-29). So the levels are made, not
asked for. One high-detail render of the sample brief (Ellis) is cut out and,
per level, shrunk to that level's figure height, quantized to that level's
colour count with no dither, outlined inside its silhouette
(`finish.finish_cell`) and enlarged back with hard pixels. Height and colours
rise by one ratio per level, so each step is as noticeable as the last.

    uv run python -m cag.ladder <render.png>

writes `cag/references/detail-level-NN.png` (the figure on magenta, key-art
canvas) and `detail-level-NN-tiles.png` (squares of its materials, the local
sample) for every level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .finish import finish_cell
from .mask import cutout

REFERENCES = Path(__file__).parent / "references"

#: Figure height in art pixels and palette size at level 1 and level 10. The
#: user chose the floor (what had been level 7 of a 43-347 px ladder) and this
#: ladder's step, 1.26x a level, over one capped at today's cell (2026-09-29).
BOTTOM_HEIGHT, TOP_HEIGHT = 173, 1380
BOTTOM_COLOURS, TOP_COLOURS = 32, 256

CANVAS = (1024, 1536)
MAGENTA = (255, 0, 255)
#: Where the figure's feet sit on the canvas, as in a key art.
FOOT_ROW = 1456

#: Squares cut from the finished figure for the local sample, as fractions of
#: its box: (centre x, centre y): knit, belt and buckle, wool, boot. Never the
#: head: a square of it carried the eyes, and a face is what gets copied.
TILES = [(0.5, 0.26), (0.5, 0.425), (0.36, 0.72), (0.3, 0.95)]
TILE_SIDE = 0.1   # of the figure's height
TILE_SHOWN = 320


def ramp(level: int, low: float, high: float) -> int:
    """Geometric: each level the same ratio above the last."""
    return round(low * (high / low) ** ((level - 1) / 9))


def rung(figure: Image.Image, level: int) -> Image.Image:
    """`figure` (RGBA, cropped) at `level`, enlarged back to TOP_HEIGHT with hard pixels."""
    h = ramp(level, BOTTOM_HEIGHT, TOP_HEIGHT)
    colours = ramp(level, BOTTOM_COLOURS, TOP_COLOURS)
    w = max(1, round(figure.width * h / figure.height))
    rgba = np.asarray(figure.resize((w, h), Image.LANCZOS)).copy()
    fig = rgba[..., 3] > 127
    rgba[..., 3] = fig * 255
    palette = Image.fromarray(np.ascontiguousarray(rgba[..., :3][fig]).reshape(1, -1, 3)).quantize(
        colors=colours, method=Image.Quantize.MEDIANCUT
    )
    done = Image.fromarray(finish_cell(rgba, palette), "RGBA")
    return done.resize((round(w * TOP_HEIGHT / h), TOP_HEIGHT), Image.NEAREST)


def on_canvas(figure: Image.Image) -> Image.Image:
    canvas = Image.new("RGBA", CANVAS, MAGENTA + (255,))
    x = (CANVAS[0] - figure.width) // 2
    canvas.alpha_composite(figure, (x, FOOT_ROW - figure.height))
    return canvas.convert("RGB")


def tiles(figure: Image.Image) -> Image.Image:
    side = round(figure.height * TILE_SIDE)
    sheet = Image.new("RGB", CANVAS, MAGENTA)
    for i, (cx, cy) in enumerate(TILES):
        x, y = round(figure.width * cx - side / 2), round(figure.height * cy - side / 2)
        square = Image.new("RGBA", (side, side), MAGENTA + (255,))
        square.alpha_composite(figure.crop((x, y, x + side, y + side)))
        square = square.convert("RGB")
        sheet.paste(square.resize((TILE_SHOWN, TILE_SHOWN), Image.NEAREST),
                    (160 + (i % 2) * 384, 300 + (i // 2) * 480))
    return sheet


def write(render: Path, into: Path = REFERENCES) -> list[Path]:
    figure = cutout(render)
    figure = figure.crop(figure.getbbox())
    written = []
    for level in range(1, 11):
        shown = rung(figure, level)
        for path, image in ((into / f"detail-level-{level:02d}.png", on_canvas(shown)),
                            (into / f"detail-level-{level:02d}-tiles.png", tiles(shown))):
            image.save(path)
            written.append(path)
    return written


if __name__ == "__main__":
    for path in write(Path(sys.argv[1])):
        print(path)
