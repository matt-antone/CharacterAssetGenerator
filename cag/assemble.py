"""Turn registered cells into the things a game and a reviewer actually use.

The sprite sheet is the deliverable and keeps its alpha. The proof is a looping
GIF flattened onto grey — it exists so a person can watch the motion and check
the loop seam, which a grid of stills cannot show.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from PIL import Image

from .geometry import CELL_HEIGHT, CELL_WIDTH

#: Flat backdrop for the proof only. Never applied to a delivered cell.
PROOF_BACKDROP = (68, 68, 68)


def sprite_sheet(
    cells: Sequence[Path | str],
    dst: Path | str,
    columns: int | None = None,
    cell: tuple[int, int] = (CELL_WIDTH, CELL_HEIGHT),
) -> Path:
    """Lay the cells out left to right, top to bottom, in frame order."""
    width, height = cell
    cells = [Path(cell) for cell in cells]
    if not cells:
        raise ValueError("no cells to assemble")
    columns = columns or len(cells)
    rows = -(-len(cells) // columns)

    sheet = Image.new("RGBA", (width * columns, height * rows), (0, 0, 0, 0))
    for index, path in enumerate(cells):
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(f"{path} is {image.size}, not the {width}x{height} cell")
            sheet.paste(image.convert("RGBA"), (index % columns * width, index // columns * height))

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, format="PNG")
    return dst


def split_sheet(sheet: Path | str, count: int, columns: int | None = None) -> list[Image.Image]:
    """Cut a sprite sheet back into cells. The inverse of `sprite_sheet`."""
    columns = columns or count
    with Image.open(sheet) as image:
        image = image.convert("RGBA")
        return [
            image.crop(
                (
                    index % columns * CELL_WIDTH,
                    index // columns * CELL_HEIGHT,
                    (index % columns + 1) * CELL_WIDTH,
                    (index // columns + 1) * CELL_HEIGHT,
                )
            )
            for index in range(count)
        ]


def gif_proof(cells: Sequence[Path | str], dst: Path | str, fps: int, loop: bool = True) -> Path:
    """A watchable loop at the declared rate, flattened onto grey."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    frames = []
    for cell in cells:
        with Image.open(cell) as image:
            flat = Image.new("RGB", image.size, PROOF_BACKDROP)
            flat.paste(image.convert("RGBA"), (0, 0), image.convert("RGBA"))
            frames.append(flat.convert("P", palette=Image.ADAPTIVE))
    if not frames:
        raise ValueError("no cells to assemble")

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        dst,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0 if loop else 1,
    )
    return dst


#: Portrait side as a share of the figure's height: crown to about mid-chest.
PORTRAIT_SHARE = 0.3

#: The game's two portrait sizes, by the unit name its files carry, drawn at 4x.
PORTRAIT_SIZES = {34: 136, 80: 320}


def portrait(key_cell: Path | str, dst: Path | str, size: int) -> Path:
    """Cut a square head-and-shoulders portrait from the key art's cell.

    Cut, not drawn: portraits drawn on their own drifted off-model and nothing
    said so. Centred on the head, read as the top eighth of the figure, then
    resized nearest-neighbour to `size`, so every character ships the same
    dimensions whatever their height.
    """
    image = Image.open(key_cell).convert("RGBA")
    alpha = image.getchannel("A")
    _, top, _, bottom = alpha.getbbox()
    side = round((bottom - top) * PORTRAIT_SHARE)
    # ponytail: head read as the top eighth's box; a tall hair tower or a raised prop drags the centre
    head = alpha.crop((0, top, image.width, top + max(1, (bottom - top) // 8))).getbbox()
    centre = (head[0] + head[2]) // 2
    left = min(max(0, centre - side // 2), image.width - side)
    top = max(0, top - side // 20)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    crop = image.crop((left, top, left + side, top + side))
    crop.resize((size, size), Image.Resampling.NEAREST).save(dst)
    return dst


GALLERY = """<!doctype html>
<meta charset="utf-8"><title>{name}</title>
<style>
 body{{background:#1b1b1f;color:#e8e6ef;font:15px/1.5 system-ui,sans-serif;margin:0;padding:32px}}
 h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:14px;text-transform:uppercase;
   letter-spacing:.1em;color:#9a96ac;margin:36px 0 12px;font-weight:600}}
 p{{color:#9a96ac;margin:0}}
 .views{{display:flex;flex-wrap:wrap;gap:16px}}
 figure{{margin:0}} figcaption{{color:#9a96ac;font-size:12px;margin-top:6px}}
 img{{background:
   conic-gradient(#2a2a31 25%,#23232a 0 50%,#2a2a31 0 75%,#23232a 0) 0 0/24px 24px;
   border-radius:4px;max-width:100%}}
 .views img{{height:280px;width:auto}} .sheet{{overflow-x:auto}}
</style>
<h1>{name}</h1><p>{height} &middot; {description}</p>
<h2>Projection</h2><div class="views">{views}</div>
{sets}
"""

SET_BLOCK = """<h2>{set_name} &middot; {frames} frames at {fps} fps</h2>
<figure><img src="{proof}" alt="{set_name} loop"><figcaption>proof, {fps} fps</figcaption></figure>
<div class="sheet"><img src="{sheet}" alt="{set_name} sprite sheet"></div>
"""


MANIFEST = "manifest.json"


def manifest(dst: Path | str, name: str, height: str, views: dict, sets: list) -> Path:
    """What a front end needs to play this character: fps, frame counts, file names.

    The fps here is the one the frames were rendered for, and `cag edit` rewrites
    it when a proof is rebuilt at another rate, so the front end never has to
    guess a playback speed.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        json.dumps(
            {
                "name": name,
                "height": height,
                "cell": [CELL_WIDTH, CELL_HEIGHT],
                "views": {view: str(path) for view, path in views.items()},
                "sets": {
                    block["set_name"]: {
                        key: block[key] for key in ("frames", "fps", "columns", "sheet", "proof")
                    }
                    for block in sets
                },
            },
            indent=2,
        )
        + "\n"
    )
    return dst


def gallery(dst: Path | str, name: str, height: str, description: str, views: dict, sets: list) -> Path:
    """A page for looking at what came out. Paths are relative to `dst`."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        GALLERY.format(
            name=name,
            height=height,
            description=description,
            views="".join(
                f'<figure><img src="{path}" alt="{view}"><figcaption>{view}</figcaption></figure>'
                for view, path in views.items()
            ),
            sets="".join(SET_BLOCK.format(**block) for block in sets),
        )
    )
    return dst
