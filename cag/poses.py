"""The pose reference, cut out of the bundle's own sprite sheet.

MotionArtist renders every frame of a set into one sheet: hands, feet hinged at
the ball, the pelvis and rib-cage boxes that carry the twist, head facing, and
the character's own left and right limbs in different colours so a crossed limb
says which side it passes on. That sheet is the pose reference. cag cuts it up
and hands the pieces over; it does not draw poses.

It used to draw them. `skeleton.py` rebuilt each pose from the raw landmarks as
twelve line segments and a head circle, which threw away every hand, every
heel, and both girdles — and then the generator was asked to draw a dance whose
whole engine is the pelvis turning against the shoulders. Redrawing what the
bundle already ships is how that happened, so this reads and never redraws.

The layout is declared, never measured. The manifest's `spritesheet` block
gives the grid and the tile geometry in SVG units, and `scale` converts to PNG
pixels. Recovering that from the image — thresholding the backdrop and reading
the gaps between frame numbers — worked, but it made cag a second guess at
MotionArtist's renderer, which is the same mistake `skeleton.py` was.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

#: One pose card. Four across and two down is 1536x1024, the landscape canvas the
#: generator draws a sheet on, so card N sits exactly where figure N is drawn. It
#: is also MotionArtist's tile shape (163x220) to within a pixel, so the figure
#: fills its card; the 560x560 sprite cell left it a third of the width.
CARD_WIDTH = 384
CARD_HEIGHT = 512


class PoseSheetError(ValueError):
    """Raised when a sprite sheet cannot be cut into one tile per frame."""


@dataclass(frozen=True)
class SheetLayout:
    """Where every frame sits on the sprite sheet, in PNG pixels.

    MotionArtist states this in SVG units and the sheet is rendered at `scale`,
    so every measurement is multiplied on the way in and nothing downstream has
    to remember which space it is in.
    """

    columns: int
    rows: int
    tile_w: float
    tile_h: float
    cell_w: float
    cell_h: float
    label_h: float

    @classmethod
    def from_manifest(cls, block: dict) -> "SheetLayout":
        missing = {"cols", "rows", "tile_w", "tile_h", "cell_w", "cell_h"} - set(block)
        if missing:
            raise PoseSheetError(
                f"the manifest's spritesheet block is missing {', '.join(sorted(missing))}"
            )
        scale = float(block.get("scale", 1))
        return cls(
            columns=int(block["cols"]),
            rows=int(block["rows"]),
            tile_w=float(block["tile_w"]) * scale,
            tile_h=float(block["tile_h"]) * scale,
            cell_w=float(block["cell_w"]) * scale,
            cell_h=float(block["cell_h"]) * scale,
            # A sheet rendered with --no-labels reserves no band and says so.
            label_h=float(block.get("label_h", 0)) * scale,
        )

    def box(self, index: int) -> tuple[int, int, int, int]:
        """One frame's figure, with the frame-number band left behind.

        Tiles are on a fixed pitch and every frame is drawn against one shared
        bounding box, so the floor line lands at the same height in every cell.
        """
        row, column = divmod(index, self.columns)
        left = column * self.tile_w
        top = row * self.tile_h + self.label_h
        return (round(left), round(top), round(left + self.cell_w), round(top + self.cell_h))


def cut(sheet_path: Path | str, layout: SheetLayout, count: int) -> list[Image.Image]:
    """The first `count` figures of a sprite sheet, in frame order."""
    if count > layout.columns * layout.rows:
        raise PoseSheetError(
            f"{Path(sheet_path).name} is a {layout.columns}x{layout.rows} grid, too small "
            f"for {count} frames"
        )
    with Image.open(sheet_path) as handle:
        sheet = handle.convert("RGB")
        # The sheet's canvas stops after the last tile rather than after a
        # trailing gap, so the final row and column can fall a few pixels short
        # of the declared pitch. Nothing is drawn out there, but the crop still
        # has to be told it may come back small.
        return [sheet.crop(layout.box(index)) for index in range(count)]


def write_photos(photos: list[Path] | tuple[Path, ...], out_dir: Path | str) -> list[Path]:
    """Letterbox each traced video frame onto a card, numbered in frame order.

    One scale for the whole set, as `write_poses` does, so the performer keeps
    the sizes they were really filmed at from one frame to the next. Fitting
    each frame to its own card would flatten exactly the travel the sheet is
    there to show.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [Image.open(photo).convert("RGB") for photo in photos]
    # Never past 1: a thumbnail smaller than the card is pasted at the size it
    # was shipped at, letterboxed. Blowing it up adds no detail, softens the
    # edges the pose is read from, and is not what the amplitude was measured
    # on — those renders were fed thumbnails at their native 200px.
    scale = min(
        CARD_WIDTH / max(f.width for f in frames),
        CARD_HEIGHT / max(f.height for f in frames),
        1.0,
    )

    paths = []
    for index, frame in enumerate(frames):
        size = (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
        card = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0))
        card.paste(frame.resize(size, Image.LANCZOS), ((CARD_WIDTH - size[0]) // 2, 0))
        path = out_dir / f"{index:02d}.png"
        card.save(path, format="PNG")
        paths.append(path)
        frame.close()
    return paths


def write_poses(
    sheet_path: Path | str, layout: SheetLayout, count: int, out_dir: Path | str
) -> list[Path]:
    """Cut the sheet into one card per frame, numbered in frame order.

    Every tile is scaled by the same factor, so the figures keep the sizes they
    were drawn at relative to each other — the whole set is laid out at one
    scale, and fitting each frame to the cell on its own would throw that away.
    Cells are not square, so the figure is letterboxed rather than stretched.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = cut(sheet_path, layout, count)
    backdrop = cells[0].getpixel((0, 0))
    scale = min(CARD_WIDTH / layout.cell_w, CARD_HEIGHT / layout.cell_h)

    paths = []
    for index, cell in enumerate(cells):
        size = (max(1, round(cell.width * scale)), max(1, round(cell.height * scale)))
        card = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), backdrop)
        card.paste(cell.resize(size, Image.LANCZOS), ((CARD_WIDTH - size[0]) // 2, 0))
        path = out_dir / f"{index:02d}.png"
        card.save(path, format="PNG")
        paths.append(path)
    return paths
