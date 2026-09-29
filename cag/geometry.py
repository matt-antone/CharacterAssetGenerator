"""Canonical cell geometry. Every render, mask and assembly step agrees on this.

Carried over from KaraokeParty-Graphics, which had these numbers right.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

#: A square cell. The width is free — scale is measured off the height alone —
#: so widening it costs nothing but gives a reaching or striding pose room that a
#: 480px portrait cropped into.
CELL_WIDTH = 560
CELL_HEIGHT = 560

#: The full cell height represents a 9'0" character: the figure is drawn at its true
#: scale and the rest is headroom, so a hat, a raised arm or a stride has somewhere
#: to go instead of meeting the edge.
CELL_HEIGHT_INCHES = 108
PX_PER_INCH = CELL_HEIGHT / CELL_HEIGHT_INCHES  # 5.185...
PX_PER_FOOT = PX_PER_INCH * 12  # 62.22...

#: Row the supporting heel sits on for grounded poses.
CONTACT_ROW = 550

#: An animation frame maps the same 560px of canvas to 8'6" of world, bottom edge
#: to top. That is half a foot less than the static cell, so a character renders
#: slightly larger in an animation frame than on the static sheet.
ANIM_CELL_HEIGHT_INCHES = 102
ANIM_PX_PER_INCH = CELL_HEIGHT / ANIM_CELL_HEIGHT_INCHES  # 5.490...

#: The floor sits this far up from the bottom edge, so a heel is never flush
#: against it and a shadow or a trailing foot has somewhere to go.
ANIM_FLOOR_MARGIN_INCHES = 6
ANIM_CONTACT_ROW = CELL_HEIGHT - round(ANIM_FLOOR_MARGIN_INCHES * ANIM_PX_PER_INCH)

#: Render backdrop. Generators drift off the exact value; masking tolerates that.
MAGENTA = "#FF00FF"


#: The detail ladder (`cag.ladder`): the sample figure's height in art pixels and
#: its palette at level 1 and level 10, rising by one ratio a level. The sample
#: is Ellis, 5'10", so a level is also a density in pixels per inch.
LADDER_HEIGHTS = (173, 1380)
LADDER_COLOURS = (32, 256)
LADDER_FIGURE_INCHES = 70

#: The cell finish's palette when no level says otherwise.
DEFAULT_COLOURS = 64


def ramp(level: int, low: float, high: float) -> int:
    """Geometric: each level the same ratio above the last."""
    return round(low * (high / low) ** ((level - 1) / 9))


@dataclass(frozen=True)
class Geometry:
    """One cell geometry: the square cell and every row and scale derived from it.

    The cell always spans the same world (9'0" static, 8'6" animated); only its
    pixel density changes, so a character stands the same share of every cell
    and a raised arm has the same headroom at any level.
    """

    px_per_inch: float
    #: The cell finish's palette size.
    colours: int = DEFAULT_COLOURS
    #: The detail level it was made for, if any.
    level: int | None = None

    @property
    def cell(self) -> int:
        return round(CELL_HEIGHT_INCHES * self.px_per_inch)

    @property
    def contact_row(self) -> int:
        return self.cell - round(self.cell * (CELL_HEIGHT - CONTACT_ROW) / CELL_HEIGHT)

    @property
    def anim_px_per_inch(self) -> float:
        return self.cell / ANIM_CELL_HEIGHT_INCHES

    @property
    def anim_contact_row(self) -> int:
        return self.cell - round(ANIM_FLOOR_MARGIN_INCHES * self.anim_px_per_inch)

    def subject_height_px(self, height_inches: float) -> int:
        if height_inches <= 0:
            raise ValueError("height must be positive")
        return round(height_inches * self.px_per_inch)

    def anim_subject_height_px(self, height_inches: float) -> int:
        if height_inches <= 0:
            raise ValueError("height must be positive")
        return round(height_inches * self.anim_px_per_inch)


#: The 560px cell every build used before detail levels drew their own, and what
#: anything outside a build still sees.
LEGACY = Geometry(PX_PER_INCH)


def at_level(level: int) -> Geometry:
    """The cell geometry that draws a character at detail `level` of the ladder."""
    if not 1 <= level <= 10:
        raise ValueError(f"detail level must be 1-10, not {level!r}")
    return Geometry(
        ramp(level, *LADDER_HEIGHTS) / LADDER_FIGURE_INCHES,
        ramp(level, *LADDER_COLOURS),
        level,
    )


def from_cell(size: int) -> Geometry:
    """The geometry of a package whose cells are `size` px square (its manifest's `cell`)."""
    return Geometry(size / CELL_HEIGHT_INCHES)


_CURRENT: ContextVar[Geometry] = ContextVar("geometry", default=LEGACY)


def current() -> Geometry:
    """The geometry in force: a build's detail level inside `using`, else `LEGACY`."""
    return _CURRENT.get()


@contextmanager
def using(geometry: Geometry):
    """Draw everything inside at `geometry`."""
    token = _CURRENT.set(geometry)
    try:
        yield geometry
    finally:
        _CURRENT.reset(token)


def subject_height_px(height_inches: float) -> int:
    """Crown-to-heel pixel span for a character of this physical height."""
    return current().subject_height_px(height_inches)


def anim_subject_height_px(height_inches: float) -> int:
    """Crown-to-heel pixel span in an animation frame, which spans 8'6\"."""
    return current().anim_subject_height_px(height_inches)


def parse_height(text: str) -> float:
    """Parse a spec height such as `5' 9"` or `6'` into inches."""
    feet, _, inches = text.partition("'")
    total = int(feet.strip()) * 12
    inches = inches.strip().rstrip('"').strip()
    if inches:
        total += int(inches)
    return float(total)
