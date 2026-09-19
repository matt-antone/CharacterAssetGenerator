"""Canonical cell geometry. Every render, mask and assembly step agrees on this.

Carried over from KaraokeParty-Graphics, which had these numbers right.
"""

from __future__ import annotations

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


def subject_height_px(height_inches: float) -> int:
    """Crown-to-heel pixel span for a character of this physical height."""
    if height_inches <= 0:
        raise ValueError("height must be positive")
    return round(height_inches * PX_PER_INCH)


def anim_subject_height_px(height_inches: float) -> int:
    """Crown-to-heel pixel span in an animation frame, which spans 8'6\"."""
    if height_inches <= 0:
        raise ValueError("height must be positive")
    return round(height_inches * ANIM_PX_PER_INCH)


def parse_height(text: str) -> float:
    """Parse a spec height such as `5' 9"` or `6'` into inches."""
    feet, _, inches = text.partition("'")
    total = int(feet.strip()) * 12
    inches = inches.strip().rstrip('"').strip()
    if inches:
        total += int(inches)
    return float(total)
