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

#: An animation frame spans more world than a static cell does. The static sheet
#: is a portrait: the character stands, and 7'0" of canvas is all it ever needs.
#: A performance jumps, reaches and wears a hat, so the same 560px of canvas is
#: mapped to 10'0" instead, and every character is correspondingly smaller here.
#: It stays one foot taller than the static cell: the static sheet sets the scale,
#: and a frame must never render a character larger than the sheet it is measured
#: against.
ANIM_CELL_HEIGHT_INCHES = 120
ANIM_PX_PER_INCH = CELL_HEIGHT / ANIM_CELL_HEIGHT_INCHES  # 4.666...

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
    """Crown-to-heel pixel span in an animation frame, which spans 8' not 7'."""
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
