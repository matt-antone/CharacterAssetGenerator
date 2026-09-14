"""Canonical cell geometry. Every render, mask and assembly step agrees on this.

Carried over from KaraokeParty-Graphics, which had these numbers right.
"""

from __future__ import annotations

CELL_WIDTH = 480
CELL_HEIGHT = 560

#: The full cell height represents a 7'0" character.
CELL_HEIGHT_INCHES = 84
PX_PER_INCH = CELL_HEIGHT / CELL_HEIGHT_INCHES  # 6.666...
PX_PER_FOOT = PX_PER_INCH * 12  # 80.0

#: Row the supporting heel sits on for grounded poses.
CONTACT_ROW = 550

#: Render backdrop. Generators drift off the exact value; masking tolerates that.
MAGENTA = "#FF00FF"


def subject_height_px(height_inches: float) -> int:
    """Crown-to-heel pixel span for a character of this physical height."""
    if height_inches <= 0:
        raise ValueError("height must be positive")
    return round(height_inches * PX_PER_INCH)


def parse_height(text: str) -> float:
    """Parse a spec height such as `5' 9"` or `6'` into inches."""
    feet, _, inches = text.partition("'")
    total = int(feet.strip()) * 12
    inches = inches.strip().rstrip('"').strip()
    if inches:
        total += int(inches)
    return float(total)
