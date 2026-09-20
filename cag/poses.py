"""The pose reference: one card per frame, cut from the bundle's traced footage.

MotionArtist ships one video frame per motion frame under `thumbs/`. That is
the pose reference. cag letterboxes each onto a card and hands the pieces over;
it does not draw poses.

It used to draw them. `skeleton.py` rebuilt each pose from the raw landmarks as
twelve line segments and a head circle, which threw away every hand, every
heel, and both girdles — and then the generator was asked to draw a dance whose
whole engine is the pelvis turning against the shoulders. Redrawing what the
bundle already ships is how that happened, so this reads and never redraws.

A drawn sprite sheet used to be the other way in, cut up by a declared grid.
Measured against the trace it carried 0.43-0.70 of the movement where the
photographs carry 0.9-1.2, so a set that ranked perfectly still read as a sway.
MotionArtist has since deleted its figure renderer and tiles its sheet from the
footage too, so nothing produced that input any more and the code that read it
is gone.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

#: One pose card. Four across and two down is 1536x1024, the landscape canvas the
#: generator draws a sheet on, so card N sits exactly where figure N is drawn. It
#: is also MotionArtist's tile shape (163x220) to within a pixel, so the figure
#: fills its card; the 560x560 sprite cell left it a third of the width.
CARD_WIDTH = 384
CARD_HEIGHT = 512


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
