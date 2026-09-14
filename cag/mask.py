"""Cut the character off the magenta backdrop and register it into the cell.

The cutout is Apple's Vision framework, not a chroma key: it segments the
foreground subject directly, so a magenta-ish costume colour cannot eat a limb.
macOS only.

Scale is deliberately *not* derived per frame. A crouching pose has a shorter
bounding box than a standing one, so fitting every frame to the same box would
shrink and grow the character as it moves. One scale factor comes from the key
art and every frame of that character uses it.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from .geometry import CELL_HEIGHT, CELL_WIDTH, CONTACT_ROW, subject_height_px

#: Alpha at or below this counts as background when measuring the subject.
ALPHA_FLOOR = 8


class MaskError(RuntimeError):
    """Raised when no subject could be separated from the background."""


def cutout(src: Path | str) -> Image.Image:
    """Return `src` as RGBA with everything but the foreground subject removed."""
    import Quartz  # noqa: PLC0415 - macOS frameworks, imported at call time
    import Vision
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(Path(src).resolve()))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
    request = Vision.VNGenerateForegroundInstanceMaskRequest.alloc().init()
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise MaskError(f"Vision request failed for {src}: {error}")

    observations = request.results()
    if not observations:
        raise MaskError(f"no foreground subject found in {src}")

    observation = observations[0]
    buffer, error = observation.generateMaskedImageOfInstances_fromRequestHandler_croppedToInstancesExtent_error_(
        observation.allInstances(), handler, False, None
    )
    if buffer is None:
        raise MaskError(f"Vision could not render a masked image for {src}: {error}")

    ci_image = Quartz.CIImage.imageWithCVPixelBuffer_(buffer)
    cg_image = Quartz.CIContext.context().createCGImage_fromRect_(ci_image, ci_image.extent())
    data = Quartz.CFDataCreateMutable(None, 0)
    destination = Quartz.CGImageDestinationCreateWithData(data, "public.png", 1, None)
    Quartz.CGImageDestinationAddImage(destination, cg_image, None)
    Quartz.CGImageDestinationFinalize(destination)
    return Image.open(io.BytesIO(bytes(data))).convert("RGBA")


def subject_box(image: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of the visible subject, as PIL's exclusive (l, t, r, b)."""
    box = image.convert("RGBA").getchannel("A").point(lambda a: 255 * (a > ALPHA_FLOOR)).getbbox()
    if box is None:
        raise MaskError("cutout is fully transparent")
    return box


def key_art_scale(key_art: Image.Image, height_inches: float) -> float:
    """Pixels-per-source-pixel that puts this character at their real height.

    Read once from a standing key art, then reused for every later render of
    the same character so that scale never drifts between poses or views.
    """
    left, top, right, bottom = subject_box(key_art)
    source_height = bottom - top
    if source_height <= 0:
        raise MaskError("key art subject has no height")
    return subject_height_px(height_inches) / source_height


def register(image: Image.Image, scale: float) -> Image.Image:
    """Place a cutout on the canonical cell: scaled, centred, heel on the baseline."""
    left, top, right, bottom = subject_box(image)
    subject = image.crop((left, top, right, bottom))
    width = max(1, round(subject.width * scale))
    height = max(1, round(subject.height * scale))
    subject = subject.resize((width, height), Image.LANCZOS)

    # Resampling softens the edges, so measure the subject again and place it by
    # what is actually visible rather than by the resized canvas.
    left, top, right, bottom = subject_box(subject)

    # ponytail: every frame lands its lowest visible pixel on the contact row.
    # Airborne frames would need the motion sheet's own floor offset instead.
    # Pasted without a mask: the cell is empty, and using the subject as its own
    # mask would multiply alpha by itself and eat the antialiased edge.
    cell = Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
    cell.paste(subject, ((CELL_WIDTH - (right - left)) // 2 - left, CONTACT_ROW + 1 - bottom))
    return cell


def mask_to_cell(src: Path | str, dst: Path | str, scale: float) -> Path:
    """Cut `src` out and save it registered into the cell at `dst`."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    register(cutout(src), scale).save(dst, format="PNG")
    return dst
