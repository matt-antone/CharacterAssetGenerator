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

import numpy
from PIL import Image

from .geometry import CELL_HEIGHT, CELL_WIDTH, CONTACT_ROW, subject_height_px

#: Alpha at or below this counts as background when measuring the subject.
ALPHA_FLOOR = 8

#: How close a pixel must sit to the measured backdrop colour to be read as
#: backdrop rather than costume, as a max per-channel difference.
BACKDROP_TOLERANCE = 24

#: Width of the border band the backdrop colour is measured from.
BORDER_PIXELS = 8

#: Alpha this high is rounded up to solid. A sprite should not come out faintly
#: transparent throughout.
ALPHA_CEILING = 250

#: Colour distance from the backdrop at which a pixel counts as fully subject.
#: Between the tolerance and this, alpha ramps, which keeps antialiased edges
#: soft rather than jagged.
#:
#: ponytail: a fixed threshold, because the obvious alternative is worse.
#: Scaling the ramp to the image's own contrast made almost the whole subject
#: semi-transparent — 478k partial pixels against 2.5k. The ceiling is that an
#: edge pixel half covered by a high-contrast colour is already past this
#: distance, so it reads as solid and keeps a little backdrop tint. On pixel art
#: with hard edges that is a couple of thousand pixels and invisible; on soft
#: painterly art it would need real chroma maths keyed to the backdrop hue.
SUBJECT_DISTANCE = 72

#: A key that keeps less than this share of the canvas found no character, so
#: the render falls back to Vision.
MIN_SUBJECT_SHARE = 0.02


class MaskError(RuntimeError):
    """Raised when no subject could be separated from the background."""


def cutout(src: Path | str) -> Image.Image:
    """Return `src` as RGBA with the background removed.

    Keys out the backdrop sampled from the border, the way the old project did
    and with no cleanup afterwards. Vision is the fallback for a render the key
    cannot read — it segments the subject semantically, so it survives a
    backdrop the character happens to share a colour with.
    """
    with Image.open(src) as image:
        source = image.convert("RGB")
        keyed = key_out(source, backdrop_colour(source))
    share = numpy.count_nonzero(numpy.array(keyed)[:, :, 3]) / (keyed.width * keyed.height)
    if share >= MIN_SUBJECT_SHARE:
        return keyed
    return vision_cutout(src)


def key_out(image: Image.Image, backdrop: numpy.ndarray) -> Image.Image:
    """Remove a flat backdrop by colour, and divide its share back out."""
    rgb = numpy.array(image.convert("RGB"), dtype=numpy.float64)
    distance = numpy.max(numpy.abs(rgb - backdrop), axis=2)
    alpha = numpy.clip(
        (distance - BACKDROP_TOLERANCE) / (SUBJECT_DISTANCE - BACKDROP_TOLERANCE), 0, 1
    )
    alpha[alpha * 255 >= ALPHA_CEILING] = 1.0

    subject = numpy.divide(
        rgb - (1 - alpha[:, :, None]) * backdrop,
        alpha[:, :, None],
        out=numpy.zeros_like(rgb),
        where=alpha[:, :, None] > 0,
    )
    pixels = numpy.concatenate(
        [numpy.clip(subject, 0, 255), (alpha * 255)[:, :, None]], axis=2
    )
    pixels[alpha == 0] = 0
    return Image.fromarray(pixels.astype(numpy.uint8), "RGBA")


def vision_cutout(src: Path | str) -> Image.Image:
    """Segment the subject with Apple Vision. macOS only."""
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
    masked = Image.open(io.BytesIO(bytes(data))).convert("RGBA")
    with Image.open(src) as source:
        return unmix(masked, backdrop_colour(source))


def backdrop_colour(image: Image.Image) -> numpy.ndarray:
    """The backdrop the character was drawn on, read off the source border.

    Measured rather than assumed. The old project keyed on magenta because its
    masker was a chroma key; Vision segments the subject instead, so the
    backdrop can be any colour and this only needs to know which.
    """
    pixels = numpy.array(image.convert("RGB"), dtype=numpy.int16)
    band = numpy.concatenate(
        [
            pixels[:BORDER_PIXELS].reshape(-1, 3),
            pixels[-BORDER_PIXELS:].reshape(-1, 3),
            pixels[:, :BORDER_PIXELS].reshape(-1, 3),
            pixels[:, -BORDER_PIXELS:].reshape(-1, 3),
        ]
    )
    return numpy.median(band, axis=0)


def unmix(image: Image.Image, backdrop: numpy.ndarray) -> Image.Image:
    """Remove the backdrop's contribution to the cutout.

    Three things, in order. Any pixel still the backdrop colour is background
    Vision enclosed inside the subject — the gap inside a hand holding a
    microphone — so it goes fully transparent. Every remaining partial pixel is
    a blend of subject over backdrop, so the backdrop's share is divided back
    out, which is what clears the coloured fringe along the outline. Finally
    near-solid alpha is rounded up, so the sprite is not faintly see-through.
    """
    pixels = numpy.array(image, dtype=numpy.float64)
    rgb, alpha = pixels[:, :, :3], pixels[:, :, 3]

    is_backdrop = numpy.max(numpy.abs(rgb - backdrop), axis=2) <= BACKDROP_TOLERANCE
    alpha[is_backdrop] = 0

    share = (alpha / 255)[:, :, None]
    subject = numpy.divide(
        rgb - (1 - share) * backdrop,
        share,
        out=numpy.zeros_like(rgb),
        where=share > 0,
    )

    alpha[alpha >= ALPHA_CEILING] = 255
    pixels[:, :, :3] = numpy.clip(subject, 0, 255)
    pixels[:, :, 3] = alpha
    pixels[alpha == 0] = 0
    return Image.fromarray(pixels.astype(numpy.uint8), "RGBA")




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
    # Nearest neighbour, per the rendering contract: smooth resampling would
    # blur the pixel grid that makes this arcade art rather than a cartoon.
    subject = subject.resize((width, height), Image.NEAREST)

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
