"""Cut the character off the magenta backdrop and register it into the cell.

The cutout is a chroma key, with Apple's Vision framework behind it for a render
the key cannot read — Vision segments the subject semantically, so it survives a
backdrop the character happens to share a colour with. macOS only.

Both paths produce a hard matte and then cut one pixel into the character's
outline. The pixel where that outline was antialiased against the backdrop is a
real blend of the two and no threshold can separate it from costume in the same
hue, so it is thrown away rather than judged — `STYLE` mandates an outline about
two pixels wide so there is one to spare. That is what keeps magenta off the
outline; see `CUT_IN`.

Scale is never derived from a bounding box per frame. A crouching pose has a
shorter box than a standing one, so fitting every frame to the same box would
shrink and grow the character as it moves. A static view is always standing, so
one factor read off the key art covers the whole sheet.

An animation frame gets no such luxury. Each one is a separate generation, drawn
at whatever size the generator felt like, so a fixed pixels-per-source-pixel
factor drifts from frame to frame — and the frame spans 8' of world where the
static cell spans 7'. So animation measures per frame after all, but along the
motion sheet's own skeleton rather than the box: heel to crown through the
bones, a length the pose cannot change. See `frame_scale`.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy
from PIL import Image

from .geometry import (
    ANIM_CONTACT_ROW,
    CELL_HEIGHT,
    CELL_WIDTH,
    CONTACT_ROW,
    anim_subject_height_px,
    subject_height_px,
)
from .skeleton import pose_extent, stature

#: Alpha at or below this counts as background when measuring the subject.
ALPHA_FLOOR = 8

#: Width of the border band the backdrop colour is measured from.
BORDER_PIXELS = 8

#: Backdrop share above which a pixel counts as backdrop rather than character.
#: The matte is a straight in-or-out decision. This art has hard pixel edges by
#: contract and `register` resamples nearest-neighbour, so a soft alpha ramp is
#: mostly thrown away at the downscale anyway — and a ramp is what let the
#: half-magenta rim through as solid costume in the first place.
KEY_THRESHOLD = 0.5

#: How far into the outline the matte is cut back, in source pixels.
#:
#: The pixel where the outline was antialiased against the backdrop is a genuine
#: blend of the two. No threshold can clean it, because "backdrop darkened by the
#: outline" and "costume in the backdrop's hue" are the same colour — so it is
#: discarded rather than judged. `STYLE` mandates an outline about two pixels
#: wide precisely so there is one to spare; renders measure 4-6 source pixels of
#: it, well clear of this. An outline thinner than this would be eaten.
CUT_IN = 1

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


def backdrop_share(rgb: numpy.ndarray, backdrop: numpy.ndarray) -> numpy.ndarray:
    """How much of each pixel is backdrop, as a 0..1 map.

    Measured as how far the backdrop's two strong channels run ahead of its weak
    one, not as an RGB distance. That distinction is the whole point: where the
    character's black outline was antialiased against the backdrop, a rim pixel
    is literally half backdrop, but in plain RGB it sits ~95 away from magenta
    and read as solid costume. Darkening a colour does not change how far its
    channels are spread, so this sees that pixel for the half backdrop it is.

    Taking the *minimum* of the strong channels is what keeps costume out of it.
    A crimson jacket is bright in red but not in blue, so it scores 0.05 here,
    where projecting onto the backdrop's colour axis called it 0.37 backdrop and
    would have made the whole jacket translucent.

    A backdrop with no colour to it scores nothing anywhere; `keyable` is the
    guard for that case, not this.
    """
    weak, strong, spread = _channels(backdrop)
    if spread <= 0:
        return numpy.zeros(rgb.shape[:2])
    return numpy.clip((rgb[:, :, strong].min(axis=2) - rgb[:, :, weak]) / spread, 0, 1)


def keyable(backdrop: numpy.ndarray) -> bool:
    """Whether this backdrop has enough colour in it to key against at all.

    Black, white and grey have none: there is no channel spread to separate them
    from a subject, so keying would keep the whole canvas. `cutout` hands those
    to Vision instead.
    """
    return _channels(backdrop)[2] > 0


def _channels(backdrop: numpy.ndarray) -> tuple[int, list[int], float]:
    """The backdrop's weak channel, its two strong ones, and the gap between."""
    weak = int(numpy.argmin(backdrop))
    strong = [channel for channel in range(3) if channel != weak]
    return weak, strong, float(backdrop[strong].min() - backdrop[weak])


def cut_in(keep: numpy.ndarray, depth: int = CUT_IN) -> numpy.ndarray:
    """`keep` shrunk by `depth` pixels, taking the blended rim off with it."""
    for _ in range(depth):
        padded = numpy.pad(keep, 1, constant_values=False)
        keep = (
            padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:] & keep
        )
    return keep


def matte(rgb: numpy.ndarray, keep: numpy.ndarray) -> Image.Image:
    """`rgb` as RGBA: opaque where `keep`, fully transparent everywhere else."""
    pixels = numpy.concatenate([rgb, (keep * 255)[:, :, None]], axis=2)
    pixels[~keep] = 0
    return Image.fromarray(pixels.astype(numpy.uint8), "RGBA")


def key_out(image: Image.Image, backdrop: numpy.ndarray) -> Image.Image:
    """Remove a flat backdrop by colour, cutting one pixel into the outline."""
    rgb = numpy.array(image.convert("RGB"), dtype=numpy.float64)
    if not keyable(backdrop):
        # Nothing to key against, so key nothing and let `cutout` reach Vision.
        return Image.new("RGBA", image.size, (0, 0, 0, 0))
    return matte(rgb, cut_in(backdrop_share(rgb, backdrop) < KEY_THRESHOLD))


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
        return trim_matte(masked, backdrop_colour(source))


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


def trim_matte(image: Image.Image, backdrop: numpy.ndarray) -> Image.Image:
    """Clean up Vision's matte the same way the colour key cleans its own.

    Vision segments the subject but traces it a pixel wide, so the rim it hands
    back is the same backdrop-over-outline blend the colour key leaves behind,
    and it also encloses background the subject wraps around — the gap inside a
    hand holding a microphone. Both go the same way: anything still backdrop
    coloured is out, then one pixel is cut off the edge.
    """
    pixels = numpy.array(image, dtype=numpy.float64)
    rgb, alpha = pixels[:, :, :3], pixels[:, :, 3]
    keep = alpha > ALPHA_FLOOR
    if keyable(backdrop):
        keep &= backdrop_share(rgb, backdrop) < KEY_THRESHOLD
    return matte(rgb, cut_in(keep))


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


def register(
    image: Image.Image,
    scale: float,
    contact_row: int = CONTACT_ROW,
    anchor: tuple[float, float] | None = None,
) -> Image.Image:
    """Place a cutout on the canonical cell: scaled, heel on the baseline.

    `anchor` is `(source_x, cell_x)`: put the body point found at `source_x` in
    the incoming image on column `cell_x` of the cell. Pass it for anything that
    moves. Without it the bounding box is centred instead, and a box is a poor
    thing to line a body up by — an outflung arm widens it on one side only, so
    centring the box shoves the body the other way. Worse, it fights the
    performance: a step to the left widens the box to the left, which pushes the
    figure back to the right, and the motion comes out at half amplitude.

    The default suits a single standing view, where there is no motion to lose.
    """
    origin_left, top, right, bottom = subject_box(image)
    left = origin_left
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
    if anchor is None:
        offset_x = (CELL_WIDTH - (right - left)) // 2 - left
    else:
        source_x, cell_x = anchor
        offset_x = round(cell_x - (source_x - origin_left) * scale)
    cell.paste(subject, (offset_x, contact_row + 1 - bottom))
    return cell


def mask_to_cell(
    src: Path | str, dst: Path | str, scale: float, contact_row: int = CONTACT_ROW
) -> Path:
    """Cut `src` out and save it registered into the cell at `dst`."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    register(cutout(src), scale, contact_row).save(dst, format="PNG")
    return dst


def frame_scale(
    subject: Image.Image,
    pts: dict[str, list[float]],
    floor_y: float,
    body_h: float,
    height_inches: float,
) -> float:
    """Scale for one animation frame, read off the pose the frame was drawn from.

    The generator draws the pose it was shown, at whatever size it likes, so the
    cutout's box is the pose's full extent at an unknown magnification. The
    motion sheet knows that same pose exactly: what fraction of its extent is
    height rather than an outflung arm. Apply the fraction, and what falls out is
    the character's crown-to-heel in source pixels, whatever the generator did.

    This trusts the generator to have followed the skeleton's proportions, which
    is the reason a skeleton is attached to every frame. It does not trust it to
    have followed the skeleton's *size*, which it never has.
    """
    left, top, right, bottom = subject_box(subject)
    drawn = bottom - top
    extent = pose_extent(pts, floor_y, body_h)
    standing = stature(pts, body_h)
    if drawn <= 0 or extent <= 0 or standing <= 0:
        raise MaskError("frame has no measurable pose")
    return anim_subject_height_px(height_inches) / (drawn * standing / extent)


def pose_to_cell(
    src: Path | str,
    dst: Path | str,
    pts: dict[str, list[float]],
    floor_y: float,
    body_h: float,
    height_inches: float,
    fallback: float,
) -> Path:
    """Cut an animation frame out and register it at the scale its pose asks for.

    `fallback` covers a sheet that carries no landmarks: nothing to measure, so
    the key art's one factor is all there is.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subject = cutout(src)
    scale = fallback
    if pts and body_h:
        scale = frame_scale(subject, pts, floor_y, body_h, height_inches)
    register(subject, scale, ANIM_CONTACT_ROW).save(dst, format="PNG")
    return dst
