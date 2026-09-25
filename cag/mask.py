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
factor drifts from frame to frame — and the frame spans 8'6" of world where the
static cell spans 9'. So animation measures per frame after all, but along the
motion sheet's own skeleton rather than the box: heel to crown through the
bones, a length the pose cannot change. See `frame_scale`.
"""

from __future__ import annotations

import io
import statistics
from pathlib import Path
from typing import Callable

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
#: Alpha at or below this counts as background when measuring the subject.
ALPHA_FLOOR = 8

#: Head radius, in body_h units, for finding the crown above the ears.
HEAD_RADIUS = 0.07

#: Ankle joint height above the floor, in body_h units. MediaPipe gives no heel
#: landmark, so the last span from ankle down to the ground is assumed.
# ponytail: anthropometric average; tune per rig if a character's feet read wrong.
ANKLE_RISE = 0.039

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

#: On a sheet with no landmarks, frames within this factor of the shortest
#: figure count as standing. Measured standing frames of one sheet spread about
#: 3%; arms overhead added 6-15%.
STANDING_SPREAD = 1.05


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
    # An airborne frame is lifted off the contact row, and a lift that pushes the
    # crown past the top of the cell cut hype-man's head off on three hiphop-04
    # frames. The figure stays whole; the lift gives way.
    # ponytail: clamps the top only; a figure too tall for the cell still clips at the bottom.
    offset_y = max(contact_row + 1 - bottom, -top)
    cell.paste(subject, (offset_x, offset_y))
    return cell


def mask_to_cell(
    src: Path | str, dst: Path | str, scale: float, contact_row: int = CONTACT_ROW
) -> Path:
    """Cut `src` out and save it registered into the cell at `dst`."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    register(cutout(src), scale, contact_row).save(dst, format="PNG")
    return dst


def mid(a: list[float], b: list[float]) -> list[float]:
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]


def span(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def crown(pts: dict[str, list[float]], body_h: float) -> list[float]:
    """Top of the skull: one head radius past the ears, along the neck's own axis.

    Not straight up. A tilted head carries its crown sideways, and measuring to a
    point directly above the ears would lose exactly the height the tilt costs.
    """
    ear = mid(pts["earL"], pts["earR"])
    neck = mid(pts["shL"], pts["shR"])
    dx, dy = ear[0] - neck[0], ear[1] - neck[1]
    reach = span(ear, neck)
    if reach < 1e-9:
        return [ear[0], ear[1] - HEAD_RADIUS * body_h]
    step = HEAD_RADIUS * body_h / reach
    return [ear[0] + dx * step, ear[1] + dy * step]


def stature(pts: dict[str, list[float]], body_h: float) -> float:
    """Heel-to-crown measured along the bones, so the pose cannot change it.

    Summing segment lengths is what makes this pose-invariant. A vertical extent
    shortens the moment a character crouches or leans, and normalising on one
    inflates them back to full height; the leg is the same leg either way.

    The supporting leg is measured, the torso down the midline, and the two are
    added rather than walked through — a chain that detoured via the supporting
    hip would pick up half a pelvis width that is not part of anyone's height.
    """
    lower = "L" if pts["anL"][1] > pts["anR"][1] else "R"
    ankle, knee, hip = pts[f"an{lower}"], pts[f"kn{lower}"], pts[f"hip{lower}"]
    leg = ANKLE_RISE * body_h + span(ankle, knee) + span(knee, hip)
    torso = span(mid(pts["hipL"], pts["hipR"]), mid(pts["shL"], pts["shR"]))
    return leg + torso + span(mid(pts["shL"], pts["shR"]), crown(pts, body_h))


def pose_extent(pts: dict[str, list[float]], floor_y: float, body_h: float) -> float:
    """Vertical span of one pose, crown and floor included: what gets drawn."""
    ys = [point[1] for point in pts.values()] + [floor_y, crown(pts, body_h)[1]]
    return max(ys) - min(ys)


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


#: The four joints that box the torso. Their mean is where the body is, whatever
#: the arms and legs are doing.
TORSO = ("shL", "shR", "hipL", "hipR")

#: Every landmark that can be the lowest point of a foot.
SOLES = ("anL", "anR", "heelL", "heelR", "toeL", "toeR")


def torso_x(pts: dict[str, list[float]]) -> float:
    return statistics.fmean(pts[name][0] for name in TORSO)


def home_x(poses: dict[int, dict[str, list[float]]]) -> float | None:
    """Where the performer's torso sits on average across the set, or None."""
    xs = [torso_x(pts) for pts in poses.values() if pts]
    return statistics.fmean(xs) if xs else None


def placement(
    subject: Image.Image,
    pts: dict[str, list[float]],
    floor_y: float,
    body_h: float,
    height_inches: float,
    home: float | None,
    airborne: bool,
) -> tuple[tuple[float, float] | None, int]:
    """Where a drawn frame goes in its cell: `register`'s anchor and contact row.

    Across: the torso is found in the drawing and put where the pose says the
    torso is, so a kicked leg widens the box without shoving the body the other
    way, and the performer's own side-to-side travel survives. Up: a frame the
    tracer calls airborne is lifted off the contact row by as far as its lowest
    sole is off the floor. Grounded frames are not lifted at all — their soles
    sit a few percent off `floor_y` by noise, and that would come out as jitter.
    """
    if not pts or not body_h or home is None:
        return None, ANIM_CONTACT_ROW
    px = anim_subject_height_px(height_inches) / body_h
    row = ANIM_CONTACT_ROW
    if airborne:
        lowest = max(pts[name][1] for name in SOLES if name in pts)
        row -= max(0, round((floor_y - lowest) * px))

    # The torso's rows in the drawing, read off the pose. The box runs crown to
    # lowest landmark: a generator draws feet, not the floor under a jump.
    left, top, right, bottom = subject_box(subject)
    ys = [point[1] for point in pts.values()] + [crown(pts, body_h)[1]]
    low, high = min(ys), max(ys)
    rows = [
        top + round((statistics.fmean(pts[a][1] for a in pair) - low) / (high - low) * (bottom - top))
        for pair in (TORSO[:2], TORSO[2:])
    ]
    alpha = numpy.array(subject.convert("RGBA").getchannel("A")) > ALPHA_FLOOR
    # Row by row, the middle of the widest unbroken run is the torso; the median
    # over the band outvotes the few rows an outstretched arm is joined to it.
    # Counting every pixel instead let that arm drag the body 6px on the shuffle.
    middles = []
    for line in alpha[min(rows) : max(rows) + 1]:
        filled = numpy.flatnonzero(line)
        if len(filled):
            run = max(numpy.split(filled, numpy.flatnonzero(numpy.diff(filled) > 1) + 1), key=len)
            middles.append((run[0] + run[-1]) / 2)
    if not middles:
        return None, row
    return (statistics.median(middles), CELL_WIDTH / 2 + (torso_x(pts) - home) * px), row


def pose_to_cell(
    src: Path | str,
    dst: Path | str,
    pts: dict[str, list[float]],
    floor_y: float,
    body_h: float,
    height_inches: float,
    fallback: float,
    home: float | None = None,
    airborne: bool = False,
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
    anchor, row = placement(subject, pts, floor_y, body_h, height_inches, home, airborne)
    register(subject, scale, row, anchor).save(dst, format="PNG")
    return dst


#: A run of backdrop rows or columns thinner than this does not separate two
#: figures. The gap between a raised arm and its own torso is backdrop too.
MIN_GAP = 12


def _runs(filled: numpy.ndarray, min_gap: int = MIN_GAP) -> list[tuple[int, int]]:
    """Spans of True in `filled`, as (start, end), gaps shorter than `min_gap` bridged."""
    runs: list[list[int]] = []
    start = None
    for i, on in enumerate(filled):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(filled)])
    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < min_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [(a, b) for a, b in merged]


def _row_runs(row: numpy.ndarray) -> list[tuple[int, int]]:
    """Spans of True in one row of the mask, as (start, end)."""
    edges = numpy.flatnonzero(numpy.diff(numpy.concatenate(([0], row.view(numpy.int8), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def _blobs(character: numpy.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the touching regions in the mask, as (left, top, right, bottom).

    Row runs are labelled and any two that touch between one row and the next
    are joined, so a region is followed whatever shape it takes. A figure is
    usually one region; a detached spike of hair or a held prop is its own.
    """
    parent: dict[int, int] = {}

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    spans: list[tuple[int, int, int, int]] = []
    previous: list[tuple[int, int, int]] = []
    for y in range(character.shape[0]):
        current = []
        for start, end in _row_runs(character[y]):
            label = len(spans)
            parent[label] = label
            spans.append((start, y, end, y + 1))
            for was_start, was_end, was_label in previous:
                if start < was_end and was_start < end:
                    a, b = root(was_label), root(label)
                    if a != b:
                        parent[b] = a
            current.append((start, end, label))
        previous = current
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for label, span in enumerate(spans):
        owner = root(label)
        if owner in boxes:
            have = boxes[owner]
            boxes[owner] = (
                min(have[0], span[0]),
                min(have[1], span[1]),
                max(have[2], span[2]),
                max(have[3], span[3]),
            )
        else:
            boxes[owner] = span
    return list(boxes.values())


#: A region smaller than this share of a figure's area is a piece of one, not a
#: figure: a detached hair spike, a thrown highlight, a prop held clear of the hand.
FRAGMENT_SHARE = 0.25


def find_figures(character: numpy.ndarray, count: int) -> list[tuple[int, int, int, int]]:
    """Figure boxes in reading order, found from the mask alone.

    Splitting the sheet by rows and columns of backdrop cannot separate two
    figures whose boxes overlap — a raised hand reaching up beside the boots of
    the figure above it leaves no clear row to cut along. So regions are
    followed by touch instead, and the small ones are folded into the nearest
    figure so a detached hair spike never counts as one.
    """
    regions = _blobs(character)
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])  # noqa: E731
    ranked = sorted(regions, key=area, reverse=True)
    biggest = ranked[:count] or ranked
    if not biggest:
        return []
    median = sorted(area(box) for box in biggest)[len(biggest) // 2]
    figures = [list(box) for box in ranked if area(box) >= FRAGMENT_SHARE * median]
    for left, top, right, bottom in ranked[len(figures) :]:
        def apart(figure: list[int]) -> int:
            across = max(figure[0] - right, left - figure[2], 0)
            down = max(figure[1] - bottom, top - figure[3], 0)
            return across * across + down * down

        nearest = min(figures, key=apart)
        nearest[0] = min(nearest[0], left)
        nearest[1] = min(nearest[1], top)
        nearest[2] = max(nearest[2], right)
        nearest[3] = max(nearest[3], bottom)
    return [tuple(box) for box in _reading_order(figures)]


def _reading_order(figures: list[list[int]]) -> list[list[int]]:
    """Left to right within a row, row by row down the sheet.

    A row is gathered by where each figure's middle falls, not by its top: two
    figures stand side by side in the same row while one crouches and the other
    reaches, and sorting on the top edge alone would shuffle them together with
    the row below.
    """
    ordered: list[list[int]] = []
    row: list[list[int]] = []
    floor = 0
    for figure in sorted(figures, key=lambda box: (box[1] + box[3]) / 2):
        middle = (figure[1] + figure[3]) / 2
        if row and middle > floor:
            ordered.extend(sorted(row, key=lambda box: box[0]))
            row, floor = [], 0
        if not row:
            floor = figure[3]
        row.append(figure)
    ordered.extend(sorted(row, key=lambda box: box[0]))
    return ordered


def slice_frame_sheet(sheet: Path | str, count: int, pad: int = 2 * BORDER_PIXELS) -> list[Image.Image]:
    """Cut one render holding several figures into one source image per figure.

    Figures are found, never assumed: regions of character are followed through
    the backdrop and come out in reading order. Each is set on fresh backdrop
    with a border wide enough for `backdrop_colour` to read, so nothing
    downstream can tell it from a figure rendered alone.
    """
    with Image.open(sheet) as image:
        source = image.convert("RGB")
    backdrop = backdrop_colour(source)
    if not keyable(backdrop):
        raise MaskError(f"{Path(sheet).name}: backdrop has no colour to find the figures against")
    rgb = numpy.array(source, dtype=numpy.float64)
    character = backdrop_share(rgb, backdrop) < KEY_THRESHOLD
    boxes = find_figures(character, count)
    if len(boxes) != count:
        raise MaskError(f"{Path(sheet).name} holds {len(boxes)} figures, not {count}")
    fill = tuple(int(v) for v in backdrop)
    cells = []
    for box in boxes:
        figure = source.crop(box)
        cell = Image.new("RGB", (figure.width + 2 * pad, figure.height + 2 * pad), fill)
        cell.paste(figure, (pad, pad))
        cells.append(cell)
    return cells


def set_to_cells(
    sources: dict[int, Path],
    dst_for: Callable[[int], Path],
    poses: dict[int, dict[str, list[float]]],
    floor_y: float,
    body_h: float,
    height_inches: float,
    frame_sheets: list[list[int]] | None = None,
    airborne: frozenset[int] = frozenset(),
) -> dict[int, Path]:
    """Register frames drawn together, one scale per sheet they were drawn on.

    `airborne` names the frames the tracer says have both feet off the floor;
    see `placement` for what is done with them.

    Figures on one sheet were drawn at one magnification, so one factor is
    measured per sheet — the median of what each frame's pose says — and
    applied to every frame from it. Measuring every frame on its own, as
    `pose_to_cell` does, would put measurement noise back between frames the
    generator had already drawn the same size. Two frame sheets of the same set do
    not share a magnification, though: the first came back 8% larger than the
    second, and one factor across both put a size pop at the seam.

    `frame_sheets` lists the frame indices drawn together; absent, every frame is
    taken as one sheet.

    A sheet with no landmarks has nothing to read a pose from, and the key
    art's factor is no use either: it assumes the key art's magnification, and
    a figure sharing a canvas with seven others is drawn a third that size. So
    the sheet is its own ruler: the largest group of frames drawn within a few
    percent of each other's height is taken as standing, and its median is
    standing height; on a tie the shorter group wins. Arms raised overhead add
    up to a sixth to a figure's box, and a victory set that reached on half its
    frames had its median pulled up by that and came out 7% small. A single
    crouched frame is a group of one and is outvoted. A set that crouches on
    most of its frames would still read tall.
    """
    subjects = {index: cutout(source) for index, source in sorted(sources.items())}
    cells = {}
    home = home_x(poses)
    for group in frame_sheets or [list(subjects)]:
        if body_h and all(poses.get(index) for index in group):
            scale = statistics.median(
                frame_scale(subjects[index], poses[index], floor_y, body_h, height_inches)
                for index in group
            )
        else:
            # ponytail: largest height cluster as stature, O(n²) over <=16 frames; measure the key art's proportions if a set fools it
            heights = sorted(subject_box(subjects[index])[3] - subject_box(subjects[index])[1] for index in group)
            # max() keeps the first of equal clusters, and heights are sorted, so a tie goes to the shorter
            standing = max(([x for x in heights if h <= x <= h * STANDING_SPREAD] for h in heights), key=len)
            scale = anim_subject_height_px(height_inches) / statistics.median(standing)
        for index in group:
            dst = dst_for(index)
            dst.parent.mkdir(parents=True, exist_ok=True)
            anchor, row = placement(
                subjects[index], poses.get(index, {}), floor_y, body_h, height_inches,
                home, index in airborne,
            )
            register(subjects[index], scale, row, anchor).save(dst, format="PNG")
            cells[index] = dst
    return cells
