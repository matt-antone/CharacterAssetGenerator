"""The drive video: a bundle's source clip, cut and resampled for SCAIL-2.

SCAIL-2 animates the set reference by following a video of the performer, at
video rate. The traced frames are too sparse for that — club-01's fifteen are a
sixth of a second apart — so the drive is cut from the source clip itself: one
2:3 box for the whole clip, resampled to the drive rate, sized to the machine
profile, written as one animated PNG beside a drive mask of the performer's
silhouette.

Everything here is a function of the clip and the profile, not the character,
so one drive is shared by every character that dances the same bundle on the
same machine. Nothing but ffmpeg decoding touches the outside world; the mask
pass, when the footage needs one, is handed in by the caller.

What the research established, on club-01 only: a white studio backdrop makes a
threshold mask (`LIGHT`) good enough, the drive rate is 16, and the SCAIL frame
for traced frame i is `round((t_i - t_0) * rate)`. Footage without a light
backdrop needs the mask pass, which no render has exercised yet.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy
from PIL import Image, ImageSequence

from .mask import BORDER_PIXELS

if TYPE_CHECKING:
    from .machines import Machine
    from .motion import Clip, MotionSheet


class DriveError(RuntimeError):
    """Raised when a bundle's source clip cannot make a drive video."""


#: SCAIL-2's own ceiling on one job's frames (`WanSCAILToVideo.length`: step 4,
#: max 81). Past it a drive would have to be drawn in chunks, which is not built.
SCAIL_MAX_LENGTH = 81

#: Bumped whenever the way a drive is cut changes, so every cached drive goes
#: stale at once rather than one being quietly reused under the new rules.
DRIVE_VERSION = "drive/1"

#: The record a finished drive leaves; written last, so its presence means done.
DRIVE_RECORD = "drive.json"
DRIVE_VIDEO = "drive.png"
DRIVE_MASK = "drive-mask.png"
MASK_PASS_DIR = "mask-pass"

#: Channel value above which a pixel is backdrop-bright. The research rule: on
#: club-01's white studio a pixel is the performer when any channel is below it.
LIGHT = 200

#: Share of the drive's border pixels that must be backdrop-bright for the
#: footage to count as shot on a light backdrop. club-01 measures 1.00.
LIGHT_SHARE = 0.90

#: A drive mask frame whose figure covers less or more of the frame than this
#: has lost the performer or swallowed the backdrop. club-01 runs 0.20-0.23.
FIGURE_SHARE = (0.03, 0.60)

#: The largest touching region must hold this share of the figure. Less is two
#: people, or a backdrop that the threshold broke into pieces.
ONE_BODY = 0.80

#: Overlap with the frame before, as intersection over union. A silhouette that
#: jumps further than this between drive frames is a mask that lost track, not
#: a dance. club-01 at 16 fps never drops below 0.76.
STEADY = 0.5

#: The figure's colour in both masks, and the drive mask's background: the
#: research's convention, which SCAIL-2's animation mode reads for the drive.
MASK_FIGURE = (0, 0, 255)
DRIVE_MASK_BACKGROUND = (0, 0, 0)

#: The reference mask's background. SCAIL-2's own convention for animation mode
#: is white (`nodes_scail.py`); every render so far used black and worked, and
#: changing it needs two rolls, so it stays named here rather than inlined.
REF_MASK_BACKGROUND = (0, 0, 0)

#: The magenta backdrop the set reference is drawn on, see `draw.py`.
MAGENTA = (255, 0, 255)

#: Headroom on a frame count, so a span that fills a capped length exactly is
#: not pushed to the next 4k+1 by rounding error.
_EPSILON = 1e-6


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _length(span: float, rate: float) -> int:
    """Frames that cover `span` seconds at `rate`, rounded up to SCAIL's 4k+1."""
    steps = max(math.ceil(span * rate - _EPSILON), 0)
    return 4 * math.ceil(steps / 4) + 1


def plan(times: Sequence[float], m: Machine, name: str = "") -> tuple[float, int]:
    """The drive rate and length for traced frames at `times`, on machine `m`.

    The rate is the profile's, lowered only when the profile's length cap cannot
    hold the span at it. SCAIL's own 81-frame ceiling is measured at the full
    rate: a span that needs more is refused rather than drawn at a crawl.
    """
    label = name or "this bundle"
    if len(times) < 2:
        raise DriveError(f"{label}: a drive needs at least two traced frames with times")
    span = times[-1] - times[0]
    if span <= 0:
        raise DriveError(f"{label}: traced frame times do not advance ({times[0]} .. {times[-1]})")
    needed = math.ceil(span * m.rate - _EPSILON) + 1
    if _length(span, m.rate) > SCAIL_MAX_LENGTH:
        raise DriveError(
            f"{label}: {span:.3f} s at {m.rate} fps needs {needed} drive frames; SCAIL-2 draws "
            f"at most {SCAIL_MAX_LENGTH} in one job and chunking is not built"
        )
    cap = 4 * ((int(m.max_length) - 1) // 4) + 1
    if cap < 5:
        raise DriveError(f"{label}: a length cap of {m.max_length} leaves no room for a drive")
    rate = min(float(m.rate), (cap - 1) / span)
    return rate, _length(span, rate)


def trace_index(
    times: Sequence[float], rate: float, length: int, *, capped: bool = False
) -> list[int]:
    """The SCAIL frame at each traced frame's time: `round((t_i - t_0) * rate)`.

    Two traced frames landing on one SCAIL frame means the rate is too low to
    tell them apart. That is refused at the profile's rate and only reported
    under a capped one, where it is the price of fitting the length.
    """
    index = [round((t - times[0]) * rate) for t in times]
    if index[-1] > length - 1 or min(index) < 0:
        raise DriveError(
            f"trace index {index} runs outside the {length} drive frames at {rate:.3f} fps"
        )
    if any(b < a for a, b in zip(index, index[1:])):
        raise DriveError(f"traced frame times run backwards: {list(times)}")
    repeats = [i for i in range(1, len(index)) if index[i] == index[i - 1]]
    if repeats:
        message = (
            f"traced frames {repeats} share a SCAIL frame with the one before at "
            f"{rate:.3f} fps"
        )
        if not capped:
            raise DriveError(message)
        _log(f"{message}; the profile's length cap lowered the rate, so they repeat")
    return index


def decode(path: Path | str) -> numpy.ndarray:
    """Every frame of a video, in order, as an (n, h, w, 3) uint8 array.

    Decoded at the native rate with nothing dropped or doubled, so array index j
    is the clip's frame j. Frames come out as a PPM stream, which carries its own
    size, so a clip with rotation metadata is read the way ffmpeg displays it.
    """
    try:
        run = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
                "-map", "0:v:0", "-fps_mode", "passthrough",
                "-f", "image2pipe", "-c:v", "ppm", "-",
            ],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise DriveError("ffmpeg is not installed; the video path decodes source clips with it") from None
    except subprocess.CalledProcessError as exc:
        raise DriveError(f"ffmpeg could not decode {path}: {exc.stderr.decode().strip()}") from None
    data = run.stdout
    if not data.startswith(b"P6"):
        raise DriveError(f"ffmpeg returned no frames for {path}")
    fields = data.split(maxsplit=4)
    width, height = int(fields[1]), int(fields[2])
    header = len(b"P6 %d %d 255\n" % (width, height))
    step = header + width * height * 3
    if len(data) % step:
        raise DriveError(f"{path} decodes to frames of more than one size")
    frames = numpy.frombuffer(data, numpy.uint8).reshape(-1, step)[:, header:]
    return frames.reshape(-1, height, width, 3)


def source_frames(
    clip: Clip, t0: float, rate: float, length: int, available: int
) -> list[int]:
    """The clip frame under each drive frame: drive frame k is at `t0 + k / rate`.

    A drive frame past either end of the clip is refused. It is never filled by
    repeating the last frame: SCAIL follows the drive frame by frame, and a held
    frame is a performer who stopped.
    """
    picks = [clip.frame_at(t0 + k / rate) for k in range(length)]
    if picks[0] < 0 or picks[-1] > available - 1:
        raise DriveError(
            f"clip pad too short: the drive needs clip frames {picks[0]}..{picks[-1]} and "
            f"{clip.path} holds 0..{available - 1}"
        )
    return picks


def _place(centre: float, length: int, frame: int) -> int:
    """Start of a span of `length` centred on `centre`, kept inside the frame.

    A span longer than the frame is kept covering it instead, so the pad is only
    ever what the frame genuinely lacks.
    """
    low, high = sorted((0, frame - length))
    return min(max(round(centre - length / 2), low), high)


def drive_box(
    size: tuple[int, int], box: tuple[int, int, int, int] | None = None
) -> tuple[int, int, int, int]:
    """The 2:3 box the drive is cut from, as (x, y, w, h) in clip pixels.

    `box` is the clip box, in the same form. Landscape footage gives up its
    sides: full height, centred across on the clip box. Portrait footage (a
    Short) gives up its top and bottom: full width, one and a half widths tall,
    centred down on the clip box. Either way the box is clamped into the frame;
    if it is still bigger than the frame it runs past the edges, and `cut` pads
    what is missing.
    """
    width, height = size
    left, top, box_w, box_h = box if box else (0, 0, width, height)
    if width >= height:
        cut_w, cut_h = round(height * 2 / 3), height
    else:
        cut_w, cut_h = width, round(width * 3 / 2)
    x = _place(left + box_w / 2, cut_w, width)
    y = _place(top + box_h / 2, cut_h, height)
    return x, y, cut_w, cut_h


def border(frames: numpy.ndarray | Sequence[numpy.ndarray], depth: int = BORDER_PIXELS) -> numpy.ndarray:
    """Every pixel in a band `depth` wide round the edge of each frame, as (n, 3)."""
    stack = numpy.asarray(frames)
    return numpy.concatenate(
        [
            stack[:, :depth].reshape(-1, 3),
            stack[:, -depth:].reshape(-1, 3),
            stack[:, depth:-depth, :depth].reshape(-1, 3),
            stack[:, depth:-depth, -depth:].reshape(-1, 3),
        ]
    )


def cut(
    frame: numpy.ndarray, box: tuple[int, int, int, int], fill: Sequence[int]
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The (x, y, w, h) box out of one frame, and where it ran past the frame and was padded."""
    left, top, box_w, box_h = box
    right, bottom = left + box_w, top + box_h
    height, width = frame.shape[:2]
    out = numpy.empty((box_h, box_w, 3), numpy.uint8)
    out[:] = fill
    pad = numpy.ones(out.shape[:2], bool)
    x0, y0, x1, y1 = max(left, 0), max(top, 0), min(right, width), min(bottom, height)
    out[y0 - top : y1 - top, x0 - left : x1 - left] = frame[y0:y1, x0:x1]
    pad[y0 - top : y1 - top, x0 - left : x1 - left] = False
    return out, pad


def light_backdrop(frames: numpy.ndarray | Sequence[numpy.ndarray]) -> bool:
    """Was this footage shot against a light backdrop, so a threshold finds the performer?"""
    return float((border(frames).min(axis=1) > LIGHT).mean()) >= LIGHT_SHARE


def figure_mask(frame: numpy.ndarray, pad: numpy.ndarray | None = None) -> numpy.ndarray:
    """The performer on a light backdrop: any pixel with a channel below `LIGHT`.

    Padding is backdrop whatever colour it came out, so it is never figure.
    """
    figure = numpy.asarray(frame).min(axis=2) < LIGHT
    return figure & ~pad if pad is not None else figure


def _component_sizes(mask: numpy.ndarray) -> list[int]:
    """Pixel count of each touching region of `mask`, largest first.

    Row runs are labelled and joined wherever two overlap between one row and
    the next — the same walk as `mask._blobs`, counting pixels instead of boxes.
    """
    parent: list[int] = []
    sizes: list[int] = []

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    previous: list[tuple[int, int, int]] = []
    padded = numpy.zeros((mask.shape[0], mask.shape[1] + 2), numpy.int8)
    padded[:, 1:-1] = mask
    edges = numpy.diff(padded, axis=1)
    for y in range(mask.shape[0]):
        starts = numpy.flatnonzero(edges[y] == 1)
        ends = numpy.flatnonzero(edges[y] == -1)
        current = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            label = len(parent)
            parent.append(label)
            sizes.append(end - start)
            for was_start, was_end, was_label in previous:
                if start < was_end and was_start < end:
                    a, b = root(was_label), root(label)
                    if a != b:
                        parent[b] = a
            current.append((start, end, label))
        previous = current
    totals: dict[int, int] = {}
    for label, size in enumerate(sizes):
        owner = root(label)
        totals[owner] = totals.get(owner, 0) + size
    return sorted(totals.values(), reverse=True)


def mask_stats(masks: Sequence[numpy.ndarray]) -> list[dict[str, float | None]]:
    """Per drive mask frame: figure share, largest region's share, overlap with the one before."""
    stats: list[dict[str, float | None]] = []
    before = None
    for mask in masks:
        area = int(mask.sum())
        regions = _component_sizes(mask) if area else []
        overlap = None
        if before is not None:
            union = int((mask | before).sum())
            overlap = round(int((mask & before).sum()) / union, 4) if union else 0.0
        stats.append(
            {
                "figure": round(area / mask.size, 4),
                "largest": round(regions[0] / area, 4) if regions else 0.0,
                "iou": overlap,
            }
        )
        before = mask
    return stats


def check_masks(
    masks: Sequence[numpy.ndarray], stats: Sequence[dict[str, float | None]] | None = None
) -> list[str]:
    """What is wrong with a drive mask, frame by frame; empty when nothing is."""
    problems = []
    low, high = FIGURE_SHARE
    for k, stat in enumerate(stats if stats is not None else mask_stats(masks)):
        if not low <= stat["figure"] <= high:
            problems.append(
                f"frame {k:02d}: the figure covers {stat['figure']:.0%} of the frame, "
                f"outside {low:.0%}-{high:.0%}"
            )
        elif stat["largest"] < ONE_BODY:
            problems.append(
                f"frame {k:02d}: the largest region is {stat['largest']:.0%} of the figure, "
                f"under {ONE_BODY:.0%}; two people, or a broken mask"
            )
        if stat["iou"] is not None and stat["iou"] < STEADY:
            problems.append(
                f"frame {k:02d}: overlaps the frame before by {stat['iou']:.2f}, under {STEADY}"
            )
    return problems


def paint(mask: numpy.ndarray, background: Sequence[int] = DRIVE_MASK_BACKGROUND) -> Image.Image:
    """A mask as SCAIL-2 reads one: `MASK_FIGURE` on `background`."""
    out = numpy.empty(mask.shape + (3,), numpy.uint8)
    out[:] = background
    out[mask] = MASK_FIGURE
    return Image.fromarray(out, "RGB")


def reference_mask(image: Image.Image | Path | str) -> Image.Image:
    """The set reference's silhouette: everything but the magenta backdrop is figure."""
    if not isinstance(image, Image.Image):
        with Image.open(image) as opened:
            image = opened.convert("RGB")
    rgb = numpy.asarray(image.convert("RGB")).astype(numpy.int16)
    backdrop = (rgb[..., 0] > 170) & (rgb[..., 1] < 110) & (rgb[..., 2] > 170)
    return paint(~backdrop, REF_MASK_BACKGROUND)


def fit(reference: Image.Image, width: int, height: int) -> Image.Image:
    """The set reference at the drive's size, padded with magenta to its shape first.

    SCAIL-2 centre-crops a reference that is not the drive's shape, which would
    cut into the character; padding with the backdrop costs nothing but canvas.
    """
    image = reference.convert("RGB")
    w, h = image.size
    if w * height != h * width:
        wide = round(h * width / height)
        tall = round(w * height / width)
        canvas = (wide, h) if wide >= w else (w, tall)
        padded = Image.new("RGB", canvas, MAGENTA)
        padded.paste(image, ((canvas[0] - w) // 2, (canvas[1] - h) // 2))
        _log(f"set reference is {w}x{h}, not {width}:{height}; padded with magenta to {canvas[0]}x{canvas[1]}")
        image = padded
    return image.resize((width, height), Image.LANCZOS)


def write_apng(frames: Sequence[Image.Image], path: Path | str, ms: int) -> Path:
    """Frames as one animated PNG, every frame kept.

    Pillow merges a frame identical to the one before into a longer duration,
    and a drive one frame short leaves SCAIL's tail unguided — the research's
    mannequin drive lost a frame that way. So a repeat has the lowest bit of
    one pixel's blue flipped, which no model can see, and the count is checked.
    """
    images = [frame.convert("RGB").copy() for frame in frames]
    if not images:
        raise DriveError("no frames to write")
    for i in range(1, len(images)):
        if images[i].tobytes() == images[i - 1].tobytes():
            r, g, b = images[i].getpixel((0, 0))
            images[i].putpixel((0, 0), (r, g, b ^ 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path, format="PNG", save_all=True, append_images=images[1:], duration=ms, loop=0
    )
    with Image.open(path) as check:
        written = getattr(check, "n_frames", 1)
    if written != len(images):
        raise DriveError(f"{path} holds {written} frames, not {len(images)}")
    return path


def read_apng(path: Path | str) -> list[numpy.ndarray]:
    """Every frame of an animated PNG, as RGB arrays."""
    with Image.open(path) as image:
        return [numpy.asarray(frame.convert("RGB")) for frame in ImageSequence.Iterator(image)]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def drive_digest(
    clip: Clip, t0: float, rate: float, length: int, width: int, height: int
) -> str:
    """Everything a drive is a function of. Change any of it and the drive is redrawn."""
    parts = (
        clip.sha256 or _sha(clip.path),
        f"{clip.start:.6f}",
        f"{clip.t_offset:.6f}",
        list(clip.box) if clip.box else None,
        f"{t0:.6f}",
        f"{rate:.6f}",
        length,
        width,
        height,
        DRIVE_VERSION,
    )
    return hashlib.sha256("\t".join(map(str, parts)).encode()).hexdigest()


@dataclass(frozen=True)
class Drive:
    """A drive video and its drive mask, ready to hand to SCAIL-2."""

    root: Path
    digest: str
    rate: float
    length: int
    size: tuple[int, int]
    #: The 2:3 box the drive was cut from, as (x, y, w, h) in source-clip pixels.
    box: tuple[int, int, int, int]
    #: "light-backdrop" or "mask-pass": how the drive mask was made.
    method: str
    #: The SCAIL frame for each traced frame; see `trace_index`.
    index: tuple[int, ...]

    @property
    def video(self) -> Path:
        return self.root / DRIVE_VIDEO

    @property
    def mask(self) -> Path:
        return self.root / DRIVE_MASK

    @classmethod
    def read(cls, root: Path) -> Drive:
        data = json.loads((root / DRIVE_RECORD).read_text())
        return cls(
            root=root,
            digest=data["digest"],
            rate=float(data["rate"]),
            length=int(data["length"]),
            size=(int(data["width"]), int(data["height"])),
            box=tuple(data["box"]),
            method=data["method"],
            index=tuple(data["trace_index"]),
        )


#: `(drive video, into, length) -> images`: runs the mask pass on the drive video
#: and returns its `length` mask frames, in order. The caller owns the Comfy job.
MaskPass = Callable[[Path, Path, int], Sequence[Path]]


def _mask_pass_masks(
    mask_pass: MaskPass | None, here: Path, length: int, size: tuple[int, int], pad: numpy.ndarray
) -> list[numpy.ndarray]:
    """The drive mask from the mask pass, reusing one already downloaded."""
    if mask_pass is None:
        raise DriveError(
            f"{here.name}: the footage has no light backdrop, so its drive mask needs the "
            "mask pass, and none was given"
        )
    into = here / MASK_PASS_DIR
    done = sorted(into.glob("*.png"))
    paths = done if len(done) == length else list(mask_pass(here / DRIVE_VIDEO, into, length))
    if len(paths) != length:
        raise DriveError(f"{here.name}: the mask pass returned {len(paths)} frames, not {length}")
    masks = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != size:
                image = image.resize(size, Image.NEAREST)
            # SCAIL2ColoredMask paints object 0 in its first palette colour on
            # black; any colour at all is the performer.
            masks.append((numpy.asarray(image).max(axis=2) > 127) & ~pad)
    return masks


def build_drive(
    motion: MotionSheet,
    m: Machine,
    root: Path | str,
    mask_pass: MaskPass | None = None,
    decode: Callable[[Path], numpy.ndarray] = decode,
) -> Drive:
    """The drive video and drive mask for `motion` on machine `m`, cached under `root`.

    Cached at `root/<bundle>-<digest12>/`, keyed on the clip and the profile's
    size and length, not the character, so every character dancing this bundle
    on this machine shares one drive. `drive.json` is written last and is what
    says a drive is complete.
    """
    clip = getattr(motion, "clip", None)
    times = tuple(getattr(motion, "frame_times", ()))
    if clip is None or not times:
        raise DriveError(
            f"{motion.name} carries no source clip with traced frame times; "
            f"run `cag clips {motion.name}`"
        )
    rate, length = plan(times, m, motion.name)
    index = trace_index(times, rate, length, capped=rate < m.rate)
    size = (int(m.width), int(m.height))
    digest = drive_digest(clip, times[0], rate, length, *size)
    here = Path(root) / f"{motion.name}-{digest[:12]}"
    if all((here / name).exists() for name in (DRIVE_RECORD, DRIVE_VIDEO, DRIVE_MASK)):
        return Drive.read(here)

    native = decode(clip.path)
    height, width = native.shape[1:3]
    if clip.size and tuple(clip.size) != (width, height):
        raise DriveError(
            f"{clip.path} decodes at {width}x{height}, but its manifest says "
            f"{clip.size[0]}x{clip.size[1]}; the clip box would land in the wrong place"
        )
    if clip.frame_count and clip.frame_count != len(native):
        _log(f"{clip.path} decodes to {len(native)} frames; its manifest says {clip.frame_count}")
    picks = source_frames(clip, times[0], rate, length, len(native))
    box = drive_box((width, height), clip.box)
    fill = numpy.median(border(native[picks]), axis=0).astype(numpy.uint8)

    video: list[Image.Image] = []
    pad = None
    for pick in picks:
        piece, pad = cut(native[pick], box, fill)
        video.append(Image.fromarray(piece, "RGB").resize(size, Image.LANCZOS))
    pad = numpy.asarray(Image.fromarray(pad).resize(size, Image.NEAREST))
    ms = round(1000 / rate)
    write_apng(video, here / DRIVE_VIDEO, ms)

    arrays = [numpy.asarray(frame) for frame in video]
    method, masks, stats = "mask-pass", None, None
    if light_backdrop(arrays):
        masks = [figure_mask(frame, pad) for frame in arrays]
        stats = mask_stats(masks)
        problems = check_masks(masks, stats)
        if not problems:
            method = "light-backdrop"
        elif mask_pass is None:
            raise DriveError(f"{motion.name}: the threshold drive mask fails: " + "; ".join(problems))
        else:
            _log(f"{motion.name}: the threshold drive mask fails ({problems[0]}); running the mask pass")
    if method == "mask-pass":
        masks = _mask_pass_masks(mask_pass, here, length, size, pad)
        stats = mask_stats(masks)
        problems = check_masks(masks, stats)
        if problems:
            raise DriveError(f"{motion.name}: the mask pass's drive mask fails: " + "; ".join(problems))
    write_apng([paint(mask) for mask in masks], here / DRIVE_MASK, ms)

    record: dict[str, Any] = {
        "bundle": motion.name,
        "digest": digest,
        "version": DRIVE_VERSION,
        "clip_sha256": clip.sha256,
        "rate": rate,
        "length": length,
        "width": size[0],
        "height": size[1],
        "ms": ms,
        "box": list(box),
        "source_frames": picks,
        "method": method,
        "trace_index": index,
        "masks": stats,
    }
    (here / DRIVE_RECORD).write_text(json.dumps(record, indent=2) + "\n")
    return Drive.read(here)
