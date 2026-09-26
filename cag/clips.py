"""Give a traced bundle its source clip, rebuilt from the video it was traced off.

A stopgap until MotionArtist writes `clip.mp4` itself. Every installed bundle
records its video's URL, the traced window and the source second of each traced
frame, but none carries the video, and the 3:4 box its traced frames were cut
through is written down nowhere. So the whole video is fetched once into
`work/sources/`, the traced window is cut out of it at native rate and size with
half a second either side, and the clip box is found by matching the bundle's
own traced frames against the clip. That match is also the proof that the times
line up: a mean below `MATCH` is refused, never written.

What lands in the bundle is `clip.mp4` and a `clip` block in its manifest,
marked `backfilled_by: "cag"`. The clip is git-ignored and the block is
committed, so a fresh clone has the block without the file, and running this
again rebuilds the file. The block's hash is the committing machine's; another
machine's cut is recorded beside the clip in `clip.sha256`, also ignored. `motion.json` is never touched, and neither is the
manifest's `files` map: that lists what MotionArtist shipped, and a clip absent
on a fresh clone would fail every check that reads it.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .motion import BUNDLE, CLIP_SHA, Bundle, sha256_of

#: Whole source videos, one per video id, shared by every bundle cut from them.
SOURCES = Path("work/sources")
#: The source clip's name inside its bundle.
CLIP = "clip.mp4"
#: Seconds of video kept either side of the traced window. SCAIL's drive video
#: resamples the clip, and a resample that runs off either end has nothing to
#: land on.
PAD = 0.5
#: Mean match of the traced frames against the clip, below which the box or the
#: times are wrong and nothing is written. Shuffled traced frames score ~0.5.
MATCH = 0.90

#: MotionArtist's own download, so the geometry and timestamps are the ones it
#: traced: best mp4 video, preferring 720p.
FORMAT = "bv*[ext=mp4]/bv*/b"
SORT = "res:720"
#: How far a download may run from the duration the trace recorded before it is
#: taken to be a different cut of the video.
DURATION_SLACK = 0.15
#: How far a stream's nominal and average rates may sit apart before it counts
#: as variable. One frame short at the end of a two-minute video is 0.04%.
RATE_SLACK = 0.002

#: The box search. Coarse runs at 1/COARSE scale over box heights from SMALLEST
#: of the frame up to all of it, each SIZE_STEP taller than the last; the refine
#: runs at full size within NUDGE pixels and RESIZE of the coarse answer, and
#: tries each traced frame SKEW native frames either side of its traced time.
COARSE = 4
SMALLEST = 0.25
SIZE_STEP = 1.03
NUDGE = 4
RESIZE = 0.02
SKEW = 1

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


class ClipError(RuntimeError):
    """Raised when a bundle's source clip cannot be rebuilt, or would be wrong."""


def video_id(url: str) -> str:
    """The YouTube id in a watch, youtu.be or Shorts link, whatever rides after it."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    parts = [p for p in parsed.path.split("/") if p]
    found = ""
    if host == "youtu.be" and parts:
        found = parts[0]
    elif host.endswith("youtube.com"):
        if parsed.path.rstrip("/") == "/watch":
            found = parse_qs(parsed.query).get("v", [""])[0]
        elif len(parts) > 1 and parts[0] in ("shorts", "embed", "live"):
            found = parts[1]
    if not _VIDEO_ID.match(found):
        raise ClipError(f"no YouTube video id in {url!r}")
    return found


@dataclass(frozen=True)
class Probe:
    """What ffprobe says about a video's first stream."""

    duration: float
    fps: Fraction
    size: tuple[int, int]
    frames: int
    variable: bool


def probe(path: Path) -> Probe:
    """Duration, rate, size and frame count of `path`, and whether its rate varies."""
    out = json.loads(
        _run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
                "-of", "json", str(path),
            ]
        )
    )
    streams = out.get("streams") or []
    if not streams:
        raise ClipError(f"{path} has no video stream")
    stream = streams[0]
    nominal, average = _rate(stream.get("r_frame_rate")), _rate(stream.get("avg_frame_rate"))
    duration = float(out.get("format", {}).get("duration") or 0.0)
    fps = nominal or average
    frames = int(stream.get("nb_frames") or 0) or round(duration * fps)
    variable = not nominal or not average or abs(nominal / average - 1) > RATE_SLACK
    return Probe(duration, fps, (int(stream["width"]), int(stream["height"])), frames, variable)


def fetch(url: str, duration: float, cache: Path = SOURCES) -> Path:
    """The whole source video, downloaded once into `cache` as `<id>.<ext>`.

    Refused when it is not the video that was traced: a duration off the one the
    trace recorded, or a variable rate, which no traced second maps onto.
    """
    ident = video_id(url)
    cache = Path(cache)
    video = _downloaded(cache, ident)
    if video is None:
        cache.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "yt-dlp", "-f", FORMAT, "-S", SORT, "--no-playlist",
                "-o", str(cache / f"{ident}.%(ext)s"),
                f"https://www.youtube.com/watch?v={ident}",
            ]
        )
        video = _downloaded(cache, ident)
        if video is None:
            raise ClipError(f"yt-dlp wrote nothing for {ident} into {cache}")
    found = probe(video)
    if abs(found.duration - duration) > DURATION_SLACK:
        raise ClipError(
            f"{video} runs {found.duration:.2f} s but the trace recorded {duration:.2f} s; "
            "it is not the cut of the video that was traced"
        )
    if found.variable:
        raise ClipError(f"{video} has a variable frame rate; traced seconds do not map onto it")
    return video


def cut(src: Path, start: float, end: float, pad: float, out: Path) -> tuple[float, int]:
    """Cut `[start - pad, end + pad]` out of `src` at its native rate and size.

    The window is widened onto whole frames, and clamped to the video. Returns
    the source second of the clip's first frame and how many frames it holds.
    """
    found = probe(src)
    fps = found.fps
    first = max(0, math.floor((start - pad) * fps))
    last = min(found.frames - 1, math.ceil((end + pad) * fps))
    count = last - first + 1
    out.parent.mkdir(parents=True, exist_ok=True)
    # Seek half a frame early: an exact seek can land a hair past the frame it
    # names and drop it. The first frame kept is then restamped to zero, or a
    # video whose stream starts late opens on a gap that the constant rate
    # fills by showing the first frame twice. One encoder thread and bitexact
    # flags keep the bytes, and so the manifest's hash, the same run to run.
    seek = max(0.0, float((first - Fraction(1, 2)) / fps))
    _run(
        [
            "ffmpeg", "-v", "error", "-y", "-ss", f"{seek:.6f}", "-i", str(src),
            "-frames:v", str(count), "-an", "-map_metadata", "-1",
            "-vf", "setpts=PTS-STARTPTS",
            "-c:v", "libx264", "-threads", "1", "-crf", "18", "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr", "-r", str(fps),
            "-fflags", "+bitexact", "-flags:v", "+bitexact",
            str(out),
        ]
    )
    got = probe(out).frames
    if got != count:
        raise ClipError(f"cut {out} holds {got} frames, not the {count} asked for")
    return float(first / fps), count


def decode(path: Path, grey: bool = False) -> np.ndarray:
    """Every frame of `path`: (n, h, w, 3) RGB, or (n, h, w) luma when `grey`."""
    width, height = probe(path).size
    depth = 1 if grey else 3
    raw = _run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "gray" if grey else "rgb24", "-",
        ],
        binary=True,
    )
    frame = width * height * depth
    if not raw or len(raw) % frame:
        raise ClipError(f"{path} decoded to {len(raw)} bytes, not whole {width}x{height} frames")
    frames = np.frombuffer(raw, np.uint8)
    shape = (len(raw) // frame, height, width)
    return frames.reshape(shape if grey else (*shape, 3))


def recover_box(
    frames: np.ndarray,
    fps: float,
    clip_start: float,
    times: Sequence[float],
    thumbs: Sequence[Path | Image.Image],
) -> tuple[tuple[int, int, int, int], float, float]:
    """Find the box, in clip pixels, that the traced frames were cut through.

    One box for every frame, at the traced frames' own aspect and inside the
    frame, found by zero-mean normalised cross-correlation of each traced frame
    against the clip frame at its traced time. Each is also tried a native frame
    either side, because a traced second lands between two frames and the
    tracer's seek may have taken either.

    Returns `(x, y, w, h)`, the clip's `t_offset` — seconds to take off a traced
    time to land on the clip frame it was traced from — and the mean match.
    """
    grey = _grey(frames)
    count, height, width = grey.shape
    pictures = [_picture(t) for t in thumbs]
    if len(pictures) != len(times) or not pictures:
        raise ClipError(f"{len(pictures)} traced frames for {len(times)} traced times")
    aspect = pictures[0].width / pictures[0].height
    base = [round((t - clip_start) * fps) for t in times]
    skews = []
    for t, j in zip(times, base):
        if not 0 <= j < count:
            raise ClipError(
                f"traced second {t:.3f} is frame {j} of a {count}-frame clip; "
                "the clip does not cover the trace"
            )
        skews.append([s for s in range(-SKEW, SKEW + 1) if 0 <= j + s < count])

    # Coarse: every box height at once over the whole frame, small.
    small_h, small_w = height // COARSE, width // COARSE
    needed = sorted({j + s for j, own in zip(base, skews) for s in own})
    coarse = {
        k: _Plane(
            grey[k, : small_h * COARSE, : small_w * COARSE]
            .reshape(small_h, COARSE, small_w, COARSE)
            .mean(axis=(1, 3))
        )
        for k in needed
    }
    candidates = [[coarse[j + s] for s in own] for j, own in zip(base, skews)]
    heights = _heights(small_h, small_w, aspect)
    found = [_best(candidates, pictures, round(box_h * aspect), box_h, skews) for box_h in heights]
    if not found:
        raise ClipError(f"a {width}x{height} clip is too small to search for a box")
    top = max(range(len(found)), key=lambda k: found[k][0])
    _, (x, y, box_w, box_h), _ = found[top]
    centre_x, centre_y = (x + box_w / 2) * COARSE, (y + box_h / 2) * COARSE
    around = box_h * COARSE

    # Refine at full size, within a couple of percent of the coarse height and a
    # few pixels of its place. One region holds every size tried, so each clip
    # frame is transformed once; heights are stepped through at a stride, then
    # pixel by pixel either side of the best. The tallest coarse height can sit
    # a whole step short of the frame (117 of 120 coarse rows at 480p), so when
    # it wins the refine reaches all the way: a full-height box is the commonest
    # there is, since MotionArtist cuts landscape footage full height.
    limit = min(height, math.floor(width / aspect))
    tallest = limit if top == len(heights) - 1 else min(limit, round(around * (1 + RESIZE)))
    shortest = min(tallest, round(around * (1 - RESIZE)))
    reach_y = tallest / 2 + NUDGE
    reach_x = round(tallest * aspect) / 2 + NUDGE
    rows = _span(centre_y, reach_y, height)
    cols = _span(centre_x, reach_x, width)
    candidates = [[_Plane(grey[j + s, rows, cols]) for s in own] for j, own in zip(base, skews)]
    tried: dict[int, tuple] = {}

    def at(box_h: int) -> tuple:
        if box_h not in tried:
            box_w = round(box_h * aspect)
            fits = box_h <= rows.stop - rows.start and box_w <= cols.stop - cols.start
            tried[box_h] = (
                _best(candidates, pictures, box_w, box_h, skews) if fits else (-1.0, None, [])
            )
        return tried[box_h]

    stride = max(1, round(around * RESIZE / 4))
    for box_h in range(shortest, tallest + 1, stride):
        at(box_h)
    at(tallest)
    centre = max(tried, key=lambda h: tried[h][0])
    for box_h in range(max(shortest, centre - stride + 1), min(tallest, centre + stride - 1) + 1):
        at(box_h)
    score, placed, shift = max(tried.values(), key=lambda f: f[0])
    if placed is None:
        raise ClipError(f"no box of the traced frames' shape fits a {width}x{height} clip")
    x, y, box_w, box_h = placed
    box = (cols.start + x, rows.start + y, box_w, box_h)
    return box, -statistics.median_low(shift) / fps, score


def bundle_roots(root: Path | str) -> dict[str, Path]:
    """Every bundle under `root`, by name, read off its manifest alone.

    Not `library()`: that checks every clip on disk against its hash, and a
    clip that fails the check is exactly what `backfill` is here to repair.
    """
    found = {}
    for manifest in sorted(Path(root).rglob(BUNDLE)):
        found[json.loads(manifest.read_text()).get("name", manifest.parent.name)] = manifest.parent
    return found


@dataclass(frozen=True)
class Backfill:
    """What `backfill` found, and whether it wrote it."""

    name: str
    box: tuple[int, int, int, int]
    match: float
    size: tuple[int, int]
    frames: int
    t_offset: float
    #: False on a dry run, and when the bundle already held this clip.
    written: bool

    def __str__(self) -> str:
        return (
            f"{self.name} box={','.join(map(str, self.box))} match={self.match:.3f} "
            f"size={self.size[0]}x{self.size[1]} frames={self.frames} OK"
        )


def backfill(
    bundle: Bundle | Path | str,
    cache: Path = SOURCES,
    pad: float = PAD,
    dry_run: bool = False,
) -> Backfill:
    """Rebuild one bundle's `clip.mp4` and write its `clip` block.

    Idempotent: a clip already on disk whose hash is the block's, or this
    machine's own cut of it (`CLIP_SHA`), is left alone. A rebuild of the same
    frames of the same video leaves the manifest byte for byte, even when this
    machine's x264 writes other bytes than the one that committed the block.
    Raises ClipError, writing nothing, for a bundle with no source recorded, no
    traced times or traced frames, or a match under `MATCH`.
    """
    root = bundle.root if isinstance(bundle, Bundle) else Path(bundle)
    if root.name == BUNDLE:
        root = root.parent
    manifest = root / BUNDLE
    data = json.loads(manifest.read_text())
    name = data.get("name", root.name)

    held = data.get("clip")
    if held and (root / CLIP).exists() and sha256_of(root / CLIP) in _accepted(root, held):
        return _result(name, held, written=False)

    source = data.get("source") or {}
    missing = [key for key in ("url", "duration") if not source.get(key)]
    if missing:
        raise ClipError(f"{manifest} records no source {' or '.join(missing)} to cut a clip from")
    times = _traced_times(root, data)
    thumbs = sorted(f for f in data["files"] if Path(f).parent.name == "thumbs")
    if len(thumbs) != len(times):
        raise ClipError(
            f"{manifest} ships {len(thumbs)} traced frames for {len(times)} traced times; "
            "there is nothing to match the clip against"
        )

    video = fetch(source["url"], float(source["duration"]), cache)
    made = Path(cache) / "cuts" / f"{name}.mp4"
    start, count = cut(video, times[0], times[-1], pad, made)
    found = probe(made)
    frames = decode(made, grey=True)
    if len(frames) != count:
        raise ClipError(f"{made} decoded to {len(frames)} frames, not {count}")
    box, t_offset, match = recover_box(
        frames, float(found.fps), start, times, [root / t for t in thumbs]
    )
    if match < MATCH:
        raise ClipError(
            f"the traced frames match the clip at {match:.3f}, under {MATCH}; "
            "the box or the traced times are wrong"
        )

    if held and _same_cut(held, start, found, count):
        # The footage the committed block describes, cut on a machine whose
        # x264 writes other bytes. The block stays as committed, so two
        # machines never take turns rewriting a tracked manifest; this
        # machine's hash is kept beside the clip instead.
        if not dry_run:
            shutil.move(made, root / CLIP)
            (root / CLIP_SHA).write_text(sha256_of(root / CLIP) + "\n")
        return _result(name, held, written=not dry_run)

    block = {
        "file": CLIP,
        "start": round(start, 6),
        "fps": float(found.fps),
        "frame_count": count,
        "size": list(found.size),
        "box": list(box),
        "t_offset": round(t_offset, 6),
        "sha256": sha256_of(made),
        "match": round(match, 4),
        "backfilled_by": "cag",
    }
    if dry_run:
        return _result(name, block, written=False)
    shutil.move(made, root / CLIP)
    (root / CLIP_SHA).write_text(block["sha256"] + "\n")
    text = json.dumps(_with_clip(data, block), indent=1)
    if text != manifest.read_text():
        manifest.write_text(text)
    return _result(name, block, written=True)


class _Plane:
    """One greyscale image, ready to be matched against: its spectrum and the
    running sums that give any window's mean and spread in constant time."""

    def __init__(self, pixels: np.ndarray):
        pixels = np.asarray(pixels, dtype=np.float64)
        pixels = pixels - pixels.mean()
        self.shape = pixels.shape
        # Zero-padded out to lengths the FFT is quick at. A template never
        # reaches past the image at a placement that counts, so the circular
        # correlation does not wrap either way.
        self.padded = tuple(_fast(n) for n in pixels.shape)
        self.spectrum = np.fft.rfft2(pixels, s=self.padded)
        self.sums = np.pad(pixels.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        self.squares = np.pad((pixels * pixels).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def ncc(self, template: _Template) -> np.ndarray:
        """Normalised cross-correlation at every place `template` fits wholly inside."""
        h, w = template.shape
        valid = (self.shape[0] - h + 1, self.shape[1] - w + 1)
        if template.norm == 0:
            return np.zeros(valid)
        corr = np.fft.irfft2(self.spectrum * template.spectrum(self.padded), s=self.padded)
        corr = corr[: valid[0], : valid[1]]
        area = h * w
        total = _window(self.sums, h, w)
        spread = np.maximum(_window(self.squares, h, w) - total * total / area, 0.0)
        denom = np.sqrt(spread) * template.norm
        return np.where(spread > 1e-3 * area, corr / np.maximum(denom, 1e-12), 0.0)


class _Template:
    """One traced frame at one box size, zero-mean, its spectrum kept per padding."""

    def __init__(self, pixels: np.ndarray):
        self.pixels = pixels - pixels.mean()
        self.shape = pixels.shape
        self.norm = math.sqrt(float((self.pixels * self.pixels).sum()))
        self._spectra: dict[tuple[int, int], np.ndarray] = {}

    def spectrum(self, padded: tuple[int, int]) -> np.ndarray:
        if padded not in self._spectra:
            self._spectra[padded] = np.conj(np.fft.rfft2(self.pixels, s=padded))
        return self._spectra[padded]


def _best(
    candidates: list[list[_Plane]],
    pictures: list[Image.Image],
    w: int,
    h: int,
    skews: list[list[int]],
) -> tuple[float, tuple[int, int, int, int], list[int]]:
    """The best single placement of a w x h box across every traced frame.

    Each traced frame scores against the best of its candidate clip frames at
    each placement. Returns the mean score, the box `(x, y, w, h)` in the
    candidates' own pixels, and which skew each traced frame took there.
    """
    total, taken = None, []
    for planes, picture in zip(candidates, pictures):
        template = _Template(_sized(picture, w, h))
        maps = np.stack([plane.ncc(template) for plane in planes])
        taken.append(np.argmax(maps, axis=0))
        best = maps.max(axis=0)
        total = best if total is None else total + best
    y, x = np.unravel_index(int(np.argmax(total)), total.shape)
    shift = [own[pick[y, x]] for own, pick in zip(skews, taken)]
    return float(total[y, x]) / len(candidates), (int(x), int(y), w, h), shift


def _window(running: np.ndarray, h: int, w: int) -> np.ndarray:
    """Sum over every h x w window, from a zero-padded 2D running sum."""
    return running[h:, w:] - running[:-h, w:] - running[h:, :-w] + running[:-h, :-w]


def _fast(n: int) -> int:
    """The smallest length at least `n` with no prime factor above 5."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _span(centre: float, reach: float, size: int) -> slice:
    """`centre` ± `reach`, slid back inside `[0, size)` rather than cut short by its edge."""
    length = min(size, math.ceil(2 * reach))
    start = min(max(0, math.floor(centre - reach)), size - length)
    return slice(start, start + length)


def _heights(rows: int, cols: int, aspect: float) -> list[int]:
    """Coarse box heights to try, each SIZE_STEP taller, the box inside the frame."""
    heights, h = [], max(8, math.ceil(rows * SMALLEST))
    while h <= rows:
        if round(h * aspect) <= cols:
            heights.append(h)
        h = max(h + 1, round(h * SIZE_STEP))
    return heights


def _sized(picture: Image.Image, w: int, h: int) -> np.ndarray:
    shrink = w < picture.width
    filt = Image.Resampling.BOX if shrink else Image.Resampling.BICUBIC
    return np.asarray(picture.resize((w, h), filt), dtype=np.float64)


def _picture(thumb: Path | Image.Image) -> Image.Image:
    if isinstance(thumb, Image.Image):
        return thumb.convert("L")
    with Image.open(thumb) as image:
        return image.convert("L")


def _grey(frames: np.ndarray) -> np.ndarray:
    """Luma, (n, h, w) uint8, from either shape `decode` returns."""
    frames = np.asarray(frames)
    if frames.ndim == 3:
        return frames
    weights = np.array([0.299, 0.587, 0.114])
    return np.stack([(frame @ weights).round().astype(np.uint8) for frame in frames])


def _traced_times(root: Path, data: dict) -> list[float]:
    """Each traced frame's source second, off the motion sheet the manifest names."""
    named = [f for f in data["files"] if Path(f).name == "motion.json"]
    if len(named) != 1:
        raise ClipError(f"{root / BUNDLE} names {len(named)} motion files, not one")
    frames = json.loads((root / named[0]).read_text()).get("frames", [])
    times = [raw.get("t") for raw in frames]
    if not times or None in times:
        raise ClipError(f"{root / named[0]} does not record the second each frame was traced at")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ClipError(f"{root / named[0]} traced times do not run forwards")
    return [float(t) for t in times]


def _accepted(root: Path, held: dict) -> set[str]:
    """The hashes a clip on disk may have: the committed block's, and this machine's own cut."""
    local = root / CLIP_SHA
    return {held.get("sha256"), local.read_text().strip() if local.exists() else None} - {None, ""}


def _same_cut(held: dict, start: float, found: Probe, count: int) -> bool:
    """Is a fresh cut the same frames of the same video as the committed block?"""
    return (
        round(start, 6) == held.get("start")
        and float(found.fps) == held.get("fps")
        and count == held.get("frame_count")
        and list(found.size) == held.get("size")
    )


def _with_clip(data: dict, block: dict) -> dict:
    """The manifest with `block` as its clip: in place if it had one, else after `source`."""
    if "clip" in data:
        return {**data, "clip": block}
    out = {}
    for key, value in data.items():
        out[key] = value
        if key == "source":
            out["clip"] = block
    out.setdefault("clip", block)
    return out


def _result(name: str, block: dict, written: bool) -> Backfill:
    return Backfill(
        name=name,
        box=tuple(block["box"]),
        match=float(block.get("match", 0.0)),
        size=tuple(block["size"]),
        frames=int(block["frame_count"]),
        t_offset=float(block.get("t_offset", 0.0)),
        written=written,
    )


def _downloaded(cache: Path, ident: str) -> Path | None:
    """The finished download for `ident`, if there is one; a `.part` is not."""
    return next(
        (p for p in sorted(cache.glob(f"{ident}.*")) if p.suffix in (".mp4", ".webm", ".mkv")),
        None,
    )


def _rate(text: str | None) -> Fraction:
    try:
        return Fraction(text or "0")
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def _run(cmd: Sequence[str], binary: bool = False) -> str | bytes:
    """Run a tool, and turn its failure into a ClipError that says what it said."""
    try:
        done = subprocess.run(list(cmd), check=True, capture_output=True, text=not binary)
    except FileNotFoundError:
        raise ClipError(f"{cmd[0]} is not installed") from None
    except subprocess.CalledProcessError as exc:
        said = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(
            errors="replace"
        )
        raise ClipError(f"{cmd[0]} failed: {said.strip()[-1500:]}") from None
    return done.stdout
