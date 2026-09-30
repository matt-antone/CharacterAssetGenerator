"""Read a MotionArtist motion sheet.

A motion sheet is a *motion source*: it describes movement only. Scale, identity,
view of the character, and prop hand come from the character package, never from
the video the sheet was traced off.

Produced by https://github.com/matt-antone/MotionArtist —
`motion_artist.py extract <video> --fps N --frames M`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path


#: Drawn first, by the keyframer. Everything else is a tweener's in-between.
LOCKED_ROLES = ("key", "pilot")


class MotionError(ValueError):
    """Raised when a motion sheet cannot drive an animation set."""


#: How a set repeats, in the one spelling everything downstream uses. Only a
#: loop cuts from the last frame back to the first, so only a loop needs a
#: clean seam; a pingpong turns around on its ends and plays back down the
#: frames it just played, which cannot jump however the trace was cut.
#:
#: The keys are every word a sheet may arrive with. `one-shot` and `final-hold`
#: are MotionArtist's (`extract --playback`), and both land on "once" here:
#: whether a set holds its last frame is the set plan's call, not the trace's.
#: `pingpong` as a word is ours, written by `write_motion` and by a brief. No
#: trace says it: MotionArtist sends a loop with `pingpong: true` beside it,
#: because the cells are a loop either way and the flag is a walk order.
PLAYBACKS = {
    "loop": "loop",
    "pingpong": "pingpong",
    "once": "once",
    "one-shot": "once",
    "oneshot": "once",
    "final-hold": "once",
}


def playback_of(declared: str) -> str:
    """Normalise what a sheet says about repetition, rather than guess at it."""
    try:
        return PLAYBACKS[str(declared).strip().lower()]
    except KeyError:
        raise MotionError(
            f"unknown playback {declared!r}; a sheet plays {', '.join(PLAYBACKS)}"
        ) from None


@dataclass(frozen=True)
class ClipPart:
    """A file the tracer ships beside its source clip, frame for frame with it.

    The bundle mask (`mask.mp4`: the performer white on black, one frame per
    clip frame) and the head boxes (`heads.json`: one `[x0, y0, x1, y1]` in clip
    pixels per clip frame, or null). Declared in the manifest's `clip` block as
    `{"file": ..., "sha256": ...}` under `mask` and `heads`. Like `clip.mp4`, a
    part whose file is not there is absent, not an error, and one whose file is
    not the one declared is stale.
    """

    path: Path
    sha256: str = ""


@dataclass(frozen=True)
class Clip:
    """A bundle's source clip: the footage the traced frames were taken from.

    Kept at the source's native rate and pixel size, uncropped, and cut a little
    wider than the traced window, so a drive video can be resampled off it at
    whatever rate the video path asks for. Every time here is a second of the
    original video, the same clock a traced frame's `t` is on.
    """

    path: Path
    #: The source second of the clip's first frame.
    start: float
    fps: float
    frame_count: int
    #: Width and height of the clip in pixels.
    size: tuple[int, int]
    #: The clip box, as (x, y, w, h) in clip pixels: the 3:4 box the traced
    #: frames were cut from. It always lies inside the frame. None when nobody
    #: recorded it.
    box: tuple[int, int, int, int] | None
    #: How far a traced frame's `t` sits from the clip frame that matches it,
    #: in seconds. MotionArtist seeks by millisecond and lands on a frame
    #: boundary; the backfill measures the difference rather than assume none.
    t_offset: float = 0.0
    sha256: str = ""
    #: The bundle mask and the head boxes, when the tracer shipped them. On a
    #: clip read off disk, each is there only when its file is and matches its
    #: hash; on a declared clip, whenever the block names it.
    mask: ClipPart | None = None
    heads: ClipPart | None = None
    #: Why a declared mask or head boxes on disk were not taken: their file is
    #: not the one declared. Only the video path reads it, and fails with it.
    problem: str = ""
    #: Who cut the clip when cag did (`"cag"`, from `cag clips`); empty for a
    #: clip the tracer shipped, which cag never cuts again.
    backfilled_by: str = ""

    @property
    def from_tracer(self) -> bool:
        """Shipped with the bundle by MotionArtist, rather than backfilled here."""
        return not self.backfilled_by

    @property
    def end(self) -> float:
        """The source second of the clip's last frame."""
        return self.start + (self.frame_count - 1) / self.fps

    def frame_at(self, t: float) -> int:
        """The clip frame shown at source second `t`. Not clamped: an index
        outside the clip is the caller's to refuse."""
        return round((t - self.start - self.t_offset) * self.fps)

    def covers(self, t: float) -> bool:
        return 0 <= self.frame_at(t) < self.frame_count


#: How far below the ears the nose sits, as a share of the ears' spread, past
#: which a traced head is bowed. Upright heads on club-01 and emote-02 measured
#: 0.02-0.31; emote-01's KO bows to 0.57-0.85 from its fifth frame on.
BOWED_PITCH = 0.45


def head_bowed(pts: dict[str, list[float]]) -> bool:
    """Whether a traced pose's head is bowed, read off its landmarks."""
    try:
        ear_y = (pts["earL"][1] + pts["earR"][1]) / 2
        spread = abs(pts["earL"][0] - pts["earR"][0])
        return spread > 0 and (pts["nose"][1] - ear_y) / spread > BOWED_PITCH
    except (KeyError, IndexError, TypeError):
        return False


@dataclass(frozen=True)
class Frame:
    index: int
    role: str
    pace: str
    cue: str
    note: str = ""
    #: MediaPipe landmark positions for this pose, keyed by joint name.
    pts: dict[str, list[float]] = field(default_factory=dict)
    #: The tracer's own call that both feet are off the floor. Read, never
    #: re-derived from `pts`: MotionArtist decides contact from the soles.
    airborne: bool = False
    #: The source second this frame was traced at. None for a motion sheet that does
    #: not say, which is every written sheet.
    t: float | None = None
    #: The frame of the bundle's `clip.mp4` this frame was traced from, counted
    #: from 0. Exact where `t` is a rounded second; None when the tracer does
    #: not say, which is every bundle cut before MotionArtist shipped its clip.
    clip_frame: int | None = None

    @property
    def is_locked(self) -> bool:
        return self.role in LOCKED_ROLES

    @property
    def instruction(self) -> str:
        """What this frame is told to do: its cue, its note, and its pace.

        The pace is traced off the footage — a frame the performer held, or one
        they passed through fast. Without it a held frame is drawn as one more
        in-between and the settle averages away.
        """
        said = {
            "hold": "The pose settles here and does not travel.",
            "fast": "A fast transition, not a hold.",
        }.get(self.pace, "")
        return " ".join(part for part in (self.cue, self.note, said) if part).strip()


@dataclass(frozen=True)
class MotionSheet:
    name: str
    fps: int
    view: str
    #: One of the values in `PLAYBACKS`: "loop", "pingpong" or "once".
    playback: str
    arc: str
    frames: tuple[Frame, ...]
    #: The traced video frames, one per motion frame: the pose reference the
    #: keyframer is shown. Measured against the trace, renders made from them
    #: carry 0.9-1.2 of the movement in it. A sheet read straight off disk
    #: rather than out of a bundle has none.
    photos: tuple[Path, ...] = ()
    #: Where the performer's floor sits, and their body height, both normalised.
    floor_y: float = 0.0
    body_h: float = 0.0
    #: How the last frame meets the first. "clean" cuts back; anything else is
    #: the tracer saying it does not, and the proof will show the jump.
    seam: str = ""
    #: The footage the frames were traced from, when the bundle carries it.
    clip: Clip | None = None
    #: Why a bundle that declares a clip has none to give: the file on disk is
    #: not the one it declares. Only the video path reads it, and fails with it.
    clip_problem: str = ""
    #: The command that puts the clip back when it is missing or stale; see
    #: `clip_remedy`. Empty on a sheet read straight off disk.
    clip_remedy: str = ""

    @property
    def clip_frames(self) -> tuple[int, ...]:
        """Every frame's clip frame, in order; empty unless every frame has one."""
        picks = tuple(frame.clip_frame for frame in self.frames)
        return () if None in picks else picks

    @property
    def frame_times(self) -> tuple[float, ...]:
        """Every frame's traced second, in order; empty unless every frame has one.

        With a clip and a clip frame on every frame, the second is the clip
        frame's own, `start + t_offset + clip_frame / fps`, which `Clip.frame_at`
        takes back to that exact frame. Otherwise it is each frame's `t`. A
        partial set of either would pair a frame with the wrong moment of the
        clip, so it is dropped entirely, the way a partial set of photographs is.
        """
        picks = self.clip_frames
        if self.clip is not None and picks:
            clip = self.clip
            return tuple(clip.start + clip.t_offset + pick / clip.fps for pick in picks)
        times = tuple(frame.t for frame in self.frames)
        return () if None in times else times

    @property
    def has_poses(self) -> bool:
        return all(frame.pts for frame in self.frames)

    @property
    def travel(self) -> float:
        """Mean joint travel over the loop, in body heights.

        How much movement the sheet actually holds, independent of where it
        goes: a crouch and a sideways sway both count. 0.0 for a sheet with no
        landmarks, which has nothing to measure.
        """
        if not self.has_poses or not self.body_h:
            return 0.0
        spans = []
        for joint in {name for frame in self.frames for name in frame.pts}:
            seen = [frame.pts[joint] for frame in self.frames if joint in frame.pts]
            xs, ys = [p[0] for p in seen], [p[1] for p in seen]
            spans.append(max(max(xs) - min(xs), max(ys) - min(ys)) / self.body_h)
        return sum(spans) / len(spans) if spans else 0.0

    @property
    def loops(self) -> bool:
        """Cuts from the last frame back to the first. A pingpong does not."""
        return self.playback == "loop"

    @property
    def seams_cleanly(self) -> bool:
        """A loop the tracer did not flag. An unset seam is not a complaint."""
        return not self.loops or self.seam in ("", "clean")

    @property
    def locked(self) -> tuple[Frame, ...]:
        return tuple(frame for frame in self.frames if frame.is_locked)

    def neighbours(self, frame: Frame) -> tuple[int, int]:
        """Indices of the frames either side of an in-between.

        The one before is whatever was drawn last; the one after is the next
        locked frame, wrapping past the loop seam when the set loops.
        """
        after = next(
            (f.index for f in self.frames if f.index > frame.index and f.is_locked), None
        )
        if after is None:
            after = self.locked[0].index if self.loops else self.frames[-1].index
        return frame.index - 1 if frame.index else self.frames[-1].index, after


def load_motion(path: Path | str) -> MotionSheet:
    """Read and validate a MotionArtist `motion.json`."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise MotionError(f"{path} is not valid JSON: {exc}") from exc

    missing = {"fps", "view", "playback", "frames"} - set(data)
    if missing:
        raise MotionError(f"{path} is missing {', '.join(sorted(missing))}")
    if data.get("missing_frames"):
        raise MotionError(f"{path} has gaps at frames {data['missing_frames']}")

    for raw in data["frames"]:
        pick = raw.get("clip_frame")
        if pick is not None and (
            isinstance(pick, bool)
            or not isinstance(pick, (int, float))
            or not float(pick).is_integer()
            or pick < 0
        ):
            raise MotionError(
                f"{path} frame {raw.get('i')} has clip_frame {pick!r}; it is a frame "
                "number of clip.mp4, a whole number from 0"
            )
    frames = tuple(
        Frame(
            index=int(raw["i"]),
            role=raw["role"],
            pace=raw.get("pace", ""),
            cue=raw["cue"].strip(),
            note=raw.get("note", "").strip(),
            pts=raw.get("pts", {}),
            airborne=bool(raw.get("features", {}).get("airborne", False)),
            t=None if raw.get("t") is None else float(raw["t"]),
            clip_frame=None if raw.get("clip_frame") is None else int(raw["clip_frame"]),
        )
        for raw in data["frames"]
    )
    if not frames:
        raise MotionError(f"{path} has no frames")
    if [frame.index for frame in frames] != list(range(len(frames))):
        raise MotionError(f"{path} frames must be numbered 0..{len(frames) - 1} in order")
    if not any(frame.is_locked for frame in frames):
        raise MotionError(f"{path} has no key or pilot frames for the keyframer to draw")

    # MotionArtist says a pingpong twice: `playback` stays the loop an unaware
    # consumer would play, and `pingpong` is the boolean beside it saying the
    # return leg reverses the out leg. One word from here on, because everything
    # downstream asks the sheet a single question.
    playback = "pingpong" if data.get("pingpong") else playback_of(data["playback"])

    return MotionSheet(
        name=data.get("name", path.parent.name),
        fps=int(data["fps"]),
        view=data["view"],
        playback=playback,
        arc=data.get("arc", "").strip(),
        frames=frames,
        floor_y=float(data.get("floor_y", 0.0)),
        body_h=float(data.get("body_h", 0.0)),
        seam=str(data.get("seam", "")).strip(),
    )


#: The file that says a directory is a motion bundle, and what is in it.
BUNDLE = "manifest.json"
#: Beside a bundle's `clip.mp4`, and git-ignored like it: the sha256 of the clip
#: as this machine cut it. The manifest's hash is the machine that committed the
#: block, and x264 writes its own build into every file, so a clip cut again on
#: another machine is the same footage in different bytes.
CLIP_SHA = "clip.sha256"

#: Bundle layouts this reads. A bundle that declares anything else is refused
#: rather than guessed at.
# motion-artist/2 gave landmarks a third float, z. Every reader here takes [0] and
# [1] and leaves the rest, so both versions load the same way and a 2D sheet keeps
# working; the depth only shows up where the skeleton sorts bones by it.
SCHEMAS = ("motion-artist/1", "motion-artist/2")


@dataclass(frozen=True)
class Bundle:
    """A traced motion bundle, read from its manifest.

    The manifest is the bundle: it declares the layout, the header a brief
    picks a sheet by, and the files it contains. The motion data is whichever
    file the manifest names — reaching straight for `motion.json` assumed a
    filename no bundle ever promised.

    It is a record of what was traced, not a config file. How a set plays is a
    rule, and a rule is read off the motion sheet the manifest names — never off
    the manifest, which would be a second place to write the same thing down.
    """

    name: str
    title: str
    fps: int
    frame_count: int
    view: str
    seam: str
    #: The manifest's own directory, and the motion data it names inside it.
    root: Path
    sheet: Path
    #: One traced video frame per motion frame, in order, when the bundle ships
    #: a complete set. The pose reference; see `photos` on MotionSheet.
    photos: tuple[Path, ...] = ()
    #: The source clip, when it is on disk and matches its hash. Its `sha256` is
    #: the file's own, which is what a drive is keyed on.
    clip: Clip | None = None
    #: Why a clip on disk was not taken; see `clip_status`.
    clip_problem: str = ""
    #: The clip the manifest declares, whether or not its file is there.
    #: `clip.mp4` is not tracked, so on a fresh clone every bundle has this and
    #: no `clip` until it is pulled or cut again; see `clip_remedy`.
    declared_clip: Clip | None = None

    @property
    def clip_remedy(self) -> str:
        """The command that puts this bundle's clip back; see `clip_remedy`."""
        return clip_remedy(self.root, self.name, self.declared_clip)

    def load(self) -> MotionSheet:
        """The sheet itself, checked against the header that advertised it."""
        motion = load_motion(self.sheet)
        if (motion.fps, len(motion.frames)) != (self.fps, self.frame_count):
            raise MotionError(
                f"{self.root / BUNDLE} advertises {self.frame_count} frames at {self.fps} fps, "
                f"but {self.sheet.name} holds {len(motion.frames)} at {motion.fps}"
            )
        picks = motion.clip_frames
        if self.clip and picks:
            if any(b <= a for a, b in zip(picks, picks[1:])):
                raise MotionError(f"{self.sheet} clip frames are not strictly increasing")
            outside = [
                f.index for f in motion.frames if not 0 <= f.clip_frame < self.clip.frame_count
            ]
            if outside:
                raise MotionError(
                    f"{self.root / BUNDLE} clip holds frames 0-{self.clip.frame_count - 1}, "
                    f"which does not cover the clip frames of traced frames {outside}"
                )
        elif self.clip and motion.frame_times:
            times = motion.frame_times
            if any(b <= a for a, b in zip(times, times[1:])):
                raise MotionError(f"{self.sheet} traced times are not strictly increasing")
            outside = [f.index for f in motion.frames if not self.clip.covers(f.t)]
            if outside:
                raise MotionError(
                    f"{self.root / BUNDLE} clip spans {self.clip.start:.3f}-{self.clip.end:.3f}s, "
                    f"which does not cover traced frames {outside}"
                )
        return replace(
            motion,
            photos=self.photos,
            clip=self.clip,
            clip_problem=self.clip_problem,
            clip_remedy=self.clip_remedy,
        )


def read_bundle(path: Path | str) -> Bundle:
    """Read one bundle's manifest."""
    path = Path(path)
    manifest = path / BUNDLE if path.is_dir() else path
    try:
        data = json.loads(manifest.read_text())
    except FileNotFoundError:
        raise MotionError(f"{manifest} is not there; a motion bundle is its manifest") from None
    except json.JSONDecodeError as exc:
        raise MotionError(f"{manifest} is not valid JSON: {exc}") from exc

    schema = data.get("schema")
    if schema not in SCHEMAS:
        raise MotionError(
            f"{manifest} declares schema {schema!r}; this reads {', '.join(SCHEMAS)}"
        )
    missing = {"fps", "frame_count", "view", "files"} - set(data)
    if missing:
        raise MotionError(f"{manifest} is missing {', '.join(sorted(missing))}")

    named = [f for f in data["files"] if Path(f).name == "motion.json"]
    if len(named) != 1:
        raise MotionError(
            f"{manifest} names {len(named)} motion files; a bundle carries exactly one"
        )

    # The traced frames. A photograph of the performer carries what a drawn
    # skeleton cannot — the whole body at once, at the size and commitment it
    # was really done at. Taken only when there is one per frame, in order: a
    # partial set would silently pair figure n with the wrong frame, so it is
    # dropped entirely and the set is drawn with no pose reference at all.
    # A `spritesheet` block in the manifest is ignored; nothing reads it.
    thumbs = sorted(f for f in data["files"] if Path(f).parent.name == "thumbs")
    photos = (
        tuple(manifest.parent / f for f in thumbs)
        if len(thumbs) == int(data["frame_count"])
        else ()
    )
    name = data.get("name", manifest.parent.name)
    declared = read_clip(manifest, data["clip"]) if data.get("clip") else None
    clip, problem = None, ""
    if declared and declared.path.exists():
        # The clip's hash lives in its own block, never in `files`: `files` is
        # what a fresh clone must already hold, and clip.mp4 is not tracked.
        # A mismatch is not raised here: every build and `cag motions` read the
        # whole library, and one stale clip would stop all of them. The video
        # path raises it, for this bundle only.
        actual = sha256_of(declared.path)
        local = declared.path.with_name(CLIP_SHA)
        cut_here = local.read_text().strip() if local.exists() else ""
        if not declared.sha256 or actual in (declared.sha256, cut_here):
            parts, problems = {}, []
            for kind in PARTS:
                parts[kind], said = _adopt(manifest, getattr(declared, kind), kind, name)
                problems += [said] if said else []
            clip = replace(declared, sha256=actual, problem="; ".join(problems), **parts)
        else:
            problem = (
                f"{manifest} declares clip {declared.path.name} with sha256 "
                f"{declared.sha256[:12]}, but the file on disk is not it; "
                f"run `{clip_remedy(manifest.parent, name, declared)}` to "
                + ("pull it again" if declared.from_tracer else "cut it again")
            )
    return Bundle(
        name=name,
        title=data.get("title", ""),
        fps=int(data["fps"]),
        frame_count=int(data["frame_count"]),
        view=data["view"],
        seam=str(data.get("seam", "")).strip(),
        root=manifest.parent,
        sheet=manifest.parent / named[0],
        photos=photos,
        clip=clip,
        clip_problem=problem,
        declared_clip=declared,
    )


#: The files a `clip` block may declare beside the clip, by the key they sit under.
PARTS = ("mask", "heads")


def _adopt(
    manifest: Path, part: ClipPart | None, kind: str, name: str
) -> tuple[ClipPart | None, str]:
    """A declared mask or head boxes as found on disk, and why it was not taken.

    Not there is absent and says nothing. There but not the declared file is
    absent too, with the reason, which only the video path raises.
    """
    if part is None or not part.path.exists():
        return None, ""
    actual = sha256_of(part.path)
    if not part.sha256 or actual == part.sha256:
        return replace(part, sha256=actual), ""
    return None, (
        f"{manifest} declares {kind} {part.path.name} with sha256 {part.sha256[:12]}, "
        f"but the file on disk is not it; pull {name} again with "
        f"`cag motions pull {manifest.parent.parent.name}/{name}`"
    )


def read_clip(manifest: Path, block: dict) -> Clip:
    """A manifest's `clip` block, checked for sense but not for the file.

    The block records the source clip: `file` beside the manifest, `start` (the
    source second of its first frame), `fps`, `frame_count`, `size` as [w, h],
    and optionally `box` as [x, y, w, h], `t_offset`, `sha256`, and the `mask`
    and `heads` the tracer ships beside it, each `{"file": ..., "sha256": ...}`.
    `backfilled_by` says cag cut the clip rather than the tracer. Anything else
    in it (`match`) is a note for people and is not read. A missing file is not
    an error: the footage is untracked and fetched again (`cag clips` for a
    backfilled clip, `cag motions pull` for a shipped one), so the manifest
    outlives it.
    """
    missing = {"file", "start", "fps", "frame_count", "size"} - set(block)
    if missing:
        raise MotionError(f"{manifest} clip block is missing {', '.join(sorted(missing))}")
    fps, count = float(block["fps"]), int(block["frame_count"])
    if fps <= 0 or count < 1:
        raise MotionError(f"{manifest} clip block has {count} frames at {fps} fps")
    w, h = (int(n) for n in block["size"])
    box = None
    if block.get("box") is not None:
        x, y, bw, bh = (int(n) for n in block["box"])
        if bw <= 0 or bh <= 0 or x < 0 or y < 0 or x + bw > w or y + bh > h:
            raise MotionError(
                f"{manifest} clip box {[x, y, bw, bh]} does not lie inside the {w}x{h} frame"
            )
        box = (x, y, bw, bh)
    parts = {}
    for kind in PARTS:
        part = block.get(kind)
        if part is None:
            continue
        if not isinstance(part, dict) or not part.get("file"):
            raise MotionError(f"{manifest} clip block's {kind} names no file")
        parts[kind] = ClipPart(manifest.parent / part["file"], str(part.get("sha256", "")))
    return Clip(
        path=manifest.parent / block["file"],
        start=float(block["start"]),
        fps=fps,
        frame_count=count,
        size=(w, h),
        box=box,
        t_offset=float(block.get("t_offset", 0.0)),
        sha256=str(block.get("sha256", "")),
        backfilled_by=str(block.get("backfilled_by", "") or ""),
        **parts,
    )


def read_heads(path: Path, count: int) -> list[list[float] | None]:
    """A bundle's head boxes: one `[x0, y0, x1, y1]` in clip pixels, or null, per clip frame.

    Raises MotionError when the file does not read, or does not hold exactly
    `count` entries of that shape.
    """
    path = Path(path)
    try:
        boxes = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise MotionError(f"{path.name} does not read ({error})") from None
    if not isinstance(boxes, list) or len(boxes) != count:
        held = len(boxes) if isinstance(boxes, list) else type(boxes).__name__
        raise MotionError(f"{path.name} holds {held} entries for {count} clip frames")
    for j, head in enumerate(boxes):
        if head is not None and not (
            isinstance(head, list)
            and len(head) == 4
            and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in head)
        ):
            raise MotionError(f"{path.name} entry {j} is {head!r}, not [x0, y0, x1, y1] or null")
    return boxes


def sha256_of(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def clip_remedy(root: Path, name: str, declared: Clip | None) -> str:
    """The command that puts back the clip of the bundle `name` at `root`.

    A clip the tracer shipped is pulled again, `cag motions pull <set>/<name>`,
    and never cut here: `cag clips` refuses it. A clip cag backfilled, or a
    bundle that has never had one, is cut by `cag clips <name>`.
    """
    if declared is not None and declared.from_tracer:
        return f"cag motions pull {Path(root).parent.name}/{name}"
    return f"cag clips {name}"


def clip_status(bundle: Bundle) -> str:
    """One word for what a bundle holds of its source clip, as `cag motions` shows it.

    `ok` is a clip on disk that matches its hash. `missing` is a manifest that
    declares one whose file is not there, which is every clip on a fresh clone
    until it is put back. `stale` is a file there that is not the one declared.
    Either is put back by `Bundle.clip_remedy`: a pull for a clip the tracer
    shipped, `cag clips <name>` for one cag backfilled. `none` is a bundle that
    has never had one, which only `cag clips` can backfill.
    """
    if bundle.clip:
        return "ok"
    if bundle.clip_problem:
        return "stale"
    return "missing" if bundle.declared_clip else "none"


def part_status(bundle: Bundle, kind: str) -> str:
    """`clip_status` for the bundle mask (`"mask"`) or the head boxes (`"heads"`).

    Read off the file and the declared hash alone, so it answers whether or
    not the clip itself is on disk.
    """
    declared = getattr(bundle.declared_clip, kind, None) if bundle.declared_clip else None
    if declared is None:
        return "none"
    if not declared.path.exists():
        return "missing"
    if declared.sha256 and sha256_of(declared.path) != declared.sha256:
        return "stale"
    return "ok"


def summary(name: str, bundle: Bundle) -> str:
    """The one-line install check AGENTS.md prints for each bundle."""
    motion = bundle.load()
    return (
        f"{name}: {bundle.frame_count}f @ {bundle.fps}fps {bundle.view} {motion.playback} "
        f"seam={bundle.seam!r} photos={len(motion.photos)} "
        f"airborne={[f.index for f in motion.frames if f.airborne]} "
        f"travel={motion.travel:.3f} clip={clip_status(bundle)}"
        + "".join(
            f" {kind}={part_status(bundle, kind)}"
            for kind in PARTS
            if getattr(bundle.declared_clip, kind, None)
        )
    )


def library(root: Path | str) -> dict[str, Bundle]:
    """Every motion bundle under `root`, by the name a brief would call it.

    Found by manifest at any depth, not by guessing at filenames: a capture
    session installs its clips as `dance-2/dance-2-4/`, and a one-level scan
    saw none of them. A bundle that will not read raises: skipping quietly
    made a library that had lost three of its four sheets read as a library
    of one.
    """
    return {
        bundle.name: bundle
        for bundle in (read_bundle(p) for p in sorted(Path(root).rglob(BUNDLE)))
    }
