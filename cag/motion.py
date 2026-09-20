"""Read a MotionArtist motion sheet.

A motion sheet is a *motion source*: it describes movement only. Scale, identity,
view of the character, and prop hand come from the character package, never from
the video the sheet was traced off.

Produced by https://github.com/matt-antone/MotionArtist —
`motion_artist.py extract <video> --fps N --frames M`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from .poses import SheetLayout

#: Drawn first, by the keyframer. Everything else is a tweener's in-between.
LOCKED_ROLES = ("key", "pilot")


class MotionError(ValueError):
    """Raised when a motion sheet cannot drive an animation set."""


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
    playback: str
    arc: str
    frames: tuple[Frame, ...]
    #: The bundle's sprite sheet and where each frame sits on it. This is the
    #: pose reference the keyframer is shown; a sheet read straight off disk
    #: rather than out of a bundle has no sprite sheet and so carries neither.
    poses: Path | None = None
    pose_layout: SheetLayout | None = None
    #: The traced video frames, one per motion frame. Preferred over the drawn
    #: sprite sheet: measured against the trace, renders made from photographs
    #: carry 0.9-1.2 of the movement in it, where the drawn cards carry 0.43-0.70
    #: — the poses rank correctly either way, but the drawn ones come out small
    #: enough to read as a sway rather than the motion that was traced.
    photos: tuple[Path, ...] = ()
    #: Where the performer's floor sits, and their body height, both normalised.
    floor_y: float = 0.0
    body_h: float = 0.0
    #: How the last frame meets the first. "clean" cuts back; anything else is
    #: the tracer saying it does not, and the proof will show the jump.
    seam: str = ""

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

    frames = tuple(
        Frame(
            index=int(raw["i"]),
            role=raw["role"],
            pace=raw.get("pace", ""),
            cue=raw["cue"].strip(),
            note=raw.get("note", "").strip(),
            pts=raw.get("pts", {}),
            airborne=bool(raw.get("features", {}).get("airborne", False)),
        )
        for raw in data["frames"]
    )
    if not frames:
        raise MotionError(f"{path} has no frames")
    if [frame.index for frame in frames] != list(range(len(frames))):
        raise MotionError(f"{path} frames must be numbered 0..{len(frames) - 1} in order")
    if not any(frame.is_locked for frame in frames):
        raise MotionError(f"{path} has no key or pilot frames for the keyframer to draw")

    return MotionSheet(
        name=data.get("name", path.parent.name),
        fps=int(data["fps"]),
        view=data["view"],
        playback=data["playback"],
        arc=data.get("arc", "").strip(),
        frames=frames,
        floor_y=float(data.get("floor_y", 0.0)),
        body_h=float(data.get("body_h", 0.0)),
        seam=str(data.get("seam", "")).strip(),
    )


#: The file that says a directory is a motion bundle, and what is in it.
BUNDLE = "manifest.json"

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
    """

    name: str
    title: str
    fps: int
    frame_count: int
    playback: str
    view: str
    seam: str
    #: The manifest's own directory, and the motion data it names inside it.
    root: Path
    sheet: Path
    #: The sprite sheet of every frame's figure, and the layout that cuts it up.
    #: Both come from the manifest, so a bundle exported before sprite sheets
    #: existed simply has no pose reference rather than a guessed one.
    poses: Path | None = None
    pose_layout: SheetLayout | None = None
    #: One traced video frame per motion frame, in order, when the bundle ships
    #: a complete set. The preferred pose reference; see `photos` on MotionSheet.
    photos: tuple[Path, ...] = ()

    def load(self) -> MotionSheet:
        """The sheet itself, checked against the header that advertised it."""
        motion = load_motion(self.sheet)
        if (motion.fps, len(motion.frames)) != (self.fps, self.frame_count):
            raise MotionError(
                f"{self.root / BUNDLE} advertises {self.frame_count} frames at {self.fps} fps, "
                f"but {self.sheet.name} holds {len(motion.frames)} at {motion.fps}"
            )
        return replace(
            motion, poses=self.poses, pose_layout=self.pose_layout, photos=self.photos
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
    missing = {"fps", "frame_count", "playback", "view", "files"} - set(data)
    if missing:
        raise MotionError(f"{manifest} is missing {', '.join(sorted(missing))}")

    named = [f for f in data["files"] if Path(f).name == "motion.json"]
    if len(named) != 1:
        raise MotionError(
            f"{manifest} names {len(named)} motion files; a bundle carries exactly one"
        )

    # The sprite sheet is the pose reference. It is taken only when the manifest
    # both names the image and declares its layout: without the layout the grid
    # would have to be measured off the picture, which is cag guessing at
    # MotionArtist's renderer instead of reading what it published.
    drawn = [f for f in data["files"] if Path(f).name.endswith("-spritesheet.png")]
    block = data.get("spritesheet")
    poses = manifest.parent / drawn[0] if len(drawn) == 1 and block else None

    # The traced frames themselves. A photograph of the performer carries what a
    # drawn skeleton cannot — the whole body at once, at the size and commitment
    # it was really done at — and renders drawn from it came back at 0.9-1.2 of
    # the movement in the trace where skeleton cards sat at 0.43-0.70. Taken only
    # when there is one per frame, in order: a partial set would silently pair
    # figure n with the wrong frame.
    thumbs = sorted(f for f in data["files"] if Path(f).parent.name == "thumbs")
    photos = (
        tuple(manifest.parent / f for f in thumbs)
        if len(thumbs) == int(data["frame_count"])
        else ()
    )
    return Bundle(
        name=data.get("name", manifest.parent.name),
        title=data.get("title", ""),
        fps=int(data["fps"]),
        frame_count=int(data["frame_count"]),
        playback=data["playback"],
        view=data["view"],
        seam=str(data.get("seam", "")).strip(),
        root=manifest.parent,
        sheet=manifest.parent / named[0],
        poses=poses,
        pose_layout=SheetLayout.from_manifest(block) if poses else None,
        photos=photos,
    )


def library(root: Path | str) -> dict[str, Bundle]:
    """Every motion bundle under `root`, by the name a brief would call it.

    Found by manifest, not by guessing at filenames. A bundle that will not
    read raises: skipping quietly made a library that had lost three of its
    four sheets read as a library of one.
    """
    return {
        bundle.name: bundle
        for bundle in (read_bundle(p) for p in sorted(Path(root).glob(f"*/{BUNDLE}")))
    }
