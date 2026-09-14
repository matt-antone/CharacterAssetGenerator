"""Read a MotionArtist motion sheet.

A motion sheet is a *motion source*: it describes movement only. Scale, identity,
view of the character, and prop hand come from the character package, never from
the video the sheet was traced off.

Produced by https://github.com/matt-antone/MotionArtist —
`motion_artist.py extract <video> --fps N --frames M`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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

    @property
    def is_locked(self) -> bool:
        return self.role in LOCKED_ROLES


@dataclass(frozen=True)
class MotionSheet:
    name: str
    fps: int
    view: str
    playback: str
    arc: str
    frames: tuple[Frame, ...]

    @property
    def loops(self) -> bool:
        return self.playback == "loop"

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
    )
