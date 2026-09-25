"""How closely a rendered set dances its motion, bone by bone.

Both sides are traced by MotionArtist's own `trace` command — the character's
cells and the bundle's photographs, the same raw method on each, so the two
compare like with like. The motion.json landmarks are not used here: extract
rescales and exaggerates them, and the photographs are what the generator saw.

The measure is each bone's direction on screen. A bone's angle does not depend
on its length, so a long-legged character dancing the same move scores the same
as the performer; widths and joint positions do not have that property, and
read costume and build as pose error. On Belter against club-01, renders from
full-card photographs landed at 4.9-5.8 degrees mean across five runs, the
same inputs rolled again — that spread is the generator's floor, and the band a
new character should be read against. The same set drawn from 200px thumbnails
tracked its arms at r +0.30.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

#: MotionArtist's tracer. It needs mediapipe, which cag does not carry, so it
#: runs under whichever interpreter has it.
MOTION_ARTIST = Path(
    os.environ.get(
        "MOTION_ARTIST",
        Path.home() / "Development/MotionArtist/motion-artist/scripts/motion_artist.py",
    )
)
MOTION_ARTIST_PYTHON = os.environ.get("MOTION_ARTIST_PYTHON", "python3")

BONES = {
    "upper arm L": ("shL", "elL"),
    "forearm L": ("elL", "wrL"),
    "upper arm R": ("shR", "elR"),
    "forearm R": ("elR", "wrR"),
    "thigh L": ("hipL", "knL"),
    "shin L": ("knL", "anL"),
    "thigh R": ("hipR", "knR"),
    "shin R": ("knR", "anR"),
    "shoulders": ("shR", "shL"),
    "hips": ("hipR", "hipL"),
}


def bone_angles(pts: dict[str, list[float]]) -> dict[str, float]:
    """Each bone's direction on screen, in degrees, y down."""
    return {
        name: math.degrees(math.atan2(pts[b][1] - pts[a][1], pts[b][0] - pts[a][0]))
        for name, (a, b) in BONES.items()
    }


def angle_gap(a: float, b: float) -> float:
    """Smallest difference between two directions, 0-180."""
    return abs((a - b + 180) % 360 - 180)


def bone_errors(drawn: dict[str, list[float]], traced: dict[str, list[float]]) -> dict[str, float]:
    got, want = bone_angles(drawn), bone_angles(traced)
    return {name: angle_gap(got[name], want[name]) for name in BONES}


def trace(images: list[Path]) -> list[dict | None]:
    """Landmarks for each image, in order; None where MotionArtist found no body."""
    with tempfile.TemporaryDirectory() as tmp:
        for index, image in enumerate(images):
            (Path(tmp) / f"{index:03d}{image.suffix}").symlink_to(image.resolve())
        out = Path(tmp) / "pts.json"
        # Under `uv run`, "python3" is cag's own venv, which has no mediapipe:
        # the tracer runs on the interpreter found with that venv taken off PATH.
        env = dict(os.environ)
        venv = env.pop("VIRTUAL_ENV", None)
        if venv:
            env["PATH"] = os.pathsep.join(
                p for p in env["PATH"].split(os.pathsep) if not p.startswith(venv)
            )
        run = subprocess.run(
            [MOTION_ARTIST_PYTHON, str(MOTION_ARTIST), "trace", tmp, str(out)],
            env=env,
            capture_output=True,
            text=True,
        )
        # mediapipe can exit non-zero while tearing down after a clean write,
        # so the file is the verdict, not the exit code.
        if not out.exists():
            raise RuntimeError(f"MotionArtist trace wrote nothing:\n{run.stderr[-2000:]}")
        traced = json.loads(out.read_text())
    return [traced.get(str(index)) for index in range(len(images))]


def report(cells: list[Path], photos: list[Path]) -> str:
    """Per-frame bone-angle error of the drawn set against its photographs."""
    drawn, traced = trace(cells), trace(photos)
    lines = [f"{'frame':>5}  {'mean':>5}  worst"]
    means = []
    for index, (got, want) in enumerate(zip(drawn, traced)):
        if got is None or want is None:
            side = "character" if got is None else "photograph"
            lines.append(f"{index:5d}  {'-':>5}  no body found in the {side}")
            continue
        errors = bone_errors(got, want)
        worst = max(errors, key=errors.get)
        means.append(statistics.fmean(errors.values()))
        lines.append(f"{index:5d}  {means[-1]:5.1f}  {worst} {errors[worst]:.0f}°")
    if means:
        lines.append(
            f"set: mean {statistics.fmean(means):.1f}°, median frame {statistics.median(means):.1f}°"
            f" over {len(means)} of {len(cells)} frames"
        )
    return "\n".join(lines)
