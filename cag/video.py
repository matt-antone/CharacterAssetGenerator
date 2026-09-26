"""The video path: a traced set drawn from its source clip through SCAIL-2.

The pose-edit path re-poses the set reference into each traced frame, one
render apiece, and every render guesses the in-between motion afresh. This path
animates the set reference once, as video: SCAIL-2 follows a drive video cut
from the footage the frames were traced off, and returns one SCAIL frame per
drive frame. The frames the set needs are the SCAIL frames at the traced times
(the trace index), and each is then restyled — redrawn by Qwen-Image-2.1 in the
set reference's own art — because SCAIL-2 keeps the character's shape and
colours but softens the pixel art.

Three kinds of work are cached separately, each keyed on what it is a function
of, so a failure or a change in one never pays again for the others:

    work/drive/<bundle>-<d12>/          the drive video and drive mask; see `cag.drive`
    work/<char>/video/<set>/<v12>/      the SCAIL video, and the frames picked from it
    work/<char>/source/<set>/NN.png     the restyles, stamped in `drawn.sha`

What is established is one roll per character on one dance, from scratch
scripts. See AGENTS.md, "The video path".
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image

from . import comfy, drive
from .animation import AnimationState, claim_frames, frame_path, motion_digest, motion_stamp
from .draw import DrawError
from .prompts import MASK_PASS_PROMPT, RESTYLE, scail_prompt

#: Bumped whenever the way a restyle is asked for changes outside the graph and
#: the prompt, so every set drawn the old way is superseded at once.
VIDEO_VERSION = "video/1"

#: A SCAIL video's own files, beside its `NNN.png` frames.
PENDING = "pending.json"
DONE = "done.json"
TRACE_INDEX = "trace-index.json"


def _log(state: AnimationState, message: str) -> None:
    print(f"[{state['set_name']}] {message}", file=sys.stderr, flush=True)


def _sha(*parts: bytes | str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if isinstance(part, bytes) else part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def drive_root(state: AnimationState) -> Path:
    """Beside the characters, not inside one: every character dancing a bundle shares its drive."""
    return state["work_dir"].parent / "drive"


def scail_frames(folder: Path, length: int) -> list[Path]:
    """A finished SCAIL video in `folder`, or nothing.

    Finished is every frame on disk and no job still pending: `render_frames`
    removes its pending record only once the last frame is written.
    """
    frames = sorted(folder.glob("[0-9][0-9][0-9].png"))
    if len(frames) != length or (folder / PENDING).exists():
        return []
    return frames


def video_frames(
    state: AnimationState,
    draw_fn: Callable[..., Path],
    frames_fn: Callable[..., Sequence[Path]] | None = None,
    decode: Callable[[Path], object] | None = None,
) -> AnimationState:
    """Draw a traced set from its source clip: one SCAIL video, then one restyle per traced frame.

    `frames_fn` runs a Comfy job that saves many images (`comfy.render_frames`)
    and `decode` reads the source clip (`drive.decode`); both are looked up
    when called, so a test can stand in for either.

    A restyle that fails is not the end of the set: the rest are drawn, then the
    set fails naming the frames it lacks, and a rebuild draws only those.
    """
    frames_fn = frames_fn or comfy.render_frames
    motion = state["motion"]
    machine = state["machine"]
    graphs = state["machine_graphs"]
    local = state.get("local", machine.backend == "local")
    reference = Path(state["key_art"])

    mask_graph = Path(graphs["mask"])
    mask_extra = machine.placeholders("mask")

    def mask_pass(video: Path, into: Path, length: int) -> Sequence[Path]:
        return frames_fn(
            MASK_PASS_PROMPT, into, [video], comfy.load_workflow(mask_graph),
            machine.mask_timeout, raw=[True], extra=mask_extra, expect=length,
            pending=into.parent / drive.MASK_PENDING, seed=machine.seed, local=local,
        )

    made = drive.build_drive(
        motion, machine, drive_root(state), mask_pass, decode=decode or drive.decode,
        mask_key=_sha(
            mask_graph.read_bytes(), MASK_PASS_PROMPT, json.dumps(mask_extra, sort_keys=True),
            str(machine.seed),
        ),
    )
    if len(made.index) != len(motion.frames):
        raise DrawError(
            f"{state['set_name']}: the drive's trace index holds {len(made.index)} SCAIL frames "
            f"for {len(motion.frames)} traced frames"
        )
    _log(
        state,
        f"video path: {motion.name}, {len(motion.frames)} traced -> {made.length} SCAIL frames "
        f"at {made.rate:g} fps (mask: {made.method})",
    )

    # The SCAIL video: one job, keyed on everything it is drawn from.
    prompt = scail_prompt(state["set_name"], state["bible"])
    video_graph = Path(graphs["video"])
    extra = {**machine.placeholders("video"), "$length": made.length}
    video_key = _sha(
        made.digest, reference.read_bytes(), video_graph.read_bytes(), prompt,
        json.dumps(extra, sort_keys=True),
    )
    folder = state["work_dir"] / "video" / state["set_name"] / video_key[:12]
    frames = scail_frames(folder, made.length)
    if frames:
        _log(state, f"SCAIL video cached at {folder}")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        with Image.open(reference) as image:
            fitted = drive.fit(image, machine.width, machine.height)
        fitted.save(folder / "ref.png")
        drive.reference_mask(fitted).save(folder / "ref-mask.png")
        frames = list(
            frames_fn(
                prompt, folder,
                [folder / "ref.png", folder / "ref-mask.png", made.video, made.mask],
                comfy.load_workflow(video_graph), machine.video_timeout,
                raw=[False, False, True, True], extra=extra, expect=made.length,
                pending=folder / PENDING, seed=machine.seed, local=local,
            )
        )
    (folder / DONE).write_text(
        json.dumps(
            {"key": video_key, "drive": made.digest, "machine": machine.name,
             "length": made.length, "rate": made.rate, "prompt": prompt},
            indent=2,
        )
        + "\n"
    )

    # The restyles: one per traced frame, of the SCAIL frame at its traced time.
    restyle_graph = Path(graphs["restyle"])
    restyle_extra = machine.placeholders("restyle")
    claim_frames(
        state,
        "video\t"
        + _sha(
            VIDEO_VERSION, motion_digest(motion), video_key, json.dumps(list(made.index)),
            restyle_graph.read_bytes(), RESTYLE, json.dumps(restyle_extra, sort_keys=True),
            str(machine.seed),
        ),
    )
    with Image.open(reference) as image:
        size = image.size
    sources, failed = {}, []
    for frame, pick in zip(motion.frames, made.index, strict=True):
        # At the reference's size and shape: the restyle's canvas is its first
        # image's. Named by the SCAIL frame, not the traced one, so a trace
        # index that moves can never pick up another frame's file.
        traced = folder / f"picked-{pick:03d}.png"
        if not traced.exists():
            with Image.open(frames[pick]) as image:
                drive.unfit(image, size).save(traced)
        try:
            sources[frame.index] = draw_fn(
                RESTYLE,
                frame_path(state, "source", frame.index),
                references=[traced, reference],
                workflow=restyle_graph,
                timeout=machine.restyle_timeout,
                seed=machine.seed,
                extra=restyle_extra,
            )
        except DrawError as error:
            failed.append(frame.index)
            _log(state, f"restyle {frame.index:02d} failed: {error}")
            continue
        # Kept as drawn: snapping to the key art's grid (`cag.snap`) made the
        # faces blocky and the set was judged better without it (2026-09-26).

    stamp = motion_stamp(state)
    stamp.write_text(motion_digest(motion) + "\n")
    (folder / TRACE_INDEX).write_text(json.dumps(list(made.index)) + "\n")
    if failed:
        raise DrawError(
            f"{state['set_name']}: {len(failed)} of {len(motion.frames)} restyles failed: "
            f"{', '.join(f'{index:02d}' for index in failed)}; rebuild to redraw only these"
        )
    return {
        "sources": sources,
        "frame_sheets": [[index] for index in sorted(sources)],
        "shared_canvas": True,
    }
