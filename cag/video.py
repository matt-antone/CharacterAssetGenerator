"""The video path: a traced set drawn from its source clip through SCAIL-2.

The pose-edit path re-poses the set reference into each traced frame, one
render apiece, and every render guesses the in-between motion afresh. This path
animates the set reference once, as video: SCAIL-2 follows a drive video cut
from the footage the frames were traced off, and returns one SCAIL frame per
drive frame. The frames the set needs are the SCAIL frames at the traced times
(the trace index), and each is then restyled — redrawn by Qwen-Image-2.1 in the
set reference's own art — because SCAIL-2 keeps the character's shape and
colours but softens the pixel art. Under `scail_only` (`--no-restyle`) the
SCAIL frames are the set's frames, with no restyle at all.

Three kinds of work are cached separately, each keyed on what it is a function
of, so a failure or a change in one never pays again for the others:

    work/drive/<bundle>-<d12>/          the drive video and drive mask; see `cag.drive`
    work/<char>/video/<set>/<v12>/      the SCAIL video, and the frames picked from it
    work/<char>/source/<set>/NN.png     the restyles, or the SCAIL frames, stamped in `drawn.sha`

What is established is one roll per character on one dance, from scratch
scripts. See AGENTS.md, "The video path".
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image

from . import comfy, drive
from .animation import AnimationState, claim_frames, frame_path, motion_digest, motion_stamp
from .draw import DrawError
from .motion import head_bowed
from .prompts import MASK_PASS_PROMPT, RESTYLE, RESTYLE_BOWED, scail_prompt

#: Bumped whenever the way a restyle is asked for changes outside the graph and
#: the prompt, so every set drawn the old way is superseded at once.
VIDEO_VERSION = "video/1"

#: A SCAIL video's own files, beside its `NNN.png` frames.
PENDING = "pending.json"
DONE = "done.json"
TRACE_INDEX = "trace-index.json"


def _log(state: AnimationState, message: str) -> None:
    print(f"[{state['set_name']}] {message}", file=sys.stderr, flush=True)


#: How far a restyle may raise the figure's crown above the SCAIL frame it
#: redraws, as a share of the figure's height. The head never moves in a
#: restyle, so a crown that rose is a head it lifted: Belter's KO frames 12 and
#: 13 came back looking out at +0.14 and +0.16. At 0.10 that was caught, but
#: frames 9-11 then rose +0.03 to +0.05 each on their own roll, and the bowed
#: head bobbed up and down across the hold. Every frame of her dance and victory
#: measured +0.008 or less (2026-09-30).
MAX_CROWN_RISE = 0.03

#: How far a restyle may move the head sideways from where the SCAIL frame put
#: it, as a share of the figure's height. KO frames 8, 10 and 14 swung the head
#: 50-77 px at a figure of about 1250 px, and every other frame kept within 12.
MAX_HEAD_SHIFT = 0.02

#: How much bigger a restyle may draw the head and hair than the SCAIL frame did.
#: Those same KO frames swelled the head-and-hair band to 1.30-1.34x, so the
#: bowed head grew and swung while the body held still; the rest measured 1.06x
#: or less (2026-09-30).
MAX_HEAD_GROWTH = 1.10

#: The top of the figure measured as the head: the head and the hair around it.
HEAD_BAND = 0.22

#: Restyles drawn per frame before the one closest to its SCAIL frame is kept.
RESTYLE_TRIES = 5


def _head(path: Path) -> tuple[int, int, float, int] | None:
    """The figure's top row and height, and its head band's centre column and area."""
    import numpy as np

    from .mask import cutout

    alpha = np.asarray(cutout(path))[..., 3] > 0
    rows = np.flatnonzero(alpha.any(axis=1))
    if not len(rows) or rows[-1] <= rows[0]:
        return None
    top, height = int(rows[0]), int(rows[-1] - rows[0])
    band = alpha[top : top + max(1, int(height * HEAD_BAND))]
    return top, height, float(np.flatnonzero(band.any(axis=0)).mean() if band.any() else 0), int(band.sum())


def head_drift(scail: Path, restyled: Path) -> tuple[float, float, float]:
    """How far `restyled`'s head moved from `scail`'s: crown rise and sideways
    shift as shares of the SCAIL figure's height, and the head band's growth."""
    before, after = _head(scail), _head(restyled)
    if not before or not after or not before[3]:
        return 0.0, 0.0, 1.0
    height = before[1]
    return (
        (before[0] - after[0]) / height,
        abs(after[2] - before[2]) / height,
        after[3] / before[3],
    )


def head_kept(drift: tuple[float, float, float]) -> bool:
    rise, shift, growth = drift
    return rise <= MAX_CROWN_RISE and shift <= MAX_HEAD_SHIFT and growth <= MAX_HEAD_GROWTH


def _miss(drift: tuple[float, float, float]) -> float:
    """How far past the limits a drift is, for picking the closest try."""
    rise, shift, growth = drift
    return (max(0.0, rise - MAX_CROWN_RISE) / MAX_CROWN_RISE
            + max(0.0, shift - MAX_HEAD_SHIFT) / MAX_HEAD_SHIFT
            + max(0.0, growth - MAX_HEAD_GROWTH) / (MAX_HEAD_GROWTH - 1))


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

    Under `scail_only` each traced frame's SCAIL frame is the set's frame, at
    the set reference's size, and nothing is restyled. The two are stamped
    apart (`scail` and `video` in `drawn.sha`), so switching moves the other
    mode's frames to `superseded/`; the SCAIL video is the same in both.

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
    scail_only = bool(state.get("scail_only"))

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
        f"at {made.rate:g} fps (mask: {made.method}), "
        + ("SCAIL only, no restyle" if scail_only else "then a restyle per traced frame"),
    )

    # The SCAIL video: one job, keyed on everything it is drawn from.
    prompt = scail_prompt(state["set_name"], state["bible"])
    video_graph = Path(graphs["video"])
    extra = {**machine.placeholders("video"), "$length": made.length}
    video_key = _sha(
        made.digest, reference.read_bytes(), video_graph.read_bytes(), prompt,
        json.dumps(extra, sort_keys=True),
        # The drive's shared folder survives a changed mask pass, but its
        # mask and face blur do not. Carry that dependency into SCAIL too.
        # No extra part for threshold/bundle masks: their existing keys hold.
        *((made.mask_key,) if made.mask_key else ()),
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

    with Image.open(reference) as image:
        size = image.size

    def picked(pick: int) -> Path:
        # At the reference's size and shape: the restyle's canvas is its first
        # image's. Named by the SCAIL frame, not the traced one, so a trace
        # index that moves can never pick up another frame's file.
        path = folder / f"picked-{pick:03d}.png"
        if not path.exists():
            with Image.open(frames[pick]) as image:
                drive.unfit(image, size).save(path)
        return path

    sources, failed = {}, []
    if scail_only:
        claim_frames(
            state,
            "scail\t" + _sha(VIDEO_VERSION, motion_digest(motion), video_key, json.dumps(list(made.index))),
        )
        for frame, pick in zip(motion.frames, made.index, strict=True):
            out = frame_path(state, "source", frame.index)
            if not out.exists():
                shutil.copyfile(picked(pick), out)
            sources[frame.index] = out
    else:
        # The restyles: one per traced frame, of the SCAIL frame at its traced time.
        restyle_graph = Path(graphs["restyle"])
        restyle_extra = machine.placeholders("restyle")
        style = Path(state.get("style_reference") or reference)
        # The trace says which frames bow the head (`RESTYLE_BOWED`). A set with
        # none keeps the stamp it had before bowed frames were told apart.
        bowed = [frame.index for frame in motion.frames if head_bowed(frame.pts)]
        claim_frames(
            state,
            "video\t"
            + _sha(
                VIDEO_VERSION, motion_digest(motion), video_key, json.dumps(list(made.index)),
                restyle_graph.read_bytes(), RESTYLE, json.dumps(restyle_extra, sort_keys=True),
                str(machine.seed), style.read_bytes(),
                *((RESTYLE_BOWED, json.dumps(bowed)) if bowed else ()),
            ),
        )
        if bowed:
            _log(state, f"head bowed in frames {', '.join(f'{i:02d}' for i in bowed)}: "
                        "their restyles keep the head's tilt")
        unkept: list[int] = []
        for frame, pick in zip(motion.frames, made.index, strict=True):
            dst = frame_path(state, "source", frame.index)
            cached = dst.exists()
            tries: list[tuple[float, tuple[float, float, float], Path]] = []
            try:
                for attempt in range(RESTYLE_TRIES):
                    out = dst if attempt == 0 else dst.with_name(f"{dst.stem}.try-{attempt}.png")
                    draw_fn(
                        RESTYLE_BOWED if frame.index in bowed else RESTYLE,
                        out,
                        references=[picked(pick), style],
                        workflow=restyle_graph,
                        timeout=machine.restyle_timeout,
                        seed=machine.seed + 1000 * attempt,
                        extra=restyle_extra,
                    )
                    if cached:
                        # Drawn by an earlier build: measured, never redrawn.
                        if not head_kept(head_drift(picked(pick), out)):
                            unkept.append(frame.index)
                        break
                    drift = head_drift(picked(pick), out)
                    tries.append((_miss(drift), drift, out))
                    if head_kept(drift):
                        break
                    _log(state, f"restyle {frame.index:02d} moved the head (crown +{drift[0]:.2f}, "
                                f"shift {drift[1]:.2f}, size {drift[2]:.2f}x); drawing it again")
            except DrawError as error:
                failed.append(frame.index)
                _log(state, f"restyle {frame.index:02d} failed: {error}")
                continue
            if tries:
                _, drift, best = min(tries, key=lambda t: t[0])
                if not head_kept(drift):
                    unkept.append(frame.index)
                if best != dst:
                    dst.rename(dst.with_name(f"{dst.stem}.lifted-0.png"))
                    best.rename(dst)
                for _, _, other in tries:
                    if other != best and other.exists() and ".try-" in other.name:
                        other.rename(other.with_name(other.name.replace(".try-", ".lifted-")))
            sources[frame.index] = dst
        # A frame no seed could draw with the head where SCAIL put it takes the
        # nearest frame that could: KO frame 10 swelled the head 1.32-1.46x on
        # all five seeds, and in a held pose a repeated frame does not show
        # where a swollen head does. Its closest try is kept beside it.
        kept = sorted(set(sources) - set(unkept) - set(failed))
        for index in unkept:
            if not kept:
                break
            near = min(kept, key=lambda k: (abs(k - index), k > index))
            dst = sources[index]
            shutil.copy2(dst, dst.with_name(f"{dst.stem}.closest.png"))
            shutil.copy2(sources[near], dst)
            _log(state, f"restyle {index:02d}: no try kept the head where SCAIL put it; "
                        f"it repeats frame {near:02d}")
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
