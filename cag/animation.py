"""One animation set, drawn the way a studio draws one.

The motion director binds the motion source to this character. The keyframer
draws the extremes and the fastest transitions. The tweener fills the gaps, each
in-between drawn from its two locked neighbours — drawn, not interpolated. No
cross-fade, no optical flow, no frame grid used as a creative source.
"""

from __future__ import annotations

import hashlib
import math
import sys
from functools import partial
from pathlib import Path
from typing import Callable, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .assemble import tile
from .draw import DrawError, draw
from .geometry import ANIM_PX_PER_INCH, PX_PER_INCH
from .mask import (
    MaskError,
    canvas_to_cells,
    home_x,
    pose_to_cell,
    set_to_cells,
    slice_frame_sheet,
)
from .machines import Machine
from .motion import Frame, MotionError, MotionSheet
from .prompts import (
    DIRECTOR_SYSTEM,
    EMPTY_HANDS,
    FRAME_VIEWS,
    POSE_EDIT,
    director_request,
    frame_prompt,
    frame_sheet_prompt,
)
from .poses import CARD_HEIGHT, CARD_WIDTH, write_photos
from .props import clauses
from .snap import snap_file
from .style import detail_frame
from .spec import CharacterSpec


class AnimationState(TypedDict, total=False):
    spec: CharacterSpec
    #: Locked appearance and scale, carried over from the static sheet.
    bible: str
    key_art: Path
    scale: float
    motion: MotionSheet
    set_name: str
    work_dir: Path
    #: Standing instruction for every frame in this set.
    set_note: str
    #: Pose reference per frame index, when the sheet carries poses.
    poses: dict[int, Path]
    #: The last frame of the set drawn before this one, when the build chains
    #: its sets. The first sheet of this set continues from it exactly as its
    #: own later sheets continue from each other.
    carry: Path
    #: Whether those references are photographs of the performer rather than
    #: drawn figures. They are described very differently to the generator: one
    #: is a person to copy a pose off and take nothing else from, the other a
    #: colour-coded diagram carrying no costume at all.
    photographic: bool
    #: Whether frames get the detail sample beside the key art. Missing means
    #: true. See `DETAIL_FRAMES`.
    detail_after_key: bool
    #: The most figures one sheet-mode render takes. Missing means the
    #: eight-figure batching; see `sheet_layout`.
    frames_per_sheet: int
    #: A workflow that re-poses the set's reference into each traced photograph,
    #: one frame per render. Set, a traced set is drawn that way instead of on a
    #: frame sheet; see `pose_edit_frames`.
    pose_workflow: Path
    #: Put each re-posed frame back on the reference's pixel grid and palette
    #: (see `cag.snap`): the pose workflow keeps the character, not the art.
    snap_to_key: bool
    #: Every frame is its own render of the same reference on the same canvas,
    #: so the set is registered with one transform; see `canvas_to_cells`.
    shared_canvas: bool
    #: The machine profile a traced set is drawn from its source clip on. Set,
    #: a traced set takes the video path (`cag.video`) instead of the pose-edit
    #: path, and one with no source clip fails rather than falling back.
    machine: Machine
    #: The video path's graphs as this machine patched them, by stage: `video`,
    #: `restyle` and `mask` (see `cag.machines.materialise`).
    machine_graphs: dict[str, Path]
    #: Whether those graphs run on your own ComfyUI rather than Comfy Cloud.
    local: bool
    #: Raw magenta-backdrop frames, by frame index.
    sources: dict[int, Path]
    #: Frame indices drawn together on one render, per render, in sheet mode.
    frame_sheets: list[list[int]]
    #: Masked frames registered into the cell, by frame index.
    cells: dict[int, Path]


def frame_path(state: AnimationState, kind: str, index: int) -> Path:
    return state["work_dir"] / kind / state["set_name"] / f"{index:02d}.png"


def prop_clause(state: AnimationState) -> str:
    """What this character holds in this set, or the empty-hands instruction.

    A set the brief gives no props is drawn empty-handed. That is how a prop is
    kept out of one set without being taken off the character everywhere.

    Saying nothing is not the same as saying empty, and this was the last path
    still saying nothing: the director and the per-set key art already send
    EMPTY_HANDS, so a hand rising near the face came back holding a microphone
    only in the frames themselves.
    """
    held = state["spec"].animation_props.get(state["set_name"], ())
    return clauses(held, state["set_name"]) if held else EMPTY_HANDS


def view_clause(state: AnimationState) -> str:
    view = state["motion"].view
    if view not in FRAME_VIEWS:
        raise KeyError(f"motion sheet asks for an unknown view {view!r}")
    return FRAME_VIEWS[view]


def pose_cards(state: AnimationState) -> AnimationState:
    """One pose card per frame, cut from the bundle's traced footage.

    Shown, not told: a cue is a paragraph and an image generator will quietly
    flatten a paragraph back towards a neutral standing pose. A picture of the
    pose is not negotiable in the same way.

    The photographs win where it counts. Both references rank the poses about
    equally against the trace, but measured on amplitude — how big the drawn
    movement is beside the real one — the drawn cards come back at 0.43-0.70
    and the photographs at 0.9-1.2. A set that ranks perfectly at half size
    still reads as a sway. The cost is that a photograph also carries the
    performer's own costume, which bleeds: a lifted foot came back wearing the
    dancer's white trainer instead of the character's boot until the prompt
    named the footwear positively.
    """
    motion = state["motion"]
    if not motion.photos:
        return {"poses": {}}
    work = state["work_dir"] / "poses" / state["set_name"]
    return {"poses": dict(enumerate(write_photos(motion.photos, work))), "photographic": True}


def set_note_path(state: AnimationState) -> Path:
    return state["work_dir"] / "motion" / f"{state['set_name']}.note.txt"


def motion_stamp(state: AnimationState) -> Path:
    return state["work_dir"] / "source" / state["set_name"] / "motion.sha"


def motion_digest(motion: MotionSheet) -> str:
    """Everything the drawing reads off the sheet: the view, every frame's cue, and
    the pose photographs.

    The photographs are the pose reference on the photo route, and the cues
    never reach that prompt. Hashing the cues alone meant a bundle re-cut with
    new photos and the same cues kept its old frames: every cast dance drawn
    from the 200px thumbnails would have survived their replacement. A set with
    no photographs hashes as it always did, so its stamp still matches.
    """
    frames = "\n".join(f"{f.index}\t{f.role}\t{f.cue}\t{f.note}" for f in motion.frames)
    digest = hashlib.sha256(f"{motion.view}\n{frames}".encode())
    for photo in motion.photos:
        digest.update(Path(photo).read_bytes())
    return digest.hexdigest()


def drawn_stamp(state: AnimationState) -> Path:
    return state["work_dir"] / "source" / state["set_name"] / "drawn.sha"


def claim_frames(state: AnimationState, key: str, adopt: bool = False) -> None:
    """Leave `source/<set>/` holding only frames drawn under `key`, and stamp it so.

    `key` is the path's name, a tab, and a digest of everything its frames are
    a function of. A frame on disk is otherwise reused by `draw` for whatever
    reason it is there, which is how a different motion with the same frame
    count once kept the old art without a word. Frames stamped with another
    key — or with none, unless `adopt` vouches for them — are moved with
    everything drawn beside them (`.raw.png`, `.txt`, `.ruled.png`,
    `.unusable-N.png`) into `superseded/<path>-<digest8>/`, kept to compare
    against, never deleted. The stamp is written before anything is drawn, so a
    set stopped halfway resumes under the same key.
    """
    stamp = drawn_stamp(state)
    held = stamp.read_text().strip() if stamp.exists() else ""
    if held == key:
        return
    folder = stamp.parent
    drawn = sorted(folder.glob("[0-9][0-9].*"))
    if drawn and (held or not adopt):
        kind, _, digest = held.partition("\t")
        into = folder / "superseded" / (f"{kind}-{digest[:8]}" if held else "unstamped")
        into = next(
            candidate
            for candidate in (into, *(into.with_name(f"{into.name}-{n}") for n in range(1, 1000)))
            if not candidate.exists()
        )
        into.mkdir(parents=True)
        for path in drawn:
            path.rename(into / path.name)
        print(
            f"[{state['set_name']}] {len(drawn)} files drawn under another stamp moved to {into}",
            file=sys.stderr,
            flush=True,
        )
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(key + "\n")


def draws_by_video(state: AnimationState) -> bool:
    """Whether this set takes the video path: a machine is named and the set is traced."""
    return bool(state.get("machine") and state.get("photographic") and state.get("poses"))


def sources_are_current(state: AnimationState) -> bool:
    """Frames on disk that this motion sheet would only draw again the same way.

    Reuse used to key on the frame index alone, so handing a set a different
    sheet with the same frame numbers silently kept the old art: a traced sheet
    swapped in for a written one re-masked renders drawn from the prose it was
    meant to replace. A stamp that is missing counts as current, so a roster
    drawn before this existed keeps its renders instead of redrawing itself.
    """
    if not all(frame_path(state, "source", f.index).exists() for f in state["motion"].frames):
        return False
    stamp = motion_stamp(state)
    return not stamp.exists() or stamp.read_text().strip() == motion_digest(state["motion"])


def direct(state: AnimationState, model: BaseChatModel) -> AnimationState:
    """Bind the motion source to this character before a frame is drawn.

    The note only ever goes into a prompt, and a render already on disk is
    returned without its prompt being used. So a set whose frames are all drawn
    needs no note at all, and asking for one spends a model call per set on
    every rebuild — a re-mask of the whole roster was paying for thirty-two.
    A note that is asked for is kept, so the next run does not ask again.

    The note binds one motion source, naming its arc, rate, length and view, so
    it is only kept while the frames it was written for are the ones on disk.
    A set about to be redrawn from a different sheet asks for a new one.
    """
    record = set_note_path(state)
    if draws_by_video(state):
        # Nothing on the video path reads a note: the footage is the direction.
        return {"set_note": ""}
    if sources_are_current(state):
        return {"set_note": record.read_text().strip() if record.exists() else ""}
    motion = state["motion"]
    reply = model.invoke(
        [
            SystemMessage(DIRECTOR_SYSTEM),
            HumanMessage(
                director_request(
                    state["bible"], motion.arc, motion.fps, len(motion.frames), motion.view,
                    # The brief already says what this set holds. The director
                    # was reading it out of the identity text instead.
                    prop_clause(state),
                )
            ),
        ]
    )
    note = str(reply.content).strip()
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(note + "\n")
    return {"set_note": note}


def _draw_frame(
    state: AnimationState,
    frame: Frame,
    references: list[Path],
    draw_fn: Callable[..., Path],
) -> Path:
    spec = state["spec"]
    cue = frame.instruction
    pose = state.get("poses", {}).get(frame.index)
    detail = detail_frame(spec.detail_level) if state.get("detail_after_key", True) else None
    prompt = frame_prompt(
        spec,
        state["bible"],
        state["set_note"],
        view_clause(state),
        cue,
        frame.role,
        pose_reference=pose is not None,
        detail_level=spec.detail_level,
        detail_reference=detail is not None,
        props=prop_clause(state),
    )
    # The pose goes last, because the prompt calls it "the last reference image".
    references = [*references, *( [detail] if detail else [] ), *([pose] if pose else [])]
    return draw_fn(prompt, frame_path(state, "source", frame.index), references=references)


def keyframe(state: AnimationState, draw_fn: Callable[..., Path]) -> AnimationState:
    """Draw the extremes and fastest transitions first; they set the set's shape."""
    sources = dict(state.get("sources", {}))
    for frame in state["motion"].locked:
        sources[frame.index] = _draw_frame(state, frame, [state["key_art"]], draw_fn)
    return {"sources": sources}


def tween(state: AnimationState, draw_fn: Callable[..., Path]) -> AnimationState:
    """Fill the in-betweens in order, each continuing from the frame before it.

    Only the previous frame comes along, not the locked frame ahead. Both ends
    used to be attached, and between them and the key art the pose skeleton was
    outvoted three to one by references showing a character standing still —
    the set came back barely moving.
    """
    motion = state["motion"]
    sources = dict(state["sources"])
    for frame in motion.frames:
        if frame.is_locked:
            continue
        before, _ = motion.neighbours(frame)
        references = [state["key_art"]]
        if before in sources:
            references.append(sources[before])
        sources[frame.index] = _draw_frame(state, frame, references, draw_fn)
    return {"sources": sources}


#: Figures per render in sheet mode, drawn as a grid `FIGURES_PER_ROW` wide.
#:
#: 12 is the ceiling, measured up the ladder 8/12/16/24/32 on one 4-column grid.
#: Two separate things break above it. At 16 the count is still right but the
#: generator stops reading distinct poses and tiles one across the final row —
#: figures 12-15 came back at silhouette IoU 0.88-0.95, the same pose four
#: times. At 24 and 32 it will not draw the count asked for at all (20 and 28),
#: which also makes every per-frame measurement meaningless, because figure `n`
#: is no longer frame `n`. 8 and 12 both came back clean with no repeats.
#:
#: 12 costs resolution: its figures land ~344px tall against ~476px at 8, and
#: the cell holds a ~390px character, so 12 is upscaled into the cell where 8 is
#: downsampled into it. Twelve was chosen anyway, to cover a 24-frame set in two
#: renders rather than three.
#:
#: That trade was wrong. Belter's dance at 12 came back with the dancer's white
#: socks above the character's boots in 11 of 24 figures, and denim shorts in
#: place of her jeans on the second sheet — while the prompt named the boots and
#: the jeans positively and the key art beside it showed both. The costume
#: anchor is a sentence and the photograph is a picture, and at 344px the
#: picture stops being read as this performer's clothing and starts being read
#: as the character's. Eight does not argue with the photograph; it makes each
#: one big enough to read correctly.
FRAME_SHEET_SIZE = 8
FIGURES_PER_ROW = 4

#: The most figures one render takes when a set is drawn whole, as the Comfy
#: backend draws it. Batches of eight are separate rolls, and Nano Banana's
#: rolls drift apart in style — the hair fuller on one batch, the jacket pinker
#: on the next — where figures on one canvas are drawn together. So a set goes
#: in one render, eight to a row, and only one longer than this is split, into
#: equal parts. The ladder above was measured on codex, not on Nano Banana.
WHOLE_SET_SHEET = 24


def sheet_layout(frame_count: int, most: int = FRAME_SHEET_SIZE) -> tuple[int, int]:
    """Figures per render and per row, for a set of `frame_count` frames.

    `most` at `FRAME_SHEET_SIZE` is the eight-figure batching, unchanged. Below
    it, batches of `most` are laid out as near square as they go: four in a row
    on a landscape canvas leaves each figure the slot too narrow for Belter's
    hair and mic arm that eight across did. Above it the set is split into as
    few renders as `most` allows, as evenly as they go, and laid out eight to a row.
    """
    if most == FRAME_SHEET_SIZE:
        return most, FIGURES_PER_ROW
    if most < FRAME_SHEET_SIZE:
        return most, max(2, math.ceil(math.sqrt(most)))
    size = -(-frame_count // -(-frame_count // most))
    if size <= FRAME_SHEET_SIZE:
        return size, FIGURES_PER_ROW
    rows = -(-size // FRAME_SHEET_SIZE)
    return size, -(-size // rows)

#: Draws allowed per sheet before the set is given up on.
SHEET_ATTEMPTS = 4


def pose_edit_frames(state: AnimationState, draw_fn: Callable[..., Path]) -> AnimationState:
    """Draw a traced set one frame per render, each an edit of the set's reference.

    Every frame is the approved reference re-posed into one traced photograph,
    at full size. Nothing is drawn from a description, so nothing drifts
    between frames the way separate renders of a sheet did; and nothing shares
    a canvas, so no figure is shrunk to fit.

    The frames are registered together, with one transform for the set (see
    `canvas_to_cells`): every render is an edit of the same reference on the
    same canvas, so the generator's own sizes and positions are the ones to
    keep. They were not always: Belter's `shuffle-01` dance once came back
    between 1178 and 1430px tall, which is why frames were each rescaled off
    their traced pose, and why a frame that far out is now reported by name
    instead of being quietly rescaled.

    A frame on disk is kept, as `draw` keeps every render, so an interrupted set
    resumes where it stopped — but only one drawn this way from this motion
    (`claim_frames`). Anything else in the folder is moved aside first. A set
    from before the stamp is adopted when its `motion.sha` still matches.
    """
    motion = state["motion"]
    digest = motion_digest(motion)
    stamp = motion_stamp(state)
    claim_frames(
        state,
        "pose-edit\t" + digest,
        adopt=stamp.exists() and stamp.read_text().strip() == digest,
    )
    sources = {}
    for frame in motion.frames:
        sources[frame.index] = draw_fn(
            POSE_EDIT,
            frame_path(state, "source", frame.index),
            references=[state["key_art"], state["poses"][frame.index]],
            workflow=state["pose_workflow"],
        )
        if state.get("snap_to_key"):
            snap_file(sources[frame.index], state["key_art"])
    stamp.write_text(digest + "\n")
    return {
        "sources": sources,
        "frame_sheets": [[index] for index in sorted(sources)],
        "shared_canvas": True,
    }


def frame_sheet(state: AnimationState, draw_fn: Callable[..., Path]) -> AnimationState:
    """Draw the set as one image of every frame, then slice it into sources.

    The generator is shown all the poses at once and told nothing about size:
    it cannot follow a measurement, but it keeps figures it can see side by
    side the same size unasked. Layout is found afterwards by the backdrop
    between figures, never assumed, so a sheet that came back with the wrong
    number of figures is thrown away and drawn again.

    A traced set under a pose workflow is drawn frame by frame instead; see
    `pose_edit_frames`. Under a machine profile it is drawn from its source
    clip; see `cag.video`. A written set has no photographs to pose from, so it
    still comes here.
    """
    if draws_by_video(state):
        motion = state["motion"]
        if motion.clip is None or not motion.frame_times:
            # No quiet fallback to the pose-edit path: a machine was asked for,
            # and a set drawn some other way would pass for its output.
            raise MotionError(
                motion.clip_problem
                or f"{motion.name} carries no source clip; --machine draws traced sets from "
                f"video. Run `cag clips {motion.name}`, or build without --machine"
            )
        # Imported here: the video path builds on this module's paths and stamps.
        from .video import video_frames

        return video_frames(state, draw_fn)
    if state.get("pose_workflow") and state.get("photographic") and state.get("poses"):
        return pose_edit_frames(state, draw_fn)
    spec = state["spec"]
    motion = state["motion"]
    poses = state.get("poses", {})
    detail = detail_frame(spec.detail_level) if state.get("detail_after_key", True) else None
    sources = {}
    frame_sheets = []
    digest = motion_digest(motion)
    reusable = sources_are_current(state)
    #: The last figure of the sheet before this one. Each sheet is an independent
    #: render off the same key art, so the second one has never seen the first and
    #: wanders: this carries the character as actually drawn a moment earlier.
    #: Continuity only — the prompt says to take no pose from it, because the
    #: single-frame path measured a second figure reference outvoting the pose
    #: cards and flattening the movement. The set before this one, when there was
    #: one, hands over its last frame to start the chain off.
    carry: Path | None = state.get("carry")
    size, per_row = sheet_layout(
        len(motion.frames), state.get("frames_per_sheet") or FRAME_SHEET_SIZE
    )
    for start in range(0, len(motion.frames), size):
        chunk = motion.frames[start : start + size]
        frame_sheets.append([frame.index for frame in chunk])
        # Every frame of this chunk is already cut out and on disk. Redrawing the sheet
        # to slice it again would spend a render to arrive back at these same files, and
        # would do it with whatever note this run happens to hold.
        drawn_already = {
            frame.index: frame_path(state, "source", frame.index) for frame in chunk
        }
        if reusable and all(path.exists() for path in drawn_already.values()):
            sources.update(drawn_already)
            carry = drawn_already[chunk[-1].index]
            continue
        cues = [(frame.role, frame.instruction) for frame in chunk]
        pose_grid = None
        if poses:
            pose_grid = tile(
                [poses[frame.index] for frame in chunk],
                state["work_dir"] / "poses" / state["set_name"] / f"pose-grid-{start:02d}.png",
                columns=per_row,
                cell=(CARD_WIDTH, CARD_HEIGHT),
            )
        prompt = frame_sheet_prompt(
            spec,
            state["bible"],
            state["set_note"],
            view_clause(state),
            cues,
            pose_reference=pose_grid is not None,
            detail_level=spec.detail_level,
            detail_reference=detail is not None,
            per_row=per_row,
            props=prop_clause(state),
            photographic=state.get("photographic", False),
            carry_reference=carry is not None,
        )
        references = [
            state["key_art"],
            *([carry] if carry else []),
            *([detail] if detail else []),
            *([pose_grid] if pose_grid else []),
        ]
        # The render is of this chunk of this sheet, so the name says so. `draw` keeps
        # any render already at the path it is given, which is what resumes an
        # interrupted set — but under a name that only counted frames, a different
        # motion sheet resumed into the last one's art and sliced it up as its own.
        path = (
            state["work_dir"] / "source" / state["set_name"] / f"sheet-{start:02d}-{digest[:8]}.png"
        )
        # A sheet with the wrong figure count cannot be salvaged frame by frame, so it is drawn
        # again. Each draw is an independent roll: at the rate measured over a full eight-character
        # run, two tries lost about one set in six and four lose closer to one in forty. Only a
        # failure costs the extra call. The reject is kept beside the prompt, so what came back can
        # be read against it.
        for attempt in range(SHEET_ATTEMPTS):
            drawn = draw_fn(prompt, path, references=references)
            try:
                cells = slice_frame_sheet(drawn, len(chunk))
                break
            except MaskError as error:
                drawn.rename(path.with_name(f"{path.stem}.rejected-{attempt}.png"))
                last_error = error
        else:
            raise DrawError(f"could not draw {path.name}: {last_error}")
        for frame, cell in zip(chunk, cells):
            source = frame_path(state, "source", frame.index)
            cell.save(source, format="PNG")
            sources[frame.index] = source
        carry = sources[chunk[-1].index]
    stamp = motion_stamp(state)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(motion_digest(motion) + "\n")
    return {"sources": sources, "frame_sheets": frame_sheets}


def mask_frames(state: AnimationState, single_scale: bool = False) -> AnimationState:
    """Register every frame at the scale its own pose asks for.

    `single_scale` is for frames drawn together on one sheet: one factor for
    all of them, measured across the set. See `set_to_cells`.

    The static sheet's scale only comes along as a fallback. It is pixels per
    source pixel, read off one standing key art and calibrated to the 7' static
    cell; an animation frame is neither standing, nor drawn at the key art's
    size, nor 7' tall. Converting it to this frame's scale is the most it can
    say about a set with no landmarks to measure.
    """
    motion = state["motion"]
    fallback = state["scale"] * ANIM_PX_PER_INCH / PX_PER_INCH
    poses = {frame.index: frame.pts for frame in motion.frames}
    airborne = frozenset(frame.index for frame in motion.frames if frame.airborne)
    if state.get("shared_canvas"):
        cells, outliers = canvas_to_cells(
            state["sources"],
            partial(frame_path, state, "cells"),
            poses,
            motion.floor_y,
            motion.body_h,
            state["spec"].height_inches,
            airborne,
        )
        if outliers:
            print(
                f"[{state['set_name']}] frames {outliers} are more than 10% off the set's "
                "height: the generator may have drawn them at another zoom",
                file=sys.stderr,
                flush=True,
            )
        return {"cells": cells}
    if single_scale:
        return {
            "cells": set_to_cells(
                state["sources"],
                partial(frame_path, state, "cells"),
                poses,
                motion.floor_y,
                motion.body_h,
                state["spec"].height_inches,
                state.get("frame_sheets"),
                airborne,
            )
        }
    home = home_x(poses)
    return {
        "cells": {
            index: pose_to_cell(
                source,
                frame_path(state, "cells", index),
                poses.get(index, {}),
                motion.floor_y,
                motion.body_h,
                state["spec"].height_inches,
                fallback,
                home,
                index in airborne,
            )
            for index, source in sorted(state["sources"].items())
        }
    }


def build_animation_graph(
    model: BaseChatModel, draw_fn: Callable[..., Path] | None = None, frame_sheet_mode: bool = True
):
    """Sheet mode draws the whole set in one render; off, it draws a frame at a time."""
    # Resolved here, not as a default, so the module attribute stays swappable.
    draw_fn = draw_fn or draw
    graph = StateGraph(AnimationState)
    graph.add_node("poses", pose_cards)
    graph.add_node("direct", partial(direct, model=model))
    graph.add_node("mask", partial(mask_frames, single_scale=frame_sheet_mode))
    graph.add_edge(START, "poses")
    graph.add_edge("poses", "direct")
    if frame_sheet_mode:
        graph.add_node("frame_sheet", partial(frame_sheet, draw_fn=draw_fn))
        graph.add_edge("direct", "frame_sheet")
        graph.add_edge("frame_sheet", "mask")
    else:
        graph.add_node("keyframe", partial(keyframe, draw_fn=draw_fn))
        graph.add_node("tween", partial(tween, draw_fn=draw_fn))
        graph.add_edge("direct", "keyframe")
        graph.add_edge("keyframe", "tween")
        graph.add_edge("tween", "mask")
    graph.add_edge("mask", END)
    return graph.compile()
