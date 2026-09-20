"""One animation set, drawn the way a studio draws one.

The motion director binds the motion source to this character. The keyframer
draws the extremes and the fastest transitions. The tweener fills the gaps, each
in-between drawn from its two locked neighbours — drawn, not interpolated. No
cross-fade, no optical flow, no frame grid used as a creative source.
"""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path
from typing import Callable, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .assemble import sprite_sheet
from .draw import DrawError, draw
from .geometry import ANIM_PX_PER_INCH, PX_PER_INCH
from .mask import MaskError, home_x, pose_to_cell, set_to_cells, slice_sheet
from .motion import Frame, MotionSheet
from .prompts import DIRECTOR_SYSTEM, FRAME_VIEWS, director_request, frame_prompt, sheet_prompt
from .poses import CARD_HEIGHT, CARD_WIDTH, write_photos
from .props import clauses
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
    #: Whether those references are photographs of the performer rather than
    #: drawn figures. They are described very differently to the generator: one
    #: is a person to copy a pose off and take nothing else from, the other a
    #: colour-coded diagram carrying no costume at all.
    photographic: bool
    #: Raw magenta-backdrop frames, by frame index.
    sources: dict[int, Path]
    #: Frame indices drawn together on one render, per render, in sheet mode.
    sheets: list[list[int]]
    #: Masked frames registered into the cell, by frame index.
    cells: dict[int, Path]


def frame_path(state: AnimationState, kind: str, index: int) -> Path:
    return state["work_dir"] / kind / state["set_name"] / f"{index:02d}.png"


def prop_clause(state: AnimationState) -> str:
    """What this character holds in this set, or "" for empty hands.

    A set the brief gives no props is drawn empty-handed. That is how a prop is
    kept out of one set without being taken off the character everywhere.
    """
    held = state["spec"].animation_props.get(state["set_name"], ())
    return clauses(held, state["set_name"]) if held else ""


def view_clause(state: AnimationState) -> str:
    view = state["motion"].view
    if view not in FRAME_VIEWS:
        raise KeyError(f"motion sheet asks for an unknown view {view!r}")
    return FRAME_VIEWS[view]


def pose_sheets(state: AnimationState) -> AnimationState:
    """One pose card per frame: the traced footage when the bundle carries it,
    the drawn sprite sheet otherwise.

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
    """Everything the drawing reads off the sheet: the view and every frame's cue."""
    frames = "\n".join(f"{f.index}\t{f.role}\t{f.cue}\t{f.note}" for f in motion.frames)
    return hashlib.sha256(f"{motion.view}\n{frames}".encode()).hexdigest()


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
    detail = detail_frame(spec.detail_level)
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
        photographic=state.get("photographic", False),
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
#: 12 costs some resolution: its figures land ~344px tall against ~476px at 8,
#: and the cell holds a ~390px character, so 12 is upscaled into the cell where
#: 8 is downsampled into it. Twelve is chosen anyway because it covers a
#: 24-frame set in two renders.
SHEET_FRAMES = 12
FIGURES_PER_ROW = 4

#: Draws allowed per sheet before the set is given up on.
SHEET_ATTEMPTS = 4


def sheet(state: AnimationState, draw_fn: Callable[..., Path]) -> AnimationState:
    """Draw the set as one image of every frame, then slice it into sources.

    The generator is shown all the poses at once and told nothing about size:
    it cannot follow a measurement, but it keeps figures it can see side by
    side the same size unasked. Layout is found afterwards by the backdrop
    between figures, never assumed, so a sheet that came back with the wrong
    number of figures is thrown away and drawn again.
    """
    spec = state["spec"]
    motion = state["motion"]
    poses = state.get("poses", {})
    detail = detail_frame(spec.detail_level)
    sources = {}
    sheets = []
    digest = motion_digest(motion)
    reusable = sources_are_current(state)
    for start in range(0, len(motion.frames), SHEET_FRAMES):
        chunk = motion.frames[start : start + SHEET_FRAMES]
        sheets.append([frame.index for frame in chunk])
        # Every frame of this chunk is already cut out and on disk. Redrawing the sheet
        # to slice it again would spend a render to arrive back at these same files, and
        # would do it with whatever note this run happens to hold.
        drawn_already = {
            frame.index: frame_path(state, "source", frame.index) for frame in chunk
        }
        if reusable and all(path.exists() for path in drawn_already.values()):
            sources.update(drawn_already)
            continue
        cues = [(frame.role, frame.instruction) for frame in chunk]
        pose_grid = None
        if poses:
            pose_grid = sprite_sheet(
                [poses[frame.index] for frame in chunk],
                state["work_dir"] / "poses" / state["set_name"] / f"sheet-{start:02d}.png",
                columns=FIGURES_PER_ROW,
                cell=(CARD_WIDTH, CARD_HEIGHT),
            )
        prompt = sheet_prompt(
            spec,
            state["bible"],
            state["set_note"],
            view_clause(state),
            cues,
            pose_reference=pose_grid is not None,
            detail_level=spec.detail_level,
            detail_reference=detail is not None,
            per_row=FIGURES_PER_ROW,
            props=prop_clause(state),
            photographic=state.get("photographic", False),
        )
        references = [state["key_art"], *([detail] if detail else []), *([pose_grid] if pose_grid else [])]
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
                cells = slice_sheet(drawn, len(chunk))
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
    stamp = motion_stamp(state)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(motion_digest(motion) + "\n")
    return {"sources": sources, "sheets": sheets}


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
    if single_scale:
        return {
            "cells": set_to_cells(
                state["sources"],
                partial(frame_path, state, "cells"),
                poses,
                motion.floor_y,
                motion.body_h,
                state["spec"].height_inches,
                state.get("sheets"),
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
    model: BaseChatModel, draw_fn: Callable[..., Path] | None = None, sheet_mode: bool = True
):
    """Sheet mode draws the whole set in one render; off, it draws a frame at a time."""
    # Resolved here, not as a default, so the module attribute stays swappable.
    draw_fn = draw_fn or draw
    graph = StateGraph(AnimationState)
    graph.add_node("poses", pose_sheets)
    graph.add_node("direct", partial(direct, model=model))
    graph.add_node("mask", partial(mask_frames, single_scale=sheet_mode))
    graph.add_edge(START, "poses")
    graph.add_edge("poses", "direct")
    if sheet_mode:
        graph.add_node("sheet", partial(sheet, draw_fn=draw_fn))
        graph.add_edge("direct", "sheet")
        graph.add_edge("sheet", "mask")
    else:
        graph.add_node("keyframe", partial(keyframe, draw_fn=draw_fn))
        graph.add_node("tween", partial(tween, draw_fn=draw_fn))
        graph.add_edge("direct", "keyframe")
        graph.add_edge("keyframe", "tween")
        graph.add_edge("tween", "mask")
    graph.add_edge("mask", END)
    return graph.compile()
