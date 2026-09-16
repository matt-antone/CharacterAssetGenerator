"""One animation set, drawn the way a studio draws one.

The motion director binds the motion source to this character. The keyframer
draws the extremes and the fastest transitions. The tweener fills the gaps, each
in-between drawn from its two locked neighbours — drawn, not interpolated. No
cross-fade, no optical flow, no frame grid used as a creative source.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .assemble import sprite_sheet
from .draw import DrawError, draw
from .geometry import ANIM_PX_PER_INCH, PX_PER_INCH
from .mask import MaskError, pose_to_cell, set_to_cells, slice_sheet
from .motion import Frame, MotionSheet
from .prompts import DIRECTOR_SYSTEM, FRAME_VIEWS, director_request, frame_prompt, sheet_prompt
from .skeleton import write_skeletons
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
    #: Stick-figure pose reference per frame index, when the sheet carries poses.
    poses: dict[int, Path]
    #: Raw magenta-backdrop frames, by frame index.
    sources: dict[int, Path]
    #: Frame indices drawn together on one render, per render, in sheet mode.
    sheets: list[list[int]]
    #: Masked frames registered into the cell, by frame index.
    cells: dict[int, Path]


def frame_path(state: AnimationState, kind: str, index: int) -> Path:
    return state["work_dir"] / kind / state["set_name"] / f"{index:02d}.png"


def view_clause(state: AnimationState) -> str:
    view = state["motion"].view
    if view not in FRAME_VIEWS:
        raise KeyError(f"motion sheet asks for an unknown view {view!r}")
    return FRAME_VIEWS[view]


def pose_sheets(state: AnimationState) -> AnimationState:
    """Draw the motion sheet's poses, so each frame is shown its pose, not told it."""
    motion = state["motion"]
    if not motion.has_poses:
        return {"poses": {}}
    paths = write_skeletons(
        [frame.pts for frame in motion.frames],
        motion.floor_y,
        motion.body_h,
        state["work_dir"] / "poses" / state["set_name"],
    )
    return {"poses": dict(enumerate(paths))}


def set_note_path(state: AnimationState) -> Path:
    return state["work_dir"] / "motion" / f"{state['set_name']}.note.txt"


def _every_source_drawn(state: AnimationState) -> bool:
    return all(
        frame_path(state, "source", frame.index).exists() for frame in state["motion"].frames
    )


def direct(state: AnimationState, model: BaseChatModel) -> AnimationState:
    """Bind the motion source to this character before a frame is drawn.

    The note only ever goes into a prompt, and a render already on disk is
    returned without its prompt being used. So a set whose frames are all drawn
    needs no note at all, and asking for one spends a model call per set on
    every rebuild — a re-mask of the whole roster was paying for thirty-two.
    A note that is asked for is kept, so the next run does not ask again.
    """
    record = set_note_path(state)
    if record.exists():
        return {"set_note": record.read_text().strip()}
    if _every_source_drawn(state):
        return {"set_note": ""}
    motion = state["motion"]
    reply = model.invoke(
        [
            SystemMessage(DIRECTOR_SYSTEM),
            HumanMessage(
                director_request(
                    state["bible"], motion.arc, motion.fps, len(motion.frames), motion.view
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
    cue = f"{frame.cue} {frame.note}".strip()
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


#: Figures per render in sheet mode. The whole set in one image when it fits,
#: so every figure is drawn at one size against its neighbours.
# ponytail: 8 fills a 4x2 grid; drop to 4 if the outline comes back thinner than CUT_IN can spare.
SHEET_FRAMES = 8
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
    for start in range(0, len(motion.frames), SHEET_FRAMES):
        chunk = motion.frames[start : start + SHEET_FRAMES]
        sheets.append([frame.index for frame in chunk])
        # Every frame of this chunk is already cut out and on disk. Redrawing the sheet
        # to slice it again would spend a render to arrive back at these same files, and
        # would do it with whatever note this run happens to hold.
        drawn_already = {
            frame.index: frame_path(state, "source", frame.index) for frame in chunk
        }
        if all(path.exists() for path in drawn_already.values()):
            sources.update(drawn_already)
            continue
        cues = [(frame.role, f"{frame.cue} {frame.note}".strip()) for frame in chunk]
        pose_grid = None
        if poses:
            pose_grid = sprite_sheet(
                [poses[frame.index] for frame in chunk],
                state["work_dir"] / "poses" / state["set_name"] / f"sheet-{start:02d}.png",
                columns=FIGURES_PER_ROW,
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
        )
        references = [state["key_art"], *([detail] if detail else []), *([pose_grid] if pose_grid else [])]
        path = state["work_dir"] / "source" / state["set_name"] / f"sheet-{start:02d}.png"
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
            )
        }
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
