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

from .draw import draw
from .mask import mask_to_cell
from .motion import Frame, MotionSheet
from .prompts import DIRECTOR_SYSTEM, FRAME_VIEWS, director_request, frame_prompt
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


def direct(state: AnimationState, model: BaseChatModel) -> AnimationState:
    """Bind the motion source to this character before a frame is drawn."""
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
    return {"set_note": str(reply.content).strip()}


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


def mask_frames(state: AnimationState) -> AnimationState:
    return {
        "cells": {
            index: mask_to_cell(source, frame_path(state, "cells", index), state["scale"])
            for index, source in sorted(state["sources"].items())
        }
    }


def build_animation_graph(model: BaseChatModel, draw_fn: Callable[..., Path] | None = None):
    # Resolved here, not as a default, so the module attribute stays swappable.
    draw_fn = draw_fn or draw
    graph = StateGraph(AnimationState)
    graph.add_node("poses", pose_sheets)
    graph.add_node("direct", partial(direct, model=model))
    graph.add_node("keyframe", partial(keyframe, draw_fn=draw_fn))
    graph.add_node("tween", partial(tween, draw_fn=draw_fn))
    graph.add_node("mask", mask_frames)
    graph.add_edge(START, "poses")
    graph.add_edge("poses", "direct")
    graph.add_edge("direct", "keyframe")
    graph.add_edge("keyframe", "tween")
    graph.add_edge("tween", "mask")
    graph.add_edge("mask", END)
    return graph.compile()
