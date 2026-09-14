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
    cue = f"{frame.cue} {frame.note}".strip()
    prompt = frame_prompt(
        state["spec"], state["bible"], state["set_note"], view_clause(state), cue, frame.role
    )
    return draw_fn(prompt, frame_path(state, "source", frame.index), references=references)


def keyframe(state: AnimationState, draw_fn: Callable[..., Path]) -> AnimationState:
    """Draw the extremes and fastest transitions first; they set the set's shape."""
    sources = dict(state.get("sources", {}))
    for frame in state["motion"].locked:
        sources[frame.index] = _draw_frame(state, frame, [state["key_art"]], draw_fn)
    return {"sources": sources}


def tween(state: AnimationState, draw_fn: Callable[..., Path]) -> AnimationState:
    """Fill the in-betweens in order, each from the frames either side of it."""
    motion = state["motion"]
    sources = dict(state["sources"])
    for frame in motion.frames:
        if frame.is_locked:
            continue
        before, after = motion.neighbours(frame)
        references = [state["key_art"]]
        references += [sources[i] for i in (before, after) if i in sources]
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
    graph.add_node("direct", partial(direct, model=model))
    graph.add_node("keyframe", partial(keyframe, draw_fn=draw_fn))
    graph.add_node("tween", partial(tween, draw_fn=draw_fn))
    graph.add_node("mask", mask_frames)
    graph.add_edge(START, "direct")
    graph.add_edge("direct", "keyframe")
    graph.add_edge("keyframe", "tween")
    graph.add_edge("tween", "mask")
    graph.add_edge("mask", END)
    return graph.compile()
