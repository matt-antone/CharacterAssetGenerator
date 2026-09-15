"""The static half of a character package: key art, then the projection sheet.

A LangGraph run, because the animation half branches off the same state. The
order is load-bearing: key art is drawn and measured first, and every later
render references it, so identity and scale are fixed before anything else is
committed to.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .draw import draw
from .mask import cutout, key_art_scale, mask_to_cell
from .prompts import BIBLE_SYSTEM, KEY_VIEW, VIEWS, bible_request, view_prompt
from .sets import REQUIRED_VIEWS, wanted
from .style import detail_frame
from .spec import CharacterSpec


def projection_views() -> list[str]:
    """The views to draw beside the key art, minus any switched off in the config."""
    return wanted([v for v in VIEWS if v != KEY_VIEW], "views", REQUIRED_VIEWS)


class StaticState(TypedDict, total=False):
    spec: CharacterSpec
    work_dir: Path
    bible: str
    #: Raw magenta-backdrop renders, by view.
    sources: dict[str, Path]
    #: Masked renders registered into the cell, by view.
    cells: dict[str, Path]
    #: Source-pixels-to-cell-pixels factor, measured once from the key art.
    scale: float


def source_path(state: StaticState, view: str) -> Path:
    return state["work_dir"] / "source" / f"{view}.png"


def cell_path(state: StaticState, view: str) -> Path:
    return state["work_dir"] / "cells" / f"{view}.png"


def write_bible(state: StaticState, model: BaseChatModel) -> StaticState:
    """Lock the character's appearance in words before drawing anything.

    A bible already on disk is the one every existing frame was drawn against,
    so it is read back rather than rewritten. Asking for a second description of
    the same brief returns different words, and a set rendered later would then
    quote a different identity than the sets beside it.
    """
    record = state["work_dir"] / "bible.txt"
    if record.exists():
        return {"bible": record.read_text().strip()}
    reply = model.invoke(
        [SystemMessage(BIBLE_SYSTEM), HumanMessage(bible_request(state["spec"]))]
    )
    bible = str(reply.content).strip()
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(bible + "\n")
    return {"bible": bible}


def draw_key_art(state: StaticState, draw_fn: Callable[..., Path]) -> StaticState:
    spec = state["spec"]
    detail = detail_frame(spec.detail_level)
    path = draw_fn(
        view_prompt(
            spec,
            state["bible"],
            KEY_VIEW,
            detail_level=spec.detail_level,
            detail_reference=detail is not None,
        ),
        source_path(state, KEY_VIEW),
        references=[detail] if detail else [],
    )
    return {"sources": {KEY_VIEW: path}}


def measure_scale(state: StaticState) -> StaticState:
    """Read the character's working scale off the key art, once, for good."""
    key_art = cutout(state["sources"][KEY_VIEW])
    return {"scale": key_art_scale(key_art, state["spec"].height_inches)}


def draw_projection(state: StaticState, draw_fn: Callable[..., Path]) -> StaticState:
    spec = state["spec"]
    key_art = state["sources"][KEY_VIEW]
    detail = detail_frame(spec.detail_level)
    sources = dict(state["sources"])
    for view in projection_views():
        sources[view] = draw_fn(
            view_prompt(
                spec,
                state["bible"],
                view,
                detail_level=spec.detail_level,
                detail_reference=detail is not None,
            ),
            source_path(state, view),
            references=[key_art, detail] if detail else [key_art],
        )
    return {"sources": sources}


def mask_views(state: StaticState) -> StaticState:
    return {
        "cells": {
            view: mask_to_cell(source, cell_path(state, view), state["scale"])
            for view, source in state["sources"].items()
        }
    }


def build_static_graph(model: BaseChatModel, draw_fn: Callable[..., Path] | None = None):
    # Resolved here, not as a default, so the module attribute stays swappable.
    draw_fn = draw_fn or draw
    graph = StateGraph(StaticState)
    graph.add_node("bible", partial(write_bible, model=model))
    graph.add_node("key_art", partial(draw_key_art, draw_fn=draw_fn))
    graph.add_node("scale", measure_scale)
    graph.add_node("projection", partial(draw_projection, draw_fn=draw_fn))
    graph.add_node("mask", mask_views)
    graph.add_edge(START, "bible")
    graph.add_edge("bible", "key_art")
    graph.add_edge("key_art", "scale")
    graph.add_edge("scale", "projection")
    graph.add_edge("projection", "mask")
    graph.add_edge("mask", END)
    return graph.compile()
