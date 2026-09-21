"""The static half of a character package: key art, then the projection sheet.

A LangGraph run, because the animation half branches off the same state. The
order is load-bearing: key art is drawn and measured first, and every later
render references it, so identity and scale are fixed before anything else is
committed to. A human signs off on the key art in between: it is the reference
every other render quotes, so a wrong one is a whole wrong character.
"""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path
from typing import Callable, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .draw import draw
from .mask import cutout, key_art_scale, mask_to_cell
from .prompts import (
    BIBLE_SYSTEM,
    EMPTY_HANDS,
    KEY_FRAME_VIEW,
    KEY_VIEW,
    VIEWS,
    assemble_bible,
    bible_request,
    view_prompt,
)
from .props import clauses
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


def source_path(work_dir: Path, view: str) -> Path:
    return Path(work_dir) / "source" / f"{view}.png"


def cell_path(work_dir: Path, view: str) -> Path:
    return Path(work_dir) / "cells" / f"{view}.png"


class ApprovalRequired(RuntimeError):
    """Raised when the key art on disk has not been signed off on."""


def approval_record(work_dir: Path) -> Path:
    return Path(work_dir) / "key-approved.txt"


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def is_approved(work_dir: Path, key_art: Path) -> bool:
    """Has this exact key art been signed off on?

    The record holds the art's digest, not just a flag, so redrawing the key art
    revokes the approval instead of inheriting the old one.
    """
    record = approval_record(work_dir)
    return record.exists() and record.read_text().strip() == _digest(key_art)


def approve(work_dir: Path) -> Path:
    """Sign off on the key art currently on disk. Returns what was approved."""
    key_art = source_path(work_dir, KEY_VIEW)
    if not key_art.exists():
        raise FileNotFoundError(f"no key art at {key_art}; build it first")
    record = approval_record(work_dir)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(_digest(key_art) + "\n")
    return key_art


def write_bible(state: StaticState, model: BaseChatModel) -> StaticState:
    """Lock the character's appearance in words before drawing anything.

    A brief that states its own appearance assembles the bible from those
    fields, and nothing is cached: the text is a function of the spec, so the
    spec is the record, and it is a record git can show you. The freeze below
    exists only for the model path, which is the only thing here that returns
    different words when asked twice.
    """
    assembled = assemble_bible(state["spec"])
    if assembled:
        return {"bible": assembled}
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
            props=clauses(spec.props, None),
        ),
        source_path(state["work_dir"], KEY_VIEW),
        references=[detail] if detail else [],
    )
    return {"sources": {KEY_VIEW: path}}


def set_key_art(
    spec: CharacterSpec,
    set_name: str,
    bible: str,
    work_dir: Path,
    key_art: Path,
    draw_fn: Callable[..., Path] | None = None,
    view: str = KEY_FRAME_VIEW,
) -> Path:
    """The key art with this set's hands and facing, drawn once per set that needs it.

    The prop sits on the character in the key art, because that is where a
    designer puts it. Every frame prompt then says to match the reference for
    prop hand — so a set the brief draws empty-handed inherited the microphone
    from its own reference, and nothing in the prompt contradicted the picture.
    Belter's dance came back holding one in almost every figure while its
    director note said, in the same prompt, that both hands were empty.

    Words did not win that argument and were never going to: the reference is a
    picture of the character holding it. So the set gets a reference whose hands
    are its own. Drawn from the approved key art, so identity and scale come
    with it, and skipped entirely for a set whose hands already match.

    Facing is the same argument again, with a picture on the other side. The key
    art is a three-quarter view; a set traced in profile was still drawn against
    it, and the one photograph in the reference list beat the view clause every
    time — profile frames came back at three-quarters. So a set whose motion
    faces another way gets its reference redrawn at that facing, and the
    three-quarter key art is not among the frame references at all.
    """
    held = spec.animation_props.get(set_name, ())
    if held == spec.props and view == KEY_FRAME_VIEW:
        return key_art
    dst = work_dir / "source" / f"{set_name}-key-{view.replace('/', '-')}.png"
    detail = detail_frame(spec.detail_level)
    return (draw_fn or draw)(
        view_prompt(
            spec,
            bible,
            KEY_VIEW if view == KEY_FRAME_VIEW else view,
            detail_level=spec.detail_level,
            detail_reference=detail is not None,
            props=clauses(held, set_name) if held else EMPTY_HANDS,
        ),
        dst,
        references=[key_art, *([detail] if detail else [])],
    )


def check_approval(state: StaticState) -> StaticState:
    """Stop the run until a human has approved the key art.

    Everything downstream — the scale, the projection views, every animation
    frame — is drawn against this one render, so it is the only place worth a
    gate. Nothing here prompts: builds run several at a time in the background,
    where there is no one at a terminal to answer.
    """
    key_art = state["sources"][KEY_VIEW]
    if not is_approved(state["work_dir"], key_art):
        raise ApprovalRequired(str(key_art))
    return {}


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
                props=clauses(spec.props, None),
            ),
            source_path(state["work_dir"], view),
            references=[key_art, detail] if detail else [key_art],
        )
    return {"sources": sources}


def mask_views(state: StaticState) -> StaticState:
    return {
        "cells": {
            view: mask_to_cell(source, cell_path(state["work_dir"], view), state["scale"])
            for view, source in state["sources"].items()
        }
    }


def build_static_graph(model: BaseChatModel, draw_fn: Callable[..., Path] | None = None):
    # Resolved here, not as a default, so the module attribute stays swappable.
    draw_fn = draw_fn or draw
    graph = StateGraph(StaticState)
    graph.add_node("bible", partial(write_bible, model=model))
    graph.add_node("key_art", partial(draw_key_art, draw_fn=draw_fn))
    graph.add_node("approval", check_approval)
    graph.add_node("scale", measure_scale)
    graph.add_node("projection", partial(draw_projection, draw_fn=draw_fn))
    graph.add_node("mask", mask_views)
    graph.add_edge(START, "bible")
    graph.add_edge("bible", "key_art")
    graph.add_edge("key_art", "approval")
    graph.add_edge("approval", "scale")
    graph.add_edge("scale", "projection")
    graph.add_edge("projection", "mask")
    graph.add_edge("mask", END)
    return graph.compile()
