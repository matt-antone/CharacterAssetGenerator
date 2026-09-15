"""`cag build` — one brief in, one character package out."""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .animation import build_animation_graph
from .assemble import gallery, gif_proof, sprite_sheet
from .chat_codex import ChatCodex
from .motion import load_motion
from .motion_writer import write_motion
from .prompts import KEY_VIEW
from .sets import plan_for, wanted
from .spec import CharacterSpec, load_spec
from .static_sheet import (
    ApprovalRequired,
    approve,
    build_static_graph,
    projection_views,
)

#: Wrap long sets so the sheet stays a reasonable shape to open.
SHEET_COLUMNS = 8


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def motion_for(spec: CharacterSpec, set_name: str, work_dir: Path, supplied: Path | None):
    """A traced sheet when one was supplied, otherwise one written from the brief."""
    if supplied:
        return load_motion(supplied)
    intent = spec.animations.get(set_name)
    if not intent:
        raise KeyError(f"{spec.name} has no {set_name!r} animation in their brief")
    return write_motion(
        ChatCodex(),
        spec,
        set_name,
        intent,
        plan_for(set_name),
        work_dir / "motion" / f"{set_name}.json",
    )


def render_set(
    spec: CharacterSpec,
    set_name: str,
    static: dict,
    work_dir: Path,
    supplied: Path | None,
    sheet_mode: bool = True,
) -> dict:
    """Draw and mask one animation set. Safe to run alongside other sets."""
    motion = motion_for(spec, set_name, work_dir, supplied)
    log(f"[{set_name}] {len(motion.frames)} frames at {motion.fps} fps, {motion.view} view")
    animated = build_animation_graph(ChatCodex(), sheet_mode=sheet_mode).invoke(
        {
            "spec": spec,
            "bible": static["bible"],
            "key_art": static["sources"][KEY_VIEW],
            "scale": static["scale"],
            "motion": motion,
            "set_name": set_name,
            "work_dir": work_dir,
        }
    )
    log(f"[{set_name}] done")
    return {"set_name": set_name, "motion": motion, "cells": animated["cells"]}


def build(
    spec_path: Path,
    motion_path: Path | None,
    set_names: list[str] | None,
    work_root: Path,
    out_root: Path,
    jobs: int = 1,
    sheet_mode: bool = True,
) -> Path:
    spec = load_spec(spec_path)
    work_dir = work_root / spec.slug
    out_dir = out_root / spec.slug
    # An explicit --set is the operator asking for that set by name, so it overrides the
    # config. Without one, the config decides which of the brief's animations are drawn.
    chosen = set_names if set_names is not None else wanted(sorted(spec.animations), "animations")
    if motion_path and len(chosen) != 1:
        raise SystemExit("--motion applies to a single --set; other sets write their own sheet")

    log(f"[static] {spec.name}: bible, key art, projection")
    try:
        static = build_static_graph(ChatCodex()).invoke({"spec": spec, "work_dir": work_dir})
    except ApprovalRequired as gate:
        raise SystemExit(
            f"[static] {spec.name}: key art is waiting for approval at {gate}\n"
            f"  approve it:  cag approve {spec_path}\n"
            f"  or redraw it: rm {gate} and build again"
        ) from None

    views = {}
    for view in [KEY_VIEW, *projection_views()]:
        views[view] = Path("views") / f"{view}.png"
        destination = out_dir / views[view]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(static["cells"][view], destination)

    results = []
    if chosen:
        log(f"[sets] {', '.join(chosen)} across {min(jobs, len(chosen))} lane(s)")
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = {
                name: pool.submit(
                    render_set, spec, name, static, work_dir, motion_path, sheet_mode
                )
                for name in chosen
            }
            for name, future in futures.items():
                try:
                    results.append(future.result())
                except Exception as error:  # one bad set must not lose the others
                    log(f"[{name}] FAILED: {error}")

    sets = []
    for result in sorted(results, key=lambda r: chosen.index(r["set_name"])):
        name, motion = result["set_name"], result["motion"]
        cells = [result["cells"][index] for index in sorted(result["cells"])]
        block = {
            "set_name": name,
            "frames": len(cells),
            "fps": motion.fps,
            "sheet": f"{name}-sheet.png",
            "proof": f"{name}-proof.gif",
        }
        sprite_sheet(cells, out_dir / block["sheet"], columns=min(SHEET_COLUMNS, len(cells)))
        gif_proof(cells, out_dir / block["proof"], motion.fps, loop=motion.loops)
        sets.append(block)

    page = gallery(
        out_dir / "index.html",
        spec.name,
        spec.height,
        spec.description,
        {view: str(path) for view, path in views.items()},
        sets,
    )
    log(str(page))
    return page


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="render a character package")
    build_parser.add_argument("spec", type=Path, help="path to a character brief")
    build_parser.add_argument(
        "--set",
        dest="set_names",
        action="append",
        help="animation set to render; repeatable. Default: every set in the brief",
    )
    build_parser.add_argument(
        "--motion", type=Path, help="a traced MotionArtist motion.json, for a single --set"
    )
    build_parser.add_argument(
        "--jobs", type=int, default=1, help="sets to render at once (default 1)"
    )
    build_parser.add_argument(
        "--per-frame",
        dest="sheet_mode",
        action="store_false",
        help="draw one render per frame instead of one sheet per set",
    )
    build_parser.add_argument("--work", type=Path, default=Path("work"))
    build_parser.add_argument("--out", type=Path, default=Path("outputs"))

    approve_parser = sub.add_parser(
        "approve", help="sign off on a character's key art so the rest can be drawn"
    )
    approve_parser.add_argument("spec", type=Path, help="path to a character brief")
    approve_parser.add_argument("--work", type=Path, default=Path("work"))

    args = parser.parse_args(argv)
    if args.command == "approve":
        spec = load_spec(args.spec)
        log(f"[static] {spec.name}: approved {approve(args.work / spec.slug)}")
        return 0
    build(args.spec, args.motion, args.set_names, args.work, args.out, args.jobs, args.sheet_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
