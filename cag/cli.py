"""`cag build` — one brief in, one character package out."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from functools import partial
from pathlib import Path

from .animation import build_animation_graph
from .assemble import (
    MANIFEST,
    PORTRAIT_SIZES,
    gallery,
    gif_proof,
    location,
    manifest,
    portrait,
    tile,
)
from .chat_codex import ChatCodex
from .draw import draw
from .edit import serve
from .motion import BUNDLE, MotionError, library, load_motion, read_bundle
from .motion_writer import write_motion
from .prompts import KEY_VIEW
from .sets import plan_for, wanted
from .spec import CharacterSpec, load_spec
from .static_sheet import (
    ApprovalRequired,
    approve,
    build_static_graph,
    projection_views,
    set_key_art,
)

#: Wrap long sets so the sheet stays a reasonable shape to open.
SHEET_COLUMNS = 8

#: Where a sheet a brief names by name is looked up, laid out as MotionArtist
#: lays them out: <root>/<name>/motion.json. Sheets live in this repo, not in
#: the checkout that traced them — a brief that outlives its footage still
#: builds, and a sheet cannot be cleaned away from under the roster.
MOTION_ROOT = Path("motions")

#: What a brief writes instead of a sheet name to have one chosen for it.
AUTO = "auto"

#: Where briefs are filed. The folder a brief sits in inside this root is its
#: theme, and the package lands under a folder of that name.
SPECS_ROOT = Path("specs")


def out_for(out_root: Path, spec_path: Path, slug: str, specs_root: Path = SPECS_ROOT) -> Path:
    """Where this brief's package lands: `<out>/<theme>/<slug>`.

    The theme is the folder the brief is filed in under `specs/` —
    `specs/halloween/mort.json` writes `outputs/halloween/mort/`. Anything else
    — a brief loose in `specs/`, a fixture, a one-off passed by path — has no
    theme and writes `outputs/<slug>/` as before.

    Only a folder's *name* is ever used, never its path, so a brief reached
    through `..` or from outside the repo still writes inside the output folder.
    """
    parent = spec_path.resolve().parent
    theme = parent.name if parent.parent == specs_root.resolve() else ""
    return out_root / theme / slug


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def motion_for(
    spec: CharacterSpec,
    set_name: str,
    work_dir: Path,
    supplied: Path | None,
    motion_root: Path = MOTION_ROOT,
):
    """The sheet this set is drawn from: supplied, named by the brief, or written.

    `--motion` is the operator pointing at one file for this run, so it wins over
    what the brief names, the same way `--set` wins over the config.
    """
    if supplied:
        return load_motion(supplied)
    named = spec.motions.get(set_name)
    if named == AUTO:
        name, bundle = auto_bundle(spec, set_name, motion_root)
        log(f"[{set_name}] auto: the {name!r} sheet")
        return bundle.load()
    if named:
        bundle = motion_root / named
        if not (bundle / BUNDLE).exists():
            raise MotionError(
                f"{spec.name} names the {named!r} motion sheet for {set_name}, "
                f"but there is no bundle at {bundle}"
            )
        return read_bundle(bundle).load()
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


def auto_bundle(spec: CharacterSpec, set_name: str, motion_root: Path) -> tuple[str, "Bundle"]:
    """Choose a traced sheet for a set whose brief did not name one.

    Every traced sheet reads as a coherent performance — that is what tracing
    buys, and it is the difference a written sheet cannot make up. So this
    spreads the library across the roster rather than ranking it: the same
    character keeps the same dance between runs, and twelve of them do not all
    dance the same one. Which sheet suits which character is a choreography
    call, and a brief that makes it names the sheet instead.
    """
    sheets = library(motion_root)
    if not sheets:
        raise MotionError(
            f"{spec.name} asks for an automatic sheet for {set_name}, "
            f"but there are none under {motion_root}"
        )
    names = sorted(sheets)
    seed = hashlib.sha256(f"{spec.slug}\t{set_name}".encode()).hexdigest()
    name = names[int(seed, 16) % len(names)]
    return name, sheets[name]


def render_set(
    spec: CharacterSpec,
    set_name: str,
    static: dict,
    work_dir: Path,
    supplied: Path | None,
    frame_sheet_mode: bool = True,
    draw_backend: str = "codex",
    motion_root: Path = MOTION_ROOT,
    carry: Path | None = None,
) -> dict:
    """Draw and mask one animation set, continuing from the set drawn before it.

    `carry` is that set's last frame. It rides along with this set's key art as
    a picture of the character as actually drawn a moment ago, and this set
    hands its own last frame on in turn. That chain is why the sets are drawn in
    order rather than in parallel lanes.
    """
    motion = motion_for(spec, set_name, work_dir, supplied, motion_root)
    log(f"[{set_name}] {len(motion.frames)} frames at {motion.fps} fps, {motion.view} view")
    if not motion.seams_cleanly:
        # The tracer compared the last frame to the first and they do not meet.
        # Nothing downstream can fix that, so it is said once, where it is chosen.
        log(f"[{set_name}] the {motion.name!r} sheet loops on a {motion.seam!r} seam: "
            "the proof will jump from the last frame back to the first")
    # None keeps callers pointed at each module's own `draw` name (unpatched, that's
    # the seam tests replace) instead of forcing a swap when nothing was asked for.
    draw_fn = partial(draw, backend=draw_backend) if draw_backend != "codex" else None
    # The reference this set is drawn against has this set's hands, not the key
    # art's: a prop on the character follows it into every frame that quotes it.
    key_art = set_key_art(
        spec, set_name, static["bible"], work_dir, static["sources"][KEY_VIEW], draw_fn, motion.view
    )
    log(f"[{set_name}] reference: {key_art.name}")
    animated = build_animation_graph(ChatCodex(), draw_fn=draw_fn, frame_sheet_mode=frame_sheet_mode).invoke(
        {
            "spec": spec,
            "bible": static["bible"],
            "key_art": key_art,
            "scale": static["scale"],
            "motion": motion,
            "set_name": set_name,
            "work_dir": work_dir,
            **({"carry": carry} if carry else {}),
        }
    )
    log(f"[{set_name}] done")
    sources = animated["sources"]
    return {
        "set_name": set_name,
        "motion": motion,
        "cells": animated["cells"],
        "last": sources[max(sources)],
    }


def build(
    spec_path: Path,
    motion_path: Path | None,
    set_names: list[str] | None,
    work_root: Path,
    out_root: Path,
    frame_sheet_mode: bool = True,
    draw_backend: str = "codex",
    motion_root: Path = MOTION_ROOT,
) -> Path:
    spec = load_spec(spec_path)
    work_dir = work_root / spec.slug
    out_dir = out_for(out_root, spec_path, spec.slug)
    # An explicit --set is the operator asking for that set by name, so it overrides the
    # config. Without one, the config decides which of the brief's animations are drawn.
    chosen = set_names if set_names is not None else wanted(sorted(spec.animations), "animations")
    if motion_path and len(chosen) != 1:
        raise SystemExit("--motion applies to a single --set; other sets write their own sheet")

    log(f"[static] {spec.name}: bible, key art, projection")
    try:
        draw_fn = partial(draw, backend=draw_backend) if draw_backend != "codex" else None
        static = build_static_graph(ChatCodex(), draw_fn=draw_fn).invoke(
            {"spec": spec, "work_dir": work_dir}
        )
    except ApprovalRequired as gate:
        raise SystemExit(
            f"[static] {spec.name}: key art is waiting for approval at {gate}\n"
            f"  approve it:  uv run cag approve {spec_path}\n"
            f"  or redraw it: rm {gate} and build again"
        ) from None

    views = {}
    for view in [KEY_VIEW, *projection_views()]:
        views[view] = Path("views") / f"{view}.png"
        destination = out_dir / views[view]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(static["cells"][view], destination)

    background = None
    if static.get("location"):
        background = "location.png"
        log(f"[location] {spec.name}: {location(static['location'], out_dir / background)}")

    results = []
    if chosen:
        log(f"[sets] {' -> '.join(chosen)}, each drawn from the one before it")
        # Each set continues from the last frame of the set before it, so they are drawn
        # in order. A set asked for on its own starts from the key art, as the first one
        # does: the chain is what a full build has, not a promise every invocation keeps.
        carry = None
        for name in chosen:
            try:
                result = render_set(
                    spec,
                    name,
                    static,
                    work_dir,
                    motion_path,
                    frame_sheet_mode,
                    draw_backend,
                    motion_root,
                    carry,
                )
            except Exception as error:  # one bad set must not lose the others
                log(f"[{name}] FAILED: {error}")
                continue
            results.append(result)
            carry = result["last"]

    sets = []
    for result in sorted(results, key=lambda r: chosen.index(r["set_name"])):
        name, motion = result["set_name"], result["motion"]
        cells = [result["cells"][index] for index in sorted(result["cells"])]
        block = {
            "set_name": name,
            "frames": len(cells),
            "fps": motion.fps,
            "columns": min(SHEET_COLUMNS, len(cells)),
            "sheet": f"{name}-sheet.png",
            "proof": f"{name}-proof.gif",
        }
        tile(cells, out_dir / block["sheet"], columns=block["columns"])
        gif_proof(cells, out_dir / block["proof"], motion.fps, loop=motion.loops)
        sets.append(block)

    # Last, once every set is in: the portraits are cut from the key art, not drawn.
    for units, size in PORTRAIT_SIZES.items():
        views[f"portrait-{units}"] = Path("views") / f"portrait-{units}.png"
        portrait(static["cells"][KEY_VIEW], out_dir / views[f"portrait-{units}"], size)

    manifest(out_dir / MANIFEST, spec.name, spec.height, views, sets, background)
    page = gallery(
        out_dir / "index.html",
        spec.name,
        spec.height,
        spec.description,
        {view: str(path) for view, path in views.items()},
        sets,
        background,
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
        "--per-frame",
        dest="frame_sheet_mode",
        action="store_false",
        help="draw one render per frame instead of one sheet per set",
    )
    build_parser.add_argument(
        "--motion-root",
        type=Path,
        default=MOTION_ROOT,
        help=f"where sheets a brief names by name are found (default {MOTION_ROOT})",
    )
    build_parser.add_argument("--work", type=Path, default=Path("work"))
    build_parser.add_argument("--out", type=Path, default=Path("outputs"))
    build_parser.add_argument(
        "--draw-backend",
        choices=["codex", "agy"],
        default="codex",
        help="CLI agent that draws the art: codex (ChatGPT sub) or agy (Antigravity sub)",
    )

    motions_parser = sub.add_parser(
        "motions", help="list the traced sheets a brief can name"
    )
    motions_parser.add_argument("--motion-root", type=Path, default=MOTION_ROOT)

    approve_parser = sub.add_parser(
        "approve", help="sign off on a character's key art so the rest can be drawn"
    )
    approve_parser.add_argument("spec", type=Path, help="path to a character brief")
    approve_parser.add_argument("--work", type=Path, default=Path("work"))

    edit_parser = sub.add_parser(
        "edit", help="nudge frames of a set's sheet in the browser; Save rebuilds its proof"
    )
    edit_parser.add_argument(
        "out", type=Path, nargs="?", default=Path("outputs"), help="default: outputs/, every character"
    )
    edit_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)
    if args.command == "motions":
        sheets = library(args.motion_root)
        if not sheets:
            log(f"no motion sheets under {args.motion_root}")
            return 1
        print(
            f"{'name':16s} {'title':18s} {'frames':>6} {'fps':>4} {'view':>7} "
            f"{'travel':>7}  {'poses':5s}  seam"
        )
        for name, bundle in sorted(sheets.items()):
            # The header is the manifest's; travel and poses are only in the landmarks.
            motion = bundle.load()
            print(
                f"{name:16s} {bundle.title[:18]:18s} {bundle.frame_count:6d} "
                f"{bundle.fps:4d} {bundle.view:>7s} {motion.travel:7.3f}  "
                f"{'yes' if motion.has_poses else 'no':5s}  {bundle.seam or '-'}"
            )
        return 0
    if args.command == "edit":
        serve(args.out, args.port)
        return 0
    if args.command == "approve":
        spec = load_spec(args.spec)
        log(f"[static] {spec.name}: approved {approve(args.work / spec.slug)}")
        return 0
    build(
        args.spec,
        args.motion,
        args.set_names,
        args.work,
        args.out,
        args.frame_sheet_mode,
        args.draw_backend,
        args.motion_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
