"""`cag build` — one brief in, one character package out."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from . import clips, comfy
from .animation import WHOLE_SET_SHEET, build_animation_graph
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
from .chat_session import ChatSession, TextPending
from .draw import BACKENDS, WORKFLOW_BACKENDS, draw
from .edit import serve
from .fidelity import report as fidelity_report
from .machines import STAGES, Machine, MachineError, load_machine, machine_names, materialise
from .motion import BUNDLE, MotionError, clip_status, library, load_motion, read_bundle
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

    However the sheet arrives, it is the sheet that says how the set plays: the
    motion spec is a prompt like the brief is, and the rule belongs to whichever
    of the two is driving. A set with a motion spec takes it from there; a set
    writing its own takes it from the brief, or from the set plan when the brief
    is silent. Never from both, so there is nothing to reconcile.
    """
    if supplied:
        # A traced bundle is its manifest, and the manifest is what names the
        # traced frames — so reading the sheet straight off disk left the pose
        # photographs behind and drew the set from skeletons instead. A sheet
        # written for a set has no manifest beside it and still loads bare.
        manifest = (supplied if supplied.is_dir() else supplied.parent) / BUNDLE
        if manifest.exists():
            return read_bundle(manifest).load()
        return load_motion(supplied)
    named = spec.motions.get(set_name)
    if named == AUTO:
        name, bundle = auto_bundle(spec, set_name, motion_root)
        log(f"[{set_name}] auto: the {name!r} sheet")
        return bundle.load()
    if named:
        found = library(motion_root).get(named)
        if not found:
            raise MotionError(
                f"{spec.name} names the {named!r} motion sheet for {set_name}, "
                f"but no bundle under {motion_root} calls itself that"
            )
        return found.load()
    intent = spec.animations.get(set_name)
    if not intent:
        raise KeyError(f"{spec.name} has no {set_name!r} animation in their brief")
    plan = plan_for(set_name)
    if set_name in spec.playbacks:
        plan = replace(plan, playback=spec.playbacks[set_name])
    return write_motion(
        text_model(work_dir),
        spec,
        set_name,
        intent,
        plan,
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
    draw_fn: Callable[..., Path] | None = None,
    motion_root: Path = MOTION_ROOT,
    carry: Path | None = None,
    backend: str = "codex",
    frames_per_sheet: int | None = None,
    machine: Machine | None = None,
    graphs: dict[str, Path] | None = None,
) -> dict:
    """Draw and mask one animation set, continuing from the set drawn before it.

    `carry` is that set's last frame. It rides along with this set's key art as
    a picture of the character as actually drawn a moment ago, and this set
    hands its own last frame on in turn. That chain is why the sets are drawn in
    order rather than in parallel lanes.
    """
    motion = motion_for(spec, set_name, work_dir, supplied, motion_root)
    log(
        f"[{set_name}] {len(motion.frames)} frames at {motion.fps} fps, "
        f"{motion.view} view, {motion.playback}"
    )
    if not motion.seams_cleanly:
        # The tracer compared the last frame to the first and they do not meet.
        # Nothing downstream can fix that, so it is said once, where it is chosen.
        log(f"[{set_name}] the {motion.name!r} sheet loops on a {motion.seam!r} seam: "
            "the proof will jump from the last frame back to the first")
    # The reference this set is drawn against has this set's hands, not the key
    # art's: a prop on the character follows it into every frame that quotes it.
    key_art = set_key_art(
        spec, set_name, static["bible"], work_dir, static["sources"][KEY_VIEW], draw_fn,
        motion.view, detail_after_key=backend == "codex",
    )
    log(f"[{set_name}] reference: {key_art.name}")
    animated = build_animation_graph(text_model(work_dir), draw_fn=draw_fn, frame_sheet_mode=frame_sheet_mode).invoke(
        {
            "spec": spec,
            "bible": static["bible"],
            "key_art": key_art,
            "scale": static["scale"],
            "motion": motion,
            "set_name": set_name,
            "work_dir": work_dir,
            **backend_state(backend, frames_per_sheet, machine, graphs),
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
    draw_fn: Callable[..., Path] | None = None,
    motion_root: Path = MOTION_ROOT,
    backend: str = "codex",
    frames_per_sheet: int | None = None,
    machine: Machine | None = None,
    graphs: dict[str, Path] | None = None,
) -> Path:
    spec = load_spec(spec_path)
    work_dir = work_root / spec.slug
    out_dir = out_for(out_root, spec_path, spec.slug)
    # An explicit --set is the operator asking for that set by name, so it overrides the
    # config. Without one, the config decides which of the brief's animations are drawn.
    chosen = set_names if set_names is not None else wanted(list(spec.sets), "animations")
    if motion_path and len(chosen) != 1:
        raise SystemExit("--motion applies to a single --set; other sets write their own sheet")

    log(f"[static] {spec.name}: bible, key art, projection")
    try:
        static = build_static_graph(text_model(work_dir), draw_fn=draw_fn).invoke(
            {"spec": spec, "work_dir": work_dir, "detail_after_key": backend == "codex"}
        )
    except ApprovalRequired as gate:
        raise SystemExit(
            f"[static] {spec.name}: key art is waiting for approval at {gate}\n"
            f"  approve it:  uv run cag approve {spec_path}\n"
            f"  or redraw it: rm {gate} and build again"
        ) from None
    except TextPending as pending:
        raise SystemExit(f"[static] {spec.name}: {pending}") from None

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
                    draw_fn,
                    motion_root,
                    carry,
                    backend,
                    frames_per_sheet,
                    machine,
                    graphs,
                )
            except TextPending as pending:
                # Not a failure: the session's AI answers it and the build goes on.
                # The chain stops here, because every later set starts from this
                # one's last frame and would be drawn, and cached, from the wrong one.
                log(f"[{name}] {pending}")
                later = chosen[chosen.index(name) + 1 :]
                if later:
                    log(f"[sets] {', '.join(later)} wait for {name}")
                break
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
            "playback": motion.playback,
            "columns": min(SHEET_COLUMNS, len(cells)),
            "sheet": f"{name}-sheet.png",
            "proof": f"{name}-proof.gif",
        }
        tile(cells, out_dir / block["sheet"], columns=block["columns"])
        gif_proof(cells, out_dir / block["proof"], motion.fps, block["playback"])
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
        choices=list(BACKENDS),
        default=os.environ.get("CAG_DRAW_BACKEND"),
        help="what draws the art: codex (ChatGPT sub), comfy (a ComfyUI workflow on Comfy "
        "Cloud) or local (the same, on your own ComfyUI). Default: $CAG_DRAW_BACKEND. There "
        "is no fallback: a build that names neither stops and asks",
    )
    build_parser.add_argument(
        "--comfy-workflow",
        type=Path,
        help="API-format ComfyUI workflow for --draw-backend comfy. "
        f"Default: $CAG_COMFY_WORKFLOW, else {comfy.DEFAULT_WORKFLOW}",
    )
    build_parser.add_argument(
        "--comfy-pose-workflow",
        type=Path,
        help="API-format ComfyUI workflow that re-poses a traced set frame by frame. "
        f"Default: $CAG_COMFY_POSE_WORKFLOW, else {comfy.POSE_WORKFLOW}",
    )
    build_parser.add_argument(
        "--machine",
        choices=machine_names(),
        default=os.environ.get("CAG_MACHINE"),
        help="draw traced sets from their source clip, through SCAIL-2 and a restyle, with "
        "this machine profile's model files and sizes (comfy/machines/<name>.json). "
        "Default: $CAG_MACHINE; without one, traced sets take the pose-edit path",
    )
    for stage, flag in (("video", "scail"), ("restyle", "restyle"), ("mask", "mask")):
        variable, default = STAGES[stage]
        build_parser.add_argument(
            f"--comfy-{flag}-workflow",
            type=Path,
            help=f"API-format ComfyUI workflow for the video path's {stage} stage, before the "
            f"machine profile patches it. Default: ${variable}, else {default}",
        )
    build_parser.add_argument(
        "--frames-per-sheet",
        type=int,
        help="most frames drawn in one sheet-mode render. Default: 8 under codex, "
        f"the whole set (up to {WHOLE_SET_SHEET}) under comfy",
    )

    motions_parser = sub.add_parser(
        "motions", help="list the traced sheets a brief can name"
    )
    motions_parser.add_argument("--motion-root", type=Path, default=MOTION_ROOT)

    machines_parser = sub.add_parser(
        "machines", help="list the machine profiles the video path can draw on"
    )
    machines_parser.add_argument(
        "--check",
        metavar="NAME",
        help="ask that machine's ComfyUI for every node and model file its graphs load, "
        "and list what is missing: the download checklist",
    )
    machines_parser.add_argument("--work", type=Path, default=Path("work"))

    clips_parser = sub.add_parser(
        "clips",
        help="cut traced bundles' source clips back out of the videos they were traced off",
    )
    clips_parser.add_argument("names", nargs="*", help="bundles, by the name a brief uses")
    clips_parser.add_argument(
        "--cast", action="store_true", help=f"every bundle a brief under {SPECS_ROOT} names"
    )
    clips_parser.add_argument(
        "--check", action="store_true", help="say what each bundle holds of its clip; fetch nothing"
    )
    clips_parser.add_argument(
        "--dry-run", action="store_true", help="find each clip and its box, but write nothing"
    )
    clips_parser.add_argument("--motion-root", type=Path, default=MOTION_ROOT)
    clips_parser.add_argument("--specs", type=Path, default=SPECS_ROOT)
    clips_parser.add_argument("--cache", type=Path, default=clips.SOURCES)

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

    fidelity_parser = sub.add_parser(
        "fidelity", help="bone-angle error of a rendered set against its motion's photographs"
    )
    fidelity_parser.add_argument("spec", type=Path, help="path to a character brief")
    fidelity_parser.add_argument("--set", dest="set_name", default="dance")
    fidelity_parser.add_argument("--work", type=Path, default=Path("work"))
    fidelity_parser.add_argument("--motion-root", type=Path, default=MOTION_ROOT)

    args = parser.parse_args(argv)
    if args.command == "fidelity":
        spec = load_spec(args.spec)
        name = spec.motions.get(args.set_name)
        if name is None:
            log(f"{spec.name}'s {args.set_name} set names no motion; there is nothing to compare")
            return 1
        photos = library(args.motion_root)[name].photos
        cells = [
            args.work / spec.slug / "cells" / args.set_name / f"{index:02d}.png"
            for index in range(len(photos))
        ]
        print(fidelity_report(cells, list(photos)))
        return 0
    if args.command == "motions":
        sheets = library(args.motion_root)
        if not sheets:
            log(f"no motion sheets under {args.motion_root}")
            return 1
        print(
            f"{'name':16s} {'title':18s} {'frames':>6} {'fps':>4} {'view':>7} "
            f"{'travel':>7}  {'poses':5s}  {'clip':7s}  seam"
        )
        for name, bundle in sorted(sheets.items()):
            # The header is the manifest's; travel and poses are only in the landmarks.
            motion = bundle.load()
            print(
                f"{name:16s} {bundle.title[:18]:18s} {bundle.frame_count:6d} "
                f"{bundle.fps:4d} {bundle.view:>7s} {motion.travel:7.3f}  "
                f"{'yes' if motion.has_poses else 'no':5s}  {clip_status(bundle):7s}  "
                f"{bundle.seam or '-'}"
            )
        return 0
    if args.command == "machines":
        return check_machine(args.check, args.work) if args.check else list_machines()
    if args.command == "clips":
        return clip_bundles(args)
    if args.command == "edit":
        serve(args.out, args.port)
        return 0
    if args.command == "approve":
        spec = load_spec(args.spec)
        log(f"[static] {spec.name}: approved {approve(args.work / spec.slug)}")
        return 0
    if not args.draw_backend:
        # Each backend is a different model, bill and licence, so it is never
        # assumed: an agent running the build asks the person which one.
        raise SystemExit(
            "[build] no draw backend: pass --draw-backend "
            f"{{{','.join(BACKENDS)}}} or set CAG_DRAW_BACKEND. Which one draws is the "
            "person's call; an agent asks rather than picks"
        )
    if args.comfy_pose_workflow:
        # Every set reads it through `comfy.pose_workflow_path`, as the env var does.
        os.environ["CAG_COMFY_POSE_WORKFLOW"] = str(args.comfy_pose_workflow)
    machine, graphs = None, None
    if args.machine:
        machine, graphs = machine_with(
            args.draw_backend,
            args.machine,
            args.work,
            {
                "video": args.comfy_scail_workflow,
                "restyle": args.comfy_restyle_workflow,
                "mask": args.comfy_mask_workflow,
            },
        )
    build(
        args.spec,
        args.motion,
        args.set_names,
        args.work,
        args.out,
        args.frame_sheet_mode,
        draw_with(args.draw_backend, args.comfy_workflow),
        args.motion_root,
        backend=args.draw_backend,
        frames_per_sheet=args.frames_per_sheet,
        machine=machine,
        graphs=graphs,
    )
    return 0


def text_model(work_dir: Path) -> BaseChatModel:
    """Who writes a build's text: the agent session running it, unless told otherwise.

    `CAG_TEXT_MODEL=codex` sends it to the codex CLI instead, as every build did
    before; anything else is answered by the session (see `cag.chat_session`).
    """
    if os.environ.get("CAG_TEXT_MODEL") == "codex":
        return ChatCodex()
    return ChatSession(folder=work_dir / "text")


def backend_state(
    backend: str,
    frames_per_sheet: int | None = None,
    machine: Machine | None = None,
    graphs: dict[str, Path] | None = None,
) -> dict:
    """How the graphs draw for this backend, where the backends differ.

    Under Comfy, Nano Banana copies the person out of the detail sample (see
    `DETAIL_FRAMES`), and its separate renders of one set drift apart in style,
    so a set is drawn whole (see `WHOLE_SET_SHEET`). Codex keeps both as they
    were measured. `frames_per_sheet`, when given, overrides either default.

    A `machine`, with the `graphs` `machine_with` wrote for it, sends every
    traced set down the video path instead of the pose-edit path.
    """
    state = {} if backend not in WORKFLOW_BACKENDS else {
        "detail_after_key": False,
        "frames_per_sheet": WHOLE_SET_SHEET,
        # A traced set is drawn a frame at a time by re-posing its reference; see
        # `pose_edit_frames`. Only a written set still goes on a frame sheet.
        "pose_workflow": comfy.pose_workflow_path(),
        "snap_to_key": True,
    }
    if frames_per_sheet:
        state["frames_per_sheet"] = frames_per_sheet
    if machine:
        state.update(machine=machine, machine_graphs=dict(graphs or {}), local=backend == "local")
    return state


def machine_with(
    backend: str,
    name: str,
    work_root: Path,
    given: dict[str, Path | None] | None = None,
    client: comfy.Client | None = None,
) -> tuple[Machine, dict[str, Path]]:
    """The machine profile a build draws the video path on, its graphs written and checked.

    Checked here, once, before anything is drawn: a profile for the other
    backend, a missing ffmpeg, or — on your own ComfyUI — a node or model file
    the server lacks each stops the build by name, every missing file at once,
    instead of failing each set in turn an hour in.
    """
    if backend not in WORKFLOW_BACKENDS:
        raise SystemExit(
            f"[build] --machine draws through ComfyUI graphs; --draw-backend {backend} has none"
        )
    try:
        machine = load_machine(name)
    except MachineError as error:
        raise SystemExit(f"[build] {error}") from None
    if machine.backend != backend:
        raise SystemExit(
            f"[build] machine {name!r} runs on --draw-backend {machine.backend}, not {backend}"
        )
    if not shutil.which("ffmpeg"):
        raise SystemExit("[build] the video path cuts drive videos with ffmpeg, which is not installed")
    try:
        graphs = {
            stage: materialise(machine, stage, work_root / "comfy", (given or {}).get(stage))
            for stage in STAGES
        }
        if backend == "local":
            missing = comfy.preflight(client or comfy.Client(local=True), graphs.values())
            if missing:
                raise SystemExit(
                    f"[local] machine {name!r} cannot draw; the server lacks:\n  "
                    + "\n  ".join(missing)
                    + f"\n  (uv run cag machines --check {name} lists this again)"
                )
    except comfy.ComfyError as error:
        raise SystemExit(f"[{backend}] {error}") from None
    log(
        f"[{backend}] traced sets with a clip: SCAIL-2 + restyle on {name}"
        + ("" if machine.verified else " (UNVERIFIED profile)")
    )
    return machine, graphs


def list_machines() -> int:
    names = machine_names()
    if not names:
        log("no machine profiles")
        return 1
    print(f"{'name':10s} {'backend':7s} {'verified':8s} {'video':>9s} {'fps':>4s} {'cap':>4s} "
          f"{'steps':>5s} {'restyle':>8s}")
    for name in names:
        m = load_machine(name)
        print(
            f"{name:10s} {m.backend:7s} {'yes' if m.verified else 'no':8s} "
            f"{f'{m.width}x{m.height}':>9s} {m.rate:4g} {m.max_length:4d} {m.steps:5d} "
            f"{f'{m.restyle_resolution}/{m.restyle_steps}':>8s}"
        )
    return 0


def check_machine(name: str, work_root: Path, client: comfy.Client | None = None) -> int:
    """List every node and model file `name`'s graphs load that its ComfyUI lacks."""
    try:
        machine = load_machine(name)
        graphs = [materialise(machine, stage, work_root / "comfy") for stage in STAGES]
        missing = comfy.preflight(client or comfy.Client(local=machine.backend == "local"), graphs)
    except comfy.ComfyError as error:
        log(f"[{name}] {error}")
        return 1
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg is not installed")
    for line in missing:
        print(line)
    log(f"[{name}] " + (f"{len(missing)} missing" if missing else "nothing missing"))
    return 1 if missing else 0


def clip_bundles(args: argparse.Namespace) -> int:
    """`cag clips`: backfill, or with --check report, each asked-for bundle's source clip."""
    roots = clips.bundle_roots(args.motion_root)
    names = list(args.names)
    if args.cast:
        names += cast_bundles(args.specs)
    names = list(dict.fromkeys(names))
    if not names:
        log("name bundles, or pass --cast for every bundle a brief names")
        return 1
    failed = 0
    for name in names:
        if name not in roots:
            print(f"{name} REFUSED: no bundle under {args.motion_root} calls itself that")
            failed += 1
            continue
        if args.check:
            try:
                status = clip_status(read_bundle(roots[name] / BUNDLE))
            except MotionError as error:
                status = f"bad ({error})"
            print(f"{name} clip={status}")
            failed += status != "ok"
            continue
        try:
            print(clips.backfill(roots[name], args.cache, dry_run=args.dry_run))
        except clips.ClipError as error:
            print(f"{name} REFUSED: {error}")
            failed += 1
    return 1 if failed else 0


def cast_bundles(specs_root: Path = SPECS_ROOT) -> list[str]:
    """Every bundle a brief under `specs_root` names by name, in brief order."""
    names = []
    for path in sorted(specs_root.rglob("*.json")):
        if path.name.endswith(".schema.json"):
            continue
        names += [n for n in load_spec(path).motions.values() if n and n != AUTO]
    return list(dict.fromkeys(names))


def draw_with(backend: str, workflow: Path | None = None) -> Callable[..., Path] | None:
    """The draw function a build hands its graphs, checked before anything is drawn.

    None means codex, and keeps each module on its own `draw` name — the seam the
    tests replace. A workflow backend's workflows and key are checked here, once,
    so a missing one stops the build by name instead of failing every set in turn.
    """
    if backend == "codex":
        return None
    local = backend == "local"
    path = comfy.workflow_path(workflow, local=local)
    try:
        loaded = comfy.load_workflow(path)
        comfy.load_workflow(comfy.pose_workflow_path())
        comfy.Client(local=local)
    except comfy.ComfyError as error:
        raise SystemExit(f"[{backend}] {error}") from None
    log(f"[{backend}] drawing with {path}, {comfy.reference_slots(loaded)} reference slots; "
        f"traced sets frame by frame with {comfy.pose_workflow_path()}")
    return partial(draw, backend=backend, workflow=path)


if __name__ == "__main__":
    raise SystemExit(main())
