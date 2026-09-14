"""`cag build` — one brief in, one character package out."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .animation import build_animation_graph
from .assemble import gallery, gif_proof, sprite_sheet
from .chat_codex import ChatCodex
from .motion import load_motion
from .prompts import KEY_VIEW, VIEWS
from .spec import load_spec
from .static_sheet import build_static_graph

#: Wrap long sets so the sheet stays a reasonable shape to open.
SHEET_COLUMNS = 8


def build(
    spec_path: Path,
    motion_path: Path | None,
    set_name: str,
    work_root: Path,
    out_root: Path,
) -> Path:
    spec = load_spec(spec_path)
    work_dir = work_root / spec.slug
    out_dir = out_root / spec.slug
    model = ChatCodex()

    print(f"[static] {spec.name}: bible, key art, projection", file=sys.stderr)
    static = build_static_graph(model).invoke({"spec": spec, "work_dir": work_dir})

    views = {}
    for view in [KEY_VIEW, *(v for v in VIEWS if v != KEY_VIEW)]:
        views[view] = Path("views") / f"{view}.png"
        destination = out_dir / views[view]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(static["cells"][view], destination)

    sets = []
    if motion_path:
        motion = load_motion(motion_path)
        print(
            f"[{set_name}] {len(motion.frames)} frames at {motion.fps} fps: "
            f"direct, keyframe, tween, mask",
            file=sys.stderr,
        )
        animated = build_animation_graph(model).invoke(
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
        cells = [animated["cells"][index] for index in sorted(animated["cells"])]
        columns = min(SHEET_COLUMNS, len(cells))
        sets.append(
            {
                "set_name": set_name,
                "frames": len(cells),
                "fps": motion.fps,
                "sheet": f"{set_name}-sheet.png",
                "proof": f"{set_name}-proof.gif",
            }
        )
        sprite_sheet(cells, out_dir / sets[0]["sheet"], columns=columns)
        gif_proof(cells, out_dir / sets[0]["proof"], motion.fps, loop=motion.loops)

    page = gallery(
        out_dir / "index.html",
        spec.name,
        spec.height,
        spec.description,
        {view: str(path) for view, path in views.items()},
        sets,
    )
    print(page, file=sys.stderr)
    return page


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="render a character package")
    build_parser.add_argument("spec", type=Path, help="path to a character brief")
    build_parser.add_argument(
        "--motion", type=Path, help="a MotionArtist motion.json to animate from"
    )
    build_parser.add_argument("--set", dest="set_name", default="dance", help="animation set name")
    build_parser.add_argument("--work", type=Path, default=Path("work"))
    build_parser.add_argument("--out", type=Path, default=Path("outputs"))

    args = parser.parse_args(argv)
    build(args.spec, args.motion, args.set_name, args.work, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
