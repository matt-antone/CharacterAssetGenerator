"""Render a PNG with the codex CLI's built-in image generation.

Mechanism only. What to draw is decided upstream; this module gets the bytes on
disk and confirms they are a real image. Sources always land on a magenta
backdrop — the cutout happens later, once, in `cag.mask`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from PIL import Image

from .geometry import MAGENTA

INSTRUCTIONS = f"""Generate exactly one image with your built-in image generation tool and \
save it to {{filename}} in the working directory.

The whole background must be flat {MAGENTA} magenta, edge to edge, with no border, frame, \
vignette, drop shadow, rounded corner or ground shadow. Nothing but the subject sits on \
the magenta. Portrait orientation.

Write no other file. Reply with the filename and nothing else."""


class DrawError(RuntimeError):
    """Raised when image generation produced no usable PNG."""


def draw(
    prompt: str,
    out_path: Path | str,
    references: Sequence[Path | str] = (),
    timeout: int = 900,
    attempts: int = 2,
    reuse: bool = True,
) -> Path:
    """Generate one image for `prompt` and save it at `out_path`.

    `references` are attached to the prompt as reference images — the approved
    key art, a neighbouring frame, whatever locks identity for this render.

    A render that already exists is kept, never redrawn and never overwritten,
    so a run interrupted at frame twelve resumes at frame twelve. Delete the
    file to force a redraw, or pass `reuse=False` to make its presence an error.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if reuse:
            return _verify(out_path)
        raise DrawError(f"{out_path} already exists; refusing to overwrite a source")

    argv = [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--sandbox",
        "workspace-write",
        "--cd",
        str(out_path.parent),
    ]
    for reference in references:
        argv += ["--image", str(Path(reference).resolve())]
    argv.append("-")

    full_prompt = f"{prompt}\n\n{INSTRUCTIONS.format(filename=out_path.name)}"
    last_error = ""
    for _ in range(attempts):
        try:
            result = subprocess.run(
                argv, input=full_prompt, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            last_error = f"timed out after {timeout}s"
            continue
        if result.returncode != 0:
            last_error = result.stderr.strip()[-2000:]
            continue
        if out_path.exists():
            return _verify(out_path)
        last_error = "codex reported success but wrote no file"
    raise DrawError(f"could not draw {out_path.name}: {last_error}")


def _verify(path: Path) -> Path:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # PIL raises a grab-bag of types here
        path.unlink(missing_ok=True)
        raise DrawError(f"{path.name} is not a readable image: {exc}") from exc
    return path
