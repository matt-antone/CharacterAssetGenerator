"""Render a PNG with the codex CLI's built-in image generation.

Mechanism only. What to draw is decided upstream; this module gets the bytes on
disk and confirms they are a real image. Sources always land on a magenta
backdrop — the cutout happens later, once, in `cag.mask`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import numpy
from PIL import Image

from .style import OUTLINE


INSTRUCTIONS = """Generate exactly one image with your built-in image generation tool and \
save it to {filename} in the working directory.

Write no other file. Reply with the filename and nothing else."""


#: How much the border may vary and still count as one flat backdrop, as a
#: per-channel spread across the sampled band. Scenery behind the character
#: breaks the cutout.
BACKDROP_SPREAD = 32

#: How far the backdrop must sit from the outline colour. Vision does not chroma
#: key, so the exact hue is free — but a backdrop that matches the mandatory
#: black outline cannot be told apart from it, and masking removes the outline
#: along with the background.
OUTLINE_CLEARANCE = 96

#: Border band sampled when checking the backdrop.
BORDER_PIXELS = 8


class DrawError(RuntimeError):
    """Raised when image generation produced no usable PNG."""


def backdrop_is_usable(image: Image.Image) -> tuple[bool, str]:
    """Can this render be masked? Returns the verdict and, if not, why not."""
    pixels = numpy.array(image.convert("RGB"), dtype=numpy.int16)
    band = numpy.concatenate(
        [
            pixels[:BORDER_PIXELS].reshape(-1, 3),
            pixels[-BORDER_PIXELS:].reshape(-1, 3),
            pixels[:, :BORDER_PIXELS].reshape(-1, 3),
            pixels[:, -BORDER_PIXELS:].reshape(-1, 3),
        ]
    )
    spread = numpy.percentile(band, 95, axis=0) - numpy.percentile(band, 5, axis=0)
    if numpy.max(spread) > BACKDROP_SPREAD:
        return False, "there is scenery behind the character, not a flat backdrop"

    backdrop = numpy.median(band, axis=0)
    clearance = numpy.max(numpy.abs(backdrop - numpy.array(OUTLINE, dtype=numpy.int16)))
    if clearance < OUTLINE_CLEARANCE:
        return False, (
            f"the backdrop {[int(v) for v in backdrop]} is too close to the black outline, so "
            "masking cannot tell them apart"
        )
    return True, ""


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

    # Written before the call, so a frame that comes back wrong can be read back
    # against what was actually asked for.
    out_path.with_suffix(".txt").write_text(
        full_prompt
        + "\n\n--- references, in the order attached ---\n"
        + ("\n".join(str(Path(r).resolve()) for r in references) or "(none)")
        + "\n"
    )

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
            try:
                return _verify(out_path)
            except DrawError as error:
                # This attempt is unusable, so clear it and draw again.
                out_path.unlink(missing_ok=True)
                last_error = str(error)
                continue
        last_error = "codex reported success but wrote no file"
    raise DrawError(f"could not draw {out_path.name}: {last_error}")


def _verify(path: Path) -> Path:
    """Check a render is usable. Never deletes — the caller decides that."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            usable, why_not = backdrop_is_usable(image)
    except Exception as exc:  # PIL raises a grab-bag of types here
        raise DrawError(f"{path.name} is not a readable image: {exc}") from exc
    if not usable:
        raise DrawError(f"{path.name}: {why_not}")
    return path
