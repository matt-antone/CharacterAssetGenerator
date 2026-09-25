"""Render a PNG with codex's built-in image generation or a Comfy Cloud workflow.

Mechanism only. What to draw is decided upstream; this module gets the bytes on
disk and confirms they are a real image. Sources always land on a magenta
backdrop — the cutout happens later, once, in `cag.mask`.

Two backends, one at a time: `codex` (the ChatGPT subscription, via `codex exec`,
an agentic CLI with its own image tool) and `comfy` (a ComfyUI workflow run on
Comfy Cloud, see `cag.comfy`).
"""

from __future__ import annotations

import subprocess
from functools import partial
from pathlib import Path
from typing import Literal, Sequence

import httpx
import numpy
from PIL import Image

from . import comfy
from .style import OUTLINE

Backend = Literal["codex", "comfy"]


INSTRUCTIONS = """Generate exactly one image with your built-in image generation tool and \
save it to {filename} in the working directory. Pass every attached image to the tool as a \
reference image. You will shorten this message to write the tool's prompt; when you do, keep \
every "character-left" and "character-right" exactly as written, never as a bare "left" or \
"right", and keep the pose list and everything said about the pose reference in full. Shorten \
the character description first.

Then view {filename} and check the background. Every pixel that is not the character must be \
magenta, out to all four canvas edges and inside every gap between hair, limbs and props. If any \
of it came out black, dark, a stage, a spotlight, a vignette or a cast shadow, generate the image \
again until the background is magenta. The arcade style applies to the character only, never to \
the space around it.

Write no other file. Reply with the filename and nothing else."""


#: `INSTRUCTIONS` for a render that is a place rather than a character. A
#: location is delivered whole and nothing is ever cut out of it, so the magenta
#: contract is not just unnecessary here, it is the wrong picture — what has to
#: be checked instead is that the room came back empty.
SCENE_INSTRUCTIONS = """Generate exactly one image with your built-in image generation tool and \
save it to {filename} in the working directory. You will shorten this message to write the \
tool's prompt; when you do, keep everything said about what must not appear in full, and \
shorten the description of the place first.

Then view {filename} and check it. There must be no person, character, figure, silhouette or \
crowd anywhere in it, however small or far away, and no text, logo, watermark or signature. If \
any of that came out, generate the image again until it is gone.

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

#: How far a magenta backdrop's red and blue must run ahead of its green. The
#: chroma key scores each pixel against that gap and keys out anything past
#: half of it, so a washed-out orchid — Nano Banana's reading of "magenta" on a
#: frame sheet, gap 79 — keyed away 8% of Belter, jacket highlights first. Her
#: costume runs to 145 in the same measure; pure #FF00FF is 255.
MIN_MAGENTA_GAP = 160

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

    red, green, blue = backdrop
    gap = min(red, blue) - green
    if 0 < gap < MIN_MAGENTA_GAP:
        return False, (
            f"the backdrop {[int(v) for v in backdrop]} is a washed-out magenta, so the key "
            "would cut away costume in the same hues; it must be close to #FF00FF"
        )
    return True, ""


def _codex_argv(out_path: Path, references: Sequence[Path | str]) -> list[str]:
    """The prompt goes in on stdin: it is far too long for an argument."""
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
    return argv


def _run_codex(argv: list[str], full_prompt: str, timeout: int) -> str:
    """One codex turn. Returns why it failed, or "" if it exited cleanly."""
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, input=full_prompt
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout}s"
    if result.returncode != 0:
        return result.stderr.strip()[-2000:] or f"codex exited {result.returncode}"
    return ""


def _run_comfy(
    prompt: str,
    out_path: Path,
    references: Sequence[Path | str],
    workflow: dict,
    timeout: int,
    scene: bool,
) -> str:
    """One Comfy Cloud job. Returns why it failed, or "" if it saved an image."""
    try:
        comfy.render(prompt, out_path, references, workflow, timeout, rules=not scene)
    except (comfy.ComfyError, httpx.HTTPError) as error:
        return str(error)
    return ""


def draw(
    prompt: str,
    out_path: Path | str,
    references: Sequence[Path | str] = (),
    timeout: int = 900,
    attempts: int = 2,
    reuse: bool = True,
    backend: Backend = "codex",
    scene: bool = False,
    workflow: Path | str | None = None,
) -> Path:
    """Generate one image for `prompt` and save it at `out_path`.

    `references` are attached to the prompt as reference images — the approved
    key art, a neighbouring frame, whatever locks identity for this render.

    `backend` picks what does the drawing: `codex` (default, the ChatGPT
    subscription) or `comfy` (the Comfy Cloud workflow at `workflow`, see
    `cag.comfy`). One at a time — there is no fallback between them.

    `scene` draws a place instead of a character: no magenta backdrop is asked
    for and none is checked for, because nothing is cut out of a location.

    A render that already exists is kept, never redrawn and never overwritten,
    so a run interrupted at frame twelve resumes at frame twelve. Delete the
    file to force a redraw, or pass `reuse=False` to make its presence an error.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if reuse:
            return _verify(out_path, backdrop=not scene)
        raise DrawError(f"{out_path} already exists; refusing to overwrite a source")

    if backend == "comfy":
        # A workflow is not an agent: nobody reads instructions about saving a
        # file or checking the backdrop, so only the prompt is sent. A backdrop
        # that comes back wrong is caught by `_verify` below and drawn again.
        try:
            loaded = comfy.load_workflow(comfy.workflow_path(workflow))
            comfy.check_references(loaded, len(references))
        except comfy.ComfyError as error:
            raise DrawError(f"could not draw {out_path.name}: {error}") from error
        sent = prompt
        run = partial(_run_comfy, prompt, out_path, references, loaded, timeout, scene)
    else:
        instructions = SCENE_INSTRUCTIONS if scene else INSTRUCTIONS
        sent = f"{prompt}\n\n{instructions.format(filename=out_path.name)}"
        run = partial(_run_codex, _codex_argv(out_path, references), sent, timeout)

    # Written before the call, so a frame that comes back wrong can be read back
    # against what was actually asked for.
    out_path.with_suffix(".txt").write_text(
        sent
        + "\n\n--- references, in the order attached ---\n"
        + ("\n".join(str(Path(r).resolve()) for r in references) or "(none)")
        + "\n"
    )

    last_error = ""
    for attempt in range(attempts):
        failure = run()
        if failure:
            last_error = failure
            continue
        if out_path.exists():
            try:
                return _verify(out_path, backdrop=not scene)
            except DrawError as error:
                # This attempt is unusable, so move it aside and draw again. It is
                # kept, not deleted: a failure nobody can look at cannot be diagnosed.
                out_path.replace(out_path.with_name(f"{out_path.stem}.unusable-{attempt}.png"))
                last_error = str(error)
                continue
        last_error = f"{backend} reported success but wrote no file"
    raise DrawError(f"could not draw {out_path.name}: {last_error}")


def _verify(path: Path, backdrop: bool = True) -> Path:
    """Check a render is usable. Never deletes — the caller decides that."""
    usable, why_not = True, ""
    try:
        with Image.open(path) as image:
            image.verify()
        if backdrop:
            with Image.open(path) as image:
                usable, why_not = backdrop_is_usable(image)
    except Exception as exc:  # PIL raises a grab-bag of types here
        raise DrawError(f"{path.name} is not a readable image: {exc}") from exc
    if not usable:
        raise DrawError(f"{path.name}: {why_not}")
    return path
