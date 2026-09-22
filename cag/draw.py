"""Render a PNG with a CLI agent's built-in image generation.

Mechanism only. What to draw is decided upstream; this module gets the bytes on
disk and confirms they are a real image. Sources always land on a magenta
backdrop — the cutout happens later, once, in `cag.mask`.

Two backends, one at a time: `codex` (the ChatGPT subscription, via `codex exec`)
and `agy` (the Antigravity subscription, via `agy`, Gemini's replacement for the
retired `gemini` CLI). Both are agentic CLIs with their own built-in image tool —
neither is a raw provider API call.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal, Sequence

import numpy
from PIL import Image

from .style import OUTLINE

Backend = Literal["codex", "agy"]

#: Model asked to draw when the backend is `agy`. Antigravity has no dedicated
#: image-gen model name; Gemini 3's built-in image tool comes along with the
#: agent model, so this just picks the strongest one.
AGY_MODEL = "gemini-3.1-pro-high"


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


def _codex_call(full_prompt: str, out_path: Path, references: Sequence[Path | str]) -> tuple:
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
    return argv, {"input": full_prompt}


#: agy's orchestrating agent does not forward its own prompt to the built-in
#: generate_image tool — left to its own judgement it writes a short paraphrase
#: as that tool's Prompt argument, dropping the style contract, palette and
#: negative constraints the image quality actually depends on. This is the only
#: lever available: no flag controls the tool-call argument, so the agent has to
#: be told, forcefully, not to summarise.
AGY_VERBATIM_INSTRUCTION = (
    "\n\nWhen you call your image-generation tool, its Prompt argument must be a verbatim, "
    "character-for-character copy of this entire message, from \"Draw\" at the top to the end of "
    "these instructions — every sentence, hex code, and clause, in order. Do not summarise, "
    "shorten, condense, paraphrase, or drop anything, even if the tool's argument ends up "
    "extremely long. A shortened prompt has repeatedly produced unusable, off-style renders."
)


def _agy_call(full_prompt: str, out_path: Path, references: Sequence[Path | str]) -> tuple:
    # agy has no --cd: unlike codex, its working directory is whatever the process
    # inherits, not something a flag sets. INSTRUCTIONS says "the working directory",
    # so the subprocess's actual cwd must be out_path.parent or the file lands wrong.
    # Both that cwd and --add-dir must be absolute: once the subprocess's cwd changes,
    # a relative --add-dir would resolve against the new cwd, not the caller's.
    out_dir = out_path.parent.resolve()
    dirs = {str(out_dir)} | {str(Path(r).resolve().parent) for r in references}
    argv = ["agy", "--model", AGY_MODEL, "--dangerously-skip-permissions"]
    for directory in sorted(dirs):
        argv += ["--add-dir", directory]
    reference_note = (
        "\n\nReference images, at these exact paths:\n"
        + "\n".join(str(Path(r).resolve()) for r in references)
        if references
        else ""
    )
    argv += [
        "--print-timeout",
        "900s",
        "-p",
        full_prompt + reference_note + AGY_VERBATIM_INSTRUCTION,
    ]
    return argv, {"cwd": str(out_dir)}


_CALL_BUILDERS = {"codex": _codex_call, "agy": _agy_call}


def draw(
    prompt: str,
    out_path: Path | str,
    references: Sequence[Path | str] = (),
    timeout: int = 900,
    attempts: int = 2,
    reuse: bool = True,
    backend: Backend = "codex",
    scene: bool = False,
) -> Path:
    """Generate one image for `prompt` and save it at `out_path`.

    `references` are attached to the prompt as reference images — the approved
    key art, a neighbouring frame, whatever locks identity for this render.

    `backend` picks the CLI agent that does the drawing: `codex` (default, the
    ChatGPT subscription) or `agy` (the Antigravity subscription). One at a
    time — there is no fallback between them.

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

    instructions = SCENE_INSTRUCTIONS if scene else INSTRUCTIONS
    full_prompt = f"{prompt}\n\n{instructions.format(filename=out_path.name)}"
    argv, run_kwargs = _CALL_BUILDERS[backend](full_prompt, out_path, references)

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
                argv, capture_output=True, text=True, timeout=timeout, **run_kwargs
            )
        except subprocess.TimeoutExpired:
            last_error = f"timed out after {timeout}s"
            continue
        if result.returncode != 0:
            last_error = result.stderr.strip()[-2000:]
            continue
        if out_path.exists():
            try:
                return _verify(out_path, backdrop=not scene)
            except DrawError as error:
                # This attempt is unusable, so clear it and draw again.
                out_path.unlink(missing_ok=True)
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
