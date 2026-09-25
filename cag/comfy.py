"""Render a PNG by running a ComfyUI workflow, on Comfy Cloud or a local server.

Mechanism only, like the rest of `cag.draw`: this module turns one prompt and
its reference images into one image on disk. Which model draws, at what
settings, is the workflow's business, and the workflow is a file — exported from
Comfy in API format — not code. Swapping models is editing that file.

cag finds its way into the workflow by placeholders. Any node input whose value
is exactly one of these strings is filled in before the workflow is sent:

    $prompt              the render's prompt
    $image1 .. $imageN   the render's reference images, in the order attached
    $seed                a fresh random seed, so a redraw is a new roll
    $width, $height      the canvas in pixels, read off the prompt
    $aspect              the same canvas as a ratio, "2:3", "3:2" or "16:9"

A render with fewer references than the workflow has slots drops the unused
`LoadImage` nodes and every link to them. A batch node left with one input is
bypassed rather than broken, and one left with none is dropped, so references
batched together shrink cleanly. A render with more references than slots is
refused.

`comfy/workflow.json` is the one cag ships: Nano Banana Pro, which takes up to
fourteen reference images and follows a long prompt closely.

The same calls reach a ComfyUI server of your own (`Client(local=True)`): local
ComfyUI serves the same `/api/...` routes Comfy Cloud does, needs no key, and
answers a download with the image instead of a redirect. What it cannot do is
run a partner node such as Nano Banana, which only exists on Comfy's servers, so
a local build names a workflow whose nodes and models are installed locally.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy
from PIL import Image, ImageOps

BASE_URL = "https://cloud.comfy.org"

#: Where a local ComfyUI listens unless `CAG_LOCAL_COMFY_URL` says otherwise.
LOCAL_URL = "http://127.0.0.1:8188"

#: Where the workflow lives when neither `--comfy-workflow` nor
#: `CAG_COMFY_WORKFLOW` says otherwise.
DEFAULT_WORKFLOW = Path("comfy/workflow.json")

#: What a `local` build draws with unless told otherwise: Qwen-Image-2.1 at full
#: precision, the one open model whose key art has been checked against a brief
#: (Belter's, 2026-09-25). Its weights are under the Qwen Research License —
#: research and evaluation only — so a package meant for anything else names an
#: Apache-licensed workflow instead.
LOCAL_WORKFLOW = Path("comfy/qwen-image-2.1.json")

#: The workflow that draws a traced set one frame at a time: Qwen-Image-Edit
#: 2511 with the AnyPose LoRAs re-poses the set's approved reference (`$image1`)
#: into one traced photograph's pose (`$image2`). Editing the same picture every
#: frame is what holds the character and its style still; Nano Banana, drawing
#: the set from words, drifted between every render and shrank the figures when
#: asked to fit a whole set in one. One fixed seed in the workflow, so frames
#: differ only by pose.
POSE_WORKFLOW = Path("comfy/pose-edit.json")


@dataclass(frozen=True)
class Canvas:
    width: int
    height: int
    aspect: str


#: The canvas each prompt shape gets. The prompt already names its canvas in
#: words; these are the same canvases in the numbers a workflow takes. The
#: first phrase found wins, so the more specific one comes first.
CANVASES = [
    ("16:9 landscape canvas", Canvas(1536, 864, "16:9")),  # a location
    ("wide landscape canvas", Canvas(1792, 768, "21:9")),  # a set drawn whole
    ("landscape canvas", Canvas(1536, 1024, "3:2")),  # a frame sheet
]
#: Everything else is one figure.
PORTRAIT = Canvas(1024, 1536, "2:3")

#: Every reference is letterboxed onto this square before upload. A batch is one
#: tensor, so ComfyUI resizes every image in it to the first one's size and crops
#: from the centre to get there — a landscape pose grid batched behind portrait
#: key art would lose its outer pose cards. A square holds either shape whole.
REFERENCE_SIDE = 1536

#: Nodes that only join images into a batch. One left with a single input by an
#: unused reference slot passes it straight through; one left with none goes.
BATCH_NODES = {"ImageBatch", "BatchImagesNode"}

#: Seconds between job status checks.
POLL_SECONDS = 3

#: Job states that will never change again.
TERMINAL = {"completed", "failed", "cancelled"}


class ComfyError(RuntimeError):
    """Raised when Comfy Cloud refuses a workflow or the job produces no image."""


def workflow_path(given: Path | str | None = None, local: bool = False) -> Path:
    """The workflow a build draws with. A local build has its own default: the
    Comfy Cloud one is a partner node that no local server can run."""
    if local:
        return Path(given or os.environ.get("CAG_LOCAL_COMFY_WORKFLOW") or LOCAL_WORKFLOW)
    return Path(given or os.environ.get("CAG_COMFY_WORKFLOW") or DEFAULT_WORKFLOW)


def pose_workflow_path(given: Path | str | None = None) -> Path:
    """The workflow that re-poses a traced set, chosen the way `workflow_path` is.

    Separate from the main workflow because it takes a different pair of inputs
    (the reference and one traced photograph), and a model that draws a set's
    key art may not be the one its traced frames are drawn with.
    """
    return Path(given or os.environ.get("CAG_COMFY_POSE_WORKFLOW") or POSE_WORKFLOW)


def load_workflow(path: Path | str) -> dict:
    """Read an API-format workflow and check it has somewhere to put the prompt."""
    path = Path(path)
    if not path.exists():
        raise ComfyError(
            f"no Comfy workflow at {path}. Export one from Comfy Cloud with "
            "Workflow > Export (API), put $prompt and $image1.. in its inputs, and save it "
            "there or pass --comfy-workflow"
        )
    workflow = json.loads(path.read_text())
    if "nodes" in workflow and "links" in workflow:
        raise ComfyError(
            f"{path} is a UI-format workflow. Export it again with Workflow > Export (API)"
        )
    if "$prompt" not in _placeholders(workflow):
        raise ComfyError(f"{path} has no input set to $prompt, so the prompt has nowhere to go")
    return workflow


def canvas(prompt: str) -> Canvas:
    """The canvas a prompt asks for. The prompt states its canvas; this sizes it."""
    return next((shape for phrase, shape in CANVASES if phrase in prompt), PORTRAIT)


def letterbox(path: Path | str) -> bytes:
    """A reference fitted whole inside a `REFERENCE_SIDE` square, as PNG bytes.

    The bars are the picture's own border colour, so a character on magenta
    stays a character on magenta instead of gaining a frame.
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    pixels = numpy.array(rgb)
    border = numpy.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    colour = tuple(int(v) for v in numpy.median(border, axis=0))
    boxed = ImageOps.pad(rgb, (REFERENCE_SIDE, REFERENCE_SIDE), color=colour)
    buffer = io.BytesIO()
    boxed.save(buffer, format="PNG")
    return buffer.getvalue()


#: Nano Banana sometimes boxes each figure of a frame sheet in black panel
#: rules, whatever the prompt says, and a sheet in boxes cannot be sliced. A
#: rule is told from a figure by shape: a straight dark run most of the way
#: across the canvas, ending at the canvas edge or at another rule. Measured on
#: Belter's sheets, rules run 6-16px thick; no figure row passed 34% dark.
RULE_DARK = 70
RULE_MAX_THICKNESS = 24
RULE_REACH = 24
RULE_PAD = 3


def _longest_runs(dark: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Per row of `dark`: the longest run of True, and where it starts and ends."""
    padded = numpy.pad(dark, ((0, 0), (1, 1)))
    edges = numpy.diff(padded.astype(numpy.int8), axis=1)
    length = numpy.zeros(dark.shape[0], dtype=int)
    start = numpy.zeros(dark.shape[0], dtype=int)
    for row in numpy.flatnonzero(dark.any(axis=1)):
        ups, downs = numpy.flatnonzero(edges[row] == 1), numpy.flatnonzero(edges[row] == -1)
        best = int(numpy.argmax(downs - ups))
        length[row], start[row] = downs[best] - ups[best], ups[best]
    return length, start, start + length


def _bands(rows: numpy.ndarray) -> list[tuple[int, int]]:
    """Consecutive indices grouped into (first, last) bands."""
    if not len(rows):
        return []
    groups = numpy.split(rows, numpy.flatnonzero(numpy.diff(rows) > 1) + 1)
    return [(int(g[0]), int(g[-1])) for g in groups]


def erase_panel_rules(image: Image.Image) -> tuple[Image.Image, int]:
    """`image` with any panel rules painted out in the backdrop colour, and how many."""
    rgb = numpy.array(image.convert("RGB"))
    dark = rgb.max(axis=2) < RULE_DARK
    height, width = dark.shape
    pixels = rgb.reshape(-1, 3)
    # Sampled at the canvas edge, skipping the rules drawn along it.
    border = numpy.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    border = border[border.max(axis=1) >= RULE_DARK]
    if not len(border):
        return image, 0
    backdrop = numpy.median(border, axis=0)

    # Horizontal rules run the whole width, edge to edge.
    length, _, _ = _longest_runs(dark)
    across = [band for band in _bands(numpy.flatnonzero(length >= 0.9 * width))
              if band[1] - band[0] < RULE_MAX_THICKNESS]
    ends = [0, height - 1, *(row for band in across for row in band)]

    # Vertical rules may span one row of panels; each end meets an edge or a rule.
    length, top, bottom = _longest_runs(dark.T)
    tall = numpy.flatnonzero(length >= 0.4 * height)
    meets = lambda y: min(abs(y - e) for e in ends) <= RULE_REACH  # noqa: E731
    # A rule's extent is read across all of its columns: an edge column is
    # anti-aliased, and its run can start a third of the way down a rule that
    # runs the full height.
    extents = [(band, int(top[band[0] : band[1] + 1].min()), int(bottom[band[0] : band[1] + 1].max()))
               for band in _bands(tall)]
    down = [(band, y0, y1) for band, y0, y1 in extents
            if band[1] - band[0] < RULE_MAX_THICKNESS and meets(y0) and meets(y1 - 1)]

    mask = numpy.zeros_like(dark)
    for first, last in across:
        mask[max(first - RULE_PAD, 0) : last + RULE_PAD + 1, :] = True
    for (first, last), y0, y1 in down:
        mask[y0:y1, max(first - RULE_PAD, 0) : last + RULE_PAD + 1] = True
    if not mask.any():
        return image, 0
    pixels[mask.reshape(-1)] = backdrop.astype(numpy.uint8)
    return Image.fromarray(rgb), len(across) + len(down)


def reference_slots(workflow: dict) -> int:
    """How many reference images the workflow can take."""
    slots = [p for p in _placeholders(workflow) if p.startswith("$image")]
    return max((int(p.removeprefix("$image")) for p in slots), default=0)


def check_references(workflow: dict, count: int) -> None:
    """Refuse a render with more reference images than the workflow can load."""
    slots = reference_slots(workflow)
    if count > slots:
        raise ComfyError(
            f"this render has {count} reference images but the workflow has {slots} "
            f"slots ($image1..$image{slots}); add LoadImage nodes up to $image{count}"
        )


def fill(
    workflow: dict,
    prompt: str,
    images: Sequence[str],
    seed: int,
    shape: Canvas,
) -> dict:
    """A copy of `workflow` with every placeholder replaced.

    `images` are names already uploaded to Comfy, one per reference.
    """
    check_references(workflow, len(images))
    values: dict[str, Any] = {
        "$prompt": prompt,
        "$seed": seed,
        "$width": shape.width,
        "$height": shape.height,
        "$aspect": shape.aspect,
        **{f"$image{index}": name for index, name in enumerate(images, start=1)},
    }
    filled = copy.deepcopy(workflow)
    unused = []
    for node_id, node in filled.items():
        for key, value in node.get("inputs", {}).items():
            if not isinstance(value, str):
                continue
            if value in values:
                node["inputs"][key] = values[value]
            elif value.startswith("$image"):
                unused.append(node_id)
    for node_id in unused:
        _drop(filled, node_id)
    return filled


def _placeholders(workflow: dict) -> set[str]:
    return {
        value
        for node in workflow.values()
        for value in node.get("inputs", {}).values()
        if isinstance(value, str) and value.startswith("$")
    }


def _links_to(value: Any, node_id: str) -> bool:
    return isinstance(value, list) and len(value) == 2 and str(value[0]) == node_id


def _drop(workflow: dict, node_id: str) -> None:
    """Remove a node and every link to it, bypassing batch nodes it fed."""
    workflow.pop(node_id, None)
    for other_id, node in list(workflow.items()):
        inputs = node.get("inputs", {})
        cut = [key for key, value in inputs.items() if _links_to(value, node_id)]
        if not cut:
            continue
        for key in cut:
            del inputs[key]
        if node.get("class_type") not in BATCH_NODES:
            continue
        remaining = [value for value in inputs.values() if isinstance(value, list)]
        if len(remaining) == 1:
            _bypass(workflow, other_id, remaining[0])
        elif not remaining:
            _drop(workflow, other_id)


def _bypass(workflow: dict, node_id: str, source: list) -> None:
    """Point everything reading `node_id` at `source` instead, and remove it."""
    workflow.pop(node_id)
    for node in workflow.values():
        inputs = node.get("inputs", {})
        for key, value in inputs.items():
            if _links_to(value, node_id):
                inputs[key] = list(source)


class Client:
    """The few ComfyUI calls a render needs, on Comfy Cloud or, `local`, your own server."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        local: bool = False,
    ):
        self.where = "local ComfyUI" if local else "Comfy Cloud"
        if local:
            # A local server has no accounts: nothing to send, nothing to bill.
            api_key = api_key or ""
            base_url = base_url or os.environ.get("CAG_LOCAL_COMFY_URL") or LOCAL_URL
        else:
            # COMFY_API_KEY is the name Comfy's own SDK and samples read.
            api_key = (
                api_key or os.environ.get("COMFY_CLOUD_API_KEY") or os.environ.get("COMFY_API_KEY")
            )
            if not api_key:
                raise ComfyError(
                    "COMFY_CLOUD_API_KEY is not set (COMFY_API_KEY also works). Make a key at "
                    "https://platform.comfy.org/profile/api-keys"
                )
            base_url = base_url or os.environ.get("COMFY_CLOUD_URL") or BASE_URL
        self.api_key = api_key
        self.http = httpx.Client(
            base_url=base_url,
            headers={"X-API-Key": api_key} if api_key else {},
            timeout=120,
            transport=transport,
        )
        # The signed download URL is on another host: it must not see the key.
        self.storage = httpx.Client(timeout=120, transport=transport, follow_redirects=True)

    def upload(self, path: Path | str, boxed: bool = True) -> str:
        """Upload one reference image and return the name a workflow loads it by.

        `boxed` letterboxes it first (see `REFERENCE_SIDE`), which only a
        workflow that batches its references needs. The name is the content
        hash, so the same reference uploaded for every render of a set is the
        same file on the server.
        """
        if boxed:
            data = letterbox(path)
        else:
            with Image.open(path) as image:
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="PNG")
            data = buffer.getvalue()
        name = f"cag-{hashlib.sha256(data).hexdigest()[:16]}.png"
        response = self.http.post(
            "/api/upload/image",
            files={"image": (name, data, "image/png")},
            data={"type": "input", "overwrite": "true"},
        )
        _check(response, "upload", self.where)
        return response.json()["name"]

    def submit(self, workflow: dict) -> str:
        # Partner nodes (GPT Image, Nano Banana, ...) bill to the same key. A local
        # server has none to send, and no partner nodes to bill.
        extra = {"extra_data": {"api_key_comfy_org": self.api_key}} if self.api_key else {}
        response = self.http.post("/api/prompt", json={"prompt": workflow, **extra})
        _check(response, "submit", self.where)
        body = response.json()
        if "prompt_id" not in body:
            raise ComfyError(f"workflow refused: {body.get('error') or body}")
        return body["prompt_id"]

    def wait(self, job_id: str, timeout: float) -> dict:
        """Poll a job until it finishes, and return its details."""
        deadline = time.monotonic() + timeout
        while True:
            response = self.http.get(f"/api/jobs/{job_id}")
            _check(response, "job status", self.where)
            job = response.json()
            if job.get("status") in TERMINAL:
                if job["status"] != "completed":
                    detail = job.get("execution_error") or job["status"]
                    raise ComfyError(f"job {job_id} {job['status']}: {detail}")
                return job
            if time.monotonic() > deadline:
                self.http.post("/api/queue", json={"delete": [job_id]})
                raise ComfyError(f"job {job_id} timed out after {timeout:.0f}s")
            time.sleep(POLL_SECONDS)

    def download(self, image: dict) -> bytes:
        response = self.http.get(
            "/api/view",
            params={
                "filename": image["filename"],
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            },
        )
        if response.is_redirect:
            response = self.storage.get(response.headers["location"])
        _check(response, "download", self.where)
        return response.content


def _check(response: httpx.Response, what: str, where: str = "Comfy Cloud") -> None:
    if response.is_success or response.is_redirect:
        return
    why = {
        401: "the API key was refused",
        402: "the account is out of credits",
        429: "the subscription is inactive",
    }.get(response.status_code, response.text[:500])
    raise ComfyError(f"{where} {what} failed ({response.status_code}): {why}")


def first_image(outputs: dict) -> dict:
    """The first saved image among a finished job's outputs."""
    images = [
        image
        for output in outputs.values()
        for image in (output or {}).get("images", [])
    ]
    saved = [image for image in images if image.get("type", "output") == "output"]
    if not (saved or images):
        raise ComfyError("the job finished without saving an image; add a Save Image node")
    return (saved or images)[0]


def render(
    prompt: str,
    out_path: Path,
    references: Sequence[Path | str],
    workflow: dict,
    timeout: float,
    client: Client | None = None,
    rules: bool = True,
    local: bool = False,
) -> None:
    """Draw `prompt` with `workflow` and write the result to `out_path` as a PNG.

    `local` runs it on your own ComfyUI server rather than Comfy Cloud.

    `rules` paints out panel rules (see `erase_panel_rules`). A location turns
    it off: a stage edge or a lighting truss is a long dark line that belongs.
    The render as drawn is kept beside it as `<name>.ruled.png`.
    """
    check_references(workflow, len(references))
    client = client or Client(local=local)
    # Letterboxing exists to survive a batch; a workflow that loads each image
    # into its own input reads each at its own shape.
    boxed = any(node.get("class_type") in BATCH_NODES for node in workflow.values())
    names = [client.upload(reference, boxed) for reference in references]
    filled = fill(workflow, prompt, names, random.randrange(2**32), canvas(prompt))
    job = client.wait(client.submit(filled), timeout)
    data = client.download(first_image(job.get("outputs") or {}))
    with Image.open(io.BytesIO(data)) as image:
        drawn = image.convert("RGB")
    erased = 0
    if rules:
        cleaned, erased = erase_panel_rules(drawn)
    if erased:
        drawn.save(out_path.with_name(f"{out_path.stem}.ruled.png"))
        cleaned.save(out_path, format="PNG")
    else:
        drawn.save(out_path, format="PNG")
