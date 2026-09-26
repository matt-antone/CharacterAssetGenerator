"""Render a PNG by running a ComfyUI workflow, on Comfy Cloud or a local server.

Mechanism only, like the rest of `cag.draw`: this module turns one prompt and
its reference images into one image on disk. Which model draws, at what
settings, is the workflow's business, and the workflow is a file — exported from
Comfy in API format — not code. Swapping models is editing that file.

cag finds its way into the workflow by placeholders. Any node input whose value
is exactly one of these strings is filled in before the workflow is sent:

    $prompt              the render's prompt
    $image1 .. $imageN   the render's reference images, in the order attached
    $seed                the seed asked for, or a fresh random one, so a redraw is a new roll
    $width, $height      the canvas in pixels, read off the prompt
    $aspect              the same canvas as a ratio, "2:3", "3:2" or "16:9"

A caller can fill more (`extra`, e.g. a video's `$length`), and its values win.
Any other input still starting with `$` once these are in is refused before
anything is sent: ComfyUI would take the placeholder as a literal value.

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
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

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
#: A job still waiting its turn: its timeout has not started.
QUEUED = "pending"


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
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """A copy of `workflow` with every placeholder replaced.

    `images` are names already uploaded to Comfy, one per reference. `extra`
    fills placeholders of the caller's own, keyed as written in the workflow
    ("$length"), and is applied last, so it can also override the canvas.
    """
    check_references(workflow, len(images))
    unkeyed = [key for key in extra or {} if not key.startswith("$")]
    if unkeyed:
        raise ComfyError(f"extra placeholders must start with $: {', '.join(unkeyed)}")
    values: dict[str, Any] = {
        "$prompt": prompt,
        "$seed": seed,
        "$width": shape.width,
        "$height": shape.height,
        "$aspect": shape.aspect,
        **{f"$image{index}": name for index, name in enumerate(images, start=1)},
        **(extra or {}),
    }
    filled = copy.deepcopy(workflow)
    unused, leftover = [], []
    for node_id, node in filled.items():
        for key, value in node.get("inputs", {}).items():
            if not isinstance(value, str):
                continue
            if value in values:
                node["inputs"][key] = values[value]
            elif value.startswith("$image"):
                unused.append(node_id)
            elif value.startswith("$"):
                leftover.append(f"{node_id}.{key}={value}")
    # Checked while filling, not after: a filled-in prompt may itself start with $.
    if leftover:
        raise ComfyError(
            f"the workflow has placeholders nothing fills: {', '.join(leftover)}. "
            "ComfyUI would read them as literal values"
        )
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

    def upload_raw(self, path: Path | str) -> str:
        """Upload a file's bytes exactly as they are, and return its name.

        `upload` re-encodes through PIL, which keeps the first frame of an
        animated PNG and drops the rest; `LoadImage` reads every frame of one
        into a batch, which is how a video goes in as a reference.
        """
        data = Path(path).read_bytes()
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

    def job(self, job_id: str) -> dict | None:
        """One look at a job: its details, or None if the server has never heard of it."""
        response = self.http.get(f"/api/jobs/{job_id}")
        if response.status_code == 404:
            return None
        _check(response, "job status", self.where)
        return response.json()

    def wait(self, job_id: str, timeout: float, cancel: bool = True) -> dict:
        """Poll a job until it finishes, and return its details.

        `timeout` is the job's own running time. Time spent queued behind other
        jobs does not count: characters are built in parallel on one server,
        and a render that times out while it waits its turn is nothing wrong
        with the render.

        On timeout the job is cancelled, unless `cancel` is off: a job whose id
        was written down to be resumed is left running, so the next build can
        pick up what it has already paid for.
        """
        deadline = None
        while True:
            job = self.job(job_id)
            if job is None:
                raise ComfyError(f"{self.where} has no job {job_id}")
            if job.get("status") in TERMINAL:
                if job["status"] != "completed":
                    detail = job.get("execution_error") or job["status"]
                    raise ComfyError(f"job {job_id} {job['status']}: {detail}")
                return job
            if deadline is None and job.get("status") != QUEUED:
                deadline = time.monotonic() + timeout
            if deadline is not None and time.monotonic() > deadline:
                if cancel:
                    self.http.post("/api/queue", json={"delete": [job_id]})
                    raise ComfyError(f"job {job_id} timed out after {timeout:.0f}s")
                raise ComfyError(
                    f"job {job_id} still running after {timeout:.0f}s; left running, "
                    "the next build resumes it"
                )
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


#: The counter ComfyUI's SaveImage puts in each name: `<prefix>_00001_.png`.
COUNTER = re.compile(r"^(.*)_(\d+)_\.\w+$")


def saved_images(outputs: dict) -> list[dict]:
    """Every saved image among a finished job's outputs, in the order they were saved.

    Saved means written to the output folder; previews are only used when a job
    saved nothing. A video job saves one image per frame under one prefix, and
    the counter in each name is frame order. It is read off `display_name`
    when there is one: on Comfy Cloud `filename` is a content hash, and sorting
    by it shuffles the frames. The counter is compared as a number, so 100000
    follows 99999. Names with no counter keep the server's order.
    """
    images = [
        image
        for output in outputs.values()
        for image in (output or {}).get("images", [])
    ]
    saved = [image for image in images if image.get("type", "output") == "output"]
    chosen = saved or images
    counters = [COUNTER.match(image.get("display_name") or image["filename"]) for image in chosen]
    if not all(counters):
        return chosen
    order = sorted(
        range(len(chosen)), key=lambda k: (counters[k].group(1), int(counters[k].group(2)))
    )
    return [chosen[k] for k in order]


def first_image(outputs: dict) -> dict:
    """The first saved image among a finished job's outputs."""
    images = saved_images(outputs)
    if not images:
        raise ComfyError("the job finished without saving an image; add a Save Image node")
    return images[0]


def render(
    prompt: str,
    out_path: Path,
    references: Sequence[Path | str],
    workflow: dict,
    timeout: float,
    client: Client | None = None,
    rules: bool = True,
    local: bool = False,
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Draw `prompt` with `workflow` and write the result to `out_path` as a PNG.

    `local` runs it on your own ComfyUI server rather than Comfy Cloud. `seed`
    fills `$seed`; left out, every render is a fresh random roll. `extra` fills
    the workflow's own placeholders, as in `fill`.

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
    seed = random.randrange(2**32) if seed is None else seed
    filled = fill(workflow, prompt, names, seed, canvas(prompt), extra)
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


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def render_frames(
    prompt: str,
    out_dir: Path,
    references: Sequence[Path | str],
    workflow: dict,
    timeout: float,
    *,
    raw: Sequence[bool],
    extra: Mapping[str, Any] | None = None,
    expect: int,
    pending: Path,
    seed: int | None = None,
    client: Client | None = None,
    local: bool = False,
) -> list[Path]:
    """Run one job that saves many images, and write them to `out_dir/000.png` on.

    For a video: one SCAIL-2 job returns every frame. None of it goes through
    `draw`'s backdrop check — a frame here is an intermediate, not a source.

    `raw` says, per reference, whether its bytes go up untouched
    (`Client.upload_raw`, an animated PNG keeping every frame) or re-encoded
    at its own shape. `extra` and `seed` fill the workflow as in `fill`.

    A long job is resumable. Its id is written to `pending` as soon as it is
    submitted, and a later call finding that file picks the job up instead of
    paying for it again: waits on it if it is still running, downloads it if
    it finished, submits afresh only if it failed or the server has forgotten
    it. Waiting never cancels, so a build stopped mid-job leaves it running.

    A job that saves other than `expect` images is refused, and what it did
    save is kept in `out_dir/rejected-N/` to be looked at.
    """
    out_dir = Path(out_dir)
    pending = Path(pending)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = client or Client(local=local)
    job_id, job = _resume(client, pending, timeout)
    resumed = job is not None
    if job is None:
        check_references(workflow, len(references))
        if len(raw) != len(references):
            raise ComfyError(f"{len(references)} references but {len(raw)} raw flags")
        names = [
            client.upload_raw(reference) if as_is else client.upload(reference, boxed=False)
            for reference, as_is in zip(references, raw)
        ]
        seed = random.randrange(2**32) if seed is None else seed
        filled = fill(workflow, prompt, names, seed, canvas(prompt), extra)
        job_id = client.submit(filled)
        # Written before the wait, so a build stopped now resumes this job.
        _write_pending(pending, job_id, client)
        job = client.wait(job_id, timeout, cancel=False)

    try:
        images = [client.download(image) for image in saved_images(job.get("outputs") or {})]
    except ComfyError as error:
        if not resumed:
            raise
        # A finished job whose images will not download twice running (the
        # server's output folder was cleared, or the cloud's copy expired)
        # would fail the same way every build; forget it so the next submits.
        pending.unlink(missing_ok=True)
        raise ComfyError(
            f"job {job_id} finished but its images will not download ({error}); "
            f"{pending.name} is cleared, so the next build draws it again"
        ) from error
    if len(images) != expect:
        kept = _next_free(out_dir, "rejected-")
        _save_all(images, kept)
        pending.unlink(missing_ok=True)
        raise ComfyError(
            f"job {job_id} saved {len(images)} images, not the {expect} asked for; "
            f"kept in {kept}"
        )
    frames = _save_all(images, out_dir)
    # Last, so a build stopped while downloading downloads again.
    pending.unlink(missing_ok=True)
    return frames


def _resume(client: Client, pending: Path, timeout: float) -> tuple[str | None, dict | None]:
    """The job `pending` names, finished, or (None, None) if there is none to pick up."""
    if not pending.exists():
        return None, None
    try:
        job_id = json.loads(pending.read_text())["job_id"]
    except (ValueError, KeyError, TypeError):
        _log(f"{pending} is unreadable; submitting again")
        return None, None
    job = client.job(job_id)
    if job is None:
        _log(f"{client.where} has no job {job_id}; submitting again")
        return None, None
    status = job.get("status")
    if status == "completed":
        _log(f"job {job_id} finished while nobody was waiting; downloading it")
        return job_id, job
    if status in TERMINAL:
        _log(f"job {job_id} {status}; submitting again")
        return None, None
    _log(f"resuming job {job_id} on {client.where}")
    return job_id, client.wait(job_id, timeout, cancel=False)


def _write_pending(pending: Path, job_id: str, client: Client) -> None:
    pending.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "job_id": job_id,
        "where": client.where,
        "url": str(client.http.base_url),
        "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    partial = pending.with_name(pending.name + ".part")
    partial.write_text(json.dumps(record, indent=2) + "\n")
    partial.replace(pending)


def _next_free(parent: Path, prefix: str) -> Path:
    """`parent/<prefix>N` for the first N not already there."""
    taken = {
        int(path.name.removeprefix(prefix))
        for path in parent.glob(f"{prefix}*")
        if path.name.removeprefix(prefix).isdigit()
    }
    return parent / f"{prefix}{next(n for n in range(len(taken) + 1) if n not in taken)}"


def _save_all(images: Sequence[bytes], into: Path) -> list[Path]:
    """Each downloaded image as `into/NNN.png`, in order."""
    into.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, data in enumerate(images):
        paths.append(into / f"{index:03d}.png")
        with Image.open(io.BytesIO(data)) as image:
            image.convert("RGB").save(paths[-1], format="PNG")
    return paths


#: Loader inputs that name a model file: checked against what the server has.
FILE_INPUT = re.compile(r"(unet|lora|clip|vae|ckpt)_name\d*")


def preflight(client: Client, workflows: Iterable[Path | str | dict]) -> list[str]:
    """Everything the workflows need that the server does not have, or [] if nothing.

    Asks the server for each node class the workflows use, and checks every
    model file a loader names against the files that node offers — one pass
    that lists all of it, instead of a job that fails on the first missing
    file after loading the rest. Meant for a local server, which answers these
    questions; a server that does not (Comfy Cloud) is not checked.
    """
    graphs = []
    for index, workflow in enumerate(workflows, start=1):
        if isinstance(workflow, dict):
            graphs.append((f"workflow {index}", workflow))
        else:
            graphs.append((Path(workflow).name, json.loads(Path(workflow).read_text())))

    info: dict[str, dict | None] = {}
    missing: list[str] = []
    for label, workflow in graphs:
        for node_id, node in workflow.items():
            kind = node.get("class_type", "")
            if kind not in info:
                try:
                    response = client.http.get(f"/api/object_info/{quote(kind, safe='')}")
                except httpx.TransportError as error:
                    raise ComfyError(
                        f"{client.where} is not answering at {client.http.base_url}: {error}"
                    ) from error
                if response.status_code == 404 and client.where != "local ComfyUI":
                    _log(f"preflight unavailable on {client.where}; nothing checked")
                    return []
                if response.status_code != 404:
                    _check(response, "node info", client.where)
                # A local server answers an unknown class with an empty object.
                info[kind] = None if response.status_code == 404 else response.json().get(kind)
            if info[kind] is None:
                missing.append(f"{label}: node {kind} is not installed")
                continue
            declared = {
                **(info[kind].get("input") or {}).get("required", {}),
                **(info[kind].get("input") or {}).get("optional", {}),
            }
            for key, value in node.get("inputs", {}).items():
                if not (FILE_INPUT.fullmatch(key) and isinstance(value, str)):
                    continue
                if value.startswith("$"):
                    continue
                options = _options(declared.get(key))
                if options is not None and value not in options:
                    missing.append(f"{label}: {kind} {key} {value} is not installed")
    return list(dict.fromkeys(missing))


def _options(spec: Any) -> list | None:
    """The choices a node input offers, from either of ComfyUI's two spellings."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    return None
