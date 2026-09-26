"""Machine profiles: which model files, at what sizes, one machine draws the video path with.

The video path runs three Comfy graphs — the SCAIL-2 job (`video`), one
Qwen-Image-2.1 edit per traced frame (`restyle`), and the SAM3 job that makes a
drive mask when the footage has no light backdrop (`mask`). Each graph is a file
under `comfy/` that names Comfy Cloud's model files outright, because that is
where the recipe was measured. A machine that cannot load those files — a 16 GB
card that needs SCAIL-2 as a GGUF, a 4 GB card that needs everything smaller —
does not get its own copy of each graph. It gets a machine profile,
`comfy/machines/<name>.json`, that patches the shared graphs by node id and
carries the numbers the graphs leave as placeholders:

    video     $width $height $steps $seed   the SCAIL video's size, steps and seed
              $length                       not here: the drive video decides it
    restyle   $resolution $steps            the restyle's size and steps; its seed
                                            is `draw`'s, so a redraw is a new roll
    mask      nothing beyond $prompt and $image1

A patch entry either merges into a node's inputs (`{"inputs": {...}}`), or
replaces the node when it names a `class_type` — a GGUF loader takes only
`unet_name`, so it keeps none of the loader it stands in for. Node ids are
checked: a patch aimed at a node the graph does not have is refused, not
ignored.

`materialise` writes the patched graph out. That file is the record of what was
sent, and its bytes are what the video path hashes into its digests, so a
changed profile redraws what it changed and nothing else.

`verified` says whether art has ever been drawn on the machine with this
profile. Only `cloud` is; the other two are the plan for hardware that has not
drawn yet, and a build on them says so.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cag import comfy

#: Where the profiles live.
MACHINES = Path("comfy/machines")

#: The graphs a machine draws the video path with, each overridden by its own
#: environment variable the way `CAG_COMFY_POSE_WORKFLOW` overrides the pose one.
STAGES = {
    "video": ("CAG_COMFY_SCAIL_WORKFLOW", Path("comfy/scail2-animate.json")),
    "restyle": ("CAG_COMFY_RESTYLE_WORKFLOW", Path("comfy/qwen21-restyle.json")),
    "mask": ("CAG_COMFY_MASK_WORKFLOW", Path("comfy/sam3-drive-mask.json")),
}

#: Every placeholder each shipped graph carries. `$prompt`, `$seed` and
#: `$imageN` are `comfy.fill`'s own; the rest are filled from the profile
#: (`Machine.placeholders`) or, for `$length`, from the drive video.
PLACEHOLDERS = {
    "video": frozenset({"$prompt", "$image1", "$image2", "$image3", "$image4",
                        "$width", "$height", "$length", "$seed", "$steps"}),
    "restyle": frozenset({"$prompt", "$image1", "$image2", "$resolution", "$seed", "$steps"}),
    "mask": frozenset({"$prompt", "$image1"}),
}

#: The longest SCAIL video one job draws. SCAIL-2 was trained on 81-frame
#: chunks, and chaining chunks is not built.
MAX_LENGTH = 81

#: Overrides every profile's seed, so a second roll of the same set is one
#: variable away rather than an edited profile.
SEED_VARIABLE = "CAG_VIDEO_SEED"

BACKENDS = ("comfy", "local")


class MachineError(comfy.ComfyError):
    """Raised for a profile that is missing, malformed, or patches a node that is not there."""


@dataclass(frozen=True)
class Machine:
    name: str
    backend: str
    verified: bool
    video_patch: Mapping[str, Any]
    restyle_patch: Mapping[str, Any]
    mask_patch: Mapping[str, Any]
    width: int
    height: int
    rate: float
    max_length: int
    steps: int
    restyle_resolution: int
    restyle_steps: int
    video_timeout: float
    restyle_timeout: float
    mask_timeout: float
    seed: int = 1234
    about: str = ""

    def patch(self, stage: str) -> Mapping[str, Any]:
        """The patch this machine applies to one stage's graph."""
        return {"video": self.video_patch, "restyle": self.restyle_patch, "mask": self.mask_patch}[
            _stage(stage)
        ]

    def placeholders(self, stage: str) -> dict[str, Any]:
        """The placeholder values this machine fills in one stage's graph.

        The restyle's seed is left out on purpose: `draw` moves it on for each
        attempt, and a value here would pin every attempt to the same roll.
        """
        return {
            "video": {"$width": self.width, "$height": self.height,
                      "$steps": self.steps, "$seed": self.seed},
            "restyle": {"$resolution": self.restyle_resolution, "$steps": self.restyle_steps},
            "mask": {},
        }[_stage(stage)]


def _stage(stage: str) -> str:
    if stage not in STAGES:
        raise MachineError(f"no stage {stage!r}; the stages are {', '.join(STAGES)}")
    return stage


def machine_names(root: Path = MACHINES) -> list[str]:
    """Every profile there is, by name."""
    return sorted(path.stem for path in Path(root).glob("*.json"))


def load_machine(name: str, root: Path = MACHINES) -> Machine:
    """Read `root/<name>.json`, checked against what the graphs can take."""
    path = Path(root) / f"{name}.json"
    if not path.exists():
        raise MachineError(
            f"no machine profile {name!r}; the profiles are: {', '.join(machine_names(root)) or 'none'}"
        )
    data = json.loads(path.read_text())
    try:
        video, restyle, mask = data["video"], data["restyle"], data["mask"]
        machine = Machine(
            name=name,
            backend=data["backend"],
            verified=bool(data["verified"]),
            video_patch=video.get("patch", {}),
            restyle_patch=restyle.get("patch", {}),
            mask_patch=mask.get("patch", {}),
            width=int(video["width"]),
            height=int(video["height"]),
            rate=float(video["rate"]),
            max_length=int(video["max_length"]),
            steps=int(video["steps"]),
            restyle_resolution=int(restyle["resolution"]),
            restyle_steps=int(restyle["steps"]),
            video_timeout=float(video["timeout"]),
            restyle_timeout=float(restyle["timeout"]),
            mask_timeout=float(mask["timeout"]),
            seed=int(os.environ.get(SEED_VARIABLE) or data.get("seed", 1234)),
            about=data.get("about", ""),
        )
    except KeyError as missing:
        raise MachineError(f"{path} has no {missing.args[0]!r}") from None

    problems = []
    if machine.backend not in BACKENDS:
        problems.append(f"backend {machine.backend!r} is not one of {', '.join(BACKENDS)}")
    # WanSCAILToVideo steps its size by 32 and its length by 4 from 1.
    if machine.width % 32 or machine.height % 32:
        problems.append(f"{machine.width}x{machine.height} is not a multiple of 32")
    if (machine.max_length - 1) % 4 or not 1 <= machine.max_length <= MAX_LENGTH:
        problems.append(f"max_length {machine.max_length} is not 4k+1 up to {MAX_LENGTH}")
    if machine.restyle_resolution % 32:
        problems.append(f"restyle resolution {machine.restyle_resolution} is not a multiple of 32")
    if machine.rate <= 0 or machine.steps < 1 or machine.restyle_steps < 1:
        problems.append("rate and both step counts must be positive")
    if problems:
        raise MachineError(f"{path}: " + "; ".join(problems))
    return machine


def workflow_path(stage: str, given: Path | str | None = None) -> Path:
    """The graph one stage draws with: `given`, else its environment variable, else the shipped file."""
    variable, default = STAGES[_stage(stage)]
    return Path(given or os.environ.get(variable) or default)


def patch_workflow(workflow: dict, patch: Mapping[str, Any]) -> dict:
    """A copy of `workflow` with `patch` applied by node id. See the module note."""
    patched = json.loads(json.dumps(workflow))
    for node_id, change in patch.items():
        if node_id not in patched:
            raise MachineError(
                f"the patch names node {node_id!r}, which the graph does not have "
                f"(its nodes: {', '.join(patched)})"
            )
        unknown = set(change) - {"class_type", "inputs", "_meta"}
        if unknown:
            raise MachineError(f"the patch for node {node_id!r} has {', '.join(sorted(unknown))}; "
                               "a patch sets class_type, inputs or _meta")
        inputs = dict(change.get("inputs", {}))
        if "class_type" in change:
            patched[node_id] = {"class_type": change["class_type"], "inputs": inputs,
                                **({"_meta": change["_meta"]} if "_meta" in change else {})}
            continue
        node = patched[node_id]
        # A misspelt input would be dropped by Comfy without a word, so only
        # inputs the node already has are merged; a new loader is a class_type.
        stray = set(inputs) - set(node.get("inputs", {}))
        if stray:
            raise MachineError(
                f"node {node_id!r} ({node['class_type']}) has no input {', '.join(sorted(stray))}; "
                "to change what the node is, give its class_type"
            )
        node.setdefault("inputs", {}).update(inputs)
        if "_meta" in change:
            node["_meta"] = change["_meta"]
    _check_links(patched)
    return patched


def _check_links(workflow: dict) -> None:
    """Refuse a graph with an input linked to a node it does not have."""
    for node_id, node in workflow.items():
        for key, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) not in workflow:
                raise MachineError(f"node {node_id!r} input {key!r} links to missing node {value[0]!r}")


def materialise(machine: Machine, stage: str, into: Path, source: Path | str | None = None) -> Path:
    """Write one stage's graph, patched for `machine`, to `into/<machine>/<stage>.json`.

    `source` is the graph to patch, `workflow_path(stage)` when not given. The
    file written is checked the way any workflow is before it is returned.
    """
    source = workflow_path(stage, source)
    workflow = comfy.load_workflow(source)
    try:
        patched = patch_workflow(workflow, machine.patch(stage))
    except MachineError as error:
        raise MachineError(f"machine {machine.name!r} cannot patch {source}: {error}") from None
    path = Path(into) / machine.name / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(patched, indent=1) + "\n")
    comfy.load_workflow(path)
    return path
