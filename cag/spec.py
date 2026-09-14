"""The character brief: everything a designer authors, and nothing else.

The old project let production settings, QA policy, scale records and approval
state creep into the spec until nobody could write one by hand. Here a brief is
a name, a height, a description, and prose intent per animation. Paths, fps,
frame counts and geometry belong to the pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import parse_height

HEIGHT_PATTERN = re.compile(r"^\d+'(\s*\d+\")?$")


class SpecError(ValueError):
    """Raised when a character brief is unusable."""


@dataclass(frozen=True)
class CharacterSpec:
    name: str
    height: str
    description: str
    #: Animation name -> prose intent, e.g. {"dance": "Relaxed two-step loop."}
    animations: dict[str, str] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")

    @property
    def height_inches(self) -> float:
        return parse_height(self.height)


def load_spec(path: Path | str) -> CharacterSpec:
    """Read and validate a character brief."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SpecError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecError(f"{path} must contain a JSON object")

    unknown = set(data) - {"name", "height", "description", "animations"}
    if unknown:
        raise SpecError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")

    for key in ("name", "height", "description"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise SpecError(f"{path} needs a non-empty {key}")

    height = data["height"].strip()
    if not HEIGHT_PATTERN.match(height):
        raise SpecError(f"""{path} height must read like 5' 9" or 6', not {height!r}""")

    animations = data.get("animations", {})
    if not isinstance(animations, dict) or not all(
        isinstance(v, str) and v.strip() for v in animations.values()
    ):
        raise SpecError(f"{path} animations must map a name to a non-empty description")

    spec = CharacterSpec(data["name"].strip(), height, data["description"].strip(), animations)
    if not spec.slug:
        raise SpecError(f"{path} name has no usable characters")
    return spec
