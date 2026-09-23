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
from .motion import PLAYBACKS
from .style import DEFAULT_DETAIL_LEVEL, DETAIL_LEVELS

HEIGHT_PATTERN = re.compile(r"^\d+'(\s*\d+\")?$")
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: The optional prose fields specs/character.schema.json publishes. A brief that
#: fills them says a costume fact once, in its own field, instead of burying it
#: in the description for a regex to dig back out.
TEXT_FIELDS = (
    "age",
    "build",
    "face",
    "hair",
    "outfit",
    "prop",
    "personality",
    "performance_style",
    "location",
)
#: The optional list fields, same schema, same reason.
LIST_FIELDS = ("palette", "recognition_cues", "avoid")


class SpecError(ValueError):
    """Raised when a character brief is unusable."""


@dataclass(frozen=True)
class CharacterSpec:
    name: str
    height: str
    description: str
    #: Animation name -> prose intent, for the sets the brief itself prompts,
    #: e.g. {"dance": "Relaxed two-step loop."} A set driven by a motion spec
    #: is not in here; it is in `motions`. Ask `sets` for all of them.
    animations: dict[str, str] = field(default_factory=dict)
    #: Props the character holds in the key art and the projection views, by
    #: name. Empty hands for a performer who carries nothing.
    props: tuple[str, ...] = ()
    #: Animation name -> the props held in that set. A set that names none is
    #: drawn empty-handed, which is how a prop is kept out of one set.
    animation_props: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Animation name -> the traced sheet that drives it, named not pathed, e.g.
    #: {"dance": "zs-loop"}. Naming a choreography is authoring; finding the file
    #: it lives in is the pipeline's job, so a brief stays portable between machines.
    motions: dict[str, str] = field(default_factory=dict)
    #: Animation name -> how that set repeats, for a set that writes its own
    #: sheet. Only here when this character's set differs from the pipeline
    #: default in `cag/sets.py`; a set driven by a traced sheet is not here at
    #: all, because the trace already carries it.
    playbacks: dict[str, str] = field(default_factory=dict)
    #: Rendering density on the ten-step scale. Nothing else about the look.
    detail_level: int = DEFAULT_DETAIL_LEVEL
    #: The brief's own slug, when it names one. Otherwise the name makes it.
    id: str = ""
    #: TEXT_FIELDS, each empty unless the brief fills it.
    age: str = ""
    build: str = ""
    face: str = ""
    hair: str = ""
    outfit: str = ""
    prop: str = ""
    personality: str = ""
    performance_style: str = ""
    #: Where this character performs, as prose. Their background plate is drawn
    #: from it, and nothing else in the pipeline reads it: it is the one authored
    #: field that never reaches a render of the character themselves.
    location: str = ""
    #: LIST_FIELDS, each empty unless the brief fills it.
    palette: tuple[str, ...] = ()
    recognition_cues: tuple[str, ...] = ()
    avoid: tuple[str, ...] = ()

    @property
    def sets(self) -> tuple[str, ...]:
        """Every animation set the brief has, whichever prompt drives each one."""
        return tuple(sorted({*self.animations, *self.motions}))

    @property
    def slug(self) -> str:
        return self.id or re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")

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

    unknown = set(data) - {
        "name",
        "height",
        "description",
        "animations",
        "detail_level",
        "props",
        "id",
        *TEXT_FIELDS,
        *LIST_FIELDS,
    }
    if unknown:
        raise SpecError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")

    for key in ("name", "height", "description"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise SpecError(f"{path} needs a non-empty {key}")

    height = data["height"].strip()
    if not HEIGHT_PATTERN.match(height):
        raise SpecError(f"""{path} height must read like 5' 9" or 6', not {height!r}""")

    animations, motions, animation_props, playbacks = _animations(
        path, data.get("animations", {})
    )
    props = _props(path, "props", data.get("props", ()))

    detail_level = data.get("detail_level", DEFAULT_DETAIL_LEVEL)
    if detail_level not in DETAIL_LEVELS:
        raise SpecError(f"{path} detail_level must be an integer 1-10, not {detail_level!r}")

    identifier = data.get("id", "")
    if not isinstance(identifier, str) or (identifier and not ID_PATTERN.match(identifier.strip())):
        raise SpecError(f"{path} id must read like a slug, e.g. velvet-lou, not {identifier!r}")

    spec = CharacterSpec(
        data["name"].strip(),
        height,
        data["description"].strip(),
        animations,
        props,
        animation_props,
        motions,
        playbacks,
        detail_level,
        identifier.strip(),
        **{key: _text(path, key, data[key]) for key in TEXT_FIELDS if key in data},
        **{key: _text_list(path, key, data[key]) for key in LIST_FIELDS if key in data},
    )
    if not spec.slug:
        raise SpecError(f"{path} name has no usable characters")
    return spec


def _text(path: Path, key: str, raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise SpecError(f"{path} {key} must be non-empty prose")
    return raw.strip()


def _text_list(path: Path, key: str, raw: object) -> tuple[str, ...]:
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise SpecError(f"{path} {key} must be a list of non-empty lines")
    return tuple(_text(path, key, value) for value in raw)


def _animations(
    path: Path, raw: object
) -> tuple[dict[str, str], dict[str, str], dict[str, tuple[str, ...]], dict[str, str]]:
    """Split each animation into what drives it and what it holds.

    A set is driven by one prompt: the brief's own prose, or a traced motion
    spec named by `motion`. Never both — two prompts for one set is two people
    directing it, and the pipeline would have to pick. So `intent` and `motion`
    are exclusive, and an animation that is bare prose is the first of the two.

    `playback` is this character's loop rule, and belongs with `intent` for the
    same reason: a motion spec already carries its own.
    """
    if not isinstance(raw, dict):
        raise SpecError(f"{path} animations must map a name to a description")
    intents: dict[str, str] = {}
    motions: dict[str, str] = {}
    props: dict[str, tuple[str, ...]] = {}
    playbacks: dict[str, str] = {}
    for name, value in raw.items():
        if isinstance(value, str):
            value = {"intent": value}
        elif not isinstance(value, dict):
            raise SpecError(f"{path} {name} must be a description or an object")

        unknown = set(value) - {"intent", "motion", "props", "playback"}
        if unknown:
            raise SpecError(f"{path} {name} has unknown fields: {', '.join(sorted(unknown))}")
        if ("intent" in value) == ("motion" in value):
            raise SpecError(
                f"{path} {name} needs an intent or a motion, not both and not neither: "
                f"one set is driven by one prompt"
            )

        if "props" in value:
            props[name] = _props(path, name, value["props"])

        if "motion" in value:
            if "playback" in value:
                raise SpecError(
                    f"{path} {name} names a motion and a playback; the motion spec carries "
                    f"its own, so the brief does not get a second say"
                )
            motions[name] = _motion(path, name, value["motion"])
            continue

        if "playback" in value:
            playbacks[name] = _playback(path, name, value["playback"])
        intent = value["intent"]
        if not isinstance(intent, str) or not intent.strip():
            raise SpecError(f"{path} {name} needs a non-empty description")
        intents[name] = intent
    return intents, motions, props, playbacks


def _motion(path: Path, where: str, raw: object) -> str:
    """The traced sheet driving this set, named rather than pathed.

    A brief that spelled out where the sheet lives would only build on the
    machine that path came from.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise SpecError(f"{path} {where} motion must be a non-empty sheet name")
    sheet = raw.strip()
    if "/" in sheet or "\\" in sheet or sheet.startswith("."):
        raise SpecError(f"{path} {where} motion must name a sheet, not a path: {sheet!r}")
    return sheet


def _playback(path: Path, where: str, raw: object) -> str:
    """How this set repeats, in the one spelling the rest of the pipeline uses."""
    if not isinstance(raw, str) or raw.strip().lower() not in PLAYBACKS:
        raise SpecError(
            f"{path} {where} playback must be one of {', '.join(PLAYBACKS)}, not {raw!r}"
        )
    return PLAYBACKS[raw.strip().lower()]


def _props(path: Path, where: str, raw: object) -> tuple[str, ...]:
    """The props named at `where`, checked to be names rather than paths."""
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise SpecError(f"{path} {where} props must be a list of prop names")
    names = []
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise SpecError(f"{path} {where} props must each be a non-empty prop name")
        name = value.strip()
        if "/" in name or "\\" in name or name.startswith("."):
            raise SpecError(f"{path} {where} names a path, not a prop: {name!r}")
        names.append(name)
    return tuple(names)
