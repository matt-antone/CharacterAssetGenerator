"""Read a prop, and say how to draw and hold it.

A prop is authored once and carried by whichever characters hold it, in
whichever sets they hold it for. Everything the render needs is in the file:
what it looks like, which of the character's own hands are on it, how it is
oriented, and how it is carried in each animation set.

A character holds a prop only where their brief names it — on the key art, or
in a named animation set. A performer who holds nothing names none, and nothing
is drawn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Props are named by a brief, never pathed, and found here by that name.
PROP_ROOT = Path("props")


class PropError(ValueError):
    """Raised when a prop cannot be drawn."""


@dataclass(frozen=True)
class Prop:
    name: str
    summary: str
    draw: dict
    hold: dict
    carriage: dict
    never: tuple[str, ...]

    def clause(self, set_name: str | None = None) -> str:
        """The prop, written for a render prompt.

        `set_name` picks the carriage for that animation set. Without one, or
        for a set the prop says nothing about, it rests: that is the key art and
        the projection views, which are not performing anything.
        """
        hands = ". ".join(f"{side}: {job.rstrip('.')}" for side, job in self.hold["hands"].items())
        carried = self.carriage.get(set_name) if set_name else None
        parts = [
            f"{self.name.capitalize()}: {self.summary}",
            self.draw["shape"],
            self.draw.get("material", ""),
            _palette(self.draw.get("palette")),
            self.draw.get("readability", ""),
            f"Held — {hands}.",
            self.hold.get("orientation", ""),
            carried or self.hold.get("rest", ""),
            "Never " + "; never ".join(self.never) + "." if self.never else "",
        ]
        return " ".join(part.strip() for part in parts if part and part.strip())


def _palette(palette: dict | None) -> str:
    if not palette:
        return ""
    return (
        f"Its colours are {palette['base']} base, "
        f"{palette['shadow']} shadow, {palette['highlight']} highlight."
    )


def load_prop(name: str, root: Path | str = PROP_ROOT) -> Prop:
    """Read `<root>/<name>.json`."""
    path = Path(root) / f"{name}.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise PropError(f"there is no {name!r} prop at {path}") from None
    except json.JSONDecodeError as exc:
        raise PropError(f"{path} is not valid JSON: {exc}") from exc

    missing = {"name", "summary", "draw", "hold"} - set(data)
    if missing:
        raise PropError(f"{path} is missing {', '.join(sorted(missing))}")
    hands = data["hold"].get("hands")
    if not isinstance(hands, dict) or not hands:
        raise PropError(f"{path} must say which of the character's own hands hold it")
    unknown = set(hands) - {"character-left", "character-right"}
    if unknown:
        raise PropError(
            f"{path} names {', '.join(sorted(unknown))}; a prop is held in the character's "
            "own hands, never a screen side"
        )
    if not data["draw"].get("shape"):
        raise PropError(f"{path} must say what shape the prop is")

    return Prop(
        name=data["name"],
        summary=data["summary"],
        draw=data["draw"],
        hold=data["hold"],
        carriage=data.get("carriage", {}),
        never=tuple(data.get("never", ())),
    )


def clauses(names: tuple[str, ...], set_name: str | None, root: Path | str = PROP_ROOT) -> str:
    """Every prop this character holds here, as one block, or "" for empty hands."""
    return "\n\n".join(load_prop(name, root).clause(set_name) for name in names)
