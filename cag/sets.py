"""What each animation set is, before anyone decides how it moves.

Ported from KaraokeParty-Graphics' production profile. The designer writes prose
intent per set in the brief; these are the technical settings that go with it,
and they belong to the pipeline rather than to the brief.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SetPlan:
    frame_count: int
    fps: int
    #: "loop" seams back to frame 0; anything else runs once.
    playback: str
    #: Which way the character faces, in MotionArtist's vocabulary.
    view: str
    #: True when the last frame is a resting state the set settles into.
    holds_final_frame: bool = False

    @property
    def loops(self) -> bool:
        return self.playback == "loop"


SET_PLANS = {
    "dance": SetPlan(8, 4, "loop", "front"),
    "sing": SetPlan(8, 4, "loop", "left"),
    "flinch": SetPlan(8, 4, "oneshot", "left"),
    "guard": SetPlan(8, 4, "oneshot", "left", holds_final_frame=True),
    "entrance": SetPlan(8, 4, "oneshot", "left"),
    "victory": SetPlan(8, 4, "oneshot", "front"),
    "ko": SetPlan(8, 4, "oneshot", "front", holds_final_frame=True),
}


def plan_for(set_name: str) -> SetPlan:
    try:
        return SET_PLANS[set_name]
    except KeyError:
        raise KeyError(
            f"no plan for set {set_name!r}; known sets: {', '.join(sorted(SET_PLANS))}"
        ) from None


#: One file to switch renders on and off, beside the specs it applies to.
SWITCHES = Path(__file__).resolve().parent.parent / "enabled.json"

#: The key art is the reference every other render quotes and the ruler the working
#: scale is read off, so it is drawn whatever the file says.
REQUIRED_VIEWS = frozenset({"key"})


def _switches(section: str) -> dict[str, bool]:
    """What the config says about one section. A missing file leaves everything on."""
    try:
        loaded = json.loads(SWITCHES.read_text())
    except FileNotFoundError:
        return {}
    return {k: bool(v) for k, v in loaded.get(section, {}).items() if not k.startswith("_")}


def wanted(names: list[str], section: str, required: frozenset[str] = frozenset()) -> list[str]:
    """`names` filtered to the ones the config leaves on, in the order given.

    A name the file does not mention is on: a set added to the pipeline renders
    without having to be listed first. Turning one off is always deliberate.
    """
    switches = _switches(section)
    return [n for n in names if n in required or switches.get(n, True)]
