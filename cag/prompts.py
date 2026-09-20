"""Prompt construction. The creative contract with the image generator.

Two rules the old project learned the hard way and this one keeps:

- `character-left` / `character-right` name the character's own sides;
  `screen-left` / `screen-right` name position in the image. A prop is locked to
  one of the character's hands and never changes hands because the view did.
- Art is never mirrored to face the other way. Every view is drawn.
"""

from __future__ import annotations

from .spec import CharacterSpec
from .style import (
    BACKDROP,
    DEFAULT_DETAIL_LEVEL,
    DETAIL_REFERENCE,
    SIDE_LANGUAGE,
    STANDING,
    STYLE,
    detail_clause,
)

#: Locked visual description, written once per character and quoted verbatim
#: into every later prompt so identity cannot drift between renders.
BIBLE_SYSTEM = """You are a character designer writing a model sheet description for an \
illustrator who will draw the same character many times and must not deviate.

Write one paragraph of 120-180 words describing only what is visible: build, posture, face, \
hair, skin, clothing with exact colours, and footwear. Name the character's own side for \
anything asymmetric, using the words "character-left" and "character-right" (never "left", \
"right", "screen-left" or "screen-right").

Do not describe background, lighting, mood, camera, pose, action, or art style. Do not invent \
a name or backstory. Say nothing about anything the character is holding: props are authored \
separately and attached per set, and a prop named here would follow the character into every \
set whether they carry it there or not. Output the paragraph and nothing else."""

VIEWS = {
    "key": "Front-left three-quarter view, the character's body angled so they face screen-left.",
    "front": "Straight-on front view, the character facing the viewer square-on.",
    "back": "Straight-on back view, the character facing directly away from the viewer.",
    "profile": "Strict side profile, the character facing screen-left, head and body in full profile.",
}

#: Drawn first and approved; every other render references it.
KEY_VIEW = "key"

#: MotionArtist names a frame's facing; art is drawn for it, never mirrored into it.
FRAME_VIEWS = {
    "front": VIEWS["front"],
    "left": VIEWS["profile"],
    "right": "Strict side profile, the character facing screen-right, head and body in full profile.",
    "3/4": VIEWS["key"],
}

DIRECTOR_SYSTEM = """You are the motion director for one animation set.

You are given a character description and the performance arc of a motion source. Write two to \
four sentences of standing instruction that every frame of this set must obey: what the character \
holds and in which of their own hands, how their costume and hair behave while they move, and \
what must not change between frames.

Do not restate the arc, describe individual frames, give timing, or mention the source video. \
Use "character-left" and "character-right" for the character's own sides. Output the \
instruction and nothing else."""


def director_request(bible: str, arc: str, fps: int, frame_count: int, view: str) -> str:
    return (
        f"Character: {bible}\n\n"
        f"Set: {frame_count} frames at {fps} fps, drawn in the {view} view.\n\n"
        f"Performance arc: {arc}"
    )


def bible_request(spec: CharacterSpec) -> str:
    return (
        f"Character: {spec.name}\n"
        f"Height: {spec.height}\n"
        f"Designer's brief: {spec.description}"
    )


def view_prompt(
    spec: CharacterSpec,
    bible: str,
    view: str,
    pose: str | None = None,
    detail_level: int = DEFAULT_DETAIL_LEVEL,
    detail_reference: bool = False,
    props: str = "",
) -> str:
    """Prompt for one static view of the character."""
    if view not in VIEWS:
        raise KeyError(f"unknown view {view!r}")
    stance = pose or (
        # Deliberately defers to the description. Inventing a stance here
        # overrode characters whose brief specified their own.
        "Use the character's own ready stance exactly as described above. If the description does "
        "not give one, a relaxed stance with the arms readable and clear of the torso."
        if view == KEY_VIEW
        else "The same neutral standing stance in every projection view: weight even on both "
        "feet, arms hanging clear of the torso, so the views can be compared."
    )
    return "\n\n".join(
        part for part in [
            f"Draw {spec.name}.",
            bible,
            props,
            VIEWS[view],
            stance,
            f"{STYLE} {STANDING}",
            BACKDROP,
            detail_clause(detail_level),
            DETAIL_REFERENCE.format(level=detail_level) if detail_reference else "",
            SIDE_LANGUAGE,
            "Keep every detail of the description above exactly as written, including which of "
            "the character's own hands holds any prop. Do not mirror the figure.",
        ] if part
    )


#: The pose cue names arm positions and the skeleton draws bare joints, so
#: between them a frame reads as an empty hand and the prop quietly disappears
#: mid-set. Neither is describing what the character is holding.
PROP_CONTINUITY = (
    "The pose cue and the pose skeleton describe limb positions only. Neither one shows what the "
    "character is holding: the skeleton has no props, and its rings mark joints, not empty hands. "
    "Anything the description above says the character holds is still in that same hand in this "
    "frame, drawn in full and clearly readable. Never replace a held prop with a bare fist or an "
    "open hand, in any frame, whatever the arm is doing."
)

POSE_REFERENCE = (
    "The last reference image is a stick-figure skeleton of this exact pose, with the floor "
    "drawn as a grey line and hollow rings on the character-right wrist and ankle. Copy the "
    "pose from it: limb angles, which knee is bent, how the weight sits, which way the head "
    "turns. It carries no identity, costume or style — take none of its look."
)


def frame_prompt(
    spec: CharacterSpec,
    bible: str,
    set_note: str,
    view_clause: str,
    cue: str,
    role: str,
    pose_reference: bool = False,
    detail_level: int = DEFAULT_DETAIL_LEVEL,
    detail_reference: bool = False,
    props: str = "",
) -> str:
    """Prompt for one animation frame."""
    return "\n\n".join(
        part for part in [
            f"Draw one animation frame of {spec.name}.",
            bible,
            props,
            set_note,
            view_clause,
            f"Pose for this frame ({role}): {cue}",
            STYLE,
            BACKDROP,
            detail_clause(detail_level),
            DETAIL_REFERENCE.format(level=detail_level) if detail_reference else "",
            POSE_REFERENCE if pose_reference else "",
            PROP_CONTINUITY,
            SIDE_LANGUAGE,
            "Match the reference images for identity, costume, colour, proportion and prop hand "
            "exactly; only the pose changes. The supporting heel stays on the floor. Do not "
            "mirror the figure and do not move the prop to the other hand.",
        ] if part
    )


POSE_SHEET_REFERENCE = (
    "The last reference image shows every pose in this sequence as a stick-figure skeleton, laid "
    "out in the same order and the same rows as the figures you draw, with the floor as a grey "
    "line and hollow rings on the character-right wrist and ankle. Copy each pose from its own "
    "skeleton: limb angles, which knee is bent, how the weight sits, which way the head turns. "
    "The skeletons carry no identity, costume or style — take none of their look."
)


def sheet_prompt(
    spec: CharacterSpec,
    bible: str,
    set_note: str,
    view_clause: str,
    cues: list[tuple[str, str]],
    pose_reference: bool = False,
    detail_level: int = DEFAULT_DETAIL_LEVEL,
    detail_reference: bool = False,
    per_row: int = 4,
    props: str = "",
) -> str:
    """Prompt for one render holding a whole animation sequence.

    Nothing in it measures. The generator cannot hold a ruler, and every number
    it was handed came back as noise — but it keeps figures it can see side by
    side the same size without being asked, and that is what the sheet is for.
    """
    poses = "\n".join(f"{n} ({role}): {cue}" for n, (role, cue) in enumerate(cues, 1))
    # Naming the grid costs nothing — the generator already lays sheets out this way — and it
    # keeps the prompt saying what the pose reference beside it shows.
    rows = -(-len(cues) // per_row)
    layout = (
        f"Arrange the figures in one row of {len(cues)}"
        if rows == 1
        else f"Arrange the figures in {rows} rows of {per_row}"
    )
    return "\n\n".join(
        part for part in [
            f"Draw {spec.name} {len(cues)} times in one image, as the consecutive frames of one "
            "animation.",
            bible,
            props,
            set_note,
            view_clause,
            f"Draw this on a wide landscape canvas, wider than it is tall. {layout} in "
            "reading order, left to right and then the next row down, with clear background "
            "between every figure. No figure touches another figure or the canvas edge. No "
            "numbers, labels, frame lines or grid lines: only the figures on the backdrop.",
            "Every figure is the same character at the same size and the same distance from the "
            f"viewer; only the pose changes from one to the next:\n{poses}",
            STYLE,
            BACKDROP,
            detail_clause(detail_level),
            DETAIL_REFERENCE.format(level=detail_level) if detail_reference else "",
            POSE_SHEET_REFERENCE if pose_reference else "",
            PROP_CONTINUITY,
            SIDE_LANGUAGE,
            "Match the reference images for identity, costume, colour, proportion and prop hand "
            "exactly. The supporting heel stays on the floor in every figure. Do not mirror any "
            "figure and do not move the prop to the other hand.",
        ] if part
    )
