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
hair, skin, clothing with exact colours, footwear, and any handheld prop. Name the character's \
own side for anything asymmetric, using the words "character-left" and "character-right" \
(never "left", "right", "screen-left" or "screen-right"). State the prop hand explicitly.

Do not describe background, lighting, mood, camera, pose, action, or art style. Do not invent \
a name or backstory. Output the paragraph and nothing else."""

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
            f"Draw {spec.name}, who is {spec.height} tall.",
            bible,
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
) -> str:
    """Prompt for one animation frame."""
    return "\n\n".join(
        part for part in [
            f"Draw one animation frame of {spec.name}, who is {spec.height} tall.",
            bible,
            set_note,
            view_clause,
            f"Pose for this frame ({role}): {cue}",
            STYLE,
            BACKDROP,
            detail_clause(detail_level),
            DETAIL_REFERENCE.format(level=detail_level) if detail_reference else "",
            POSE_REFERENCE if pose_reference else "",
            SIDE_LANGUAGE,
            "Match the reference images for identity, costume, colour, proportion and prop hand "
            "exactly; only the pose changes. The supporting heel stays on the floor. Do not "
            "mirror the figure and do not move the prop to the other hand.",
        ] if part
    )
