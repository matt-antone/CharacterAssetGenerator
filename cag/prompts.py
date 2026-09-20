"""Prompt construction. The creative contract with the image generator.

Two rules the old project learned the hard way and this one keeps:

- `character-left` / `character-right` name the character's own sides;
  `screen-left` / `screen-right` name position in the image. A prop is locked to
  one of the character's hands and never changes hands because the view did.
- Art is never mirrored to face the other way. Every view is drawn.
"""

from __future__ import annotations

import re

from .spec import LIST_FIELDS, TEXT_FIELDS, CharacterSpec
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


#: The brief's own fields, sent to the bible writer as the facts it must keep.
#: `prop` and `personality` stay out because BIBLE_SYSTEM forbids both: props are
#: attached per set, and personality is not visible. Derived from the spec's own
#: field lists so a field added there cannot be silently dropped here.
BIBLE_FIELDS = tuple(
    name for name in (*TEXT_FIELDS, *LIST_FIELDS) if name not in ("prop", "personality")
)


def bible_request(spec: CharacterSpec) -> str:
    """The brief as the bible writer sees it: the pitch, then the stated facts.

    A brief that fills none of the fields sends exactly what it always sent. One
    that fills them stops relying on the writer to infer a costume it was never
    told: the boots are in `outfit`, so the boots reach the paragraph.
    """
    facts = [
        f"{name.replace('_', ' ').capitalize()}: "
        + (value if isinstance(value, str) else "; ".join(value))
        for name in BIBLE_FIELDS
        if (value := getattr(spec, name))
    ]
    return "\n".join([
        f"Character: {spec.name}",
        f"Height: {spec.height}",
        f"Designer's brief: {spec.description}",
        *facts,
    ])


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
    "The pose cue and the pose card describe limb positions only. Neither one shows what the "
    "character is holding: the card has no props, and the shape at the end of an arm is a hand, "
    "not a statement that the hand is empty. "
    "Anything the description above says the character holds is still in that same hand in this "
    "frame, drawn in full and clearly readable. Never replace a held prop with a bare fist or an "
    "open hand, in any frame, whatever the arm is doing."
)

#: What the figure on a pose card is drawn with. Said once, because both the
#: single-frame and the sheet clause have to describe the same picture, and a
#: prompt that describes the reference wrongly is worse than one that says
#: nothing about it.
POSE_CARD = (
    "On it the character's own LEFT arm and leg are teal green and their own RIGHT arm and leg "
    "pale violet, so you can tell which side a limb belongs to and which one passes in front "
    "where they cross. The two orange outlines are the rib cage and the pelvis: copy how far "
    "each is turned, including where they disagree with each other, because that counter-turn "
    "is the movement. Hands and feet are drawn as their own shapes — a foot is hinged at the "
    "ball, so copy which way it points and whether the heel is down or lifted. The floor is a "
    "dashed line: a foot drawn above it is off the floor and is drawn off the floor. These "
    "colours are a code, not costume, and the figure carries no identity or style — take none "
    "of its look, and draw nothing teal, violet, orange or grey because of it."
)

POSE_REFERENCE = (
    "The last reference image is a figure holding this exact pose, on a dark card. Copy the "
    "pose from it: limb angles, which knee is bent, how wide the feet are set, how the weight "
    f"sits, and which way the head turns. {POSE_CARD}"
)

#: Whichever produced the cue — a traced skeleton or a written pose description — the key art
#: is still sitting in the reference list purely for identity, and its own stance has a way of
#: winning anyway. Said once, unconditionally, so a written set with no skeleton to point at is
#: covered too, not just the sets `POSE_REFERENCE` fires for.
STANCE_REFERENCE = (
    "The key art shows the character in one stance only; that stance is not locked. Stance and "
    "foot spacing for this frame come from the pose described above, not from the key art, even "
    "where that means standing narrower or wider than the key art does."
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
    photographic: bool = False,
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
            (PHOTO_REFERENCE if photographic else POSE_REFERENCE) if pose_reference else "",
            STANCE_REFERENCE,
            # A set with nothing to hold has no prop to keep, and arguing for one invites it in.
            PROP_CONTINUITY if props else "",
            SIDE_LANGUAGE,
            "Match the reference images for identity, costume, colour, proportion and prop hand "
            "exactly; only the pose changes. Do not "
            "mirror the figure and do not move the prop to the other hand.",
        ] if part
    )


#: The single-frame counterpart to `PHOTO_SHEET_REFERENCE`.
PHOTO_REFERENCE = (
    "The last reference image is a photograph of a real performer holding this exact pose. Copy "
    "the pose from it: limb angles, which knee is bent, how high a foot leaves the floor, how "
    "wide the feet are set, where the weight sits, how far each arm reaches and in which "
    "direction, and which way the head and shoulders turn. Draw the movement at the size the "
    "photograph has it, not a smaller, more cautious version of it. Take nothing else from that "
    "photograph: not the performer's face, hair, body, build, clothing, footwear, or the room "
    "behind them. The character keeps their own costume from the key art, including on a foot "
    "that is off the floor."
)

POSE_SHEET_REFERENCE = (
    "The last reference image shows every pose in this sequence as a figure on a dark card, laid "
    "out in the same order and the same rows as the figures you draw. Copy each pose from its "
    "own card: limb angles, which knee is bent, how wide the feet are set, how the weight sits, "
    f"and which way the head turns. {POSE_CARD}"
)

#: Said of a sheet of photographs instead. A photograph carries a whole person,
#: so the boundary has to be drawn explicitly: everything about the pose, and
#: nothing about who is holding it. Saying only what NOT to take is not enough —
#: the dancer's trainers still arrived on a lifted foot while the prompt was
#: already forbidding her clothing, and it took naming the character's own
#: footwear positively to stop it.
PHOTO_SHEET_REFERENCE = (
    "The last reference image is a strip of photographs of a real performer, one per figure, laid "
    "out in the same order and the same rows as the figures you draw. Copy each photograph's pose "
    "onto the character: limb angles, which knee is bent, how high a foot leaves the floor, how "
    "wide the feet are set, where the weight sits, how far each arm reaches and in which "
    "direction, and which way the head and shoulders turn. Draw the movement at the size the "
    "photograph has it, not a smaller, more cautious version of it. Take nothing else from those "
    "photographs: not the performer's face, hair, body, build, clothing, footwear, or the room "
    "behind them. The character keeps their own costume from the key art in every figure, "
    "including on a foot that is off the floor."
)

#: Sheet-mode counterpart to `STANCE_REFERENCE`, said of every figure at once.
STANCE_SHEET_REFERENCE = (
    "The key art shows the character in one stance only; that stance is not locked. Stance and "
    "foot spacing for each figure come from its own pose cue, not from the key art, even where "
    "that means one figure stands narrower or wider than another or than the key art does."
)


#: What a photograph of somebody else will argue with. Footwear first: that is
#: the one that was measured, where a lifted foot came back in the dancer's
#: white trainer instead of the character's boot. Hair is the other thing a
#: reference carries in full view and cannot help asserting.
#: Footwear is its own group and is taken first: it is the measured failure, and
#: a bible that describes hair at length would otherwise fill the anchor with it
#: and leave the feet undefended — which is exactly what a first draft did here.
FOOTWEAR_WORDS = ("boot", "shoe", "sneaker", "trainer", "sandal", "footwear", "heel")
HAIR_WORDS = ("hair", "braid", "ponytail", "quiff", "curls")


def costume_anchor(bible: str) -> str:
    """The sentences of the bible a photographic pose reference will contradict.

    Not the whole bible: carrying that into a sheet prompt is what compressed
    the movement in the first place. Only the parts a photograph of a different
    body will overwrite, said positively — the prompt was already forbidding the
    performer's clothing when the trainers arrived, so naming what the character
    wears is doing the work that the prohibition could not.
    """
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", bible) if part.strip()]

    def first(words: tuple[str, ...]) -> list[str]:
        return next(([s] for s in sentences if any(w in s.lower() for w in words)), [])

    # ponytail: one sentence each, footwear before hair; a bible that buries
    # either deeper wants a spec field rather than more parsing here.
    kept = first(FOOTWEAR_WORDS) + first(HAIR_WORDS)
    if not kept:
        return ""
    return " ".join(kept) + (
        " That is what the character wears in every figure, whatever the pose. A foot lifted off "
        "the floor still wears the character's own footwear, drawn in full: never the footwear, "
        "hair or clothing of anyone in the reference photographs, and never bare."
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
    photographic: bool = False,
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
    columns = min(per_row, len(cues))
    # Asking for figures with gaps between them is not enough: a flung arm still
    # crosses into the next figure's space, and cutting a box around one then
    # brings the neighbour's hand along. Naming cells and saying nothing may
    # leave its own cell took overlapping boxes from three per sheet to none.
    layout = (
        f"Lay the figures out as a grid of {columns} {'column' if columns == 1 else 'columns'} and "
        f"{rows} {'row' if rows == 1 else 'rows'}, in reading order, left to right and then the "
        f"next row down. Divide the canvas into {len(cues)} equal rectangular cells, {columns} across and "
        f"{rows} down, and draw exactly one figure inside each cell. Every figure stays entirely "
        "within its own cell: no limb, hand, foot or strand of hair crosses into a neighbouring "
        "cell or touches the canvas edge. Within its cell each figure is drawn as large as it can "
        "be while still fitting whole."
    )
    # A photograph already says everything the bible, the director's note and the
    # per-frame cues approximate, and saying it again in prose costs movement:
    # the same pose cards drew 0.43-0.70 of the traced amplitude carrying all
    # three, and 0.75-1.21 without them. What a photograph cannot say is which
    # costume the character keeps while copying somebody else's body, so that one
    # part of the bible comes back as `costume_anchor` and nothing else does.
    sameness = (
        "Every figure is the same character at the same size and the same distance from the "
        "viewer; only the pose changes from one to the next."
    )
    if photographic:
        identity, standing, figures = costume_anchor(bible), "", sameness
    else:
        identity, standing, figures = bible, set_note, f"{sameness[:-1]}:\n{poses}"
    return "\n\n".join(
        part for part in [
            f"Draw {spec.name} {len(cues)} times in one image, as the consecutive frames of one "
            "animation.",
            identity,
            props,
            standing,
            view_clause,
            f"Draw this on a landscape canvas. {layout} No numbers, labels, frame lines or grid "
            "lines: only the figures on the backdrop.",
            figures,
            STYLE,
            BACKDROP,
            detail_clause(detail_level),
            DETAIL_REFERENCE.format(level=detail_level) if detail_reference else "",
            (PHOTO_SHEET_REFERENCE if photographic else POSE_SHEET_REFERENCE)
            if pose_reference
            else "",
            STANCE_SHEET_REFERENCE,
            # A set with nothing to hold has no prop to keep, and arguing for one invites it in.
            PROP_CONTINUITY if props else "",
            SIDE_LANGUAGE,
            "Match the reference images for identity, costume, colour, proportion and prop hand "
            "exactly. Do not mirror any "
            "figure and do not move the prop to the other hand.",
        ] if part
    )
