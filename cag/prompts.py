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
    SCENE_EMPTY,
    SCENE_SAFE_AREA,
    SCENE_STYLE,
    SCENE_VIEWPOINT,
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

#: The frame view the key art is already drawn at: a set traced at this facing
#: needs no reference of its own.
KEY_FRAME_VIEW = "3/4"

#: Stance for a render the character performs from. Deliberately defers to the
#: description: inventing a stance here overrode characters whose brief
#: specified their own.
READY_STANCE = (
    "Use the character's own ready stance exactly as described above. If the description does "
    "not give one, a relaxed stance with the arms readable and clear of the torso."
)

#: READY_STANCE only asks for readable arms when the brief gives no stance, so a
#: brief that gives one lost the guarantee: Belter's "free hand compact and
#: relaxed" came back on Qwen-Image-2.1 with that hand tucked behind her back.
#: Every frame is drawn from the key art, and a hand it hides is one no frame
#: has a reference for. Said on every key art, whatever the brief's stance.
#: A rule alone lost two rolls in three to "compact": it is a concrete default
#: position now, placed straight after the description, and a stance that does
#: place the free hand (a hand on the hip, a raised fist) still wins.
KEY_ARMS = (
    "Both arms and both hands are in full view, with every finger of each hand visible. If the "
    "stance below places the free hand, draw it there, in front of the body or out to the side. "
    "Otherwise the free arm hangs straight down at the character's side, clear of the torso, the "
    "open hand beside the thigh and below the hair. Never behind the back, in a pocket, or hidden "
    "by hair, a sleeve or the other arm."
)

#: The key art is the identity every later render copies, so anything it drops
#: is dropped for good. Two Qwen-Image-2.1 rolls of Belter both left out the
#: crimson waist accent, drew her near-black boots mid-brown, and gave her the
#: long upright fashion figure her brief avoids. The avoid list stays out of the
#: prompt (see `assemble_bible`), so this says the right thing positively.
KEY_COMPLETE = (
    "This drawing is the reference every other render of the character is copied from. Every "
    "garment, accessory and colour named in the description above appears in it, each where and "
    "in the colour the description gives: none left out, merged into another piece, or drawn "
    "lighter or darker than stated. The body's proportions follow the build and height described "
    "above rather than a default fashion-illustration figure."
)

#: Stance for a projection view, whose whole job is being comparable to its
#: siblings. Never for a set reference: the animation frames are drawn from
#: that picture, and an even-weight arms-down base came back through every
#: frame — Belter's dance swung its working arm a quarter as far as the motion
#: source did.
PROJECTION_STANCE = (
    "The same neutral standing stance in every projection view: weight even on both "
    "feet, arms hanging clear of the torso, so the views can be compared."
)

DIRECTOR_SYSTEM = """You are the motion director for one animation set.

You are given a character description and the performance arc of a motion source. Write two to \
four sentences of standing instruction that every frame of this set must obey: what the character \
holds and in which of their own hands, how their costume and hair behave while they move, and \
what must not change between frames.

Do not restate the arc, describe individual frames, give timing, or mention the source video. \
Use "character-left" and "character-right" for the character's own sides. Output the \
instruction and nothing else."""


#: Said when a set carries no props. DIRECTOR_SYSTEM asks the director what the
#: character holds, so silence is not an answer: left to guess, it answered from
#: whatever the identity text happened to mention, which is how a microphone
#: ended up standing instruction for `dance`, `ko` and `victory` on four
#: characters that are drawn empty-handed in all three.
EMPTY_HANDS = (
    "The character holds nothing here. Both of their hands are empty and stay empty in every "
    "figure: no microphone, no stand, no cable and no object of any kind."
)


def director_request(
    bible: str, arc: str, fps: int, frame_count: int, view: str, props: str = ""
) -> str:
    return (
        f"Character: {bible}\n\n"
        f"{props or EMPTY_HANDS}\n\n"
        f"Set: {frame_count} frames at {fps} fps, drawn in the {view} view.\n\n"
        f"Performance arc: {arc}"
    )


#: The brief's own fields, sent to the bible writer as the facts it must keep.
#: `prop`, `personality` and `location` stay out because BIBLE_SYSTEM forbids all
#: three: props are attached per set, personality is not visible, and background
#: is the one thing the bible must never mention — it is quoted into every
#: character render, where a described room would get drawn behind the figure and
#: break the cutout. Derived from the spec's own field lists so a field added
#: there cannot be silently dropped here.
BIBLE_FIELDS = tuple(
    name
    for name in (*TEXT_FIELDS, *LIST_FIELDS)
    if name not in ("prop", "personality", "location")
)


def assemble_bible(spec: CharacterSpec) -> str:
    """The bible, written from the brief's own fields instead of by a model.

    A brief that states its build, face, hair, outfit and palette has already
    said everything the bible is for, so there is nothing left to infer and no
    reason to spend a model call inferring it. That call was the only reason the
    bible had to be frozen to a file: a second one returned different words.

    `prop` is not here and cannot be. Every bible a model wrote for this roster
    named the microphone despite BIBLE_SYSTEM forbidding it, and the sentence
    was then quoted into the frames of `dance`, `ko` and `victory` — sets the
    brief draws empty-handed, where nothing else in the prompt contradicted it.
    Leaving the field out closes that structurally rather than by asking a model
    to behave. `avoid` is out too: it is a list of things not to draw, and an
    image generator handed one tends to draw them.

    Empty for a brief that fills none of the fields; those still ask a model.
    """
    said = [spec.description, spec.build, spec.face, spec.hair, spec.outfit]
    if not any(said[1:]):
        return ""
    body = " ".join(part for part in said if part)
    if spec.palette:
        body += " Colours, base/shadow/highlight per material: " + "; ".join(spec.palette) + "."
    return body


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


def location_prompt(spec: CharacterSpec) -> str:
    """Prompt for the character's location: the place they perform, without them.

    The one prompt in the pipeline that does not ask for a magenta backdrop,
    because nothing is cut out of it — a location is delivered whole and the
    character cell is composited over it.

    The bible is not quoted here and neither is the palette. Both describe the
    costume, and a room painted in a character's own colours is a room they
    vanish into.
    """
    if not spec.location:
        raise KeyError(f"{spec.name} has no location in their brief")
    return "\n\n".join([
        f"Draw the place {spec.name} performs: a background plate, with no character in it.",
        spec.location,
        SCENE_EMPTY,
        "Draw this on a 16:9 landscape canvas, filled edge to edge.",
        SCENE_SAFE_AREA,
        SCENE_STYLE,
        SCENE_VIEWPOINT,
        detail_clause(spec.detail_level),
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
    """Prompt for one static view of the character.

    `view` names a static view or a MotionArtist frame facing, so a set can be
    given a reference drawn at its own facing rather than the key art's.
    """
    line = VIEWS.get(view) or FRAME_VIEWS.get(view)
    if line is None:
        raise KeyError(f"unknown view {view!r}")
    # The bible leaves `performance_style` out (it would pin every frame to one
    # pose), so READY_STANCE's "described above" pointed at nothing and the key
    # art fell back to arms hanging: seven of eight halloween keys lost their
    # signature gesture. The key art is the one render that should quote it.
    ready = " ".join(filter(None, [spec.performance_style, READY_STANCE]))
    stance = pose or (ready if view == KEY_VIEW else PROJECTION_STANCE)
    return "\n\n".join(
        part for part in [
            f"Draw {spec.name}.",
            bible,
            KEY_ARMS if view == KEY_VIEW and not pose else "",
            props,
            line,
            stance,
            KEY_COMPLETE if view == KEY_VIEW else "",
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
            PHOTO_REFERENCE if pose_reference else "",
            STANCE_REFERENCE,
            # A set with nothing to hold has no prop to keep, and arguing for one invites it in.
            PROP_CONTINUITY if props else "",
            SIDE_LANGUAGE,
            "Match the reference images for identity, costume, colour, proportion"
            + (" and prop hand" if props else "")
            + " exactly; only the pose changes. Do not mirror the figure"
            + (" and do not move the prop to the other hand." if props else "."),
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

#: What the sheet before this one is doing in the reference list. A set longer
#: than one sheet is drawn as several independent renders off the same key art,
#: and independent is what they came back as: the second sheet is where the
#: costume wanders, because nothing in it had seen the first. Its last figure
#: comes along as the character actually drawn a moment earlier — continuity of
#: costume and colour, never of pose, which is why it says so twice.
CARRY_REFERENCE = (
    "One reference image is a single figure on a magenta backdrop: this character as already "
    "drawn in an earlier frame of this same production. It is there for continuity, not for a "
    "pose. Match its costume, colour, hair, footwear and proportion, and take nothing of its "
    "pose, stance, limb positions or facing from it."
)

#: Sheet-mode counterpart to `STANCE_REFERENCE`, said of every figure at once.
#: It names the pose reference rather than the cue: a photographic sheet carries
#: no cues, and pointing at one that is not there left the key art's wide stance
#: as the only stance in the prompt.
STANCE_SHEET_REFERENCE = (
    "The key art shows the character in one stance only; that stance is not locked. Stance and "
    "foot spacing for each figure come from its own pose reference, not from the key art, even "
    "where that means one figure stands narrower or wider than another or than the key art does."
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


def costume_anchor(bible: str, outfit: str = "", hair: str = "") -> str:
    """What the character wears, said where a photographic pose reference will
    contradict it.

    Not the whole bible: carrying that into a sheet prompt is what compressed
    the movement in the first place. Only the parts a photograph of a different
    body will overwrite, said positively — the prompt was already forbidding the
    performer's clothing when the trainers arrived, so naming what the character
    wears is doing the work that the prohibition could not.

    The brief's own `outfit` and `hair` fields come first. Two bible sentences,
    footwear and hair, left the rest of the costume out: Diva's floor-length
    gown never reached her dance prompt, and against a photograph of a dancer in
    trousers she stepped a leg out through a slit the gown does not have. The
    bible's footwear sentence still comes along when the outfit names none.
    """
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", bible) if part.strip()]

    def first(words: tuple[str, ...]) -> list[str]:
        return next(([s] for s in sentences if any(w in s.lower() for w in words)), [])

    if outfit:
        shod = any(w in outfit.lower() for w in FOOTWEAR_WORDS)
        kept = [outfit] + ([] if shod else first(FOOTWEAR_WORDS)) + ([hair] if hair else first(HAIR_WORDS))
    else:
        kept = first(FOOTWEAR_WORDS) + first(HAIR_WORDS)
    if not kept:
        return ""
    return " ".join(kept) + (
        " That is what the character wears in every figure, whatever the pose. A foot lifted off "
        "the floor still wears the character's own footwear, drawn in full: never the footwear, "
        "hair or clothing of anyone in the reference photographs, and never bare. A garment that "
        "covers the legs keeps covering them however the photographed legs move: no slit, split "
        "or bare leg the outfit does not have."
    )


def frame_sheet_prompt(
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
    carry_reference: bool = False,
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
        identity, standing, figures = costume_anchor(bible, spec.outfit, spec.hair), "", sameness
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
            # More than four across is a set drawn whole (Comfy only), and eight
            # figures across a plain landscape canvas left each one a slot too
            # narrow for Belter's hair and mic arm: they ran together and would
            # not slice. "Wide" is 21:9 there; see `cag.comfy.CANVASES`.
            f"Draw this on a {'wide ' if columns > 4 else ''}landscape canvas. {layout} "
            "No numbers, labels, frame lines or grid "
            "lines: only the figures on the backdrop.",
            figures,
            STYLE,
            BACKDROP,
            detail_clause(detail_level),
            DETAIL_REFERENCE.format(level=detail_level) if detail_reference else "",
            PHOTO_SHEET_REFERENCE if pose_reference else "",
            CARRY_REFERENCE if carry_reference else "",
            STANCE_SHEET_REFERENCE,
            # A set with nothing to hold has no prop to keep, and arguing for one invites it in.
            PROP_CONTINUITY if props else "",
            SIDE_LANGUAGE,
            "Match the reference images for identity, costume, colour, proportion"
            + (" and prop hand" if props else "")
            + " exactly. Do not mirror any figure"
            + (" and do not move the prop to the other hand." if props else "."),
        ] if part
    )


#: The prompt for re-posing a set's approved reference into one traced frame's
#: pose (`cag.comfy.POSE_WORKFLOW`). The first half is AnyPose's own wording, the
#: phrasing its LoRAs were trained against. The rest is what a photograph of a
#: real performer carries that must not come across: her clothes, her shoes, the
#: street she danced in. It names no character, because the reference image
#: already is one, and words only compete with it.
POSE_EDIT = (
    "Make the person in image 1 do the exact same pose of the person in image 2. Changing the "
    "style and background of the image of the person in image 1 is undesirable, so don't do it. "
    "The new pose should be pixel accurate to the pose we are trying to copy. The position of the "
    "arms and head and legs should be the same as the pose we are trying to copy. "
    "Keep everything else from image 1 exactly: its pixel-art style and black outline, the "
    "character's face, hair, body, clothing and footwear, anything held in the hands, and the "
    "flat pure magenta background. Take nothing from image 2 except the pose: not the performer's "
    "clothes, hair, face or shoes, not the scenery, not the camera angle. Draw the whole figure, "
    "head to feet, with nothing cropped."
)


# The video path's prompts (see `cag.video`). Each is the string a measured
# render sent, give or take the bible standing in for a hand-written costume
# sentence. One roll per character is all that is behind them, so a change is
# rolled twice before it is kept.

#: The SCAIL-2 job's prompt: one video of the set reference following the drive
#: video. `{action}` comes from `SCAIL_ACTIONS`, `{identity}` is the bible.
SCAIL_PROMPT = (
    "A 32-bit arcade pixel-art sprite of the character from the reference image {action}: "
    "{identity} The same face, hair, costume and pixel-art style with a black outline in "
    "every frame. Full body. Flat solid magenta background. Static camera."
)

#: What the character is doing in the footage, per set. Only dances have been
#: drawn this way; any other set says only that it moves.
SCAIL_ACTIONS = {"dance": "dancing in place"}


def scail_prompt(set_name: str, bible: str) -> str:
    """The SCAIL-2 prompt for one set, quoting the bible as its identity."""
    identity = bible.strip()
    if identity and identity[-1] not in ".!?":
        identity += "."
    return SCAIL_PROMPT.format(
        action=SCAIL_ACTIONS.get(set_name, "moving in place"), identity=identity
    )


#: The restyle's style clause, short on purpose: it rides beside `<image2>`,
#: which shows the style, rather than standing in for it the way `STYLE` does.
RESTYLE_STYLE = (
    "mid-1990s 32-bit arcade pixel-art sprite, visible pixel grid, banded shading, solid black "
    "outline, flat pure magenta background"
)

#: One restyle: a SCAIL frame redrawn in the set reference's art. The SCAIL frame
#: is `<image1>` because the edit's latent takes its size from the first image.
RESTYLE = (
    "Redraw <image1> in exactly the art style of <image2>: " + RESTYLE_STYLE + ". Keep "
    "everything in <image1>: the pose and position of every limb, both hands, the face and "
    "expression, the figure's size and place in the frame. Match <image2>'s colours and every "
    "costume detail."
)

#: What the mask pass tracks: SAM3 reads it as a text query, not a prompt.
MASK_PASS_PROMPT = "human"
