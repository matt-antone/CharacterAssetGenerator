"""The rendering contract: what the art looks like, independent of what it shows.

Ported from KaraokeParty-Graphics' `rendering_output_contract` and its ten-step
detail scale, which produced the look that worked. Kept in its own module so the
art direction can be read and changed without going through prompt assembly.
"""

from __future__ import annotations

from pathlib import Path

#: How the art looks. True of every render, static or animated.
#:
#: The 16-bit original, ported from KaraokeParty-Graphics' rendering contract,
#: is kept below as SIXTEEN_BIT. Everything load-bearing is the same in both:
#: the pixel grid, the hard edges, no gradients, the two-pixel black outline.
#: Only the palette depth changes. Do not soften either into "cartoon" or
#: "cel-shaded" — that gives a modern vector sprite, not an arcade one.
STYLE = (
    "Mid-1990s 32-bit arcade sprite art, the Capcom Street Fighter Alpha and Marvel vs Capcom "
    "generation rather than Street Fighter 2. A visible pixel grid and hard nearest-neighbour "
    "edges, with no antialiasing, no gradients, no airbrushing and no soft glow — but a far "
    "richer palette than the 16-bit era: four to six discrete banded tones per material, with "
    "deliberate rim light along the lit edge and reflected colour inside the shadows. A solid "
    "black silhouette outline about two pixels wide around the whole figure. Full body, head to "
    "feet, nothing cropped."
)

#: The 16-bit look this started from. Two or three flat tones per material.
SIXTEEN_BIT = (
    "Capcom Street Fighter 2 style arcade pixel art. A visible pixel grid, hard nearest-neighbour "
    "edges, and no antialiasing, gradients, airbrushing or soft glow. Two or three discrete tones "
    "per material and nothing in between. A solid black silhouette outline about two pixels wide "
    "around the whole figure. Full body, head to feet, nothing cropped."
)

#: The silhouette outline colour the style mandates. The backdrop must stay
#: well clear of it: where backdrop and outline are the same colour, nothing can
#: tell them apart, and masking eats the outline.
OUTLINE = (0, 0, 0)

#: Verbatim from KaraokeParty-Graphics' magenta-background reference, which
#: says to include it in every cutout prompt. Paraphrasing it lost: the arcade
#: style pulls the character onto a dark stage, and only naming the actual
#: failure modes — scene, vignette, cast shadow — holds it off.
BACKDROP = (
    "Magenta or magenta-like background must fill 100% of all non-character space, including "
    "gaps between hair, limbs, and props, up to every canvas boundary. No borders, frames, "
    "differently colored margins, or rounded corners. Prefer #FF00FF; nearby magenta values "
    "and generated variation are accepted. No checkerboard, scene, backdrop texture, vignette, "
    "or cast shadow."
)



#: Put in every generation prompt, word for word, exactly as the old profile
#: required. It is the one clause that keeps a prop in the right hand.
SIDE_LANGUAGE = (
    "Mandatory character-side convention: use character-left and character-right for the "
    "character's own sides. Use screen-left and screen-right only for image position. Never infer "
    "a character side from a screen side. Never use stage-left or stage-right for character "
    "anatomy."
)

#: Rendering density, on the ten-step scale from the old project. It controls
#: how much detail is drawn — never identity, pose, costume, palette or scale.
DETAIL_LEVELS = {
    1: "Minimal silhouette: flat shapes, large clusters, minimal facial and material cues.",
    2: "Very sparse: simple readable face and costume cues, mostly flat shapes, few interior marks.",
    3: "Low detail: clear face, garment, hair and prop construction with limited shading.",
    4: "Modest detail: purposeful facial and garment construction with simple two-to-three-tone shading.",
    5: "Medium detail: readable facial anatomy, hair groups, seams and material cues with controlled arcade shading.",
    6: "Moderately rich: nuanced but disciplined arcade shading with clear face, hair, garment and material structure.",
    7: "High arcade detail: strong facial specificity, clustered hair strands, tailored garments and controlled highlights.",
    8: "Illustrative pixel detail: sophisticated form and material rendering, expressive face, crisp pixel-built silhouette.",
    9: "Near-realistic illustration: very high illustrative detail, refined anatomy, tailoring, texture and lighting.",
    10: "Illustrative realism: maximum anatomy, texture, lighting and dimensional material rendering, still readable as arcade construction.",
}

DEFAULT_DETAIL_LEVEL = 10

#: The one approved detail sample that survived the old repo. Levels without a
#: frame of their own still get their written description.
DETAIL_FRAMES = {4: Path(__file__).parent / "references" / "detail-level-04.png"}


def detail_frame(level: int) -> Path | None:
    frame = DETAIL_FRAMES.get(level)
    return frame if frame and frame.exists() else None

#: Ships with the repo: the approved level-4 frame from the old project.
DETAIL_REFERENCE = (
    "One reference image is a detail-level sample at detail level {level}. Match two things from "
    "it and only two: its rendering density and pixel-art construction, and its flat magenta "
    "backdrop with the figure alone on it. Take nothing else — not identity, face, costume, "
    "palette, pose, anatomy, scale or prop hand."
)


def detail_clause(level: int) -> str:
    if level not in DETAIL_LEVELS:
        raise KeyError(f"detail level must be 1-10, not {level!r}")
    return f"Render at detail level {level}. {DETAIL_LEVELS[level]}"

#: How the character stands. Only ever true of the projection sheet — asking an
#: animation frame to stand upright overrides its own pose cue and skeleton.
STANDING = "Feet flat on an implied floor, the whole figure standing upright in frame."
