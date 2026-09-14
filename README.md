# CharacterAssetGenerator

Turns a short character brief into game-ready sprite assets: a projection sheet
of static views and a masked, looping animation set drawn from a
[MotionArtist](https://github.com/matt-antone/MotionArtist) motion sheet.

```bash
cag build specs/velvet-lou.json --motion ../MotionArtist/work/sample/motion.json --set dance
```

That writes `outputs/velvet-lou/`: four projection cells, a sprite sheet, a
looping GIF proof, and a gallery page.

## How it runs

Two LangGraph runs. The static one goes first, because everything downstream
references what it locks:

| Step | What it does |
| --- | --- |
| `bible` | One model call turns the brief into a locked visual description. Every later prompt quotes it verbatim, so identity cannot drift. |
| `key_art` | Draws the front-left three-quarter reference on a magenta backdrop. |
| `scale` | Measures the character's crown-to-heel span off the key art. Read once, reused forever. |
| `projection` | Draws front, back and profile, each with the key art attached as a reference image. |
| `mask` | Cuts every view out and registers it into the cell. |

Then the animation set, drawn the way a studio draws one:

| Role | What it does |
| --- | --- |
| `poses` | Draws each frame's pose as a stick figure from the sheet's landmarks. A generator flattens a written pose back towards neutral; it cannot argue with a picture. |
| `direct` | The motion director binds the motion source to this character: prop hand, how the costume moves, what must not change. |
| `keyframe` | The keyframer draws the frames the sheet marks `key` and `pilot` — the extremes and the fastest transitions. |
| `tween` | The tweener fills each in-between from its two locked neighbours, wrapping across the loop seam. Drawn, never interpolated. |
| `mask` | Every frame cut out and registered at the key art's scale. |

## Constraints this was built under

- **No OpenAI API.** Both the model calls and the image generation go through
  the local `codex` CLI on a ChatGPT subscription. `cag.chat_codex.ChatCodex` is
  a LangChain `BaseChatModel` that shells out to `codex exec`.
- **macOS only.** The cutout is Apple's Vision framework
  (`VNGenerateForegroundInstanceMaskRequest`) through pyobjc.

## Why magenta, and why Vision

Sources render on a full-canvas magenta backdrop and stay that way until one
masking pass at the end. Vision segments the foreground subject rather than
keying a colour, so a magenta-family costume cannot be keyed away with the
background. Two small passes clean up after it: backdrop trapped inside the
subject (the gap inside a hand holding a microphone) is cleared, and magenta
that bled into the antialiased outline is neutralised.

## Geometry

Every cell is `480x560`. The full height represents 7'0", so `80 px/ft`, and the
supporting heel sits on row `550`. Scale comes from the key art and is applied
unchanged to every frame, so a crouch renders shorter instead of being stretched
back to standing height.

`character-left` and `character-right` name the character's own sides.
`screen-left` and `screen-right` name position in the image. A prop is locked to
one of the character's hands and never changes hands because the view changed.
Art is never mirrored.

## A brief

```json
{
  "name": "Velvet Lou",
  "height": "5' 9\"",
  "description": "A cheerful lounge performer in a deep green velvet jacket ...",
  "animations": { "dance": "Relaxed club two-step that returns to frame 0." }
}
```

Name, height, description, prose intent per animation. Frame counts, fps,
paths, geometry and QA belong to the pipeline, not to the designer's file.

## Install

```bash
uv sync
uv run pytest
```

## What this deliberately is not

The predecessor to this project drowned in approval ledgers, hash-bound
acceptance bundles, revision lifecycles, a production queue and nine agent
roles. None of that is here. There is also no mechanical tweening — no
cross-fade, no optical flow, no frame grid used as a creative source.
