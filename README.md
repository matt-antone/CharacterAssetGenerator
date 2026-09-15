# CharacterAssetGenerator

Turns a short character brief into game-ready sprite assets: a projection sheet
of static views, and a masked animation set per animation the brief names.

```bash
cag build specs/belter.json --jobs 4
```

The first run stops after the key art and waits for a human. Look at
`work/belter/source/key.png`, then either sign it off or delete it and build
again for a redraw:

```bash
cag approve specs/belter.json
```

Once approved, that writes `outputs/belter/`: four projection cells under `views/`, then a
sprite sheet and a GIF proof for each of the brief's seven sets, and a gallery
page tying them together. Sets render across `--jobs` lanes, and one that fails
does not take the others down with it.

A single set, against a traced [MotionArtist](https://github.com/matt-antone/MotionArtist)
sheet instead of a written one:

```bash
cag build specs/velvet-lou.json --set dance --motion ../MotionArtist/work/sample/motion.json
```

## How it runs

Two LangGraph runs. The static one goes first, because everything downstream
references what it locks:

| Step | What it does |
| --- | --- |
| `bible` | One model call turns the brief into a locked visual description. Every later prompt quotes it verbatim, so identity cannot drift. Saved to `work/<slug>/bible.txt`, so a later run can pin the same identity instead of writing a fresh description and accepting the drift. |
| `key_art` | Draws the front-left three-quarter reference on a magenta backdrop. |
| `approval` | Stops the run until `cag approve` signs off on that key art. Every other render quotes it, so a wrong one is a whole wrong character. The record in `work/<slug>/key-approved.txt` holds the art's digest, so a redraw revokes the approval rather than inheriting it. |
| `scale` | Measures the character's crown-to-heel span off the key art. Read once, reused forever. |
| `projection` | Draws front, back and profile, each with the key art attached as a reference image. |
| `mask` | Cuts every view out and registers it into the cell. |

Each set needs a motion sheet first. MotionArtist traces those from real
footage, but only some sets have footage — so for the rest a motion director
writes the frame plan from the brief's prose intent (`cag/motion_writer.py`),
in the same shape MotionArtist emits. The graph below cannot tell where a
sheet came from, with one exception noted in `poses`.

Then each animation set, drawn the way a studio draws one:

| Role | What it does |
| --- | --- |
| `poses` | Draws each frame's pose as a stick figure from the sheet's landmarks. A generator flattens a written pose back towards neutral; it cannot argue with a picture. Only traced sheets carry landmarks, so a written set gets no skeleton — the one real quality difference between the two. |
| `direct` | The motion director binds the motion source to this character: prop hand, how the costume moves, what must not change. |
| `sheet` | One render of the whole set: eight figures on one canvas, in reading order, with the pose skeletons composed into a matching grid. The generator cannot follow a measurement, but it keeps figures it can see side by side the same size unasked — so the prompt names no size at all. Figures are found afterwards by the backdrop between them, never by a fixed grid; a render with the wrong count is kept as `sheet-NN.rejected-*.png` and drawn again. |
| `mask` | Every frame cut out and registered at one scale per sheet: the median of what each pose reads, or the median figure height where a written set has no landmarks. |

That is two image calls per sixteen-frame set instead of sixteen, and identity
cannot drift between frames the generator drew in one go. `--per-frame` keeps
the older path — the keyframer draws `key` and `pilot` frames first, the tweener
fills each in-between from the frame before it — which pays for per-frame
outlines twice as thick with sixteen chances for the costume to wander.

Frame count, fps, view and playback come from the set's plan in `cag/sets.py`,
not from the brief: `dance` and `sing` loop, `flinch`, `guard`, `entrance`,
`victory` and `ko` run once. All seven are eight frames at four fps.

## Constraints this was built under

- **No OpenAI API.** Both the model calls and the image generation go through
  the local `codex` CLI on a ChatGPT subscription. `cag.chat_codex.ChatCodex` is
  a LangChain `BaseChatModel` that shells out to `codex exec`.
- **macOS only.** The cutout's fallback path is Apple's Vision framework
  (`VNGenerateForegroundInstanceMaskRequest`) through pyobjc.

## The look

Mid-1990s 32-bit arcade sprite art — the Street Fighter Alpha and Marvel vs
Capcom generation, at detail level 10. The 16-bit contract this started from is
kept as `SIXTEEN_BIT` in `cag/style.py`. Both hold the same load-bearing rules —
visible pixel grid, hard nearest-neighbour edges, no gradients, a two-pixel
black outline — and differ only in palette depth: four to six banded tones per
material with rim light and reflected colour in the shadows, against the 16-bit
era's two or three flat ones.

## Why magenta, and how the rim comes off

Sources render on a full-canvas magenta backdrop and stay that way until one
masking pass at the end. That pass is a chroma key, with Vision behind it for a
render the key cannot read — Vision segments the subject semantically, so it
survives a backdrop the character happens to share a colour with.

The key measures the backdrop by the **spread between its strong and weak
channels**, not by RGB distance. Darkening does not change that spread, so a
half-magenta rim pixel reads as the half backdrop it is, while a crimson jacket
scores 0.05 and stays whole.

Both paths then produce a hard in-or-out matte and **cut one pixel into the
outline**. Where the two-pixel black outline was antialiased against magenta,
the rim pixel is a genuine blend of the two, and no threshold can separate it
from costume in the same hue — "backdrop darkened by the outline" and "costume
in the backdrop's hue" are the same colour. So it is discarded by position
rather than judged by colour. `STYLE` mandates an outline about two pixels wide
precisely so there is one to spare; renders measure 4-6 source pixels of it.
**An outline thinner than two pixels would be eaten** — that is the dependency
this buys the clean edge with.

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

Name, height, description, prose intent per animation. Each animation key must
name a set `cag/sets.py` has a plan for. Frame counts, fps, paths, geometry and
QA belong to the pipeline, not to the designer's file.

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
