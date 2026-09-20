# CharacterAssetGenerator

Turns a short character brief into game-ready sprite assets: a projection sheet
of static views, and a masked animation set per animation the brief names.

```bash
uv run cag build specs/belter.json --jobs 4
```

The first run stops after the key art and waits for a human. Look at
`work/belter/source/key.png`, then either sign it off or delete it and build
again for a redraw:

```bash
uv run cag approve specs/belter.json
```

Once approved, that writes `outputs/belter/`: four projection cells under `views/`, then a
sprite sheet and a GIF proof for each of the brief's seven sets, a gallery page
tying them together, and a `manifest.json` for the front end: cell size, view
paths, and per set the frame count, columns, fps and file names. Play from the
manifest's fps — `cag edit` rewrites it when a save rebuilds the proof at
another rate, so it always matches the GIF beside it. Sets render across `--jobs` lanes, and one that fails
does not take the others down with it.

By default a set writes its own sheet from the brief's prose, and nothing below
is needed. A brief that wants a traced
[MotionArtist](https://github.com/matt-antone/MotionArtist) sheet instead names
one, and every build picks it up:

```json
"dance": { "intent": "Refined lounge sway loop...", "motion": "shuffle" }
```

`"motion": "auto"` has one chosen instead, spread across the roster so a cast
does not all dance the same and stable so a character keeps its dance between
runs. See what there is to name with:

```bash
uv run cag motions
```

It names the sheet, never the file. Sheets live in `motions/<name>/motion.json`
in this repo, so a brief is portable and a sheet cannot be cleaned away from
under the roster by the checkout that traced it. Adding one is a copy — the
`motion.json` alone, not the footage beside it:

```bash
cp ../MotionArtist/work/shuffle/motion.json motions/shuffle/motion.json
```

`--motion-root` reads them from somewhere else, and `--motion` points one run at
one file, overriding whatever the brief names:

```bash
uv run cag build specs/crooner.json --set dance --motion ../MotionArtist/work/shuffle/motion.json
```

## How it runs

Two LangGraph runs. The static one goes first, because everything downstream
references what it locks:

| Step | What it does |
| --- | --- |
| `bible` | One model call turns the brief into a locked visual description. Every later prompt quotes it verbatim, so identity cannot drift. Saved to `work/<slug>/bible.txt`, so a later run can pin the same identity instead of writing a fresh description and accepting the drift. |
| `key_art` | Draws the front-left three-quarter reference on a magenta backdrop. |
| `approval` | Stops the run until `uv run cag approve` signs off on that key art. Every other render quotes it, so a wrong one is a whole wrong character. The record in `work/<slug>/key-approved.txt` holds the art's digest, so a redraw revokes the approval rather than inheriting it. |
| `scale` | Measures the character's crown-to-heel span off the key art. Read once, reused forever. |
| `projection` | Draws front, back and profile, each with the key art attached as a reference image. |
| `mask` | Cuts every view out and registers it into the cell. |

Each set needs a motion sheet first. MotionArtist traces those from real
footage, but only some sets have footage — so for the rest a motion director
writes the frame plan from the brief's prose intent (`cag/motion_writer.py`),
in the same shape MotionArtist emits. Dance is the set worth tracing: a written
one reads as a character standing still with the arms moving, because prose
flattens a pose back towards neutral and only a traced sheet carries the
landmarks `poses` needs. The graph below cannot tell where a
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
`victory` and `ko` run once. All seven are sixteen frames: `dance` plays at six
fps, the rest at eight.

## Every set needs a hand pass

Expect to nudge frames after a build. The generator draws a set's figures
side by side, but never at fixed positions: each figure is found by the
backdrop around it, cut out and registered into its cell on its own. Scale holds
across a sheet; placement does not. So a figure can sit a few pixels left, right,
high or low of its neighbours, and a loop that should stand still will jitter or
slide. The mask cannot tell a drift from a deliberate step, so it leaves both
alone. Deciding which is which is a person's job.

`uv run cag edit` serves a small editor over the outputs, or one character's
folder:

```bash
uv run cag edit
```

Open `http://127.0.0.1:8765/` (`--port` to change it) and pick a
`<set>-sheet.png` from under that folder. Every character's sheets share the
same names, so the editor matches the one you open to its folder by contents:
that folder's brief (found in `specs/` by slug) sets the height guide, and Save
writes back there. A sheet that isn't under the folder won't save.

1. **Play** to watch the loop. Changing fps while it plays takes effect at once.
2. Pause, then pick the frame that jumps: click it in the filmstrip, drag the
   slider, or step with the `‹` `›` buttons beside it, `⌘` + `←` `→`, or `,` and `.`.
3. Move it by dragging it on the stage, nudging with the arrow keys (`Shift`
   for 10px), or typing an exact x / y offset. The faded figure is the previous
   frame; line up against it, the centre line and the floor line, which sits on
   the animation contact row. The dashed line above is the brief's height: where
   the crown lands on a standing frame.
   If a frame reads bigger or smaller than its neighbours, scale it with `[` and
   `]` (`Shift` for 10%) or type a %. It scales about the floor point on the
   centre line, so feet stay planted.
4. **Save sheet** overwrites the sheet in place and rebuilds `<set>-proof.gif`
   beside it at the fps in the box, then closes the sheet and shows the result
   on the stage. If the save fails, the sheet stays open with your edits. Frames
   you moved carry a dot in the filmstrip.

| Key | Does |
| --- | --- |
| `←` `→` `↑` `↓` | nudge 1px |
| `Shift` + arrows | nudge 10px |
| `[` / `]` | scale −1% / +1% |
| `Shift` + `[` `]` | scale ±10% |
| `⌘` + `←` / `→` | previous / next frame |
| `,` / `.` | previous / next frame |
| `Space` | play / pause |

Frames are clipped to their own cell, so a nudge can push art off the edge but
never into a neighbour. The editor changes pixels only: the fps box does not
write back to the set's plan, and **a later `uv run cag build` of that set overwrites
the sheet and loses the edits**. Opened straight from disk instead of through
`uv run cag edit`, Save downloads the sheet and leaves the proof alone.

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

Every cell is a `560x560` square, static or animated. The width is free (scale
is measured off the height alone), so a reach or a stride has room.

| | Static view | Animation frame |
| --- | --- | --- |
| Cell height represents | 9'0" | 8'6" |
| Scale | `62.2 px/ft` | `65.9 px/ft` |
| Supporting heel sits on row | `550` | `527`, six inches up |

The static cell draws the character at true scale with headroom for a hat or a
raised arm. An animation frame maps the same canvas to half a foot less world,
so a character renders slightly larger there than on the static sheet, and the
floor sits six inches off the bottom edge so a trailing foot or a shadow has
somewhere to go. The editor's floor line is that `527` row.

Static views take their scale from the key art. Animation frames take one scale
per sheet, read off that set's own poses (see `mask` above), because a frame is
neither standing nor drawn at the key art's size. Either way the scale is
applied unchanged to every figure, so a crouch renders shorter instead of being
stretched back to standing height.

`character-left` and `character-right` name the character's own sides.
`screen-left` and `screen-right` name position in the image. A prop is locked to
one of the character's hands and never changes hands because the view changed.
Art is never mirrored.

## A brief

```json
{
  "name": "Crooner",
  "height": "6' 2\"",
  "description": "An unflappable lounge singer in his fifties: lean, tall, narrow ...",
  "animations": { "dance": "Refined lounge sway loop that returns to frame 0." }
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

`uv sync` installs `cag` into the project's `.venv`, not onto your PATH, so a
bare `cag` answers `command not found`. That is why every command here starts
with `uv run`. After `source .venv/bin/activate`, plain `cag` works too.

## What this deliberately is not

The predecessor to this project drowned in approval ledgers, hash-bound
acceptance bundles, revision lifecycles, a production queue and nine agent
roles. None of that is here. There is also no mechanical tweening — no
cross-fade, no optical flow, no frame grid used as a creative source.
