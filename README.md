# CharacterAssetGenerator

Turns a short character brief into game-ready sprite assets: a projection sheet
of static views, and a masked animation set per animation the brief names.

```bash
uv run cag build specs/default/belter.json
```

The first run stops after the key art and waits for a human. Look at
`work/belter/source/key.png`, then either sign it off or delete it and build
again for a redraw:

```bash
uv run cag approve specs/default/belter.json
```

Once approved, that writes `outputs/default/belter/`: four projection cells under `views/`,
`location.png` if the brief names a location, then a
sprite sheet and a GIF proof for each of the brief's seven sets, a gallery page
tying them together, and a `manifest.json` for the front end: cell size, view
paths, and per set the frame count, columns, fps, playback and file names. Play
from the manifest's fps and playback — `cag edit` rewrites the fps when a save
rebuilds the proof at another rate, so it always matches the GIF beside it.
`playback` is `loop`, `pingpong` or `once`; a pingpong sheet holds its frames
once and is played down and back up. Sets render in order, each drawn from the key art and
from the last frame of the set before it, and one that fails does not take the others down with it.

By default a set writes its own sheet from the brief's prose, and nothing below
is needed. A brief that wants a traced
[MotionArtist](https://github.com/matt-antone/MotionArtist) sheet instead names
one, and every build picks it up:

```json
"dance": { "motion": "shuffle-1" }
```

A set names a sheet *instead of* prose, never beside it: one set is driven by
one prompt, and the traced sheet is a prompt — its arc and its per-frame cues
are what the keyframer reads. A brief that writes both is refused rather than
asked which of the two to follow.

`"motion": "auto"` has one chosen instead, spread across the roster so a cast
does not all dance the same and stable so a character keeps its dance between
runs. See what there is to name with:

```bash
uv run cag motions
```

A set with no traced sheet writes its own, and the brief is where that
character's loop rule goes:

```json
"dance": { "intent": "Refined lounge sway loop...", "playback": "pingpong" }
```

`playback` is `loop`, `pingpong` or `once`. It is exclusive with `motion`: a
traced sheet already carries how it plays, so a brief that sets both is refused
rather than asked which to believe. Left out, the set plan decides.

It names the sheet, never the file. Sheets live in `motions/<name>/` in this
repo, so a brief is portable and a sheet cannot be cleaned away from under the
roster by the checkout that traced it. Each one is a MotionArtist bundle, and
its `manifest.json` is what makes it one: the layout it declares, the header a
brief picks it by, and the files it contains, including which one holds the
motion. Adding a sheet is copying the bundle:

```bash
cp -R ../MotionArtist/work/shuffle motions/shuffle-1
```

`--motion-root` reads them from somewhere else, and `--motion` points one run at
one file, overriding whatever the brief names:

```bash
uv run cag build specs/default/crooner.json --set dance --motion ../MotionArtist/work/shuffle/motion.json
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
fps, the rest at eight. The plan only decides for a set nobody traced: a traced
bundle carries its own rate and playback and those win, because the frames were
cut off footage at that speed and another rate is the dance at the wrong tempo.
So how a set repeats is written down once, in whichever of the three places owns
it: the trace, or this character's brief, or the plan. Belter's dance can
pingpong while the crooner's loops without either of them touching `sets.py`.

## Who draws

Every render goes through one backend, picked per build:

```bash
uv run cag build specs/default/belter.json                        # codex, the default
uv run cag build specs/default/belter.json --draw-backend comfy   # Comfy Cloud
uv run cag build specs/default/belter.json --draw-backend local   # your own ComfyUI
```

`CAG_DRAW_BACKEND` sets the default. The rest of a build — briefs, prompts, the
key art gate, the mask, assembly, and the text — is the same whichever draws.

**codex** is an agent with an image tool: it is handed the prompt plus
instructions to save the file and to redraw until the backdrop is magenta.

**comfy** runs a ComfyUI workflow on [Comfy Cloud](https://cloud.comfy.org).
It needs `COMFY_CLOUD_API_KEY`, made at platform.comfy.org (Standard plan or
above). The workflow it runs is `comfy/workflow.json`, which ships with the
repo: Nano Banana Pro at 2K, taking up to fourteen reference images and
following a long prompt closely. Partner nodes like it bill to the same key.

The model is the workflow's choice, not cag's, so changing models means changing
that file, or pointing `--comfy-workflow` / `CAG_COMFY_WORKFLOW` at another one.
To make one, build it in Comfy, export it with **Workflow > Export (API)**, and
set these node inputs to placeholder strings. cag fills them in for each render:

| Placeholder | Filled with |
| --- | --- |
| `$prompt` | the render's prompt. Required |
| `$image1` .. `$imageN` | the reference images, in order, on `LoadImage` nodes. A render sends up to four: key art, the carried last frame, detail, the pose grid. The prompt calls the pose grid "the last reference image", so keep them in order |
| `$seed` | a fresh random seed, so a redraw is a new roll |
| `$width`, `$height` | the canvas in pixels: 1536x864 for a location, 1536x1024 for a frame sheet, 1024x1536 for a single figure |
| `$aspect` | the same canvas as a ratio, `16:9`, `3:2` or `2:3`, for models that take one |

A render with fewer references than there are slots drops the unused
`LoadImage` nodes. A batch node left holding one image passes it straight
through, and one left with none goes, so batched references shrink on their own.
A render with more references than slots fails before anything is uploaded.

Every reference is uploaded letterboxed onto a 1536 square, in its own border
colour. ComfyUI batches images as one tensor and crops each to the first one's
size to do it, which would cut the outer pose cards off a landscape pose grid
batched behind portrait key art.

The magenta check still applies: a render with scenery behind the character is
drawn again, the same as under codex.

**local** runs the same kind of workflow on a ComfyUI server of your own, at
`http://127.0.0.1:8188` unless `CAG_LOCAL_COMFY_URL` says otherwise. It needs no
key and bills nothing, but it can only run nodes and models installed there, so
no partner nodes: Nano Banana does not exist on a local server. Its default
workflow is `comfy/qwen-image-2.1.json` (`--comfy-workflow` or
`CAG_LOCAL_COMFY_WORKFLOW` names another). Qwen-Image-2.1's weights are under
the Qwen Research License, research and evaluation only, so a package meant for
anything else draws with an Apache-licensed workflow instead.

A traced set under either workflow backend is drawn a frame at a time by
`comfy/pose-edit.json` (`--comfy-pose-workflow` or `CAG_COMFY_POSE_WORKFLOW`):
Qwen-Image-Edit-2511 at fp8 with the AnyPose LoRAs and the Lightning 4-step
LoRA, which re-poses the set's reference into each traced photograph. Every file
it loads runs on a 16 GB card, so Comfy Cloud and a local server draw the same
dance. Each frame is then snapped back onto the reference's pixel grid and
palette, with a black outline, on exact magenta (`cag.snap`): the workflow keeps
the character, not the pixel art. The render as drawn is kept as `NN.raw.png`.

## Who writes

The text steps (the bible where a brief does not assemble its own, a written
motion sheet, the motion director) are written by the AI session running the
build, whichever agent that is. cag calls no model for them. A step with no
reply writes its prompt to `work/<slug>/text/<key>.prompt.md` and the build
stops there, the way it stops for key art approval:

```
[dance] waiting for text: answer work/belter/text/3f2a….prompt.md by writing the reply to work/belter/text/3f2a….md, then build again
[sets] ko, sing, victory wait for dance
```

The session reads the request, writes the reply beside it, and builds again.
`key` is a digest of the prompt, so a reply only answers the question it was
written for. A set waiting for text stops the chain, because every later set
starts from its last frame. A written set takes two rounds: its motion sheet,
then its director. `CAG_TEXT_MODEL=codex` sends the text to the `codex` CLI
instead.

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
that folder's brief (found anywhere under `specs/` by slug) sets the height guide, and Save
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

- **No OpenAI API.** Codex image generation goes through the local `codex` CLI
  on a ChatGPT subscription, as does text under `CAG_TEXT_MODEL=codex`.
  `cag.chat_codex.ChatCodex` is a LangChain `BaseChatModel` that shells out to
  `codex exec`; `cag.chat_session.ChatSession` is the one the session answers.
- **Vision is macOS only.** The cutout's fallback path is Apple's Vision
  framework (`VNGenerateForegroundInstanceMaskRequest`) through pyobjc. pyobjc
  installs only on macOS; elsewhere the chroma key does all the cutting, and a
  render it cannot read fails instead of falling back.

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
