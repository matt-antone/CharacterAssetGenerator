# CharacterAssetGenerator

Turns a character brief into static views, animated frame sheets, GIF proofs,
and a manifest for playback.

```bash
uv run cag build specs/default/belter.json --draw-backend codex
```

The first run stops after the key art and waits for a human. Look at
`work/belter/source/key.png`, then either sign it off or delete it and build
again for a redraw:

```bash
uv run cag approve specs/default/belter.json
uv run cag build specs/default/belter.json --draw-backend codex
```

After approval, the build writes `outputs/default/belter/`: the enabled
projection cells and portraits under `views/`, `location.png` if the brief
names a location, a frame sheet and GIF proof for each completed animation set,
a gallery page, and `manifest.json`. `enabled.json` controls which views and
sets run; an explicit `--set` overrides the animation switches.

Play from the manifest's fps and playback. `playback` is `loop`, `pingpong` or
`once`; a pingpong frame sheet holds its frames once, and the player walks them
forward and back. Sets run in order. The frame-sheet path carries the previous
set's last frame as a reference; the video path animates each set reference
from its own drive video. A failed set is logged and later sets can continue.
Rebuilding retries missing renders and reuses completed work.

Keep `index.html` beside its assets: send the complete package folder or the
images themselves. The HTML alone has broken relative image links. A `--set`
build rewrites the gallery and manifest to the sets that ran; follow it with a
full build to restore the complete package listing.

## Motion bundles and written motion sheets

A set can name a traced [MotionArtist](https://github.com/matt-antone/MotionArtist)
motion bundle:

```json
"dance": { "motion": "club-01" }
```

Or it can ask the session to write a motion sheet from prose:

```json
"dance": { "intent": "Refined lounge sway loop...", "playback": "pingpong" }
```

A set chooses one source: `motion` cannot be combined with prose intent or a
playback override. A traced motion sheet already carries its rate, facing,
frames and playback. For a written motion sheet, the brief can specify
`playback` as `loop`, `pingpong` or `once`; otherwise `cag/sets.py` supplies the
default. The defaults are sixteen frames, six fps for dance and eight for the
other sets; dance and sing loop, and the others run once.

`"motion": "auto"` chooses from the installed library using the character slug
and set name. The choice stays stable while the library stays the same.

Install bundles through the checked pull command, which reads
`MOTION_ARTIST_REMOTE` (normally `kadrive:MotionArtist`):

```bash
uv run cag motions pull --list
uv run cag motions pull club/club-01
uv run cag motions
```

A bundle lands unrenamed at `motions/<set>/<name>/`. Pull checks the manifest,
file hashes, motion sheet, source clip and any bundle mask or head boxes before
replacing an installed copy. A re-cut with a new name lands beside the old one:
repoint briefs and remove the superseded bundle explicitly. Source clips and
bundle masks are git-ignored, so pull again on a fresh clone.

`cag motions` reports clip, mask and head-box status as `ok`, `missing`, `stale`
or `none`. Read playback before seam quality: a pingpong returns along the same
frames and never needs a last-to-first blend. `cag clips` only backfills old
bundles; a missing or stale tracer-supplied clip must be pulled again.

`--motion-root` reads another library. `--motion` overrides a single set's
motion source for one build, loading the bundle manifest when present:

```bash
uv run cag build specs/default/crooner.json --draw-backend codex --set dance --motion motions/club/club-01/motion.json
```

## How it runs

Two LangGraph runs: the static graph first, then one animation graph per set.

| Step | What it does |
| --- | --- |
| `bible` | Assembles identity text from the brief's `build`, `face`, `hair`, `outfit` and `palette` fields. Edit the brief to change it. Only a brief with none of those fields requests session-written text and caches it in `work/<slug>/bible.txt`. |
| `key_art` | Draws the front-left three-quarter reference on a magenta backdrop. Local builds use material swatches as the detail reference. |
| `approval` | Stops until a human approves this exact key art. The approval stores the image digest, so replacing the image revokes it. |
| `scale` | Measures the approved key art's crown-to-heel span to scale the static views. |
| `projection` | Draws enabled front, back and profile views from the key art. Local builds turn it with a dedicated view graph. |
| `mask` | Cuts the views out and registers them into cells. |

Each animation gets a set reference with its own facing and held props. A set
whose hands and facing match the key art can reuse it. Traced motion bundles
supply photographs; written motion sheets have no photographic pose reference.

The animation route depends on the backend and whether a machine is selected:

| Route | How it draws |
| --- | --- |
| Frame-sheet path | Draws chunks of up to eight figures, then finds and slices the figures by their backdrop. Traced photographs become pose cards tiled into a matching pose grid. Photographic prompts omit the bible, director note and per-frame cues. Written motion sheets use prose direction. A wrong figure count is rejected and retried once. |
| Pose-edit path | Under `comfy` or `local`, a traced set without a machine profile re-poses its set reference into each photograph, one render per frame. |
| Video path | Under `comfy` or `local` with `--machine`, a traced set animates its set reference along a drive video through SCAIL-2, then optionally restyles the selected frames. |

`--per-frame` selects the older keyframe/tween graph instead of the default
frame-sheet graph. Use the default graph for the pose-edit and video routes.
Frames are cut out and registered before assembly. Video-path cells also get
the cell finish described below.

## Who draws

Choose a draw backend for each build:

```bash
uv run cag build specs/default/belter.json --draw-backend codex   # OpenAI, via codex
uv run cag build specs/default/belter.json --draw-backend comfy   # Comfy Cloud
uv run cag build specs/default/belter.json --draw-backend local   # your own ComfyUI
```

There is no default: a build that names none, by flag or `CAG_DRAW_BACKEND`,
stops before drawing anything. The backend determines the render workflows;
the approval gate, package format and session-written text remain shared.

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

Ordinary image references are uploaded letterboxed onto a 1536 square, in
their own border colour; video inputs preserve their frames. ComfyUI batches
images as one tensor and crops each to the first one's size to do it, which
would cut the outer pose cards off a landscape pose grid batched behind
portrait key art.

The magenta check still applies: a render with scenery behind the character is
drawn again, the same as under codex.

**local** runs the same kind of workflow on a ComfyUI server of your own, at
`http://127.0.0.1:8188` unless `CAG_LOCAL_COMFY_URL` says otherwise. It needs no
key and bills nothing, but it can only run nodes and models installed there, so
no partner nodes: Nano Banana does not exist on a local server. Its default
workflow is `comfy/qwen-image-edit-2511.json`, Qwen-Image-Edit 2511
(Apache-2.0; `--comfy-workflow` or `CAG_LOCAL_COMFY_WORKFLOW` names another).
A package meant for anything but research still needs a suitable licence for
every model contributing to it.

Without a machine profile, a traced set under either workflow backend is drawn
a frame at a time by `comfy/pose-edit.json` (`--comfy-pose-workflow` or
`CAG_COMFY_POSE_WORKFLOW`):
Qwen-Image-Edit-2511 at fp8 with the AnyPose LoRAs and the Lightning 4-step
LoRA, which re-poses the set's reference into each traced photograph. Every file
it loads runs on a 16 GB card, so Comfy Cloud and a local server draw the same
dance. Each frame is then snapped back onto the reference's pixel grid and
palette, with a black outline, on exact magenta (`cag.snap`): the workflow keeps
the character, not the pixel art. The render as drawn is kept as `NN.raw.png`.

### Local views

Local key art uses material swatches from `cag/references/detail-level-10-tiles.png`
instead of a full character detail sample, which the key-art model tended to
copy. A detail level with no swatch sample uses words alone.

Other local views turn the approved key art using `comfy/view-edit-2511.json`
(`CAG_LOCAL_VIEW_WORKFLOW`): Qwen-Image-Edit 2511 with Lightning and
Multiple-Angles LoRAs. The prompt supplies facing, stance and hands; the image
supplies identity. The build checks the view graph's models before drawing.
Existing projection views and set references remain cached until their source
images are removed; changing a prompt alone does not replace them.

### The video path

A machine profile selects hardware settings and model files separately from
the draw backend:

```bash
uv run cag machines
uv run cag machines --check cloud
uv run cag build specs/default/belter.json --draw-backend comfy --machine cloud
```

`CAG_MACHINE` supplies a profile by default; `--no-machine` disables it for one
build. Profiles live in `comfy/machines/`. `cloud` is the verified cloud profile;
`local16` describes the RX 9070 setup; `smoke4` is a wiring check, never an art
quality reference. Local builds preflight the video and required restyle models.
The mask pass's SAM3 checkpoint warns if missing, since not all footage uses it.

A traced set needs its declared source clip. Missing or stale footage fails the
set with a recovery command; it never silently switches to pose editing.

The drive video is cropped and resampled from the clip. Its drive mask comes
from a valid bundle mask, a threshold on a light backdrop, or a SAM3 mask pass.
Face blur uses bundle head boxes where available and mask-based head detection
otherwise. Frames where a head cannot be found are reported. SCAIL animates the
set reference along this drive; each traced frame then selects the SCAIL frame
at its trace index. By default Qwen-Image-2.1 restyles each selection.

`--no-restyle` copies those SCAIL frames back to the set reference's dimensions
instead. Both modes finish the cells after shrinking: remove magenta fringe,
quantize to one 64-colour palette per set without dithering, and add a 1px black
outline inside the silhouette. Cut-outs live in `cells-cut/<set>/`; finished
cells in `cells/<set>/`. `--no-finish` skips that step. Both flags require a
machine profile.

The caches follow their dependencies:

- `work/drive/` shares drive videos across characters and locks cutting so
  parallel builds do the work once. Clip, bundle-mask and head-box changes
  invalidate the drive. A changed SAM3 graph or prompt rebuilds a drive that
  used the mask pass and invalidates its SCAIL video and selected frames too.
  An unused mask-pass change leaves threshold and bundle-mask builds cached.
- `work/<slug>/video/<set>/` caches SCAIL output and records submitted job IDs
  in `pending.json`, so an interrupted build resumes the job. A changed set
  reference, SCAIL graph or seed selects another video. `CAG_VIDEO_SEED` sets
  the seed for another roll.
- `source/<set>/` caches selected frames with a `drawn.sha` stamp. Changing the
  restyle graph or prompt preserves the SCAIL video but moves the old frames
  into `superseded/`. Switching `--no-restyle` does the same. Failed restyles
  are retried individually. The cell finish has its own content digest.

SCAIL caches created before the mask-pass dependency was included are replaced
on the next build that uses a SAM3 drive. Existing threshold and bundle-mask
cache keys are unchanged.

**The shipped Qwen-Image-2.1 workflow is research-only.** `--no-restyle` does
not clear that restriction for a local build: its key art, and therefore its
set reference and animation, still derive from Qwen-Image-2.1. Do not ship those
packages until the licensing is cleared. See [AGENTS.md](AGENTS.md) for measured
runs and their limits: there is no established pose-fidelity floor, and one
character's successful render is not evidence for the whole cast.

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

Review the GIF proofs after a build. On the frame-sheet path, the generator
draws figures side by side without fixed positions: each is found by its
backdrop and registered into a cell. Scale holds across a frame sheet; placement
can drift. Pose-edit and video frames share a canvas and registration transform,
but still need visual review. A figure can sit a few pixels left, right, high
or low of its neighbours, making a stationary loop jitter or slide. The mask
cannot tell drift from a deliberate step, so it leaves both alone. Deciding
which is which is a person's job.

`uv run cag edit` serves a small editor over the outputs, or one character's
folder:

```bash
uv run cag edit
```

Open `http://127.0.0.1:8765/` (`--port` to change it) and pick a
`<set>-sheet.png` from under that folder. Every character's frame sheets share the
same names, so the editor matches the one you open to its folder by contents:
that folder's brief (found anywhere under `specs/` by slug) sets the height guide, and Save
writes back there. A frame sheet that isn't under the folder won't save.

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
4. The **Save sheet** button overwrites the frame sheet and rebuilds
   `<set>-proof.gif` beside it at the fps in the box, then closes the frame sheet
   and shows the result on the stage. If the save fails, the frame sheet stays
   open with your edits. Frames you moved carry a dot in the filmstrip.
   With `CAG_OUTPUT_REMOTE` set (or
   `--output-remote`), Save then publishes that character's package, so the
   shared copy gets the edit too; `--no-publish` turns that off.

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
the frame sheet and loses the edits**. Opened straight from disk instead of through
`uv run cag edit`, Save downloads the frame sheet and leaves the proof alone.

## Publishing

Set `CAG_OUTPUT_REMOTE=kadrive:CharacterAssetGenerator/outputs`, or pass
`--output-remote`, to copy each package after a build or editor save. Publishing
uses `rclone copy --checksum`, never sync or remote deletion. `--no-publish`
disables it. Builds stopped for key-art approval publish nothing; failed sets
can leave a partial package, and publish failures warn without failing a build.

To publish existing packages without rendering:

```bash
uv run cag publish specs/default/belter.json
uv run cag publish --all
```

The remote must include its colon, and rclone must already be configured.
Encrypted configs need `RCLONE_CONFIG_PASS` because publishing is noninteractive.
Google Drive previews HTML as text: share the images or download the whole
package folder before opening its gallery locally.

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

Static views take their scale from the key art. Animation frames drawn together
take one scale per frame sheet, read from the poses or figure heights. Pose-edit
and video frames use one shared canvas transform across the set. Either way,
the scale is applied unchanged to every figure, so a crouch renders shorter
instead of being stretched back to standing height.

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
