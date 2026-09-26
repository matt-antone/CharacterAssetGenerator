# AGENTS.md

Instructions for agents working in this repo.

## The photo pose path, and what its one real render showed

A bundle that ships one thumbnail per frame uses those photographs as the pose
reference, and a photographic set sends a leaner prompt — the bible, the
director's note and every per-frame cue are dropped.

This merged once it had produced a real render through `cag build`, because
until then every number behind it came from scratch scripts that hand-assembled
their prompts differently from the shipped code. That run was belter's `dance`,
24 frames from `shuffle-1-b0ARQ5kM85Y-13.6s`, three renders of eight figures:

1. **Photo cards in frame order, photograph wording.** 24 cards in order, and
   the prompt says "strip of photographs of a real performer" — not the
   pose-diagram wording, which no longer exists.
2. **No overlapping figure boxes.** Eight figures sliced cleanly from each of
   three sheets.
3. **Costume surviving onto a lifted foot.** White pixels in the lower 45% of a
   figure: 0 of 24. Bare leg: 0 of 24. Belter's palette holds no white, so
   either reading is unambiguous bleed from the performer in the photograph.
4. **Identity across a chunk boundary.** Figures 7, 8, 15 and 16 measured
   0.93-0.98x figure 0's height with a mean colour delta at or under 8.1.

What that run does **not** establish, and nobody should claim it does:

- **Pose fidelity has no floor on the current trace.** Every recorded baseline
  was measured on a clip that has since been deleted, so no number says whether
  a rendered dance reads as that dance. Establish a fresh floor before judging
  any further prompt or reference change.
- **It is one roll.** The same clip measured zero costume bleed at twelve
  figures a render on one roll and eleven of twenty-four on the next. Roll twice
  before calling any of these numbers settled.

Ruled out with measurements, do not re-open: fixing torso rotation through
prompting. A frame traced at 28 degrees of body yaw comes back square-on in
every condition tried — 1, 4, 8 and 12 figures per render, photographs or
skeletons, and three separate rewordings.

## The video path

`--machine <profile>` (or `CAG_MACHINE`), under `--draw-backend comfy` or
`local`, draws every traced set from its bundle's source clip instead of its
pose cards: a drive video cut from the clip, one SCAIL-2 job that animates the
set reference along it, then one Qwen-Image-2.1 restyle per traced frame, of the
SCAIL frame at the trace index. Without `--machine` a traced set takes the
pose-edit path as before, and `--no-machine` takes it for one build whatever
`CAG_MACHINE` says. Like the backend, the profile is the user's call: it names
the hardware and the bill.

- **A traced set with no source clip fails,** and says to run
  `uv run cag clips <bundle>`. It never falls back to the pose-edit path, because
  a set drawn the other way would pass for this one's output. `cag motions`
  shows each bundle's clip as `ok`, `missing`, `stale` or `none`. A `stale` clip
  (a file on disk that is not the one declared) fails only the video path's
  sets of that bundle; every other build reads the library as usual.
- **Profiles** live in `comfy/machines/`. `cloud` is the only verified one — the
  research run's settings. `local16` (RX 9070) has drawn nothing yet. `smoke4`
  (GTX 1050 Ti: 256x384, 9 frames, 2 steps) is wiring only, never art: nothing
  drawn on it is judged. `uv run cag machines --check <name>` lists every node
  and model file that machine's ComfyUI lacks, and a `local` build will not
  start while anything the video or restyle graph loads is missing. The mask
  pass's SAM3 checkpoint only warns: most footage never runs it.
- **Three caches, each keyed on what it is made from.** The drive under
  `work/drive/` is shared by every character dancing that bundle, and locked
  while it is cut, so parallel builds cut it once; a mask pass is keyed on its
  graph and prompt, and one whose drive mask fails is moved aside to
  `rejected-mask-pass-N/` and run again next build. The SCAIL
  video under `work/<slug>/video/<set>/` records its job id in `pending.json`
  the moment it is submitted, so a stopped build resumes the job rather than
  paying for another. The restyles are `source/<set>/NN.png`, stamped in
  `drawn.sha`: a changed restyle graph or prompt moves them into
  `source/<set>/superseded/` and keeps the SCAIL video, and a changed set
  reference draws a new SCAIL video. A failed restyle fails the set by frame
  number, and a rebuild draws only those.
- **Only footage on a light backdrop** gets a threshold drive mask. Anything
  else runs the mask pass (SAM3). One render has exercised it, on Comfy Cloud:
  `country-01`, portrait footage in a cluttered shop, gave one clean silhouette
  per frame (figure 10% of the frame, overlap 0.82-0.95 frame to frame).
- **Qwen-Image-2.1 is licensed for research only.** Nothing the video path draws
  ships until that is cleared.

Trial renders on a branch are scratch, as below. The video path keeps each restyle as drawn; nothing is snapped to the key art's
grid (`cag.snap` still serves the pose-edit path), because snapping made the faces
blocky. What `cag build` itself has
drawn on this path, all Belter on the `cloud` profile, 2026-09-25, scratch:

1. **`club-01`, two rolls** (seed 1234, then `CAG_VIDEO_SEED=7`). The trace
   index came out `[0,3,6,8,…,36,39]`, 41 SCAIL frames at 16 fps, exactly the
   research run's. All 15 restyles passed first time. On the masked cells the
   two rolls agree: figure height within 4-5% across a set, the same foot row
   in every frame, no frame more than 2.6 off its set's mean colour, and the
   two sets' mean colours within about 3 of each other. Against the set
   reference the first roll's raw frames read 0.77-0.92 detail and 0.7-5.3
   colour delta. The second roll's backdrop drifted violet (red 177-211 rather
   than about 235), which the research `measure.py` threshold reads as figure,
   so its raw-frame numbers are not comparable; the cut-out measures each
   frame's own backdrop, so the cells are unaffected.
2. **`country-01`, one roll,** through the mask pass: 14 frames, the foot work
   (kicks, crossed steps) carried, identity held.
3. **Cost:** about 118 credits a `club-01` set, and a rebuild of a finished set
   spends none (the SCAIL video and every restyle are cached).

The SCAIL prompt quoting the bible, untested before these runs, held identity
on both dances. There is still no pose-fidelity floor for this path, and one
character is not a cast: the bulky trooper, over the 8.1 colour bar in 6 of 15
frames in the scratch-script run, has not been drawn through `cag build`.

## Real renders happen on main

A render that counts is drawn on `main`, from the committed code, into the
repo's own `work/`. A branch or a worktree renders only to try something out:
its art is scratch, it is never the version anyone ships, and a package built
there is not a package the roster has. Land the code first, then render.

A package is filed by theme: the folder a brief sits in inside `specs/` names the
folder its render lands in, so `specs/halloween/mort.json` writes
`outputs/halloween/mort/`. A brief filed loose in `specs/` writes `outputs/<slug>/`.

`outputs/` is not tracked. A build writes it wherever it is told to, and the
committed copy was 256 files that changed on every re-render.

## Build characters in parallel

Run every character that needs building at once, each as its own `uv run cag build`
process. Do not queue them one at a time.

```bash
BACKEND=comfy   # the one the user named; see "Ask which draw backend" below
for spec in specs/default/*.json; do
  uv run cag build "$spec" --draw-backend "$BACKEND" &
done
```

Characters are the unit of parallelism. Within one character the sets are drawn
in order, each continuing from the last frame of the set before it, so there is
nothing to parallelise there.

## Launch builds as tracked background tasks

Start each `uv run cag build` as a harness background task, not a detached `nohup ... &`
process. A detached process does not appear in the user's background task list,
so they cannot see what is running or stop it, and it looks like the build never
started.

## The first build stops at the key art

A build draws the key art, then exits telling you it is waiting for approval.
Nothing else — no projection views, no frames — is drawn until someone looks at
`work/<slug>/source/key.png` and runs `uv run cag approve specs/<set>/<slug>.json`, then
builds again. A key art that is wrong gets deleted instead, and the next build
redraws it. Show the user the key art and wait for their answer; do not approve
on their behalf.

## Ask which draw backend, never pick one

`cag build` has no default draw backend: it stops unless `--draw-backend` or
`CAG_DRAW_BACKEND` names `codex` (OpenAI), `comfy` (Comfy Cloud) or `local` (the
user's own ComfyUI). Each is a different model, a different bill and a different
licence, so it is the user's call. If the user has not named one, ask before the
first build, and use their answer for every build that follows. An agent that
cannot ask stops and says it needs one. The user may call it the "processor".

## A build stops for text, and the session writes it

cag calls no model for its text steps — the motion director, a written motion
sheet, a bible a brief did not assemble. The AI running the session writes them.
A step with no reply writes its prompt to `work/<slug>/text/<key>.prompt.md`,
logs `waiting for text`, and stops that set and every set after it, since each
starts from the last frame of the one before. Read the request, write the reply
to `work/<slug>/text/<key>.md` beside it, and build again. `<key>` is a digest of
the prompt, so a changed brief asks afresh. `CAG_TEXT_MODEL=codex` sends the text
to the codex CLI instead.

## A failed set is normal, retry it

A sheet render that comes back with the wrong figure count is rejected and
redrawn once, then the set is logged as `FAILED` and the run continues. Failures
are a per-render coin flip, not a broken set. Re-running the same `uv run cag build`
redraws only the missing sets, because completed frames are cached and skipped,
so repeated passes converge.

## The bible comes from the brief, not from a model

A brief that fills `build`, `face`, `hair`, `outfit` or `palette` assembles its
own bible (`assemble_bible`, `cag/prompts.py`). Nothing is cached: the text is a
function of the spec, so the spec is the record and git can show it to you. Edit
the spec, not a file under `work/`.

`work/<slug>/bible.txt` is only written for a brief that fills none of those
fields, where a model still writes the paragraph and it is frozen so later sets
quote the same identity. If such a file exists, keep it when clearing artifacts
for a fresh render; for every brief in `specs/` it is ignored and stale.

## Never hand over `index.html` on its own

`outputs/<theme>/<slug>/index.html` points at its pictures with relative paths —
`views/key.png`, `dance-sheet.png`. They resolve only while the file sits in its
own directory next to `views/` and the sheets. Send the bare `.html` to the user
and all twelve images break; the page arrives as text on a dark background and
looks like the render failed, when the render was fine.

Deliver it one of two ways:

- **Publish it as an Artifact with its assets**, so the pictures travel with the
  page. Pass every referenced file through `files`, keyed by the exact path the
  HTML asks for:

  ```
  Artifact(file_path: "outputs/default/crooner/index.html", root: "outputs/default/crooner",
           files: {"views/key.png": "views/key.png", "dance-sheet.png": "dance-sheet.png", ...})
  ```

  A finished character is about 4 MB of PNG and GIF, well inside the limits.

- **Or send the images themselves** — the key art, a sheet, a proof GIF — and
  leave the gallery on disk for the user to open locally.

The same trap catches any copy of the page: moving `index.html` anywhere without
`views/` and the sheets beside it breaks every image.

## `--set` rewrites the gallery to just that set

`uv run cag build <spec> --set victory` re-renders one set, which is what you want when
only that set is wrong. But it also rewrites `outputs/<theme>/<slug>/index.html` from
the sets that ran, so the page comes back listing `victory` alone. The other
sets' sheets and GIFs are still on disk, untouched — only the page forgot them.

Finish with a full `uv run cag build <spec>` afterwards. Completed frames are cached and
skipped, so it costs almost nothing and puts every set back on the page.

## Installing a motion bundle

Bundles arrive as zips. `library()` finds every `manifest.json` under `motions/`,
at any depth — the layout is `motions/<genre>/<genre>-NN/` — so a zip in
`motions/` is inert: nothing reads it and nothing warns you.

Installing one is a single action with four parts. Doing three of them leaves
the library lying:

1. Extract into `motions/<genre>/`, keeping the bundle's own directory name exactly.
   **Never rename on the way in.** A bundle carries its name in three places —
   the directory, the manifest's `name`, and the `name` inside `motion.json` —
   and renaming one desyncs it from the other two. `library()` keys on the
   manifest; the build log prints the motion sheet's copy.
2. Delete the zip.
3. Delete the bundle it supersedes. A re-cut that arrives under a new name
   lands *beside* the old one rather than over it, so nothing breaks and a
   brief still naming the old one silently renders the old motion. Silence is
   the failure mode here.
4. Repoint every brief that named the old bundle. A brief naming a bundle that
   is gone fails loudly and by name; a brief naming a stale one does not fail.

Then check it before trusting it:

```bash
.venv/bin/python -c "
from cag.motion import clip_status, library
for n, b in sorted(library('motions').items()):
    m = b.load()
    print(f'{n}: {b.frame_count}f @ {b.fps}fps {b.view} {m.playback} seam={b.seam!r} '
          f'photos={len(m.photos)} airborne={[f.index for f in m.frames if f.airborne]} '
          f'travel={m.travel:.3f} clip={clip_status(b)}')"
```

One pass catches everything that matters. A manifest that disagrees with its
motion sheet raises, and so does a source clip that does not span the traced
frames. A short thumb set shows as `photos=0`, which means that set renders with
no pose reference at all. `clip=none` means the video path cannot draw the
bundle until `uv run cag clips <name>` backfills it; `clip=missing` is every
backfilled bundle on a fresh clone, `clip=stale` is a file on disk that is not
the declared one, and `uv run cag clips --missing` cuts both kinds again.
`--cast` covers only the bundles a brief names. `playback` must suit the set: a `loop`
trace seams back to frame 0, a `pingpong` turns around on its ends and plays
back down the frames it just played, a `one-shot` does neither, and driving a
looping set from a one-shot cut gives a dance that plays once.

Read `playback` before you read `seam`. A `seam` verdict of `needs blend` or
`stalls` is a statement about a cut from the last frame back to the first, and a
pingpong has no such cut — its return leg is the out leg reversed. **A pingpong
never needs a blend.** `MotionSheet.seams_cleanly` already knows this and is True
for every pingpong whatever the word says, so nothing warns and nothing is
broken; the trap is a human or an agent reading the word out of the listing
above and reporting clips as flawed, or ranking a set of pingpongs by seam
quality. Both are wrong. A pingpong's ends only have to be drawable poses, which
is the `airborne` column, not the seam.

It is read off `motion.json` — the motion spec — and never off the manifest,
which records what was traced rather than ruling on it. A pingpong arrives as
two keys, `"playback": "loop"` with `"pingpong": true` beside it, because the
straight loop is what an unaware consumer plays; `load_motion` folds them into
the one word `pingpong`. `one-shot` and `final-hold` both read as `once` — what
holds a set's last frame is the set plan's call, not the trace's. A set is driven by one
prompt or the other: its motion spec, or the brief that writes it a sheet. The
set plan in `cag/sets.py` is only the default for a brief that says nothing.

The source clip is the one edit cag makes to an installed bundle. `cag clips`
fetches the source video into `work/sources/`, cuts the traced window at native
rate and size, finds the clip box by matching the bundle's own traced frames
against it, and refuses below a 0.90 match. It writes `clip.mp4`, which is
git-ignored, and a `clip` block in the manifest marked
`"backfilled_by": "cag"`, which is committed. x264 writes its own build into
every file, so another machine's cut of the same frames has another hash: that
hash goes in `clip.sha256` beside the clip, also ignored, and the committed
block is left as it is. Only a cut of different frames rewrites the block. It never touches `motion.json` or
the manifest's `files`. A MotionArtist re-export replaces the block; run
`cag clips` on the new bundle if it arrives without one.

A bundle is named `<genre>-NN` — `club-01` — and what it traced, the video id
and start second, is in its `source` block. Two cuts of one video are two
names: `club-01` and `club-04` are both from `P4QeqpsY8v8`. Names used to be
the trace itself (`shuffle-1-b0ARQ5kM85Y-13.6s`), after two different dances
collided on the bare label `shuffle` and the baselines measured against one
silently came to refer to the other. A baseline names the bundle it was
measured on.

`motions/sample` is the worked example of the format and the only motion fixture
the tests use. It is not a trace; leave it installed.

## Words

Agreed with the MotionArtist repo after one word for two things caused three
false reports between them. **Never write "sheet", "grid", "sprite sheet" or
"spritesheet" unqualified** — in code, comments, filenames or conversation.

A new term is named here before it is used.

| term | what it is |
| --- | --- |
| **motion bundle** | `motions/<genre>/<name>/`, identified by its manifest (`Bundle`) |
| **manifest** | the bundle's `manifest.json` (`read_bundle`) |
| **motion sheet** | the contents of `motion.json` (`MotionSheet`, `load_motion`). The one place "sheet" may appear, always qualified |
| **traced sheet** / **written sheet** | a motion sheet from a bundle, versus one `cag/motion_writer.py` generated from the brief's prose. A written sheet has no traced frames and therefore no pose reference at all |
| **traced frame** | one photograph of the performer, `thumbs/fNN.jpg` in a bundle |
| **source clip** | the footage a bundle was traced from, `motions/<genre>/<name>/clip.mp4`: native rate and size, uncropped, the traced window ±0.5 s. Described by the manifest's `clip` block (`Clip`, `Bundle.clip`), the one manifest edit cag makes (`"backfilled_by": "cag"`), written by `cag clips <name>`. The block is committed and the `.mp4` is git-ignored and absent from `files`, so a fresh clone loads with `clip=None` until `cag clips` runs. A file on disk whose sha256 is neither the block's nor this machine's own cut (`clip.sha256`) is `stale`, and the video path raises it |
| **clip box** | the 3:4 box, in source-clip pixels, that the traced frames were cut from (`Clip.box`). Always inside the frame |
| **pose card** | one traced frame letterboxed to 384x512, `work/<char>/poses/<set>/NN.png` (`write_photos`) |
| **pose grid** | pose cards tiled `FIGURES_PER_ROW` across, `FRAME_SHEET_SIZE` per image, handed to the generator as the last reference image — `work/<char>/poses/<set>/pose-grid-NN.png` |
| **pose reference** | the umbrella concept. Today always pose cards and pose grids made from traced frames; nothing else qualifies |
| **mannequin** | one traced frame's landmarks drawn as a flat figure in the character's own colours, on magenta (`cag/mannequin.py`). The starting latent of a Qwen-Image-2.1 frame, never a reference image: the pose comes from it, identity from the key art |
| **location** | the character's background: one 2048x1152 plate drawn from the brief's `location` prose, with no character in it. Shown cropped to the central 12:7, so the far left and right edges are croppable and carry nothing load-bearing. The only render that is not on magenta and never goes through the mask — the character cell is composited over it |
| **frame sheet** | many frames of one character drawn in one render. The chunk renders under `work/<char>/source/<set>/`, and the deliverable at `outputs/<theme>/<char>/<set>-sheet.png` |
| **key art** | the approved character reference render |
| **bible** | the identity text quoted into every prompt |
| **set** | one animation: dance, sing, flinch, guard, entrance, victory, ko |
| **draw backend** | what draws a build's art: `codex`, `comfy` or `local` (`--draw-backend`). Never assumed. The user may say "processor" |
| **set reference** | the identity image a set is drawn against, what `set_key_art` returns: `key.png`, or `source/<set>-key-<view>.png` |
| **machine profile** | `comfy/machines/<name>.json`: the model files, sizes, steps, rate, length cap and timeouts for one machine (`Machine`, `--machine`) |
| **video path** / **pose-edit path** | the two ways a traced set is drawn under a workflow backend: from its source clip through SCAIL-2 (`video_frames`), or one pose card at a time through Qwen-Image-2.1 (`pose_edit_frames`) |
| **drive video** | the source clip resampled to the drive rate, cut to a 2:3 box, sized to the machine profile, as one animated PNG: `work/drive/<bundle>-<digest12>/drive.png`. Shared across characters |
| **drive rate** | frames per second of the drive video. 16 unless the profile's length cap lowers it |
| **drive mask** | the drive video's silhouette per frame, #0000FF on black: `drive-mask.png` beside `drive.png` |
| **mask pass** | the SAM3 Comfy job that makes a drive mask when the footage has no light backdrop to threshold |
| **reference mask** | the set reference's silhouette, blue on black |
| **SCAIL video** | every image one SCAIL-2 job returns: `work/<char>/video/<set>/<digest12>/NNN.png` |
| **SCAIL frame** | one image of a SCAIL video |
| **trace index** | for each traced frame, the SCAIL frame drawn at its traced time: `round((t_i − t_0) · drive_rate)` |
| **restyle** | one Qwen-Image-2.1 edit render, SCAIL frame as `<image1>` and set reference as `<image2>`. Writes `source/<set>/NN.png` |

`tile` (`cag/assemble.py`) is a layout verb — lay cells out in a grid. It builds
both the pose grid and the frame sheet, and is never a name for either.

Retired, do not reintroduce: "sprite sheet" and "spritesheet" (they meant three
different objects), and "skeleton" (the drawn-figure pose reference, deleted).

cag tiles its own pose grid from a bundle's loose traced frames and ignores any
grid the bundle ships pre-tiled. The tiling has to match the figure layout the
same prompt asks for, and those constants are cag's render batch, not a property
of the footage.

## graft skill

This repo is indexed by graft. Use the graft skill for codebase context — `graft_find_code`, `graft_find_all`, `graft_trace_calls`, `graft_file_api`, `graft_repo_map` — before grepping or reading source files. Never commit graft caches.
