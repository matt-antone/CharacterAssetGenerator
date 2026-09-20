# AGENTS.md

Instructions for agents working in this repo.

## Build characters in parallel

Run every character that needs building at once, each as its own `uv run cag build`
process. Do not queue them one at a time.

```bash
for spec in specs/default/*.json; do
  uv run cag build "$spec" --jobs 4 &
done
```

`--jobs` parallelises the animation sets within one character. Both levels of
parallelism are wanted.

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

`outputs/<slug>/index.html` points at its pictures with relative paths —
`views/key.png`, `dance-sheet.png`. They resolve only while the file sits in its
own directory next to `views/` and the sheets. Send the bare `.html` to the user
and all twelve images break; the page arrives as text on a dark background and
looks like the render failed, when the render was fine.

Deliver it one of two ways:

- **Publish it as an Artifact with its assets**, so the pictures travel with the
  page. Pass every referenced file through `files`, keyed by the exact path the
  HTML asks for:

  ```
  Artifact(file_path: "outputs/crooner/index.html", root: "outputs/crooner",
           files: {"views/key.png": "views/key.png", "dance-sheet.png": "dance-sheet.png", ...})
  ```

  A finished character is about 4 MB of PNG and GIF, well inside the limits.

- **Or send the images themselves** — the key art, a sheet, a proof GIF — and
  leave the gallery on disk for the user to open locally.

The same trap catches any copy of the page: moving `index.html` anywhere without
`views/` and the sheets beside it breaks every image.

## `--set` rewrites the gallery to just that set

`uv run cag build <spec> --set victory` re-renders one set, which is what you want when
only that set is wrong. But it also rewrites `outputs/<slug>/index.html` from
the sets that ran, so the page comes back listing `victory` alone. The other
sets' sheets and GIFs are still on disk, untouched — only the page forgot them.

Finish with a full `uv run cag build <spec>` afterwards. Completed frames are cached and
skipped, so it costs almost nothing and puts every set back on the page.

## Installing a motion bundle

Bundles arrive as zips. `library()` globs `motions/*/manifest.json`, so a zip in
`motions/` is inert — nothing reads it and nothing warns you.

Installing one is a single action with four parts. Doing three of them leaves
the library lying:

1. Extract into `motions/`, keeping the bundle's own directory name exactly.
   **Never rename on the way in.** A bundle carries its name in three places —
   the directory, the manifest's `name`, and the `name` inside `motion.json` —
   and renaming one desyncs it from the other two. `library()` keys on the
   manifest; the build log prints the motion sheet's copy.
2. Delete the zip.
3. Delete the bundle it supersedes. A re-cut arrives under its own trace name
   and lands *beside* the old one rather than over it, so nothing breaks and a
   brief still naming the old one silently renders the old motion. Silence is
   the failure mode here.
4. Repoint every brief that named the old bundle. A brief naming a bundle that
   is gone fails loudly and by name; a brief naming a stale one does not fail.

Then check it before trusting it:

```bash
.venv/bin/python -c "
from cag.motion import library
for n, b in sorted(library('motions').items()):
    m = b.load()
    print(f'{n}: {b.frame_count}f @ {b.fps}fps {b.view} {b.playback} seam={b.seam!r} '
          f'photos={len(m.photos)} airborne={[f.index for f in m.frames if f.airborne]} '
          f'travel={m.travel:.3f}')"
```

One pass catches everything that matters. A manifest that disagrees with its
motion sheet raises. A short thumb set shows as `photos=0`, which means that set
renders with no pose reference at all. `playback` must suit the set: a `loop`
trace seams back to frame 0, a `one-shot` one does not, and driving a looping
set from a one-shot cut gives a dance that plays once.

A bundle's name is its trace — label, video id and start second — because a
label alone is a genre. Two different dances once collided on `shuffle`, and the
baselines measured against one silently came to refer to the other.

`motions/sample` is the worked example of the format and the only motion fixture
the tests use. It is not a trace; leave it installed.

## Words

Agreed with the MotionArtist repo after one word for two things caused three
false reports between them. **Never write "sheet", "grid", "sprite sheet" or
"spritesheet" unqualified** — in code, comments, filenames or conversation.

A new term is named here before it is used.

| term | what it is |
| --- | --- |
| **motion bundle** | `motions/<name>/`, identified by its manifest (`Bundle`) |
| **manifest** | the bundle's `manifest.json` (`read_bundle`) |
| **motion sheet** | the contents of `motion.json` (`MotionSheet`, `load_motion`). The one place "sheet" may appear, always qualified |
| **traced sheet** / **written sheet** | a motion sheet from a bundle, versus one `cag/motion_writer.py` generated from the brief's prose. A written sheet has no traced frames and therefore no pose reference at all |
| **traced frame** | one photograph of the performer, `thumbs/fNN.jpg` in a bundle |
| **pose card** | one traced frame letterboxed to 384x512, `work/<char>/poses/<set>/NN.png` (`write_photos`) |
| **pose grid** | pose cards tiled `FIGURES_PER_ROW` across, `FRAME_SHEET_SIZE` per image, handed to the generator as the last reference image — `work/<char>/poses/<set>/pose-grid-NN.png` |
| **pose reference** | the umbrella concept. Today always pose cards and pose grids made from traced frames; nothing else qualifies |
| **frame sheet** | many frames of one character drawn in one render. The chunk renders under `work/<char>/source/<set>/`, and the deliverable at `outputs/<char>/<set>-sheet.png` |
| **key art** | the approved character reference render |
| **bible** | the identity text quoted into every prompt |
| **set** | one animation: dance, sing, flinch, guard, entrance, victory, ko |

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
