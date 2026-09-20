# AGENTS.md

Instructions for agents working in this repo.

## Do not merge the photo pose path before one real render

On `pose-reference-from-sprite-sheet`, a bundle that ships one thumbnail per
frame now uses those photographs as the pose reference, and a photographic set
sends a leaner prompt — the bible, the director's note and every per-frame cue
are dropped. It measures far better on amplitude, which is what decides whether
a set reads as a dance or a sway. **It has never produced a real render through
`cag build`.** Every number behind it came from scratch scripts that
hand-assembled their prompts, and the shipped code assembles them differently.

Run one 12-frame set end to end before this reaches main, and check four things:

1. Photo cards in frame order, and the photograph wording in the prompt rather
   than the pose-diagram wording.
2. Zero overlapping figure boxes — every figure inside its own cell.
3. The character's costume surviving onto a foot that is off the floor. Before
   the costume anchor, a lifted foot came back wearing the reference dancer's
   white trainer, 2 of 16 figures.
4. Identity across a chunk boundary: does figure 13 look like figure 1? Scale is
   already proven fine there — a 2.0x drawn-size difference registers down to a
   4px spread — but nothing measures identity and nothing warns.

Ruled out with measurements, do not re-open: fixing torso rotation through
prompting. A frame traced at 28 degrees of body yaw comes back square-on in
every condition tried — 1, 4, 8 and 12 figures per render, photographs or
skeletons, and three separate rewordings.

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

## Keep the bible

`work/<slug>/bible.txt` is the locked visual description every frame was drawn
against. When clearing artifacts for a fresh render, delete everything else
under `work/<slug>/` and leave that file, unless a new identity is wanted.

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

## graft skill

This repo is indexed by graft. Use the graft skill for codebase context — `graft_find_code`, `graft_find_all`, `graft_trace_calls`, `graft_file_api`, `graft_repo_map` — before grepping or reading source files. Never commit graft caches.
