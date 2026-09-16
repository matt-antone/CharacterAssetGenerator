# AGENTS.md

Instructions for agents working in this repo.

## Build characters in parallel

Run every character that needs building at once, each as its own `cag build`
process. Do not queue them one at a time.

```bash
for s in belter crooner diva heavyweight hype-man idol outlaw screamer; do
  cag build "specs/$s.json" --jobs 4 &
done
```

`--jobs` parallelises the animation sets within one character. Both levels of
parallelism are wanted.

## Launch builds as tracked background tasks

Start each `cag build` as a harness background task, not a detached `nohup ... &`
process. A detached process does not appear in the user's background task list,
so they cannot see what is running or stop it, and it looks like the build never
started.

## The first build stops at the key art

A build draws the key art, then exits telling you it is waiting for approval.
Nothing else — no projection views, no frames — is drawn until someone looks at
`work/<slug>/source/key.png` and runs `cag approve specs/<slug>.json`, then
builds again. A key art that is wrong gets deleted instead, and the next build
redraws it. Show the user the key art and wait for their answer; do not approve
on their behalf.

## A failed set is normal, retry it

A sheet render that comes back with the wrong figure count is rejected and
redrawn once, then the set is logged as `FAILED` and the run continues. Failures
are a per-render coin flip, not a broken set. Re-running the same `cag build`
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

`cag build <spec> --set victory` re-renders one set, which is what you want when
only that set is wrong. But it also rewrites `outputs/<slug>/index.html` from
the sets that ran, so the page comes back listing `victory` alone. The other
sets' sheets and GIFs are still on disk, untouched — only the page forgot them.

Finish with a full `cag build <spec>` afterwards. Completed frames are cached and
skipped, so it costs almost nothing and puts every set back on the page.
