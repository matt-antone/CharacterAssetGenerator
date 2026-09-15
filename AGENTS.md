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
