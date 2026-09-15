# AGENTS.md

## Never build characters in parallel

Run `cag build` for one character at a time. Let it finish before starting the
next.

```bash
cag build specs/belter.json --jobs 4
```

This applies to repair and retry passes as well as first builds. Do not fan
characters out across background processes, shells, or subagents, and do not
start a second character because the first one is slow.

`--jobs` parallelises the animation sets *within* one character, which is fine.
The rule is about characters, not sets.
