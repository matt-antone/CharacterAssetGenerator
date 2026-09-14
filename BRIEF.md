# CharacterAssetGenerator

A LangChain/LangGraph pipeline that turns a short character brief into game-ready
sprite assets: a projection sheet of static views and a masked, looping animation
set driven by a MotionArtist motion sheet.

Successor to KaraokeParty-Graphics, which failed under its own weight (180 files,
hash-bound machine acceptance ledgers, revision lifecycles, nine agent roles, a
production queue). Two things in it worked and are kept:

1. **Character spec sheets** — but simplified to name, height, description, and
   prose intent per animation.
2. **Render on magenta, cut out locally** — proven reliable. Here the cutout is
   Apple Vision (`VNGenerateForegroundInstanceMaskRequest` via pyobjc), verified
   working on this machine.

Everything else from that repo is deliberately not carried over.

## Hard constraints

- **No OpenAI API.** Image generation and all LLM calls go through the local
  `codex` CLI, authenticated with the user's ChatGPT subscription. Verified:
  `codex features list` reports `image_generation stable true`, and
  `codex exec --sandbox workspace-write` writes a PNG to disk on request.
- **macOS only** for masking (Apple Vision).
- LangChain is the orchestration layer, reached via a `ChatCodex` adapter — a
  `BaseChatModel` subclass wrapping `codex exec --json`.

## Canonical geometry (from the old repo, kept)

- Fighter cell: `480x560` px.
- Cell height 560 px == 7 ft 0 in, i.e. `80 px/ft`.
- Supporting heel contact baseline at `y=550`.
- Sources render on a full-canvas magenta-family background (prefer `#FF00FF`,
  tolerate nearby values); masking happens once, after art is locked.
- `character-left` / `character-right` name the character's own sides.
  `screen-left` / `screen-right` name image position only. Never mix them.

## v1 deliverables

For one character spec:

- **Key art**: one front-left three-quarter reference, facing screen-left.
- **Projection sheet**: front, back, strict character-left profile, plus the
  approved key art three-quarter.
- **Dance animation**: driven by a supplied MotionArtist `motion.json`
  (`/Users/matthewantone/Development/MotionArtist/work/sample/motion.json`,
  16 frames, 4 fps, front view, clean loop seam).
- All frames masked to RGBA, registered to the 480x560 cell on the contact row.
- A sprite sheet, a looping GIF proof at declared fps, and a gallery page.

## Animation works like a studio

Roles mirror an animation studio, matching the role labels MotionArtist already
emits per frame (`key`, `pilot`, `inbetween`):

- **motion director** — reads `motion.json`, binds the arc and per-frame cues to
  this character's scale and prop hand, decides nothing about identity.
- **keyframer** — draws frames marked `key` and `pilot` first, from the approved
  key art as reference. Owns pose readability and loop-endpoint compatibility.
- **tweener** — draws each `inbetween` from its two locked neighbours. Owns
  continuity. No mechanical interpolation, cross-fade, or optical flow.
- **masker** — Apple Vision cutout and cell registration.
- **assembler** — sprite sheet, GIF proof, gallery.

## Explicit non-goals

- No approval ledgers, hash-bound acceptance bundles, or revision lifecycles.
- No production queue, run-job claiming, or multi-repo state.
- No roster batching in v1. One character, one motion sheet.
- No interpolation-based tweening.
