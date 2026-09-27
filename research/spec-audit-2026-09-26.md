# Brief audit, 2026-09-26

An audit of every brief in `specs/` (48 characters), written after the motion
library audit and the video path landing on `main`. Nothing here has been
changed yet; this is the list to work from.

Severity: **high** will visibly hurt a render or blocks the video path;
**med** is likely drift, or a constraint that matters but reaches no prompt;
**low** is tidying.

## How it was checked

- Every brief loads through `load_spec` and validates against
  `specs/character.schema.json`: 48 of 48 pass.
- Every bundle a brief names is installed. The library listing
  (`library('motions')`) loads cleanly.
- Belter's and Heavyweight's key art on disk was compared with the bible
  `assemble_bible` builds from the current brief: both match, and both are
  approved.
- The other 46 briefs were read against a written rubric (below), one reviewer
  per theme.

## Pipeline facts the findings rest on

- **`avoid`, `recognition_cues`, `age` and `personality` reach no prompt.**
  Every brief in `specs/` fills at least one bible field, so its bible is
  `assemble_bible` (`cag/prompts.py:162`): description, build, face, hair and
  outfit, then the palette. `bible_request`, the only code that reads the other
  fields, runs only for a brief that fills none of them, and there is no such
  brief. A constraint that lives only in `avoid` is never told to the image
  model. That is deliberate for `avoid` (a generator handed a list of things
  not to draw tends to draw them), so anything that matters has to be said
  positively in build, face, hair or outfit.
- **`performance_style` reaches the key art only,** as its stance line
  (`cag/prompts.py:256`). Intents that lean on "her ready stance" depend on the
  key art having drawn one.
- **The video path needs the bundle's source clip.** Clips on disk: `club-01`
  (with bundle mask and head boxes, from the tracer), `club-04`, `club-05` and
  `country-01` (backfilled). Every other bundle reads `clip=none`. The tracer's
  Drive holds only `club-01`, so `cag clips` backfill is the only source for
  the rest (no bundle mask, no head boxes: cag's own drive mask and head search
  apply).
- **A prose dance is a written sheet:** no pose reference, no source clip, no
  video path.
- **`enabled.json` renders dance, sing, victory and ko;** flinch, guard and
  entrance are off. Findings in those three intents are rated lower.
- **Defringe is not the purple-costume risk it looks like.** Several palettes
  pass the finish's `tinted` hue test (`min(r,b) − g > 22` and `b − g > 35`):
  gospel's robe, rocker's boots, rudeboy's plum highlight, and others below.
  `9be4c7e` exempts any tinted colour the set reference wears in bulk
  (`costume`, `cag/finish.py:86`), naming gospel and rocker among others. It is
  unmeasured on those characters' real renders, so check their first cells.
  The cut-out against the magenta backdrop is a separate step and is not
  covered by that fix.
- **The cell finish** shrinks a render about 3.6x, then quantizes each set to one
  64-colour palette with a 1 px black outline. Fine detail — laces, thin
  stripes, small jewellery — does not survive it.

## Cast-wide

- **Only 3 briefs can take the video path today:** belter (`club-01`), outlaw
  (`country-01`), deb (`club-04`).
- **13 briefs name a bundle with no clip:** crooner, diva, heavyweight,
  hype-man, idol, screamer, babs, brainz, chops, frank, hex, mort, spot. Fix:
  `uv run cag clips --missing --cast`.
- **31 briefs have a prose dance:** all 16 sci-fi, 8 halloween, 7 default. None
  can take the video path until cast a bundle.
- **Shared dances:** `twist-03` (heavyweight, mort), `oldies-01` (crooner,
  chops), `shakira-01` (diva, babs).
- **Airborne frames:** hype-man's `hiphop-04` has frames 10–13 and 17 airborne.
- **16 installed bundles are cast to no one:** breakin-01, butterfly-01,
  club-02, club-03, club-05, country-02, hiphop-01/02/03/05, lindey-hop-02,
  nicki-minaj-02/03, oldies-02, shuffle-01/02/04. Room to recast before
  doubling up.
- **Body type is untested on the video path.** Only Belter, a slim woman driven
  by a slim woman, has been drawn through `cag build` on it. The bulky trooper
  missed the 8.1 colour bar in 6 of 15 frames in a scratch-script run.

## default

### belter — dance: `club-01` [clip ok, mask ok, heads ok]

Ready for the video path. `club-01` is the tracer's re-cut of the same window
as before (`P4QeqpsY8v8` 77.035–79.454 s, 15 frames, pingpong).

- [med] `performance_style` is "free hand compact and relaxed." (30 characters;
  the cast runs 130–250), unchanged since the brief was written. It is the key
  art's whole stance line, and flinch ("broad composed stance"), entrance
  ("broad performance-ready stance"), sing and victory all refer back to a
  stance nothing defines. Fix: describe the stance.
- [med] No `face` field. The SCAIL video draws the set reference's face, so an
  under-specified face is carried by the key art alone. Fix: add `face`.
- [low] The boot lace detail in `outfit` ("three tone steps", "three readable
  rungs") will not survive the cell finish. Keep it for the key art; do not
  judge cells by it.
- [low] The palette now has near-white (eye white `#F3E8D8`, tooth highlight
  `#F4E9D7`). AGENTS.md's costume-bleed test says "Belter's palette holds no
  white"; the test still works on the lower 45% of the figure, but the sentence
  is no longer true.
- [low] `avoid` has "slim upright fashion-pose proportions" beside a `build` of
  "trim… lean". Dead text either way; `KEY_COMPLETE` already makes the key art
  follow the build.

### heavyweight — dance: `twist-03` [clip none]

The tailcoat fix (`6acd810`) held: the key art was redrawn five minutes after
it, matches the current bible, and is approved.

- [high] `twist-03` has no source clip, so a `--machine` build fails the dance.
  The SCAIL video in `work/heavyweight/` was drawn from the `club-01` drive, a
  `--motion` trial, not this brief. Fix: `uv run cag clips twist-03` (Mort
  uses it too).
- [med] Untested body type: the roster's largest mass driven by a slim
  performer's twist. Treat the first render as a test.
- [low] Palette: "dark" is not a material (the shoes, presumably), and "white"
  `#F1EEE5` and "shirt" `#F0EEE7` are one colour. Fix: name them shoes and
  waistcoat/bow tie, or merge.
- [low] `avoid` repeats itself — cane or weapon three times, top hat and
  monocle twice, comic obesity twice — and "Weapon or cane" is the one
  capitalised entry. Dead text, so tidying only; "No overcoat" is correctly
  stated in `outfit`.
- [low] No `hair`; `face` covers the bald crown. Fine.

### outlaw — dance: `country-01` [clip ok]

- [high] The bible never says she is a woman. Description, build, face, hair
  and outfit are all gender-neutral, and only the intents say "she". A lean,
  rangy 40-year-old in a cattleman hat will likely be drawn as a man in the key
  art. Fix: say so in `description`.
- [med] `build` ends "the leaner half of the roster beside Belter and
  Heavyweight". That sentence is quoted into every prompt, and the model knows
  neither character. Fix: remove the names.
- [med] `face` is written almost entirely in negatives ("no deep-set lines, no
  sagging…"), which invites the traits it lists. Fix: positive features.
- [med] The hat's identity is fine detail — raw-cut edge, pinch, knotted band
  with hanging ends, velvety nap. Only the curved-brim silhouette survives the
  cell finish.
- [low] "dark" and "white" palette entries match no material; belt colour
  unstated; hair colour only in the palette.
- [low] 18 `avoid` entries with duplicates ("firearm or lasso" beside "firearm,
  lasso, or weapon-like accessory"); "hat not tilted" lives only there.

### hype-man — dance: `hiphop-04` [clip none]

- [high] `build`, `performance_style` and `avoid` all demand an upright, level,
  never-crouched stance, but sing says "keep the low wide base… recover into
  the coiled stance" and victory "spring upward from the low base". Both sets
  render, so those frames will crouch. Fix: rewrite both to the upright stance.
  Ko ("coiled kinetic energy") and the three disabled sets say the same.
- [high] `hiphop-04` has no source clip. It also has airborne frames (10–13,
  17) under a wide puffer-jacket figure: untested body-type risk on SCAIL-2.
- [med] `performance_style` places no hand, no mic and no free hand, yet says
  "the energy is in… the arms".
- [med] Hair has no colour and no palette entry. The towel `#D8D4C8` sits next
  to white `#F1EEE5` and will merge with the tee; "cap" `#171A24` is next to
  "dark" `#20232B`.
- [low] The small folded towel cue will not survive the cell finish. "Rear
  shoulder" means nothing in a front view.
- [low] "Widest fighter", "like the rest of the roster": roster references in
  bible text. `avoid` repeats "narrow jacket".

### diva — dance: `shakira-01` [clip none]

- [high] `shakira-01` has no source clip.
- [med] Body-type risk: a hip-and-leg dance drives a floor-length column gown
  with a train, and every intent keeps the legs hidden. A human dancer's drive
  cannot move that hem believably.
- [med] Palette gives the earrings twice: "hoop earrings" silver `#6F747C` and
  "earring" gold `#C7A84B`. The outfit has one pair. Keep one, name the metal
  in `outfit`.
- [med] "Highlight bands stay neutral grey" and "no glossy gradient satin" live
  only in `avoid`, while `outfit` says "satin gown", which invites gloss. Fix:
  say "matte satin, flat tone bands" in `outfit`, the grey bands in `hair`.
- [low] Indigo-violet gown sits just under the defringe line (20 against 22);
  shrink and quantize could tip edge pixels over. See the defringe note above.
- [low] "dark" and "white" match no material. No `face`.

### idol — dance: `nicki-minaj-01` [clip none]

- [high] `nicki-minaj-01` has no source clip.
- [med] The description gives no gender ("young adult pop singer"); only "her
  mouth" inside `outfit` says it.
- [med] The headset boom is the only sing cue in a brief with no props, and it
  is a thin line the cell finish will drop.
- [med] Hair colour never stated (palette gives purple-black `#342A3F`); the
  skirt is "coordinated" and never named.
- [low] "A dark separating band inside the black outline" is pipeline language
  in `outfit`, and the finish already draws the outline.
- [low] "bone" `#E6E4DE` and "white" `#F1EEE5` near-duplicate; "dark" vague;
  "Merged ponytails" the one capitalised `avoid` entry.

### crooner — dance: `oldies-01` [clip none]

- [high] `oldies-01` has no source clip (chops uses it too).
- [med] No `face`. The intended expression lives only in `avoid` ("an exuberant
  grin"), which reaches no prompt. Fix: a `face` line.
- [low] "white" matches no material (the shirt is cream); "dark" is vague; the
  lapel satin and trouser stripe colours are unstated.
- [low] The undone bow tie cue is thin ribbon on cream; at risk in the finish.
- [low] `performance_style` never places the free hand; `description` (74
  characters) says nothing visual.

### screamer — dance: `shuffle-03` [clip none]

- [high] `shuffle-03` has no source clip.
- [med] Chest and arms are bare, and "no tattoos" lives only in `avoid`. Fix:
  "plain bare skin, no tattoos" in `outfit`.
- [low] ko is missing a full stop: "tip forward He holds".
- [low] "denim" `#202126` duplicates "dark" `#20232B`; "white" matches no
  material; patch colours unstated. `avoid` repeats the tattoo entry. No
  `face`. Knee tears and amber laces too small to survive.

### gospel — dance: prose

- [med] The dance intent says "the mic hand kept near the chest", but dance
  has no props and is drawn empty-handed. Naming an absent mic invites one.
  Fix: "the right hand".
- [med] Royal purple robe, the whole silhouette edge, passes the `tinted` test
  on every hex. Covered by the `costume` exemption; verify on the first render.
- [low] "Roster's broadest" competes with hype-man ("widest") and heavyweight
  ("largest mass"), and means nothing to the model. "white" should be "collar".

### rocker — dance: prose

- [med] Magenta platform boots and sash (every hex passes `tinted`) sit at the
  silhouette's bottom edge against a flat magenta backdrop. The finish's
  `costume` exemption covers the defringe; the cut-out itself is the open
  risk. Fix: shift boots and sash toward red or violet, or verify first.
- [med] The silver lightning streak across one eye — a recognition cue — is a
  few pixels wide at cell size.
- [low] "dark" matches no material.

### rudeboy — dance: prose

- [med] "Iridescent green-to-plum shot fabric" with two palette entries will
  likely render as blocks of green and plum, not a sheen. Fix: one dominant
  colour, the other named as highlight.
- [low] Plum highlight `#8A5E86` passes `tinted`; see the defringe note.
- [low] Skinny black tie and white socks are thin at cell size. Dance intent
  says "free hand" in an empty-handed set. "dark" vague.

### bebop — dance: prose

- [med] The recognition cues — round gold-wire spectacles, small polka-dot bow
  tie — are thin wire and tiny dots. Neither survives the finish. Fix: thicker
  frames, a solid or large-dot tie.
- [med] No palette for the gold spectacles or brown wingtips; "two-tone" never
  names its second tone.
- [low] Victory doffs the cap, but `hair` covers only "sides and back below the
  cap": the crown is undefined and the model will invent one.

### metalhead — dance: prose

- [med] The battle vest's identity is patches with stitched borders; at cell
  size it reads as blotchy blue. Keep the patches large and few.
- [low] No metal entry for studded wristbands and belt; grey strands have no
  entry. Victory's "horns gesture" sits oddly beside `avoid`'s "spikes, horns
  or chains". "Tallest, heaviest" competes with heavyweight (also 6'4").

### moody — dance: prose

- [med] The black-and-white stripes at hem and cuffs, a recognition cue, are
  thin stripes. Fix: bold bands, or drop as a cue.
- [low] dark `#20232B`, denim black `#2A2C33` and hair `#141418` are one
  colour; an all-black figure may flatten. The hoodie pocket that
  `performance_style` and entrance use is not in `outfit`.
- [low] Gender deliberately unstated, so the model picks; they/them used
  consistently.

### folkie — dance: prose

- [med] The "one thin braid" cue and the freckles will not survive the finish.
- [low] Oatmeal `#CDBB9A` and cream `#E6D9BC` close; hair `#A8672F` and rust
  `#A2482A` close. No palette for the brass ring (too small anyway). Dance says
  "free hand" in an empty-handed set.

### salsera — dance: prose

- [low] "Sombrero or any hat" lives only in `avoid`; `outfit` lacks "No
  headwear". Thin white trim on each tier will not survive. Red flower and red
  lips share one colour.

### default: patterns

- Only belter and outlaw can take the video path; crooner, diva, hype-man, idol
  and screamer need a clip backfill; seven have prose dances.
- "Free hand" or "mic hand" written into empty-handed sets (gospel, rudeboy,
  folkie, moody, metalhead dances). Say "left" or "right hand".
- Placeholder "dark" and "white" palette entries that match no material, or
  duplicate a named one: crooner, diva, idol, outlaw, screamer, rocker, rudeboy,
  heavyweight.
- Roster superlatives and cast names in bible text: outlaw names Belter and
  Heavyweight; gospel, hype-man, heavyweight and metalhead each claim to be
  the biggest.
- No `face`: belter, crooner, diva, hype-man, idol, screamer.
- Recognition cues that are fine detail the finish drops: spectacles, polka
  dots, braid, stripes, headset boom, lightning streak, skinny tie, towel,
  boot laces.
- Look constraints that live only in `avoid`: crooner's expression, diva's
  matte satin and grey highlights, screamer's tattoos, outlaw's hat tilt,
  salsera's no-hat.

## halloween

### deb — dance: `club-04` [clip ok]

The only halloween brief that can take the video path today.

- [med] Untested body type on the video path: sturdy, thick-waisted, knee dress,
  apron and cut-down jacket on the `club-04` drive.
- [med] `face` puts "grey lips shrunk back off long teeth in a permanent open
  grin" in the bible, against sing's mouth shapes and ko's "crestfallen".
  Fix: drop "permanent… grin".
- [low] The pigeon is in every set but dance, so the dance drifts from the
  character's identity.
- [low] Veins, hoop earrings, name tag, collar and ankle tape will not survive
  the finish; grey skin and the tangerine dress carry identity. "lash" has a
  palette entry but no lashes in `face`. 24 `avoid` lines.

### chops — dance: `oldies-01` [clip none]

- [high] No free hand. Top-level `props` are knife (held character-left) and
  mic (held character-right), but `performance_style` says "the free gloved
  character-right hand hanging open". The key art is told to hold a mic in a
  hand it is also told is empty. Fix: drop the mic from top-level `props`, or
  let the stance's right hand hold it.
- [high] Sing holds knife and mic, yet says "throw the free gloved
  character-right hand up with the fingers spread" — the mic hand. Expect a lost
  mic or swapped hands. Fix: "throw the mic hand up", or drop the knife.
- [high] `oldies-01` has no source clip.
- [med] `face` names what the mask is not ("no lacing holes, no triangular vent
  cutouts, no goalie-mask pattern"). In the bible that primes a goalie mask.
  Fix: describe only what the plate is.
- [med] "Masked slasher" in `description` and a knife with dried blood reach
  every prompt; every anti-blood line is in `avoid`.
- [med] Pushed-up sleeves show the forearms, with no skin colour anywhere.
  Fix: a skin line, or gauntlet-length gloves.
- [med] 6'4", very broad, heavy: untested body type.
- [low] A rigid plate's mouth slot cannot move; only head angle sells the sing.

### mort — dance: `twist-03` [clip none]

- [high] Three mouths. `face` (in the bible) says "grey lips shrunk back off long
  teeth in a permanent closed-mouth grin"; `performance_style` says "a
  closed-lip contented half-smile"; sing needs the mouth open wide. Fix: "grey
  lips drawn thin over long teeth", no "permanent".
- [high] `twist-03` has no source clip (shared with heavyweight).
- [high] 5'2" and "wider than he is tall through the middle": a human twist
  drive will re-proportion him. Untested.
- [low] Boutonniere (touched in entrance and victory), keys, handset and sleeve
  patch are fine detail; the patch has no palette entry. Dance is the one
  empty-handed set.

### babs — dance: `shakira-01` [clip none]

- [high] Body type: a floor-length column gown hiding both feet, a tall hair
  tower that must stay upright, and a "rigidly upright, bird-like" figure, on a
  hip-isolation drive. Also no clip. Fix: a front, low-hip bundle (twist,
  oldies), or a prose dance.
- [med] The look is the classic Bride silhouette — tower hair, white streaks,
  white gown, neck seam — and the only likeness guard ("any film's makeup
  design") is in `avoid`. Fix: state the differences positively.
- [med] "Two sharp white lightning-bolt streaks running from each temple" reads
  as four. Fix: "one… from each temple".
- [low] Neck seam and narrow silver sash will not survive; no palette for the
  brows.

### spot — dance: `party-01` [clip none]

- [high] Digitigrade wolf legs, bushy tail, long muzzle: a human drive gives
  human legs and an undriven tail.
- [low] `face` bakes in "a big open-mouthed friendly grin" against ko's
  "hangdog". Bandana `#C0302F` and plaid red `#B3262B` merge.

### frank — dance: `twist-01` [clip none]

- [med] 6'7" (the tallest), heavy-shouldered, "stiff", "knees nearly straight",
  on a twist drive that bends the knees hard. Untested; may overflow the cell.
- [med] "dark" `#20232B` names no material; trousers and "very heavy black
  boots" have no entry.
- [low] Wrist seam and electrode studs (cues) vanish at cell size. `face` bakes
  in a "hopeful, earnest expression".

### hex — dance: `twist-02` [clip none]

- [high] The recognition cue is purple-and-black striped stockings on
  stick-thin legs: thin stripes on thin legs will not survive the finish, and
  the purple `#7A4FB5` passes `tinted` right at the leg edge, where the
  `costume` exemption may not hold. Fix: wide bold stripes.
- [med] `build` says "a bent bony back held upright… never a crouch": mixed
  signal that drifts to a hunch. Fix: "a bony back held straight".
- [med] Sing asks for "the hooked nose in clear profile" in a front-view set.
- [low] Sash `#5B3A8E` passes `tinted` at the waist edge. Wart, stitched star
  and two crooked teeth are fine detail.

### brainz — dance: `lindey-hop-01` [clip none, loop]

- [med] `face` says "chin tipped down". In the bible, that pins the head in every
  frame, against victory's "tip the head back". Fix: move to
  `performance_style`.
- [med] `face` gives lipstick and grin only: no skin tone, eyes or brows,
  though the palette has them.
- [med] Thin black-and-violet tights stripes and corset lace trim will not
  survive; the violet `#7A3FC4` passes `tinted` at those edges.
- [med] A long lab coat that flares at the hem: the lindy drive moves the legs,
  not the flare the intents rely on.
- [low] "dark" vague. `avoid` bans lab equipment on her body while she wears
  goggles. `lindey-hop-01` is the only straight loop in the library.

### gorga — dance: prose

- [med] Hair is about a dozen small snakes with tiny eyes and tongues: a green
  blob after the finish, and identity rests on it. Fix: six or seven larger
  snakes.
- [low] Berry lips pass `tinted` (profile edge only); no palette for snake
  eyes, tongues or brows; the gown hides the gold sandals that have an entry;
  `face` bakes in "a warm open smile".

### sheets — dance: prose

- [high] The mic is held in a character-right fist, and the brief allows no
  visible hands, only "two small arm shapes" under the cloth. Expect a hand to
  appear or the mic to float. Fix: "the mic gripped through a fold of the sheet
  by the arm shape".
- [med] Recognition cues are a faint pale-blue pinstripe and a small star
  patch, both lost at 4'2" after the finish.
- [low] The key stance lifts the sheet edge near the mouth hole, which may hide
  the mouth sing needs.

### bayou — dance: prose

- [med] The hibiscus print (a cue) reads as turquoise with orange speckle after
  the finish. Fix: "big bold hibiscus blooms".
- [med] `face` bakes in "a big goofy grin" against ko's "bummed".
- [low] "Two-finger shaka wave" — a shaka is thumb and pinky. Gill slits, shell
  string and drips are fine detail.

### jack — dance: prose

- [med] `face` puts "a short curly green stem on top", and `outfit` puts a wide
  straw hat on top. The stem is hidden or pokes through.
- [low] Gingham, corduroy and plaid flatten in the finish. The hat reuses the
  straw entry.

### keen — dance: prose

- [med] Skin `#DCE0E4` and hair `#F2F4F6` are one colour: the pale face merges
  into the white mane. Fix: tint the skin.
- [low] "gown shade" overlaps "gown". Green braid thread, pink eye rims and
  brooch are fine detail. Ankle gown, trailing sleeves and waist-length mane
  rule out any future bundle.

### rattles — dance: prose

- [med] `description` (in the bible) says he "still combs his hair between every
  song", which invites a comb; the comb ban is only in `avoid`. Fix: drop the
  line.
- [low] Pinpoint eye glints and victory's "wink of one pinpoint glint" are
  invisible at cell size.

### vlad — dance: prose

- [low] Wine lining `#7A1E34` and brocade `#6B1A2C` merge; trousers and bow
  have no entry. Floor cape and tall collar rule out any future bundle.

### wraps — dance: prose

- [low] Bandage lines and trailing ends flatten to a tan mass; silhouette and
  turquoise eyes carry identity. "bandage shade" highlight equals "bandage"
  base.

### halloween: patterns

- Only deb can take the video path; seven name bundles with no clip; eight have
  prose dances.
- Worst body-type risks: mort (short and round), spot (digitigrade, tail), babs
  (floor gown, hair tower), frank and chops (very tall, heavy).
- `face` fields bake a fixed expression into the bible, fighting sing and ko in
  every set: deb, mort (both "permanent"), bayou, spot, gorga, frank, brainz.
  Move expressions to `performance_style`.
- Prop briefs (chops, deb, mort) carry their prop in every set but dance.
- Thin stripes and prints carry identity: hex, brainz, bayou, sheets, jack.
- [low] `location` shows what `avoid` bans — chops (woods, camp), deb (mirror
  ball), mort (enclosure rail), spot (full moon), hex (cauldron). `avoid` does
  not reach the location prompt, so the location wins; decide which was meant.

## sci-fi

All 16 have prose dances: no video path. Every `performance_style` is 159–221
characters.

### luma — dance: prose

- [high] Wrong mic hand. `performance_style` and sing both use "the free
  character-right hand", but `mic.json` holds the mic character-right. Fix:
  the free character-left hand.
- [high] A translucent body, bell and tendrils on a flat magenta backdrop:
  magenta shows through, and the cut-out and defringe read it as backdrop.
  Fix: an opaque jelly with a glow.
- [high] Dozens of thin tendrils (the cue) with pink tips (`#F07ACB` ramp
  passes `tinted`) on the silhouette edge: thin enough to fall under the
  `costume` exemption's bulk share.
- [med] "No clothing" on a slender humanoid invites nude anatomy or a refusal.
  Fix: "a smooth featureless jelly body".
- [med] Dance's "slow half-turn in place" needs body yaw, which AGENTS.md
  records as coming back square-on in every condition tried.

### oracle — dance: prose

- [high] `performance_style` raises both hands, palms up, fingers spread, with
  a mic in the key art: no hand holds it.
- [high] Belled sleeves "fall past the fingertips", against `KEY_ARMS` (every
  finger visible, never hidden by a sleeve), and hide the mic grip.
- [med] 6'5" in a floor-length pooled robe: rules out any bundle.
- [med] "Tiny white star points" (a cue) will not survive.

### doc — dance: prose

- [high] `performance_style` tucks the free hand in the lab-coat pocket;
  `KEY_ARMS` forbids a hand in a pocket, and the stance wins over it. Dance and
  entrance also start or end hand-in-pocket. Fix: hand on the lapel.
- [med] Wire-rimmed spectacles (a cue), pen and zip will not survive.
- [low] Plum jumpsuit passes `tinted` at the leg edges below the coat. Hair
  `#D6D8DC` and silver `#C8CCD2` merge against the white coat.

### glitch — dance: prose

- [high] "Sleeves bunched over the hands" and "sleeve-covered fingers" break
  `KEY_ARMS` and bury the mic grip. Fix: sleeves bunched at the wrist,
  fingers out.
- [med] Magenta streak (`#E0348C`, passes `tinted`) in spiky hair at the
  silhouette edge, small enough to miss the `costume` exemption. LED strip
  trim (a cue) too thin to survive.
- [low] `performance_style` never names the mic hand; visor frame has no
  palette entry.

### astro — dance: prose

- [high] A clear fishbowl helmet: magenta backdrop shows between head and rim,
  and the cut-out can punch holes or fill the gap. Fix: faintly tinted glass,
  solid behind the head.
- [med] Mic grey (`#6F747C`/`#B5BAC0`) against a silver suit (`#C8CDD4`):
  `mic.json` says the mic never merges into the costume.
- [med] Three round dials (a cue) become 1–2 px dots.

### nova — dance: prose

- [high] The top cue is thin cyan circuit-line tattoos, gone after the finish.
  Fix: bold glowing bands.
- [med] A full magenta palette ramp exists only for an eyeliner flick; in the
  bible it invites magenta on the costume.
- [med] Identity is asymmetric (one sleeve, an undercut side), which a written
  sheet's frames can mirror.
- [low] Purple highlight passes `tinted` at the trouser edge; tank, eyebrow
  ring and lips have no entry.

### trooper — dance: prose

- [high] Over the 8.1 colour bar in 6 of 15 frames in a scratch run, never
  drawn through `cag build`. Huge dome pauldrons plus the dance's windmill arm
  and headbang will collide or reshape. Fix: drop the windmill; trial first.
- [med] The bible says "helmet" twice in negation ("as if just out of a
  helmet", "No helmet anywhere"). Fix: "bare head, hair flattened".
- [med] "Space marine" pulls toward a rifle and unit markings; both only in
  `avoid`.

### tonnage — dance: prose

- [high] 7'6", gorilla proportions, arms to the knees, short piston legs: no
  human drive matches.
- [med] Steel `#6E7278` is the mic's `#6F747C`: the mic merges into the hands.
- [med] Faded hazard stripes (a cue) are thin diagonals.
- [low] Entrance's "shoulder vents" are not in `outfit`; its steam puff is
  semi-transparent over magenta.

### grax — dance: prose

- [high] 7'2", massive, digitigrade, a heavy floor-resting tail: same bulk class
  as trooper. Keep the prose dance; trial before any bundle.
- [med] The poncho hangs over the character-right shoulder, the mic arm.
- [med] "Bounty hunter" reaches every prompt; weapons, holster and cowboy hat
  are banned only in `avoid`.

### xyla — dance: prose

- [high] A stiff floor-length bell skirt hides the feet; the dance glides and
  curtseys.
- [med] Plates that "shift from emerald to violet to gold" invite frame-to-frame
  colour drift — the 8.1-bar failure.
- [med] A translucent amber wing collar over magenta; violets pass `tinted` at
  the skirt edge; thin antennae (a cue) will be lost.

### zorp — dance: prose

- [high] 3'6", head nearly as wide as the body, stubby limbs.
- [med] Thin antennae with pink bobs (`#FF6FB5`, passes `tinted`) on the top
  edge; both are cues.
- [low] Silver romper against the grey mic.

### beep — dance: prose

- [med] Boxy, top-heavy 4'8" robot with a cube head and tube arms.
- [med] "Built to serve drinks" reaches every prompt; "no tray" only in
  `avoid`. Fix: "empty clamp hands" in `build`.
- [med] Grey mic in chrome clamps (`#B9C0C8`).
- [low] Slatted mouth grille and push-buttons blur at cell size; sing depends
  on the grille.

### ace — dance: prose

- [med] The headset has "a small boom mic arm" — a second microphone, which
  `mic.json` forbids. Entrance flips it into place. Fix: ear cup, no boom.
- [med] "No numbers, insignia or emblem" only in `avoid`; a mecha pilot suit
  pulls toward them. Fix: "plain unmarked panels" in `outfit`.
- [low] "A full helmet" only in `avoid`. Crimson `#D0223A` and hair crimson
  `#B8202E` merge.

### captain — dance: prose

- [med] Chest insignia and franchise uniform only in `avoid`; a blue-and-gold
  starship tunic pulls straight to them. Fix: "a plain unadorned chest".
- [low] Gold piping, epaulette fringe and three cuff bands are thin. Ko's
  "epaulettes droop forward" contradicts rigid square epaulettes.

### rex — dance: prose

- [med] Mid-calf frock coat with flared skirts, plus a hornpipe hop.
- [med] The cyber-eye (top cue) is a few pixels; beard-braid rings vanish.
- [med] "Space pirate" reaches every prompt; cutlass, parrot, hook, peg leg and
  bandana are banned only in `avoid`.
- [low] Trousers, belt, grey streaks and natural eye have no entry.

### wren — dance: prose

- [med] "Cowboy hat or Stetson" only in `avoid`; a wide brim and frontier folk
  drift that way. Fix: "soft floppy canvas brim".
- [low] Thumb hooked in the belt hides it (`KEY_ARMS`). Victory takes the hat
  off in an empty-handed set, leaving the crown undefined.

### sci-fi: patterns

- Mic and hand rules broken: luma (mic hand), oracle (both hands open, sleeves
  past the fingertips), doc (pocket), glitch (sleeves over fingers), ace
  (second mic). Chops and sheets in halloween belong here too.
- Genre words in `description` — bounty hunter, space marine, pirate, starship
  captain, drinks robot — reach every prompt, while their counter-constraints
  sit in `avoid`.
- Translucent or clear parts over magenta: astro, luma, xyla.
- Grey or chrome costumes against the grey mic: astro, beep, tonnage, zorp.
- Most of these bodies (beep, grax, oracle, tonnage, trooper, xyla, zorp) would
  fit no human dancer's drive even with a bundle.

## Across all three themes

In order of what to fix first:

1. **Hand and prop contradictions** — chops, sheets, luma, oracle, doc,
   glitch, ace. Each tells the key art something it cannot draw, and every
   frame copies the key art.
2. **Bible text that fights the sets** — fixed expressions in `face` (deb,
   mort, bayou, spot, gorga, frank, brainz, crooner's grin in `avoid`), a pinned
   head pose (brainz), crouch language in an upright brief (hype-man), and
   missing gender (outlaw, idol).
3. **Constraints that live only in `avoid`** — hats, helmets, insignia, weapons,
   trays, tattoos, gloss. Restate positively where they matter.
4. **Negation in the bible** — "no goalie-mask pattern", "no deep-set lines",
   "No helmet anywhere", "No clothing": naming a thing invites it.
5. **Roster references in bible text** — outlaw names Belter and Heavyweight;
   "roster's broadest/widest/tallest/largest" in gospel, hype-man, metalhead,
   heavyweight, frank.
6. **"Free hand" in empty-handed sets** — say left or right hand.
7. **Clip backfill** for the 13 bundled briefs: `uv run cag clips --missing
   --cast`.
8. **Fine-detail recognition cues** — decide per brief whether to thicken or
   accept the loss.
9. **Palette tidying** — "dark" and "white" placeholders, near-duplicates,
   missing entries.

