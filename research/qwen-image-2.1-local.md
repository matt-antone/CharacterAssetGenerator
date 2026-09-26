# Consistent animation frames from Qwen-Image-2.1, on a 16 GB local card

Research run 2026-09-25 on Comfy Cloud, using only core ComfyUI nodes and
settings a 16 GB RX 9070 can hold, to decide whether the card is worth keeping
before it arrives. 1,184 credits across 25 jobs.

Two methods work. **The hybrid (SCAIL-2 for motion, 2.1 for style) is the
stronger**: no rerolls on either character tried, natural motion, and it held
the bulky character that 2.1 alone struggled with. The 2.1-only method is one
model and one render a frame, and stays closer to the key art's own stance.

## Method 1: SCAIL-2 motion, 2.1 style (recommended)

1. **SCAIL-2** (`WanSCAILToVideo`, core ComfyUI; Wan 2.1 14B, **MIT licence**)
   draws the whole set as one 41-frame video at 576x864: the approved front
   reference plus its silhouette mask (blue on black), and the driving video
   plus its silhouette mask, in animation mode (`replacement_mode` false). DPO
   LoRA at 1.0, lightx2v I2V 480p rank-64 distill LoRA at 0.8, shift 5, 6 steps,
   CFG 1. The masks are drawn locally from the backdrop; no SAM3 needed.
2. The frames at the traced frames' timestamps are scaled to 1024x1536 and
   **restyled by Qwen-Image-2.1 in edit mode**: the SCAIL frame as `<image1>`,
   the key art as `<image2>`, "Redraw <image1> in exactly the art style of
   <image2> ... keep everything in <image1> ... match <image2>'s colours and
   every costume detail." 2.1 copies image 1, which is now already the character.

Results, club-01, 15 frames each, fp8 throughout:

- **Belter**: detail 0.80 to 0.91 of the reference, colour delta 0.8 to 3.3, the
  lowest of any method. Livelier, more natural poses than the mannequin method,
  and it caught the raised fist 2.1 alone kept dropping. The body is curvier
  than the key art (SCAIL-2's reading of the performer), and the face turns to
  the viewer rather than holding the key art's three-quarter turn.
- **Trooper**: every frame armoured and on-model first time, bulky proportions
  kept; detail 0.76 to 0.88. One frame has a small white patch on the chest.
- Driving it with the **real clip** gave better motion than driving it with the
  mannequin and showed no performer leakage (no blonde hair, no black leggings);
  the mannequin-driven video was stiffer.
- The video itself is a 41-frame, 16 fps animation, so in-betweens come free.
- Cost on the cloud: about 10 to 20 credits a video, about 6 a restyled frame.
- A restyle by denoising the SCAIL frame at 0.5 or 0.65 changes nothing; it
  has to be edit mode.

On 16 GB: SCAIL-2's fp8 file is 17.7 GB, so ComfyUI streams the overflow; the
GGUF builds (realrebelai/SCAIL-2_GGUF, Q5_K_M 12.3 GB, Q4_K_M 10.9 GB) fit and
load through ComfyUI-GGUF, already installed. Untested on the card.

The restyle is the only part under 2.1's research licence; Qwen-Image-Edit-2511
(Apache) could do it instead, untested here.

The workflow is `comfy/scail2-animate.json`.

## Method 2: 2.1 alone, mannequin start

Per frame, one render:

| input | what it is | what it carries |
| --- | --- | --- |
| reference image (`<image1>` in `TextEncodeQwenImage21`) | the approved front reference, the only image the model sees | face, costume detail, art style |
| starting latent (`VAEEncode`, denoise **0.9**) | a **mannequin**: the traced frame drawn as a flat figure in the character's own colours (`cag/mannequin.py`) | pose, size, place on the canvas, colour layout |
| prompt | `<Name> from <image1>, full body, dancing: <the frame's cue>. Wearing the full outfit from <image1>: the same face, <costume words>, colours and proportions. Nothing in <their> hands. <style>.` | little on its own; a short prompt without the cue did as well. But see below: one wrong word undoes it |

Canvas 1024x1536, the reference's own size; 40 steps (25 is usable); CFG 1,
euler/simple, one seed for the set. The workflow is
`comfy/qwen21-mannequin-pose.json`.

A frame that comes back flat (the mannequin's shapes with no drawn detail) is
caught by `mannequin.is_flat` and redrawn with the next seed.

## Method 2 results

- **Belter, club-01, 15 frames, bf16**: every frame recognisably the key art's
  Belter in the traced pose. Colour delta against the reference 0.6 to 4.7 (the
  identity bar in AGENTS.md is 8.1). Render centre within 7 px of the
  mannequin's in every frame: position and scale are ours, not the generator's.
- **Belter, hiphop-04, 19 frames, fp8**: identity held through a jump and
  arms-overhead frames, with the figure drawn smaller than the reference.
- **fp8 DiT (`weight_dtype fp8_e4m3fn`) + `qwen3vl_8b_fp8_scaled` text encoder**:
  indistinguishable from bf16 at 40 steps. This is the configuration that fits
  16 GB.
- **Second seed**: same identity, small pose differences; the one pose a seed
  dropped, the other kept.
- **Trooper** (bulky armour, a man where the brief says a woman: a key-art miss
  by 2.1, used here as a test reference only), club-01, 15 frames, fp8: the
  bulky silhouette is much harder. First pass 7 of 15 right, the rest a smooth
  suit or a tan bare body. Three changes brought every frame home within three
  seeds: the mannequin painted in the reference's colours rather than the
  brief's, the dark bodysuit painted at the hips as the reference shows it, and
  the prompt's "bare hands" (written for Belter's microphone) replaced with
  "nothing in his hands": for a gloved character "bare" asked for bare skin.
  A second pass over a flat frame, the frame itself as the starting latent,
  adds detail when its colours are right (0.55 to 0.73) and keeps them when
  they are wrong, so a wrong-coloured frame needs a new seed, not a second pass.

## What does not work, measured

| tried | outcome |
| --- | --- |
| key art + an OpenPose stick figure as reference images (5 renders, either order) | returns the key art unposed, the stick figure's lines painted over it |
| key art + mannequin both as reference images | returns the key art unposed, mannequin limbs pasted on |
| performer photo as a reference (earlier today) | returns the performer, pixelated |
| mannequin start, denoise 0.85 or 0.8 | stays a flat cartoon |
| mannequin start, denoise 0.92 or 0.95 | identity right, pose drifts back to the key art's stance |
| mannequin start with no reference image, character LoRA only | generic figures; the LoRA trained in-job did not take |
| 768x1152, 25 steps | identity right, arms fall back to the key art's stance |
| four mannequins on one canvas, 1024x1536 or full-size 2048x3072 | consistent figures, poses invented (back views at one seed, a flat cartoon at another); SpriteForge's idea does not carry to 2.1, and SpriteForge's own LoRA is not published |
| `cag/snap.py` on the frames | worsens the face; the frames are already on a pixel grid |
| MediaPipe landmarks on the reference, to read the mannequin's colours automatically | killed by the host at under 200 MB, sandboxed or not; cause unknown |

2.1 treats every reference image as something to preserve. It copies; it does
not re-pose. The only thing that moves a figure is the latent it starts from.

## The mannequin's colours come from the reference, not the brief

At 0.9 a region comes back the colour the mannequin painted it. Trooper's brief
names white shins and boots; the reference drew them orange; a mannequin in the
brief's colours brought back a white-legged bodysuit. Painted in the
reference's colours, with dark joints where the reference shows its bodysuit,
the same frames came back as the armour. Picking the colours is a look at the
reference, like approving key art, and it is the one per-character step the
method needs.

## For the RX 9070

- Fits: fp8 DiT about 7 GB, fp8 text encoder about 9 GB, not resident together;
  ComfyUI swaps them. 62 GB of RAM holds both.
- Avoid the `int8_convrot` files on gfx1201 (black images, very slow loads,
  open issues) and `--enable-dynamic-vram` with 2.1 (corrupt output after a
  reload, open issue). Pass `--async-offload`.
- Needs no LoRA, no training and no custom nodes. A character LoRA trained on
  the six near-identical approved images did not help and is not part of it.
- Estimated 60 to 105 s a frame at 1024x1536, 40 steps. Unmeasured.
- Whether ROCm accepts the card over this Thunderbolt 3 link is decided by the
  first boot: RDNA4 no longer needs PCIe atomics given compute firmware 2090 or
  later, and this machine has 3430. The test, in order: cold boot with the
  enclosure attached; `journalctl -k | grep -i -e atomic -e kfd` shows no
  "skipped device"; `rocminfo` lists gfx1201; a torch matmul and SDPA call run;
  ComfyUI draws SDXL, then this workflow at fp8. Return triggers: the card is
  skipped for atomics, core torch ops are refused, or memory faults persist
  with `amdgpu.cwsr_enable=0`, `HSA_ENABLE_SDMA=0` and
  `DEBUG_HIP_MEM_POOL_VMHEAP=0`.

## Licence

Qwen-Image-2.1 is under the Qwen Research License: research and evaluation
only. Anything shipped commercially from these frames needs a licence from Qwen.

## Open

- The mannequin's head always faces as the key art does; nothing here turns a
  face.
- 3/4-view traces (hiphop-05) draw poorly as mannequins; not tried.
- Frames are not yet wired into `cag build`: the Comfy backend lives on the
  `worktree-comfy-cloud` branch.

## Evidence

Downscaled from the cloud renders; the full-size frames were scratch and are not kept.

- `qwen-image-2.1-local/belter-club-01.jpg`: the reference, the 15 mannequins, and the 15 frames drawn from them (bf16).
- `qwen-image-2.1-local/belter-hiphop-04-fp8.jpg`: 19 frames in the 16 GB configuration.
- `qwen-image-2.1-local/trooper-club-01-best-roll.jpg`: the most detailed of up to three rolls per frame.
- `qwen-image-2.1-local/what-fails.jpg`: one frame from each approach in the table above.
- `qwen-image-2.1-local/hybrid-belter-club-01.jpg`, `hybrid-trooper-club-01.jpg`: SCAIL-2's frames at the traced timestamps, and the same frames after the 2.1 restyle.
- `qwen-image-2.1-local/joint-render-fails.jpg`: 2.1 drawing four frames in one render.
