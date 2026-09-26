"""Draw a traced set's poses as mannequins: flat figures in the character's own colours.

Qwen-Image-2.1 copies a reference image rather than re-posing it. Given the key art
and an OpenPose stick figure, a pose photograph or a mannequin as reference images, it
returned the key art unchanged or pasted the second picture over it (2026-09-25,
five renders). What it does follow is the latent it starts from. So a traced frame
reaches the model as a mannequin, VAE-encoded as the starting latent and sampled
at denoise 0.9, with the approved key art as its only reference image:

- the mannequin carries the pose, the figure's size and its place on the canvas,
  and, at 0.9, the colour layout: whatever colour a region is drawn here is the
  colour it comes back;
- the key art carries face, costume detail and the art style.

At 0.85 the frame stays a flat cartoon; at 0.92 and above it drifts back toward
the key art's own stance. Nothing in the mannequin belongs to the performer, so
nothing of the performer can leak into the frame.

The set is fitted once, not frame by frame: one scale from nose height to the
floor, shrunk if any frame's reach (hands overhead, a kick) would leave the
canvas, and one horizontal centre. The traced size and travel therefore survive
into the frames instead of being refitted per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .motion import MotionSheet

Point = tuple[float, float]
Colour = tuple[int, int, int]

MAGENTA: Colour = (255, 0, 255)
LINE: Colour = (0x1C, 0x13, 0x10)

#: Where the floor and the mean nose height sit on the canvas, as fractions of its
#: height. Measured off Belter's approved front reference at 1024x1536.
FLOOR = 0.947
NOSE = 0.117

#: A frame whose detail falls below this share of the reference's has fallen back
#: into the mannequin's flat shapes. Every good frame measured on 2026-09-25 scored
#: 0.58 or more and every flat one 0.55 or less. A flat frame is redrawn with the
#: next seed; on the bulkiest character tried, every frame had a good roll within
#: three seeds.
DETAIL_FLOOR = 0.57


@dataclass(frozen=True)
class Look:
    """The colours a mannequin is painted in, read off the character's reference.

    Read them off the picture, not the brief: a brief's white shins that the
    reference drew orange came back white in the frame.
    """

    jacket: Colour
    sleeve: Colour
    top: Colour
    legs: Colour
    shins: Colour
    boots: Colour
    skin: Colour
    hands: Colour
    cuffs: Colour
    hair: Colour
    belt: Colour
    #: Limb and torso thickness against a trim human figure.
    bulk: float = 1.0
    #: A large hair mass behind the head, rather than a short cap.
    hair_mass: bool = True
    #: An open jacket shows the top as a strip down the whole torso; a closed
    #: one as a plate over the upper chest.
    open_front: bool = True
    #: Colour for elbow and knee joints, where armour or a bodysuit shows them.
    joints: Colour | None = None
    #: The hips and groin, when the reference shows something other than the legs
    #: there. Trooper's dark bodysuit at the hips, painted plain orange, came back
    #: as a smooth orange suit in the hands-on-hips frames.
    pelvis: Colour | None = None


def fit(sheet: MotionSheet, size: tuple[int, int]) -> Callable[[list[float]], Point]:
    """One transform from the set's normalised landmarks to canvas pixels."""
    width, height = size
    floor_px, nose_px = height * FLOOR, height * NOSE
    frames = sheet.frames
    nose = sum(f.pts["nose"][1] for f in frames) / len(frames)
    scale = (floor_px - nose_px) / (sheet.floor_y - nose)
    reach = min(min(p[1] for p in f.pts.values()) for f in frames)
    crown = min(f.pts["nose"][1] for f in frames) - 0.12 * (sheet.floor_y - nose)
    top = min(reach, crown)
    if floor_px - (sheet.floor_y - top) * scale < 0.02 * height:
        scale = (floor_px - 0.02 * height) / (sheet.floor_y - top)
    hips = [(f.pts["hipL"][0] + f.pts["hipR"][0]) / 2 for f in frames]
    centre = sum(hips) / len(hips)
    return lambda q: (width / 2 + (q[0] - centre) * scale, floor_px + (q[1] - sheet.floor_y) * scale)


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _grow(poly: list[Point], k: float) -> list[Point]:
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    return [(cx + (x - cx) * k, cy + (y - cy) * k) for x, y in poly]


def _disc(d: ImageDraw.ImageDraw, c: Point, r: float, fill: Colour, edge: int = 3) -> None:
    d.ellipse([c[0] - r - edge, c[1] - r - edge, c[0] + r + edge, c[1] + r + edge], fill=LINE)
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=fill)


def _capsule(d: ImageDraw.ImageDraw, a: Point, b: Point, w: float, fill: Colour, edge: int = 3) -> None:
    length = max(1e-3, math.hypot(b[0] - a[0], b[1] - a[1]))
    nx, ny = -(b[1] - a[1]) / length * w / 2, (b[0] - a[0]) / length * w / 2
    body = [(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny), (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)]
    for end in (a, b):
        _disc(d, end, w / 2, fill, edge)
    d.polygon(body, fill=fill, outline=LINE, width=edge)
    for end in (a, b):
        d.ellipse([end[0] - w / 2, end[1] - w / 2, end[0] + w / 2, end[1] + w / 2], fill=fill)
    d.polygon(body, fill=fill)


def draw(pts: dict[str, list[float]], look: Look, size: tuple[int, int],
         xf: Callable[[list[float]], Point], unit: float) -> Image.Image:
    """One mannequin. `unit` is the figure's height in pixels; far limbs are drawn first."""
    img = Image.new("RGB", size, MAGENTA)
    d = ImageDraw.Draw(img)
    P = {k: xf(v) for k, v in pts.items()}
    z = {k: v[2] if len(v) > 2 else 0.0 for k, v in pts.items()}
    U, B = unit, look.bulk

    head = _lerp(_lerp(P["earL"], P["earR"], 0.5), P["nose"], 0.35)
    head_w, head_h = 0.062 * U, 0.082 * U
    neck = _lerp(P["shL"], P["shR"], 0.5)
    if look.hair_mass:
        hx, top, bottom = 0.095 * U, head[1] - 0.075 * U, head[1] + 0.13 * U
        d.ellipse([head[0] - hx - 3, top - 3, head[0] + hx + 3, bottom + 3], fill=LINE)
        d.ellipse([head[0] - hx, top, head[0] + hx, bottom], fill=look.hair)

    def leg(s: str) -> None:
        hip, knee, ankle = P[f"hip{s}"], P[f"kn{s}"], P[f"an{s}"]
        _capsule(d, hip, knee, 0.085 * U * B, look.legs)
        _capsule(d, knee, _lerp(knee, ankle, 0.75), 0.068 * U * B, look.shins)
        _capsule(d, _lerp(knee, ankle, 0.72), ankle, 0.062 * U * B, look.boots)
        _capsule(d, _lerp(P[f"heel{s}"], ankle, 0.3), P[f"toe{s}"], 0.055 * U * B, look.boots)
        if look.joints:
            _disc(d, knee, 0.03 * U * B, look.joints)

    def arm(s: str) -> None:
        elbow, wrist = P[f"el{s}"], P[f"wr{s}"]
        _capsule(d, P[f"sh{s}"], elbow, 0.062 * U * B, look.sleeve, 6)
        cuff = _lerp(elbow, wrist, 0.55)
        _capsule(d, elbow, cuff, 0.055 * U * B, look.sleeve, 6)
        _capsule(d, cuff, wrist, 0.038 * U * B, look.cuffs, 5)
        if f"index{s}" in P and f"pinky{s}" in P:
            hand = _lerp(wrist, _lerp(P[f"index{s}"], P[f"pinky{s}"], 0.5), 0.6)
        else:  # a trace without finger landmarks: carry on along the forearm
            hand = _lerp(elbow, wrist, 1.12)
        _disc(d, hand, 0.024 * U * (1 + (B - 1) * 0.5), look.hands, 5)

    for s in sorted("LR", key=lambda s: -(z[f"hip{s}"] + z[f"kn{s}"] + z[f"an{s}"])):
        leg(s)
    pelvis = _grow([P["hipL"], P["hipR"], _lerp(P["hipR"], P["knR"], 0.25), _lerp(P["hipL"], P["knL"], 0.25)], 1.35)
    d.polygon(pelvis, fill=look.pelvis or look.legs, outline=LINE, width=3)

    torso_z = (z["shL"] + z["shR"] + z["hipL"] + z["hipR"]) / 4
    arms = sorted("LR", key=lambda s: -(z[f"el{s}"] + z[f"wr{s}"]))
    behind = [s for s in arms if (z[f"el{s}"] + z[f"wr{s}"]) / 2 > torso_z + 0.02]
    for s in behind:
        arm(s)
    waist_l, waist_r = _lerp(P["hipL"], P["shL"], 0.08), _lerp(P["hipR"], P["shR"], 0.08)
    d.polygon(_grow([P["shL"], P["shR"], waist_r, waist_l], 1.12 * (1 + (B - 1) * 0.6)), fill=look.jacket, outline=LINE, width=3)
    inner = [_lerp(P["shL"], P["shR"], 0.3), _lerp(P["shR"], P["shL"], 0.3), _lerp(waist_r, waist_l, 0.3), _lerp(waist_l, waist_r, 0.3)]
    if look.open_front:
        d.polygon(inner, fill=look.top)
    else:
        d.polygon(_grow([inner[0], inner[1], _lerp(inner[1], inner[2], 0.45), _lerp(inner[0], inner[3], 0.45)], 1.5), fill=look.top)
    b0, b1 = _lerp(waist_l, P["hipL"], 0.2), _lerp(waist_r, P["hipR"], 0.2)
    _capsule(d, _lerp(b0, b1, -0.12), _lerp(b1, b0, -0.12), 0.022 * U, look.belt)
    _capsule(d, neck, _lerp(neck, head, 0.6), 0.04 * U, look.skin)
    d.ellipse([head[0] - head_w / 2 - 3, head[1] - head_h / 2 - 3, head[0] + head_w / 2 + 3, head[1] + head_h / 2 + 3], fill=LINE)
    d.ellipse([head[0] - head_w / 2, head[1] - head_h / 2, head[0] + head_w / 2, head[1] + head_h / 2], fill=look.skin)
    d.chord([head[0] - head_w * 0.62, head[1] - head_h * 0.62, head[0] + head_w * 0.62, head[1] + head_h * 0.2], 180, 360, fill=look.hair)
    if B > 1.2:  # shoulder plates on a bulky figure
        for s in "LR":
            r = 0.055 * U * B
            c = P[f"sh{s}"]
            d.ellipse([c[0] - r - 4, c[1] - r * 0.8 - 4, c[0] + r + 4, c[1] + r * 0.8 + 4], fill=LINE)
            d.ellipse([c[0] - r, c[1] - r * 0.8, c[0] + r, c[1] + r * 0.8], fill=look.jacket)
    for s in arms:
        if s not in behind:
            arm(s)
    return img


def draw_set(sheet: MotionSheet, look: Look, size: tuple[int, int] = (1024, 1536)) -> list[Image.Image]:
    """One mannequin per traced frame, all under the same fit."""
    xf = fit(sheet, size)
    nose = sum(f.pts["nose"][1] for f in sheet.frames) / len(sheet.frames)
    unit = (xf([0, sheet.floor_y])[1] - xf([0, nose])[1]) / 0.9  # nose to floor is ~0.9 of the figure
    return [draw(frame.pts, look, size, xf, unit) for frame in sheet.frames]


def _figure(image: Image.Image) -> np.ndarray:
    a = np.asarray(image.convert("RGB")).astype(int)
    return ~((a[..., 0] > 170) & (a[..., 1] < 110) & (a[..., 2] > 170))


def detail(image: Image.Image) -> float:
    """Mean edge strength inside the figure, kept clear of its silhouette edge."""
    backdrop = Image.fromarray((~_figure(image) * 255).astype(np.uint8))
    inside = ~np.asarray(backdrop.filter(ImageFilter.MaxFilter(9))).astype(bool)
    edges = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES)).astype(float)
    return float(edges[inside].mean()) if inside.any() else 0.0


def is_flat(frame: Image.Image, reference: Image.Image) -> bool:
    """True for a frame that came back as the mannequin's flat shapes: redraw it with another seed."""
    return detail(frame) < DETAIL_FLOOR * detail(reference)
