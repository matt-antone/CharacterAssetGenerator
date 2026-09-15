"""Draw the motion sheet's pose as a picture.

A frame cue is a paragraph, and an image generator will quietly flatten a
paragraph back towards a neutral standing pose. The same pose as a skeleton is
not negotiable in the same way, so every frame is drawn with one attached.

The landmarks and bone list come from MotionArtist, which traces them with
MediaPipe. `L`/`R` are the performer's own sides — character-left and
character-right — never screen sides.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

BONES = (
    ("hipL", "knL"), ("knL", "anL"), ("anL", "toeL"),
    ("hipR", "knR"), ("knR", "anR"), ("anR", "toeR"),
    ("hipL", "hipR"), ("shL", "shR"),
    ("shL", "elL"), ("elL", "wrL"),
    ("shR", "elR"), ("elR", "wrR"),
)

#: Head radius, in body_h units. The drawn circle and the crown agree on it.
HEAD_RADIUS = 0.07

#: Ankle joint height above the floor, in body_h units. MediaPipe gives no heel
#: landmark, so the last span from ankle down to the ground is assumed.
# ponytail: anthropometric average; tune per rig if a character's feet read wrong.
ANKLE_RISE = 0.039

SIZE = (480, 560)
MARGIN = 40
INK = (0, 0, 0)
FLOOR_INK = (170, 170, 170)
BONE_WIDTH = 9
SPINE_WIDTH = 11


def mid(a: list[float], b: list[float]) -> list[float]:
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]


def span(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def crown(pts: dict[str, list[float]], body_h: float) -> list[float]:
    """Top of the skull: one head radius past the ears, along the neck's own axis.

    Not straight up. A tilted head carries its crown sideways, and measuring to a
    point directly above the ears would lose exactly the height the tilt costs.
    """
    ear = mid(pts["earL"], pts["earR"])
    neck = mid(pts["shL"], pts["shR"])
    dx, dy = ear[0] - neck[0], ear[1] - neck[1]
    reach = span(ear, neck)
    if reach < 1e-9:
        return [ear[0], ear[1] - HEAD_RADIUS * body_h]
    step = HEAD_RADIUS * body_h / reach
    return [ear[0] + dx * step, ear[1] + dy * step]


def stature(pts: dict[str, list[float]], body_h: float) -> float:
    """Heel-to-crown measured along the bones, so the pose cannot change it.

    Summing segment lengths is what makes this pose-invariant. A vertical extent
    shortens the moment a character crouches or leans, and normalising on one
    inflates them back to full height; the leg is the same leg either way.

    The supporting leg is measured, the torso down the midline, and the two are
    added rather than walked through — a chain that detoured via the supporting
    hip would pick up half a pelvis width that is not part of anyone's height.
    """
    lower = "L" if pts["anL"][1] > pts["anR"][1] else "R"
    ankle, knee, hip = pts[f"an{lower}"], pts[f"kn{lower}"], pts[f"hip{lower}"]
    leg = ANKLE_RISE * body_h + span(ankle, knee) + span(knee, hip)
    torso = span(mid(pts["hipL"], pts["hipR"]), mid(pts["shL"], pts["shR"]))
    return leg + torso + span(mid(pts["shL"], pts["shR"]), crown(pts, body_h))


def pose_extent(pts: dict[str, list[float]], floor_y: float, body_h: float) -> float:
    """Vertical span of one pose, crown and floor included: what gets drawn."""
    ys = [point[1] for point in pts.values()] + [floor_y, crown(pts, body_h)[1]]
    return max(ys) - min(ys)


def pose_box(frames: list[dict], floor_y: float) -> tuple[float, float, float, float]:
    """Bounds covering every pose in the set, so the skeleton never rescales."""
    xs = [point[0] for frame in frames for point in frame.values()]
    ys = [point[1] for frame in frames for point in frame.values()] + [floor_y]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def skeleton(
    pts: dict[str, list[float]],
    box: tuple[float, float, float, float],
    floor_y: float,
    body_h: float,
) -> Image.Image:
    """One pose, black on white, at a scale shared by the whole set."""
    x0, y0, width, height = box
    scale = (SIZE[1] - 2 * MARGIN) / height
    offset_x = (SIZE[0] - width * scale) / 2

    def place(point: list[float]) -> tuple[float, float]:
        return ((point[0] - x0) * scale + offset_x, (point[1] - y0) * scale + MARGIN)

    image = Image.new("RGB", SIZE, (255, 255, 255))
    pen = ImageDraw.Draw(image)

    floor = place([x0, floor_y])[1]
    pen.line([(0, floor), (SIZE[0], floor)], fill=FLOOR_INK, width=3)

    for a, b in BONES:
        pen.line([place(pts[a]), place(pts[b])], fill=INK, width=BONE_WIDTH, joint="curve")
    pen.line(
        [place(mid(pts["hipL"], pts["hipR"])), place(mid(pts["shL"], pts["shR"]))],
        fill=INK,
        width=SPINE_WIDTH,
    )

    head = place(mid(pts["earL"], pts["earR"]))
    radius = HEAD_RADIUS * body_h * scale
    pen.ellipse(
        [head[0] - radius, head[1] - radius, head[0] + radius, head[1] + radius],
        outline=INK,
        width=BONE_WIDTH,
    )
    # A stub from the head centre towards the nose: vertical reads as facing
    # front, horizontal as full profile, so the drawing carries the head's turn.
    nose = place(pts["nose"])
    ear_span = max(abs(place(pts["earR"])[0] - place(pts["earL"])[0]), 1e-6)
    dx = max(-radius, min(radius, (nose[0] - head[0]) / ear_span * 2 * radius))
    dy = (radius**2 - dx**2) ** 0.5
    pen.line([head, (head[0] + dx, head[1] + dy)], fill=INK, width=5)

    # Hollow markers on the character-right wrist and ankle, so which side is
    # which survives the trip through the image generator.
    for name in ("wrR", "anR"):
        x, y = place(pts[name])
        pen.ellipse([x - 11, y - 11, x + 11, y + 11], fill=(255, 255, 255), outline=INK, width=5)

    return image


def write_skeletons(
    frames: list[dict], floor_y: float, body_h: float, out_dir: Path | str
) -> list[Path]:
    """Render one skeleton per frame into `out_dir`, numbered in frame order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    box = pose_box(frames, floor_y)
    paths = []
    for index, pts in enumerate(frames):
        path = out_dir / f"{index:02d}.png"
        skeleton(pts, box, floor_y, body_h).save(path, format="PNG")
        paths.append(path)
    return paths
