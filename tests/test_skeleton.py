import json
import statistics
from pathlib import Path

import numpy
import pytest
from PIL import Image

from cag.skeleton import (
    FAR_INK,
    SIZE,
    crown,
    pose_box,
    pose_extent,
    skeleton,
    stature,
    write_skeletons,
)

SAMPLE = Path("tests/fixtures/sample-motion.json")
MOTION = json.loads(SAMPLE.read_text())
POSES = [frame["pts"] for frame in MOTION["frames"]]


def test_box_covers_every_pose_and_the_floor():
    x0, y0, width, height = pose_box(POSES, MOTION["floor_y"])
    assert y0 + height >= MOTION["floor_y"]
    for pose in POSES:
        for x, y, *_ in pose.values():
            assert x0 <= x <= x0 + width and y0 <= y <= y0 + height


def test_every_frame_renders_a_distinct_pose(tmp_path):
    paths = write_skeletons(POSES, MOTION["floor_y"], MOTION["body_h"], tmp_path)
    assert len(paths) == len(POSES)
    digests = {Image.open(path).tobytes() for path in paths}
    assert len(digests) == len(POSES)  # no two frames drew the same figure


def test_the_figure_is_black_ink_on_white_at_cell_size():
    image = skeleton(POSES[0], pose_box(POSES, MOTION["floor_y"]), MOTION["floor_y"], MOTION["body_h"])
    assert image.size == SIZE
    pixels = numpy.array(image.convert("L"))
    assert pixels.max() == 255  # white ground
    assert (pixels < 40).sum() > 2000  # a substantial amount of ink


def test_one_shared_scale_keeps_a_crouch_shorter_than_a_stand(tmp_path):
    """Poses are drawn against the set's bounds, not each fitted to the frame."""
    box = pose_box(POSES, MOTION["floor_y"])
    heights = []
    for pose in (POSES[0], POSES[7]):
        ink = numpy.array(skeleton(pose, box, MOTION["floor_y"], MOTION["body_h"]).convert("L")) < 40
        rows = numpy.nonzero(ink.any(axis=1))[0]
        heights.append(rows.max() - rows.min())
    assert heights[0] != heights[1]


#: A plain standing skeleton, and the same body with its knees folded. Every
#: segment keeps its length between the two: 40px shin, 40px thigh, 40px torso.
BODY_H = 140
FLOOR_Y = 182
STAND = {
    "anL": [48, 180], "anR": [52, 180],
    "knL": [48, 140], "knR": [52, 140],
    "hipL": [48, 100], "hipR": [52, 100],
    "shL": [46, 60], "shR": [54, 60],
    "elL": [44, 85], "elR": [56, 85],
    "wrL": [43, 105], "wrR": [57, 105],
    "earL": [48, 45], "earR": [52, 45], "nose": [50, 47],
}
CROUCH = {
    "anL": [48, 180], "anR": [52, 180],
    "knL": [72, 148], "knR": [76, 148],
    "hipL": [48, 116], "hipR": [52, 116],
    "shL": [46, 76], "shR": [54, 76],
    "elL": [44, 101], "elR": [56, 101],
    "wrL": [43, 121], "wrR": [57, 121],
    "earL": [48, 61], "earR": [52, 61], "nose": [50, 63],
}


def test_bending_a_knee_folds_the_leg_without_shortening_it():
    """The whole point of measuring along bones instead of down a bounding box."""
    assert stature(CROUCH, BODY_H) == pytest.approx(stature(STAND, BODY_H))


def test_the_same_crouch_does_shrink_the_vertical_extent():
    """What a bounding box would have measured, and why it misleads."""
    assert pose_extent(CROUCH, FLOOR_Y, BODY_H) < pose_extent(STAND, FLOOR_Y, BODY_H)


def test_the_crown_leans_with_the_head_rather_than_staying_overhead():
    tilted = dict(STAND, earL=[63, 52], earR=[67, 52], nose=[69, 54])
    upright, leaning = crown(STAND, BODY_H), crown(tilted, BODY_H)
    assert upright[0] == pytest.approx(50)  # squarely above the neck
    assert leaning[0] > upright[0] + 5  # carried sideways by the tilt
    assert leaning[1] > upright[1]  # and, being off the vertical, lower


def test_stature_tracks_extent_across_a_real_traced_set():
    """The landmarks are 2D, so a limb angled at the camera foreshortens and
    stature alone wobbles by ~9%. `frame_scale` divides it by the extent, and
    that ratio is several times steadier, because both shrink together."""
    statures = [stature(pose, MOTION["body_h"]) for pose in POSES]
    ratios = [
        stature(pose, MOTION["body_h"]) / pose_extent(pose, MOTION["floor_y"], MOTION["body_h"])
        for pose in POSES
    ]
    spread = lambda xs: (max(xs) - min(xs)) / statistics.median(xs)
    assert spread(statures) > 0.05  # the raw measure really does move about
    assert spread(ratios) < 0.05  # the one the scale is built on does not


def test_a_far_limb_is_drawn_grey_and_a_2d_trace_is_unchanged():
    """z on the landmarks says which crossed leg is behind; without it, nothing moves."""
    box, floor, body = pose_box(POSES, MOTION["floor_y"]), MOTION["floor_y"], MOTION["body_h"]
    flat = skeleton(POSES[0], box, floor, body)
    deep = {name: [*point, 1.0 if name.endswith("L") else -1.0] for name, point in POSES[0].items()}
    greys = numpy.array(skeleton(deep, box, floor, body).convert("L"))
    assert (numpy.abs(greys.astype(int) - FAR_INK[0]) < 10).sum() > 200  # the character-left side
    assert numpy.array_equal(numpy.array(flat), numpy.array(skeleton(POSES[0], box, floor, body)))
