import json
from pathlib import Path

import numpy
from PIL import Image

from cag.skeleton import SIZE, pose_box, skeleton, write_skeletons

SAMPLE = Path("/Users/matthewantone/Development/MotionArtist/work/sample/motion.json")
MOTION = json.loads(SAMPLE.read_text())
POSES = [frame["pts"] for frame in MOTION["frames"]]


def test_box_covers_every_pose_and_the_floor():
    x0, y0, width, height = pose_box(POSES, MOTION["floor_y"])
    assert y0 + height >= MOTION["floor_y"]
    for pose in POSES:
        for x, y in pose.values():
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
