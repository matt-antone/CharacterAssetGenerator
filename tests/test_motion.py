import json

import pytest

from cag.motion import MotionError, load_motion

SAMPLE = "tests/fixtures/sample-motion.json"


def sheet(tmp_path, **overrides):
    data = {
        "name": "dance",
        "fps": 4,
        "view": "front",
        "playback": "loop",
        "arc": "A two-step.",
        "frames": [
            {"i": 0, "role": "key", "pace": "fast", "cue": "Weight centred."},
            {"i": 1, "role": "inbetween", "pace": "steady", "cue": "Weight left."},
            {"i": 2, "role": "key", "pace": "steady", "cue": "Weight right."},
            {"i": 3, "role": "inbetween", "pace": "steady", "cue": "Recover."},
        ],
        **overrides,
    }
    path = tmp_path / "motion.json"
    path.write_text(json.dumps(data))
    return path


def test_loads_a_real_motionartist_sheet():
    motion = load_motion(SAMPLE)
    assert (motion.fps, len(motion.frames), motion.view) == (4, 16, "front")
    assert motion.loops
    assert [f.index for f in motion.locked] == [0, 2, 7, 9, 11, 13]


def test_neighbours_of_an_inbetween_are_the_frames_either_side(tmp_path):
    motion = load_motion(sheet(tmp_path))
    assert motion.neighbours(motion.frames[1]) == (0, 2)


def test_neighbours_wrap_across_a_clean_loop_seam(tmp_path):
    motion = load_motion(sheet(tmp_path))
    # Frame 3 has no locked frame after it, so it closes back onto frame 0.
    assert motion.neighbours(motion.frames[3]) == (2, 0)


def test_neighbours_of_a_one_shot_end_on_the_last_frame(tmp_path):
    motion = load_motion(sheet(tmp_path, playback="one-shot"))
    assert motion.neighbours(motion.frames[3]) == (2, 3)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"fps": None}, "missing fps"),
        ({"frames": []}, "no frames"),
        ({"missing_frames": [2]}, "gaps at frames"),
        (
            {"frames": [{"i": 0, "role": "inbetween", "cue": "x"}]},
            "no key or pilot frames",
        ),
        (
            {"frames": [{"i": 1, "role": "key", "cue": "x"}]},
            "numbered 0..0 in order",
        ),
    ],
)
def test_rejects_unusable_sheets(tmp_path, overrides, message):
    if overrides.get("fps", "keep") is None:
        data = json.loads(sheet(tmp_path).read_text())
        del data["fps"]
        path = tmp_path / "motion.json"
        path.write_text(json.dumps(data))
    else:
        path = sheet(tmp_path, **overrides)
    with pytest.raises(MotionError, match=message):
        load_motion(path)
