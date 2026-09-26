import hashlib
import json
from pathlib import Path
import shutil

from dataclasses import replace

import pytest

from cag.motion import Frame, clip_status, library, read_bundle, MotionError, load_motion

SAMPLE = "motions/sample/motion.json"


def motion_sheet(tmp_path, **overrides):
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
    motion = load_motion(motion_sheet(tmp_path))
    assert motion.neighbours(motion.frames[1]) == (0, 2)


def test_neighbours_wrap_across_a_clean_loop_seam(tmp_path):
    motion = load_motion(motion_sheet(tmp_path))
    # Frame 3 has no locked frame after it, so it closes back onto frame 0.
    assert motion.neighbours(motion.frames[3]) == (2, 0)


def test_neighbours_of_a_one_shot_end_on_the_last_frame(tmp_path):
    motion = load_motion(motion_sheet(tmp_path, playback="one-shot"))
    assert motion.neighbours(motion.frames[3]) == (2, 3)


def test_a_pingpong_is_not_a_loop_and_needs_no_seam(tmp_path):
    motion = load_motion(motion_sheet(tmp_path, playback="pingpong", seam="needs blend"))
    assert motion.playback == "pingpong"
    assert not motion.loops  # it turns around on its last frame, it does not cut back
    assert motion.seams_cleanly, "a pingpong has no seam to flag"


def test_the_spellings_of_a_one_shot_land_on_one_word(tmp_path):
    for spelling in ("once", "one-shot", "oneshot", "One-Shot"):
        assert load_motion(motion_sheet(tmp_path, playback=spelling)).playback == "once"


def test_a_pingpong_arrives_as_the_flag_beside_the_playback(tmp_path):
    """MotionArtist writes `playback: loop` plus `pingpong: true`; cag reads one word."""
    sheet = motion_sheet(tmp_path, playback="loop", pingpong=True, seam="needs blend")
    motion = load_motion(sheet)
    assert motion.playback == "pingpong"
    assert not motion.loops
    assert motion.seams_cleanly, "the return leg reverses the out leg; there is no jump"


def test_motionartists_own_playback_words_are_taken_as_written(tmp_path):
    for spelling, expected in (("loop", "loop"), ("one-shot", "once"), ("final-hold", "once")):
        assert load_motion(motion_sheet(tmp_path, playback=spelling)).playback == expected


def test_a_playback_nobody_can_play_is_refused(tmp_path):
    with pytest.raises(MotionError, match="unknown playback"):
        load_motion(motion_sheet(tmp_path, playback="pingpang"))


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
        data = json.loads(motion_sheet(tmp_path).read_text())
        del data["fps"]
        path = tmp_path / "motion.json"
        path.write_text(json.dumps(data))
    else:
        path = motion_sheet(tmp_path, **overrides)
    with pytest.raises(MotionError, match=message):
        load_motion(path)


def test_a_frames_pace_is_said_out_loud():
    """Traced pace, or the settle the performer did averages into an in-between."""
    def frame(pace):
        return Frame(index=0, role="key", pace=pace, cue="Weight centred.", note="")

    assert "settles here" in frame("hold").instruction
    assert "fast transition" in frame("fast").instruction
    steady = frame("steady").instruction
    assert steady == "Weight centred.", f"a steady frame says only its cue: {steady!r}"

    # and the cue and note survive whatever the pace
    assert "Weight centred." in frame("hold").instruction


def test_a_sheet_that_does_not_loop_cleanly_says_so():
    sheet = load_motion(SAMPLE)
    assert replace(sheet, seam="needs blend").seams_cleanly is False
    assert replace(sheet, seam="clean").seams_cleanly is True
    assert replace(sheet, seam="").seams_cleanly is True, "unset is not a complaint"
    assert replace(sheet, seam="needs blend", playback="oneshot").seams_cleanly is True


def bundle_at(tmp_path, **overrides):
    sheet = json.loads(Path(SAMPLE).read_text())
    root = tmp_path / "shuffle"
    root.mkdir(parents=True, exist_ok=True)
    (root / "motion.json").write_text(json.dumps(sheet))
    manifest = {
        "bundle": "motion-source", "schema": "motion-artist/1", "name": "shuffle",
        "title": "Shuffle", "fps": sheet["fps"], "frame_count": len(sheet["frames"]),
        "playback": sheet["playback"], "view": sheet["view"], "seam": "clean",
        "files": {"motion.json": "unchecked"},
    }
    manifest.update(overrides)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_a_manifest_is_never_asked_how_a_set_plays(tmp_path):
    """It records what was traced. The rule is the motion spec's, written down once."""
    root = bundle_at(tmp_path)
    header = json.loads((root / "manifest.json").read_text())
    del header["playback"]
    (root / "manifest.json").write_text(json.dumps(header))

    assert read_bundle(root).load().playback == "loop"  # off the sheet, not the header


def test_a_bundle_is_found_by_its_manifest(tmp_path):
    """Not by guessing at a filename the bundle never promised."""
    bundle_at(tmp_path)
    found = library(tmp_path)
    assert list(found) == ["shuffle"]
    assert found["shuffle"].sheet.name == "motion.json"
    assert found["shuffle"].title == "Shuffle"


def test_a_bundle_a_capture_session_nested_is_still_found(tmp_path):
    """A session installs its clips as `club-01/club-01-4/`, one level deeper."""
    bundle_at(tmp_path / "club-01")

    assert list(library(tmp_path)) == ["shuffle"], "a one-level scan saw none of them"


def test_a_directory_without_a_manifest_is_not_a_bundle(tmp_path):
    (tmp_path / "loose").mkdir()
    shutil.copy(SAMPLE, tmp_path / "loose" / "motion.json")
    assert library(tmp_path) == {}


def test_an_unknown_bundle_layout_is_refused(tmp_path):
    bundle_at(tmp_path, schema="motion-artist/9")
    with pytest.raises(MotionError, match="schema"):
        library(tmp_path)


def test_a_manifest_that_disagrees_with_its_sheet_is_caught(tmp_path):
    """The header a brief picks a sheet by has to be the sheet it gets."""
    root = bundle_at(tmp_path, frame_count=99)
    with pytest.raises(MotionError, match="advertises 99 frames"):
        read_bundle(root).load()


def test_a_v2_bundle_loads_and_keeps_its_depth(tmp_path):
    """motion-artist/2 only adds a z per landmark; the reader takes it in its stride."""
    root = bundle_at(tmp_path, schema="motion-artist/2")
    sheet = json.loads((root / "motion.json").read_text())
    for frame in sheet["frames"]:
        frame["pts"] = {name: [*point, -0.5] for name, point in frame["pts"].items()}
    (root / "motion.json").write_text(json.dumps(sheet))
    motion = read_bundle(root).load()
    assert all(len(point) == 3 for point in motion.frames[0].pts.values())


def timed_bundle(tmp_path, clip=None, times=None, footage=b"not really an mp4"):
    """A bundle whose frames carry traced seconds, with a clip block when given one.

    The sample's frames sit at 16.00-19.75 s, a quarter second apart.
    """
    root = bundle_at(tmp_path)
    spec = json.loads((root / "motion.json").read_text())
    for frame, t in zip(spec["frames"], times or [16.0 + 0.25 * i for i in range(16)]):
        frame["t"] = t
    (root / "motion.json").write_text(json.dumps(spec))
    if clip is not None:
        if footage is not None:
            (root / "clip.mp4").write_bytes(footage)
        block = {
            "file": "clip.mp4", "start": 15.5, "fps": 24.0, "frame_count": 109,
            "size": [1280, 720], "box": [400, 40, 480, 640], "t_offset": 0.0,
            "sha256": hashlib.sha256(footage or b"").hexdigest(),
            "backfilled_by": "cag", **clip,
        }
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["clip"] = block
        (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_a_frames_traced_second_is_read(tmp_path):
    motion = load_motion(timed_bundle(tmp_path) / "motion.json")
    assert motion.frames[1].t == 16.25
    assert motion.frame_times[0] == 16.0 and len(motion.frame_times) == 16


def test_frame_times_are_all_or_nothing(tmp_path):
    """A partial set would pair a frame with the wrong moment of the clip."""
    root = timed_bundle(tmp_path)
    spec = json.loads((root / "motion.json").read_text())
    del spec["frames"][5]["t"]
    (root / "motion.json").write_text(json.dumps(spec))
    motion = load_motion(root / "motion.json")
    assert motion.frames[4].t == 17.0
    assert motion.frame_times == ()


def test_the_sample_carries_no_clip():
    bundle = read_bundle("motions/sample")
    assert bundle.clip is None and bundle.declared_clip is None
    assert bundle.load().clip is None
    assert clip_status(bundle) == "none"


def test_a_clip_block_loads_onto_the_motion(tmp_path):
    bundle = read_bundle(timed_bundle(tmp_path, clip={}))
    assert clip_status(bundle) == "ok"
    clip = bundle.load().clip
    assert clip == bundle.clip
    assert clip.path == tmp_path / "shuffle" / "clip.mp4"
    assert (clip.size, clip.box) == ((1280, 720), (400, 40, 480, 640))
    assert clip.end == pytest.approx(15.5 + 108 / 24)
    assert clip.frame_at(16.0) == 12 and clip.frame_at(19.75) == 102


def test_a_declared_clip_that_is_not_on_disk_is_missing_not_an_error(tmp_path):
    """clip.mp4 is untracked: a fresh clone has the block and not the file."""
    bundle = read_bundle(timed_bundle(tmp_path, clip={}, footage=None))
    assert bundle.clip is None
    assert bundle.declared_clip.path.name == "clip.mp4"
    assert clip_status(bundle) == "missing"
    assert bundle.load().clip is None


def test_a_clip_that_is_not_the_one_declared_is_stale_and_only_the_video_path_refuses_it(tmp_path):
    root = timed_bundle(tmp_path, clip={})
    (root / "clip.mp4").write_bytes(b"a different cut")
    bundle = read_bundle(root)
    assert bundle.clip is None and clip_status(bundle) == "stale"
    assert "cag clips shuffle" in bundle.clip_problem
    motion = bundle.load()
    assert motion.clip is None and motion.clip_problem == bundle.clip_problem


def test_one_stale_clip_does_not_stop_the_rest_of_the_library(tmp_path):
    root = timed_bundle(tmp_path, clip={})
    (root / "clip.mp4").write_bytes(b"a different cut")
    shutil.copytree("motions/sample", tmp_path / "sample")
    found = library(tmp_path)
    assert sorted(found) == ["sample", "shuffle"]
    assert clip_status(found["shuffle"]) == "stale" and found["sample"].load()


def test_a_clip_this_machine_cut_is_its_own_even_when_the_block_is_anothers(tmp_path):
    """x264 writes its build into every file: the same footage cut elsewhere is other bytes."""
    root = timed_bundle(tmp_path, clip={})
    (root / "clip.mp4").write_bytes(b"the same frames, another x264")
    (root / "clip.sha256").write_text(hashlib.sha256(b"the same frames, another x264").hexdigest() + "\n")
    bundle = read_bundle(root)
    assert clip_status(bundle) == "ok"
    assert bundle.clip.sha256 == hashlib.sha256(b"the same frames, another x264").hexdigest(), (
        "a drive is keyed on the bytes on disk"
    )


@pytest.mark.parametrize(
    "block, message",
    [
        ({"box": [900, 40, 480, 640]}, "does not lie inside"),
        ({"fps": 0}, "at 0.0 fps"),
    ],
)
def test_a_clip_block_that_makes_no_sense_is_refused(tmp_path, block, message):
    with pytest.raises(MotionError, match=message):
        read_bundle(timed_bundle(tmp_path, clip=block))


def test_a_clip_block_missing_a_field_is_refused(tmp_path):
    root = timed_bundle(tmp_path, clip={})
    manifest = json.loads((root / "manifest.json").read_text())
    del manifest["clip"]["start"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(MotionError, match="clip block is missing start"):
        read_bundle(root)


def test_a_clip_that_does_not_span_the_trace_is_refused(tmp_path):
    """Frame 15 is at 19.75 s; a clip ending at 19.0 s never shows it."""
    bundle = read_bundle(timed_bundle(tmp_path, clip={"frame_count": 85}))
    with pytest.raises(MotionError, match=r"does not cover traced frames \[13, 14, 15\]"):
        bundle.load()


def test_traced_times_that_go_backwards_are_refused(tmp_path):
    times = [16.0 + 0.25 * i for i in range(16)]
    times[3], times[4] = times[4], times[3]
    bundle = read_bundle(timed_bundle(tmp_path, clip={}, times=times))
    with pytest.raises(MotionError, match="not strictly increasing"):
        bundle.load()


def test_a_clip_does_not_change_the_motion_digest(tmp_path):
    """The stamp every drawn set carries; a clip arriving must not stale them all."""
    from cag.animation import motion_digest

    plain = read_bundle(timed_bundle(tmp_path / "a")).load()
    clipped = read_bundle(timed_bundle(tmp_path / "b", clip={})).load()
    assert clipped.clip is not None
    assert motion_digest(clipped) == motion_digest(plain)
