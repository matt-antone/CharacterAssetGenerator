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


# What the tracer ships beside its clip: a clip frame per traced frame, the
# bundle mask and the head boxes.


def with_clip_frames(root, picks):
    spec = json.loads((root / "motion.json").read_text())
    for frame, pick in zip(spec["frames"], picks):
        if pick is None:
            frame.pop("clip_frame", None)
        else:
            frame["clip_frame"] = pick
    (root / "motion.json").write_text(json.dumps(spec))
    return root


def test_clip_frames_give_each_traced_frame_its_exact_second(tmp_path):
    """`t` is a rounded second; the clip frame is the frame itself."""
    picks = [12 + 6 * i + (i % 3 == 1) for i in range(16)]
    # Each `t` is a hair off the frame's own second, the way a seek by millisecond is.
    times = [15.5 + pick / 24 + 0.013 for pick in picks]
    root = with_clip_frames(timed_bundle(tmp_path, clip={"t_offset": 0.004}, times=times), picks)
    motion = read_bundle(root).load()
    assert [f.clip_frame for f in motion.frames] == picks
    assert motion.clip_frames == tuple(picks)
    assert motion.frame_times == pytest.approx([15.5 + 0.004 + pick / 24 for pick in picks])
    assert [motion.clip.frame_at(t) for t in motion.frame_times] == picks


def test_frame_times_fall_back_to_t_without_every_clip_frame_or_a_clip(tmp_path):
    root = with_clip_frames(timed_bundle(tmp_path, clip={}), [12 + 6 * i for i in range(15)] + [None])
    motion = read_bundle(root).load()
    assert motion.clip_frames == () and motion.frame_times[0] == 16.0
    unclipped = with_clip_frames(timed_bundle(tmp_path / "b"), [12 + 6 * i for i in range(16)])
    motion = read_bundle(unclipped).load()
    assert motion.clip is None and motion.frame_times[0] == 16.0, "no clip to count frames in"


def test_clip_frames_outside_the_clip_or_out_of_order_are_refused(tmp_path):
    picks = [12 + 6 * i for i in range(16)]
    root = with_clip_frames(timed_bundle(tmp_path, clip={}), [*picks[:-1], 109])
    with pytest.raises(MotionError, match=r"does not cover the clip frames of traced frames \[15\]"):
        read_bundle(root).load()
    root = with_clip_frames(timed_bundle(tmp_path / "b", clip={}), [picks[1], picks[0], *picks[2:]])
    with pytest.raises(MotionError, match="clip frames are not strictly increasing"):
        read_bundle(root).load()


@pytest.mark.parametrize("pick", [-1, 2.5, True, "x"])
def test_a_clip_frame_that_is_not_a_frame_number_is_refused(tmp_path, pick):
    root = timed_bundle(tmp_path)
    spec = json.loads((root / "motion.json").read_text())
    spec["frames"][2]["clip_frame"] = pick
    (root / "motion.json").write_text(json.dumps(spec))
    with pytest.raises(MotionError, match="a whole number from 0"):
        load_motion(root / "motion.json")


def test_clip_frames_do_not_change_the_motion_digest(tmp_path):
    from cag.animation import motion_digest

    plain = read_bundle(timed_bundle(tmp_path / "a", clip={})).load()
    picked = read_bundle(
        with_clip_frames(timed_bundle(tmp_path / "b", clip={}), [12 + 6 * i for i in range(16)])
    ).load()
    assert motion_digest(picked) == motion_digest(plain)


def shipped_bundle(tmp_path, mask=b"white on black", heads=b"[null]", **block):
    """A tracer's bundle: its clip block declares a bundle mask and head boxes."""
    root = timed_bundle(tmp_path, clip={"backfilled_by": None, "box": [0, 0, 1280, 720], **block})
    manifest = json.loads((root / "manifest.json").read_text())
    del manifest["clip"]["backfilled_by"]
    for kind, name, data in (("mask", "mask.mp4", mask), ("heads", "heads.json", heads)):
        manifest["clip"][kind] = {"file": name, "sha256": hashlib.sha256(data or b"").hexdigest()}
        if data is not None:
            (root / name).write_bytes(data)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_a_tracers_mask_and_head_boxes_load_onto_the_clip(tmp_path):
    from cag.motion import part_status

    bundle = read_bundle(shipped_bundle(tmp_path))
    clip = bundle.load().clip
    assert clip.from_tracer and not bundle.declared_clip.backfilled_by
    assert clip.mask.path == tmp_path / "shuffle" / "mask.mp4"
    assert clip.heads.sha256 == hashlib.sha256(b"[null]").hexdigest()
    assert clip.problem == ""
    assert (part_status(bundle, "mask"), part_status(bundle, "heads")) == ("ok", "ok")
    assert not read_bundle(timed_bundle(tmp_path / "cag", clip={})).clip.from_tracer


def test_a_missing_mask_is_absent_not_an_error(tmp_path):
    from cag.motion import part_status

    bundle = read_bundle(shipped_bundle(tmp_path, mask=None))
    assert bundle.clip.mask is None and bundle.clip.heads is not None
    assert bundle.clip.problem == "" and clip_status(bundle) == "ok"
    assert part_status(bundle, "mask") == "missing"
    assert bundle.declared_clip.mask.path.name == "mask.mp4"


def test_a_stale_mask_is_dropped_and_only_the_video_path_hears_of_it(tmp_path):
    from cag.motion import part_status

    root = shipped_bundle(tmp_path)
    (root / "heads.json").write_text("[[1, 2, 3, 4]]")
    bundle = read_bundle(root)
    assert clip_status(bundle) == "ok", "the clip itself is fine"
    assert bundle.clip.heads is None and bundle.clip.mask is not None
    assert "declares heads heads.json" in bundle.clip.problem
    assert "cag motions pull" in bundle.clip.problem
    assert part_status(bundle, "heads") == "stale"
    assert bundle.load().clip.problem == bundle.clip.problem


def test_a_mask_block_with_no_file_is_refused(tmp_path):
    root = shipped_bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["clip"]["mask"] = {"sha256": "ab"}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(MotionError, match="mask names no file"):
        read_bundle(root)
