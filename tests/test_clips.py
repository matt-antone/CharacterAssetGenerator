import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from cag import clips
from cag.clips import ClipError, Probe, backfill, cut, fetch, recover_box, video_id
from cag.motion import clip_status, read_bundle, sha256_of

FPS = 24
#: A synthetic clip: 30 frames of 320x240, whose traced frames were cut through
#: BOX at clip frames TRACED and shrunk to THUMB, the way MotionArtist cuts them.
CLIP_START = 10.0
BOX = (97, 31, 132, 176)
TRACED = [6, 10, 14, 18, 22]
THUMB = (96, 128)


def texture(rng, h=240, w=320):
    noise = rng.integers(0, 256, (h // 4, w // 4), dtype=np.uint8)
    image = Image.fromarray(noise).resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(image.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float64)


def synthetic_clip(count=30, seed=0):
    """Grey frames sharing a background, each with its own half of the texture,
    so a traced frame matches its own clip frame far better than a neighbour."""
    rng = np.random.default_rng(seed)
    shared = texture(rng)
    return np.stack(
        [(0.5 * shared + 0.5 * texture(rng)).round().astype(np.uint8) for _ in range(count)]
    )


def traced_frame(frame, box=BOX):
    x, y, w, h = box
    return Image.fromarray(frame[y : y + h, x : x + w]).resize(THUMB, Image.Resampling.LANCZOS)


def traced_times(indices=TRACED):
    return [CLIP_START + j / FPS for j in indices]


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=F5WtrVMlp1I&pp=0gcJCS8MAYcqIYzv",
        "https://youtube.com/watch?feature=share&v=F5WtrVMlp1I",
        "https://m.youtube.com/watch?v=F5WtrVMlp1I",
        "https://youtu.be/F5WtrVMlp1I?t=12",
        "https://www.youtube.com/shorts/F5WtrVMlp1I?feature=share",
        "https://youtube.com/shorts/F5WtrVMlp1I/",
    ],
)
def test_video_id_reads_every_link_form(url):
    assert video_id(url) == "F5WtrVMlp1I"


@pytest.mark.parametrize(
    "url", ["https://vimeo.com/12345678901", "https://www.youtube.com/watch?list=x", "not a url"]
)
def test_video_id_refuses_a_link_with_no_id(url):
    with pytest.raises(ClipError, match="no YouTube video id"):
        video_id(url)


def test_recover_box_finds_the_box_the_traced_frames_were_cut_through():
    frames = synthetic_clip()
    thumbs = [traced_frame(frames[j]) for j in TRACED]

    box, t_offset, score = recover_box(frames, FPS, CLIP_START, traced_times(), thumbs)

    assert all(abs(got - want) <= 2 for got, want in zip(box, BOX)), box
    assert box[2] / box[3] == pytest.approx(0.75, abs=0.01)
    assert t_offset == 0
    assert score >= 0.95


def test_recover_box_reads_rgb_frames_the_same_as_grey():
    frames = synthetic_clip()
    rgb = np.repeat(frames[..., None], 3, axis=3)
    thumbs = [traced_frame(frames[j]) for j in TRACED]

    box, _, score = recover_box(rgb, FPS, CLIP_START, traced_times(), thumbs)

    assert all(abs(got - want) <= 2 for got, want in zip(box, BOX))
    assert score >= 0.95


def test_recover_box_detects_a_trace_one_native_frame_late():
    frames = synthetic_clip()
    # The tracer's seek landed a frame after each traced second.
    thumbs = [traced_frame(frames[j + 1]) for j in TRACED]

    box, t_offset, score = recover_box(frames, FPS, CLIP_START, traced_times(), thumbs)

    assert t_offset == pytest.approx(-1 / FPS)
    # Taking the offset off each traced time lands on the frame it was traced from.
    assert [round((t - CLIP_START - t_offset) * FPS) for t in traced_times()] == [
        j + 1 for j in TRACED
    ]
    assert score >= 0.95


def test_recover_box_keeps_the_box_inside_the_frame():
    frames = synthetic_clip()
    corner = (320 - 132, 240 - 176, 132, 176)
    thumbs = [traced_frame(frames[j], corner) for j in TRACED]

    box, _, _ = recover_box(frames, FPS, CLIP_START, traced_times(), thumbs)

    x, y, w, h = box
    assert x >= 0 and y >= 0 and x + w <= 320 and y + h <= 240
    assert all(abs(got - want) <= 2 for got, want in zip(box, corner))


@pytest.mark.parametrize(
    "size, box",
    [((854, 480), (246, 0, 360, 480)), ((1024, 576), (300, 0, 432, 576))],
)
def test_recover_box_reaches_a_full_height_box_where_the_coarse_steps_stop_short(size, box):
    # At 480p the tallest coarse box is 117 of 120 rows; the refine has to reach 480.
    width, height = size
    rng = np.random.default_rng(0)
    shared = texture(rng, height, width)
    frames = np.stack(
        [(0.5 * shared + 0.5 * texture(rng, height, width)).round().astype(np.uint8) for _ in range(30)]
    )
    thumbs = [traced_frame(frames[j], box) for j in TRACED]

    found, _, score = recover_box(frames, FPS, CLIP_START, traced_times(), thumbs)

    assert all(abs(got - want) <= 2 for got, want in zip(found, box)), found
    assert found[3] == height and score >= clips.MATCH


def test_recover_box_scores_shuffled_traced_frames_under_the_floor():
    frames = synthetic_clip()
    thumbs = [traced_frame(frames[j]) for j in reversed(TRACED)]

    _, _, score = recover_box(frames, FPS, CLIP_START, traced_times(), thumbs)

    assert score < clips.MATCH


def test_recover_box_refuses_a_clip_that_does_not_cover_the_trace():
    frames = synthetic_clip(count=12)
    thumbs = [traced_frame(frames[0]) for _ in TRACED]
    with pytest.raises(ClipError, match="does not cover"):
        recover_box(frames, FPS, CLIP_START, traced_times(), thumbs)


def ffprobe_json(duration, rate="24/1", average="24/1", frames=None, size=(1280, 720)):
    stream = {"width": size[0], "height": size[1], "r_frame_rate": rate, "avg_frame_rate": average}
    if frames is not None:
        stream["nb_frames"] = str(frames)
    return json.dumps({"streams": [stream], "format": {"duration": str(duration)}})


def fake_tools(monkeypatch, answers):
    """Stand in for yt-dlp, ffprobe and ffmpeg. `answers` maps a tool to a
    function of its command line; yt-dlp and ffmpeg write their output file."""
    calls = []

    def run(cmd, binary=False):
        calls.append(list(cmd))
        tool = cmd[0]
        if tool == "yt-dlp":
            Path(cmd[cmd.index("-o") + 1].replace("%(ext)s", "mp4")).write_bytes(b"video")
        if tool == "ffmpeg" and not binary:
            Path(cmd[-1]).write_bytes(b"cut")
        return answers.get(tool, lambda cmd: "")(cmd)

    monkeypatch.setattr(clips, "_run", run)
    return calls


def test_fetch_downloads_once_by_video_id(tmp_path, monkeypatch):
    calls = fake_tools(monkeypatch, {"ffprobe": lambda cmd: ffprobe_json(102.1)})
    url = "https://www.youtube.com/watch?v=P4QeqpsY8v8&pp=abc"

    first = fetch(url, 102.14, tmp_path)
    second = fetch(url, 102.14, tmp_path)

    assert first == second == tmp_path / "P4QeqpsY8v8.mp4"
    downloads = [c for c in calls if c[0] == "yt-dlp"]
    assert len(downloads) == 1
    assert downloads[0][1:5] == ["-f", clips.FORMAT, "-S", clips.SORT]


def test_fetch_refuses_a_video_whose_duration_is_not_the_traced_one(tmp_path, monkeypatch):
    fake_tools(monkeypatch, {"ffprobe": lambda cmd: ffprobe_json(101.9)})
    with pytest.raises(ClipError, match="not the cut of the video that was traced"):
        fetch("https://youtu.be/P4QeqpsY8v8", 102.14, tmp_path)


def test_fetch_refuses_a_variable_frame_rate(tmp_path, monkeypatch):
    fake_tools(monkeypatch, {"ffprobe": lambda cmd: ffprobe_json(102.14, "30/1", "2750/100")})
    with pytest.raises(ClipError, match="variable frame rate"):
        fetch("https://youtu.be/P4QeqpsY8v8", 102.14, tmp_path)


def test_cut_snaps_the_padded_window_onto_whole_frames(tmp_path, monkeypatch):
    src, out = tmp_path / "src.mp4", tmp_path / "cuts" / "club-01.mp4"

    def ffprobe(cmd):
        if cmd[-1] == str(src):
            return ffprobe_json(102.14, "24000/1001", "24000/1001", frames=2449)
        return ffprobe_json(3.5, "24000/1001", "24000/1001", frames=83)

    calls = fake_tools(monkeypatch, {"ffprobe": ffprobe})

    start, frames = cut(src, 77.035, 79.454, 0.5, out)

    # club-01's window: frames 1835 to 1917 of the source.
    assert frames == 83
    assert start == pytest.approx(1835 * 1001 / 24000)
    assert start <= 77.035 - 0.5 and start + (frames - 1) * 1001 / 24000 >= 79.454 + 0.5
    ffmpeg = next(c for c in calls if c[0] == "ffmpeg")
    for flag in (["-an"], ["-crf", "18"], ["-pix_fmt", "yuv420p"], ["-fps_mode", "cfr"]):
        at = ffmpeg.index(flag[0])
        assert ffmpeg[at : at + len(flag)] == flag
    assert ffmpeg[ffmpeg.index("-frames:v") + 1] == "83"
    assert ffmpeg[ffmpeg.index("-r") + 1] == "24000/1001"


def test_cut_clamps_the_pad_to_the_start_of_the_video(tmp_path, monkeypatch):
    src, out = tmp_path / "src.mp4", tmp_path / "out.mp4"

    def ffprobe(cmd):
        return ffprobe_json(27.4, frames=658 if cmd[-1] == str(src) else 73)

    fake_tools(monkeypatch, {"ffprobe": ffprobe})

    start, frames = cut(src, 0.0, 2.467, 0.5, out)

    assert (start, frames) == (0.0, 73)


def make_bundle(root, frames, indices=TRACED, source=True):
    """A bundle on disk whose traced frames were cut from `frames`."""
    (root / "thumbs").mkdir(parents=True)
    motion = {
        "name": "club-09",
        "fps": 6,
        "view": "front",
        "playback": "loop",
        "frames": [
            {"i": i, "t": t, "role": "key", "cue": ""} for i, t in enumerate(traced_times())
        ],
    }
    (root / "motion.json").write_text(json.dumps(motion, indent=1))
    files = {"motion.json": "0" * 64}
    for i, j in enumerate(indices):
        traced_frame(frames[j]).convert("RGB").save(root / "thumbs" / f"f{i:02d}.jpg", quality=95)
        files[f"thumbs/f{i:02d}.jpg"] = "0" * 64
    manifest = {
        "schema": "motion-artist/2",
        "name": "club-09",
        "fps": 6,
        "frame_count": len(indices),
        "view": "front",
        "source": {
            "url": "https://www.youtube.com/watch?v=P4QeqpsY8v8",
            "start": traced_times()[0],
            "end": traced_times()[-1],
            "duration": 102.14,
        },
        "files": files,
        "pose_grid": {"cols": 4},
    }
    if not source:
        del manifest["source"]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return root


def fake_video(monkeypatch, frames):
    """Stand in for the download, the cut and the decode, counting downloads."""
    fetched = []

    def fake_fetch(url, duration, cache):
        fetched.append(url)
        return Path(cache) / "P4QeqpsY8v8.mp4"

    def fake_cut(src, start, end, pad, out):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"the same cut every time")
        return CLIP_START, len(frames)

    monkeypatch.setattr(clips, "fetch", fake_fetch)
    monkeypatch.setattr(clips, "cut", fake_cut)
    monkeypatch.setattr(
        clips, "probe", lambda path: Probe(1.25, Fraction(FPS), (320, 240), len(frames), False)
    )
    monkeypatch.setattr(clips, "decode", lambda path, grey=False: frames)
    return fetched


def test_backfill_writes_the_clip_and_its_block_after_source(tmp_path, monkeypatch):
    frames = synthetic_clip()
    root = make_bundle(tmp_path / "club-09", frames)
    motion_before = (root / "motion.json").read_bytes()
    before = json.loads((root / "manifest.json").read_text())
    fake_video(monkeypatch, frames)

    done = backfill(root, cache=tmp_path / "sources")

    assert done.written and done.match >= clips.MATCH
    assert (root / "clip.mp4").read_bytes() == b"the same cut every time"
    after = json.loads((root / "manifest.json").read_text())
    keys = list(after)
    assert keys[keys.index("source") + 1] == "clip"
    assert [k for k in keys if k != "clip"] == list(before)
    assert after["files"] == before["files"]  # clip.mp4 is git-ignored, so never listed
    block = after["clip"]
    assert block["file"] == "clip.mp4" and block["backfilled_by"] == "cag"
    assert (block["start"], block["fps"], block["frame_count"]) == (CLIP_START, FPS, 30)
    assert block["size"] == [320, 240]
    assert all(abs(got - want) <= 2 for got, want in zip(block["box"], BOX))
    assert block["t_offset"] == 0
    assert block["sha256"] == sha256_of(root / "clip.mp4")
    assert (root / "motion.json").read_bytes() == motion_before
    # The block reads back as the bundle's clip, and covers every traced time.
    clip = read_bundle(root).clip
    assert clip is not None and clip.box == tuple(block["box"])
    assert read_bundle(root).load().clip == clip
    assert str(done).startswith("club-09 box=") and str(done).endswith(" OK")


def test_backfill_is_idempotent(tmp_path, monkeypatch):
    frames = synthetic_clip()
    root = make_bundle(tmp_path / "club-09", frames)
    fetched = fake_video(monkeypatch, frames)
    backfill(root, cache=tmp_path / "sources")
    written = (root / "manifest.json").read_bytes()

    again = backfill(root, cache=tmp_path / "sources")

    assert not again.written
    assert len(fetched) == 1  # a clip that matches its hash is not rebuilt
    assert (root / "manifest.json").read_bytes() == written

    # A fresh clone: the block is committed and the clip is not.
    (root / "clip.mp4").unlink()
    rebuilt = backfill(root, cache=tmp_path / "sources")

    assert rebuilt.written and (root / "clip.mp4").exists()
    assert (root / "manifest.json").read_bytes() == written


def test_a_fresh_clone_on_another_x264_keeps_the_committed_block(tmp_path, monkeypatch):
    frames = synthetic_clip()
    root = make_bundle(tmp_path / "club-09", frames)
    fake_video(monkeypatch, frames)
    backfill(root, cache=tmp_path / "sources")
    committed = (root / "manifest.json").read_bytes()
    first_sha = json.loads(committed)["clip"]["sha256"]

    # Another machine: the clone has the block, and its x264 writes other bytes.
    (root / "clip.mp4").unlink()
    (root / "clip.sha256").unlink()

    def other_x264(src, start, end, pad, out):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"the same frames, another x264 build")
        return CLIP_START, len(frames)

    monkeypatch.setattr(clips, "cut", other_x264)
    rebuilt = backfill(root, cache=tmp_path / "sources")

    assert rebuilt.written
    assert (root / "manifest.json").read_bytes() == committed, "a tracked file is not rewritten"
    bundle = read_bundle(root)
    assert bundle.clip is not None and bundle.clip.sha256 != first_sha
    assert bundle.clip.sha256 == sha256_of(root / "clip.mp4")
    assert not backfill(root, cache=tmp_path / "sources").written, "and it is idempotent there"


def test_a_stale_clip_is_cut_again(tmp_path, monkeypatch):
    frames = synthetic_clip()
    root = make_bundle(tmp_path / "club-09", frames)
    fake_video(monkeypatch, frames)
    backfill(root, cache=tmp_path / "sources")
    (root / "clip.mp4").write_bytes(b"half a download")
    assert clip_status(read_bundle(root)) == "stale"
    assert backfill(root, cache=tmp_path / "sources").written
    assert clip_status(read_bundle(root)) == "ok"


def test_backfill_dry_run_writes_nothing(tmp_path, monkeypatch):
    frames = synthetic_clip()
    root = make_bundle(tmp_path / "club-09", frames)
    manifest = (root / "manifest.json").read_bytes()
    fake_video(monkeypatch, frames)

    done = backfill(root, cache=tmp_path / "sources", dry_run=True)

    assert not done.written and done.match >= clips.MATCH
    assert not (root / "clip.mp4").exists()
    assert (root / "manifest.json").read_bytes() == manifest


def test_backfill_refuses_shuffled_traced_frames(tmp_path, monkeypatch):
    frames = synthetic_clip()
    root = make_bundle(tmp_path / "club-09", frames, indices=list(reversed(TRACED)))
    manifest = (root / "manifest.json").read_bytes()
    fake_video(monkeypatch, frames)

    with pytest.raises(ClipError, match="under 0.9"):
        backfill(root, cache=tmp_path / "sources")

    assert not (root / "clip.mp4").exists()
    assert (root / "manifest.json").read_bytes() == manifest


def test_backfill_refuses_a_bundle_with_no_source(tmp_path, monkeypatch):
    frames = synthetic_clip()
    root = make_bundle(tmp_path / "club-09", frames, source=False)
    fake_video(monkeypatch, frames)
    with pytest.raises(ClipError, match="records no source"):
        backfill(root, cache=tmp_path / "sources")


def test_backfill_refuses_a_motion_sheet_with_no_traced_times(tmp_path, monkeypatch):
    frames = synthetic_clip()
    root = make_bundle(tmp_path / "club-09", frames)
    motion = json.loads((root / "motion.json").read_text())
    del motion["frames"][2]["t"]
    (root / "motion.json").write_text(json.dumps(motion))
    fake_video(monkeypatch, frames)
    with pytest.raises(ClipError, match="does not record the second"):
        backfill(root, cache=tmp_path / "sources")
