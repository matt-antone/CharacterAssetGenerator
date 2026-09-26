import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest
from PIL import Image

from cag import drive
from cag.drive import (
    DRIVE_MASK,
    DRIVE_RECORD,
    DRIVE_VIDEO,
    MASK_FIGURE,
    REF_MASK_BACKGROUND,
    DriveError,
    build_drive,
    check_masks,
    cut,
    decode,
    drive_box,
    figure_mask,
    fit,
    light_backdrop,
    mask_stats,
    plan,
    read_apng,
    reference_mask,
    source_frames,
    trace_index,
    write_apng,
)
from cag.machines import load_machine
from cag.motion import Clip


def machine(**overrides):
    fields = dict(name="cloud", rate=16, max_length=81, width=576, height=864)
    fields.update(overrides)
    return SimpleNamespace(**fields)


CLOUD = machine()
SMOKE4 = machine(name="smoke4", max_length=9, width=256, height=384)

#: club-01's traced frame times, as the manifest steps them.
CLUB_01 = [77.035 + 0.17279 * i for i in range(15)]
#: `work/research-2026-09-25/renders/scail-inputs/trace-index.json`.
CLUB_01_INDEX = [0, 3, 6, 8, 11, 14, 17, 19, 22, 25, 28, 30, 33, 36, 39]


def test_club_01_reproduces_the_research_trace_index():
    rate, length = plan(CLUB_01, CLOUD, "club-01")
    assert (rate, length) == (16.0, 41)
    assert trace_index(CLUB_01, rate, length) == CLUB_01_INDEX


def test_the_shipped_profiles_plan_club_01():
    assert plan(CLUB_01, load_machine("cloud"), "club-01") == (16.0, 41)
    assert plan(CLUB_01, load_machine("smoke4"), "club-01")[1] == 9


@pytest.mark.parametrize(
    "span, length",
    [(0.25, 5), (0.26, 9), (2.5, 41), (2.51, 45), (1.0 / 16, 5)],
)
def test_length_rounds_up_to_4k_plus_1(span, length):
    assert plan([0.0, span], CLOUD) == (16.0, length)


def test_a_span_past_81_frames_is_refused_by_name():
    with pytest.raises(DriveError, match=r"club-09.*needs 82.*chunking is not built"):
        plan([0.0, 81 / 16], CLOUD, "club-09")


def test_smoke4_caps_the_length_by_lowering_the_rate(capsys):
    rate, length = plan(CLUB_01, SMOKE4, "club-01")
    assert length == 9
    assert rate == pytest.approx(8 / (CLUB_01[-1] - CLUB_01[0]))
    index = trace_index(CLUB_01, rate, length, capped=True)
    assert index[0] == 0 and index[-1] == 8
    assert "repeat" in capsys.readouterr().err


def test_a_repeat_at_the_profiles_rate_is_refused():
    with pytest.raises(DriveError, match="share a SCAIL frame"):
        trace_index([0.0, 0.01, 1.0], 16.0, 17)


def test_source_frames_never_run_past_the_clip():
    clip = Clip(Path("c.mp4"), start=10.0, fps=24, frame_count=48, size=(96, 64), box=None)
    assert source_frames(clip, 10.5, 16.0, 17, 48) == [
        round((0.5 + k / 16) * 24) for k in range(17)
    ]
    with pytest.raises(DriveError, match="clip pad too short"):
        source_frames(clip, 10.0, 16.0, 41, 48)
    late = Clip(Path("c.mp4"), start=10.6, fps=24, frame_count=48, size=(96, 64), box=None)
    with pytest.raises(DriveError, match="clip pad too short"):
        source_frames(late, 10.5, 16.0, 17, 48)


def test_write_apng_keeps_every_identical_frame(tmp_path):
    frame = Image.new("RGB", (8, 12), (255, 0, 255))
    path = write_apng([frame] * 5, tmp_path / "same.png", 62)
    frames = read_apng(path)
    assert len(frames) == 5
    for back in frames:
        assert (back[1:] == (255, 0, 255)).all()
        assert tuple(back[0, 0][:2]) == (255, 0) and back[0, 0][2] in (254, 255)


@pytest.mark.parametrize(
    "size, box, expected",
    [
        # Landscape: full height, centred across on the clip box.
        ((1280, 720), (600, 100, 300, 400), (510, 0, 480, 720)),
        ((1280, 720), (0, 100, 100, 400), (0, 0, 480, 720)),
        ((1280, 720), (1200, 100, 80, 400), (800, 0, 480, 720)),
        ((1920, 1080), None, (600, 0, 720, 1080)),
        # Portrait: full width, centred down on the clip box.
        ((720, 1280), (100, 300, 500, 666), (0, 93, 720, 1080)),
        ((720, 1280), (100, 0, 500, 200), (0, 0, 720, 1080)),
        ((720, 1280), (100, 1100, 500, 180), (0, 200, 720, 1080)),
        # Taller than the frame: covers it and runs past the ends, split by the box.
        ((720, 960), (100, 200, 500, 560), (0, -60, 720, 1080)),
        ((720, 960), (100, 0, 500, 400), (0, -120, 720, 1080)),
    ],
)
def test_drive_box(size, box, expected):
    assert drive_box(size, box) == expected
    assert expected[2] * 3 == expected[3] * 2


def test_cut_pads_past_the_frame_and_says_where():
    frame = numpy.full((10, 6, 3), 50, numpy.uint8)
    piece, pad = cut(frame, (0, -2, 6, 14), (9, 9, 9))
    assert piece.shape == (14, 6, 3)
    assert (piece[:2] == 9).all() and (piece[-2:] == 9).all() and (piece[2:12] == 50).all()
    assert pad[:2].all() and pad[-2:].all() and not pad[2:12].any()


def studio(size=(32, 48), box=(10, 10, 20, 38), backdrop=(250, 250, 250), figure=(30, 20, 40)):
    frame = numpy.empty((size[1], size[0], 3), numpy.uint8)
    frame[:] = backdrop
    left, top, right, bottom = box
    frame[top:bottom, left:right] = figure
    return frame


def test_a_white_studio_is_a_light_backdrop_and_thresholds_to_the_figure():
    frames = [studio(box=(10 + k, 10, 20 + k, 38)) for k in range(4)]
    assert light_backdrop(frames)
    masks = [figure_mask(frame) for frame in frames]
    assert masks[0].sum() == 10 * 28
    assert check_masks(masks) == []


def test_a_dark_room_is_not_a_light_backdrop():
    assert not light_backdrop([studio(backdrop=(40, 30, 30), figure=(250, 250, 250))])


def test_padding_is_never_figure():
    frame = studio()
    pad = numpy.zeros(frame.shape[:2], bool)
    pad[:5] = True
    frame[:5] = 0
    assert not figure_mask(frame, pad)[:5].any()


def test_check_masks_flags_an_empty_frame_and_two_people():
    one = figure_mask(studio())
    empty = numpy.zeros_like(one)
    two = figure_mask(studio(box=(2, 10, 12, 38))) | figure_mask(studio(box=(20, 10, 30, 38)))
    problems = check_masks([one, empty, one, two])
    assert any(p.startswith("frame 01") and "covers 0%" in p for p in problems)
    assert any(p.startswith("frame 03") and "largest region is 50%" in p for p in problems)
    assert not any(p.startswith("frame 00") for p in problems)
    assert mask_stats([two])[0]["largest"] == 0.5


def test_check_masks_flags_a_silhouette_that_jumps():
    left = figure_mask(studio(box=(0, 10, 10, 38)))
    right = figure_mask(studio(box=(20, 10, 30, 38)))
    assert any("overlaps the frame before" in p for p in check_masks([left, right]))


def test_reference_mask_is_blue_on_black():
    ref = Image.new("RGB", (20, 30), (255, 0, 255))
    ref.paste((200, 150, 90), (5, 5, 15, 25))
    mask = numpy.asarray(reference_mask(ref))
    assert tuple(mask[0, 0]) == REF_MASK_BACKGROUND
    assert tuple(mask[10, 10]) == MASK_FIGURE
    assert (mask.reshape(-1, 3) == MASK_FIGURE).all(axis=1).sum() == 10 * 20


def test_fit_pads_a_square_reference_with_magenta_before_resizing(capsys):
    ref = Image.new("RGB", (100, 100), (0, 200, 0))
    out = numpy.asarray(fit(ref, 60, 90))
    assert out.shape == (90, 60, 3)
    assert tuple(out[0, 30]) == (255, 0, 255) and tuple(out[-1, 30]) == (255, 0, 255)
    assert tuple(out[45, 30]) == (0, 200, 0)
    assert "padded with magenta" in capsys.readouterr().err


def test_fit_leaves_a_2_by_3_reference_unpadded(capsys):
    assert fit(Image.new("RGB", (1024, 1536), (1, 2, 3)), 576, 864).size == (576, 864)
    assert capsys.readouterr().err == ""


def test_decode_reads_a_ppm_stream(monkeypatch):
    frames = [numpy.full((2, 3, 3), k, numpy.uint8) for k in range(3)]
    stream = b"".join(b"P6\n3 2\n255\n" + f.tobytes() for f in frames)
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, stream, b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = decode(Path("clip.mp4"))
    assert out.shape == (3, 2, 3, 3)
    assert [int(f[0, 0, 0]) for f in out] == [0, 1, 2]
    assert "passthrough" in seen[0]


# A 96x64 source clip at 24 fps, whose performer moves a pixel a frame; its
# traced frames sit 0.25 s apart from 10.5 s, so the drive is 17 frames at 16.
TIMES = [10.5 + 0.25 * i for i in range(5)]
SMALL = machine(name="small", width=32, height=48)


def native(backdrop=(250, 250, 250), figure=(30, 20, 40), count=48):
    frames = numpy.empty((count, 64, 96, 3), numpy.uint8)
    frames[:] = backdrop
    for j in range(count):
        x = 40 + j % 4
        frames[j, 12:52, x : x + 16] = figure
    return frames


def bundle(tmp_path, clip_start=10.0):
    clip = Clip(
        tmp_path / "clip.mp4", start=clip_start, fps=24, frame_count=48, size=(96, 64),
        box=(30, 4, 36, 48), sha256="ab" * 32,
    )
    return SimpleNamespace(name="club-99", clip=clip, frame_times=tuple(TIMES))


def mask_pass_writing(calls, size=(32, 48), figure=(10, 10, 22, 40)):
    """A mask pass that paints one steady figure, and counts its jobs."""

    def mask_pass(video, into, length):
        calls.append(video)
        assert len(read_apng(video)) == length
        into.mkdir(parents=True, exist_ok=True)
        paths = []
        for k in range(length):
            image = Image.new("RGB", size, (0, 0, 0))
            image.paste((0, 0, 255), figure)
            image.save(into / f"{k:03d}.png")
            paths.append(into / f"{k:03d}.png")
        return paths

    return mask_pass


def test_build_drive_on_a_light_backdrop_never_runs_the_mask_pass(tmp_path):
    calls = []
    motion = bundle(tmp_path)
    built = build_drive(motion, SMALL, tmp_path / "drive", mask_pass_writing(calls), lambda p: native())
    assert calls == []
    assert (built.rate, built.length, built.size, built.method) == (16.0, 17, (32, 48), "light-backdrop")
    assert built.root.name == f"club-99-{built.digest[:12]}"
    assert len(read_apng(built.video)) == 17 and len(read_apng(built.mask)) == 17
    record = json.loads((built.root / DRIVE_RECORD).read_text())
    assert record["source_frames"] == [round((0.5 + k / 16) * 24) for k in range(17)]
    assert record["trace_index"] == [0, 4, 8, 12, 16] == list(built.index)
    assert len(record["masks"]) == 17 and record["box"] == [26, 0, 43, 64]
    colours = {tuple(c) for c in read_apng(built.mask)[3].reshape(-1, 3)}
    assert colours <= {(0, 0, 0), (0, 0, 1), MASK_FIGURE}


def test_build_drive_reuses_its_cache(tmp_path):
    motion = bundle(tmp_path)
    first = build_drive(motion, SMALL, tmp_path / "drive", None, lambda p: native())

    def no_decode(path):
        raise AssertionError("a cached drive decoded its clip again")

    assert build_drive(motion, SMALL, tmp_path / "drive", None, no_decode) == first
    other = build_drive(motion, machine(name="other", width=40, height=60), tmp_path / "drive", None, lambda p: native())
    assert other.root != first.root


def test_build_drive_runs_the_mask_pass_once_for_a_dark_backdrop(tmp_path):
    calls = []
    motion = bundle(tmp_path)
    dark = lambda p: native(backdrop=(40, 30, 30), figure=(250, 250, 250))  # noqa: E731
    built = build_drive(motion, SMALL, tmp_path / "drive", mask_pass_writing(calls), dark)
    assert len(calls) == 1 and built.method == "mask-pass"
    build_drive(motion, SMALL, tmp_path / "drive", mask_pass_writing(calls), dark)
    assert len(calls) == 1
    # A drive whose record was lost reuses the mask pass it already paid for.
    (built.root / DRIVE_RECORD).unlink()
    build_drive(motion, SMALL, tmp_path / "drive", mask_pass_writing(calls), dark)
    assert len(calls) == 1


def test_a_dark_backdrop_without_a_mask_pass_is_refused(tmp_path):
    dark = lambda p: native(backdrop=(40, 30, 30), figure=(250, 250, 250))  # noqa: E731
    with pytest.raises(DriveError, match="needs the mask pass"):
        build_drive(bundle(tmp_path), SMALL, tmp_path / "drive", None, dark)


def test_a_clip_that_starts_after_the_first_traced_frame_is_refused(tmp_path):
    with pytest.raises(DriveError, match="clip pad too short"):
        build_drive(bundle(tmp_path, clip_start=10.6), SMALL, tmp_path / "drive", None, lambda p: native())


def test_a_motion_with_no_clip_names_the_backfill(tmp_path):
    motion = SimpleNamespace(name="club-99", clip=None, frame_times=tuple(TIMES))
    with pytest.raises(DriveError, match="cag clips club-99"):
        build_drive(motion, SMALL, tmp_path / "drive")


def test_a_clip_that_decodes_at_another_size_is_refused(tmp_path):
    with pytest.raises(DriveError, match="decodes at 64x96"):
        build_drive(
            bundle(tmp_path), SMALL, tmp_path / "drive", None,
            lambda p: native().transpose(0, 2, 1, 3),
        )


BITS = 6


def numbered(count):
    """Frames whose number is written in six black-or-white blocks."""
    frames = numpy.full((count, 64, 96, 3), 255, numpy.uint8)
    for j in range(count):
        for bit in range(BITS):
            if j >> bit & 1:
                x, y = 30 + 12 * (bit % 3), 8 + 24 * (bit // 3)
                frames[j, y : y + 24, x : x + 12] = 0
    return frames


def read_number(frame, left):
    number = 0
    for bit in range(BITS):
        x, y = 30 + 12 * (bit % 3) - left + 6, 8 + 24 * (bit // 3) + 12
        if frame[y, x].mean() < 128:
            number |= 1 << bit
    return number


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_each_drive_frame_is_the_right_native_frame_of_a_real_clip(tmp_path):
    frames = numbered(48)
    clip_path = tmp_path / "clip.mkv"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "96x64",
            "-r", "24", "-i", "-", "-c:v", "ffv1", str(clip_path),
        ],
        input=frames.tobytes(),
        check=True,
    )
    decoded = decode(clip_path)
    assert decoded.shape == (48, 64, 96, 3)
    assert [read_number(f, 0) for f in decoded] == list(range(48))

    clip = Clip(clip_path, start=10.0, fps=24, frame_count=48, size=(96, 64), box=None)
    motion = SimpleNamespace(name="numbered", clip=clip, frame_times=tuple(TIMES))
    calls = []
    # The drive is cut at its own size, so no resize blurs the blocks.
    built = build_drive(
        motion, machine(width=43, height=64), tmp_path / "drive", mask_pass_writing(calls, (43, 64), (12, 10, 30, 54))
    )
    assert built.box == (26, 0, 43, 64)
    expected = [round((10.5 + k / 16 - 10.0) * 24) for k in range(17)]
    assert expected[:4] == [12, 14, 15, 16]
    assert [read_number(f, 26) for f in read_apng(built.video)] == expected


def test_the_drive_files_are_named_as_documented():
    assert (DRIVE_VIDEO, DRIVE_MASK, DRIVE_RECORD) == ("drive.png", "drive-mask.png", "drive.json")
    assert drive.SCAIL_MAX_LENGTH == 81
