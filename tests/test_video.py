"""The video path, end to end through the animation graph, with every model faked.

The drive is cut from a fake decoded clip, SCAIL-2 is a fake `render_frames`
that saves as many frames as it is asked for, and each restyle is a fake draw.
Nothing here says anything about how the art looks; it pins what is sent where,
in what order, and what is drawn again when something changes.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from PIL import Image

from cag import animation, comfy, drive, mask, video
from cag.draw import DrawError
from cag.machines import load_machine, materialise
from cag.motion import Clip, MotionError, load_motion
from cag.prompts import MASK_PASS_PROMPT, RESTYLE
from cag.spec import load_spec
from tests.test_animation import SAMPLE, posed
from tests.test_static_sheet import flat_cutout

#: A clip at 24 fps from source second 9.5, and traced frames a tenth of a
#: second apart from second 10: the sample's 16 frames span 1.5 s, which is 25
#: drive frames at 16 fps.
CLIP_START, CLIP_FPS, CLIP_FRAMES = 9.5, 24.0, 60
TIMES = [10.0 + 0.1 * i for i in range(16)]
INDEX = [round((t - TIMES[0]) * 16) for t in TIMES]
SIZE = (240, 135)


def footage(light=True):
    """A performer, a dark block drifting right, on a light or a dark backdrop."""

    def decode(path):
        frames = numpy.full((CLIP_FRAMES, SIZE[1], SIZE[0], 3), 235 if light else 30, numpy.uint8)
        for k in range(CLIP_FRAMES):
            frames[k, 40:110, 100 + k // 6 : 130 + k // 6] = (40, 40, 60) if light else (200, 180, 160)
        return frames

    return decode


def traced(tmp_path):
    motion = posed(load_motion(SAMPLE), tmp_path)
    clip = Clip(tmp_path / "clip.mp4", CLIP_START, CLIP_FPS, CLIP_FRAMES, SIZE, (90, 20, 90, 110),
                sha256="c" * 64)
    frames = tuple(replace(f, t=t) for f, t in zip(motion.frames, TIMES))
    return replace(motion, frames=frames, clip=clip)


class Scail:
    """Stands in for `comfy.render_frames`: a job that saves `expect` frames."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, out_dir, references, workflow, timeout, **kwargs):
        self.calls.append({"prompt": prompt, "out": Path(out_dir), "refs": [Path(r) for r in references],
                           "workflow": workflow, "timeout": timeout, **kwargs})
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for k in range(kwargs["expect"]):
            paths.append(out_dir / f"{k:03d}.png")
            Image.new("RGB", (64, 96), (k * 9 % 256, 80, 120)).save(paths[-1])
        return paths


class Restyle:
    """Stands in for `draw`: records a render only when it draws one."""

    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def __call__(self, prompt, out_path, references=(), **kwargs):
        out_path = Path(out_path)
        if out_path.exists():
            return out_path
        index = int(out_path.stem)
        if index in self.fail:
            raise DrawError(f"could not draw {out_path.name}: the backdrop is not flat")
        self.calls.append({"prompt": prompt, "out": out_path, "refs": list(references), **kwargs})
        image = Image.new("RGB", (100, 200), (255, 0, 255))
        image.paste((30 + 5 * index,) * 3, (40, 20, 60, 180))
        image.save(out_path)
        return out_path


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    monkeypatch.setattr(drive, "decode", footage())
    scail = Scail()
    monkeypatch.setattr(comfy, "render_frames", scail)
    snapped = []
    monkeypatch.setattr(video, "snap_file", lambda path, key: snapped.append(Path(path).name))
    machine = replace(load_machine("cloud"), name="tiny", width=64, height=96)
    graphs = {stage: materialise(machine, stage, tmp_path / "comfy") for stage in ("video", "restyle", "mask")}
    key_art = tmp_path / "dance-key-front.png"
    Image.new("RGB", (100, 150), (255, 0, 255)).save(key_art)
    state = {
        "spec": load_spec("tests/fixtures/velvet-lou.json"),
        "bible": "A lounge performer in a velvet jacket",
        "key_art": key_art,
        "scale": 2.875,
        "motion": traced(tmp_path),
        "set_name": "dance",
        "work_dir": tmp_path / "work" / "lou",
        "snap_to_key": True,
        "machine": machine,
        "machine_graphs": graphs,
        "local": False,
    }

    def build(restyle=None, **changes):
        restyle = restyle or Restyle()
        # No replies: any call for a director's note fails the test.
        graph = animation.build_animation_graph(
            FakeMessagesListChatModel(responses=[]), draw_fn=restyle, frame_sheet_mode=True
        )
        return graph.invoke({**state, **changes}), restyle

    return {"build": build, "scail": scail, "snapped": snapped, "state": state, "tmp": tmp_path}


def test_a_traced_set_with_a_machine_is_drawn_from_its_clip(world):
    result, restyle = world["build"]()
    state, scail = world["state"], world["scail"]

    assert len(scail.calls) == 1, "one SCAIL job for the whole set"
    job = scail.calls[0]
    drive_dir = world["tmp"] / "work" / "drive"
    (made,) = drive_dir.iterdir()
    assert [r.name for r in job["refs"]] == ["ref.png", "ref-mask.png", "drive.png", "drive-mask.png"]
    assert job["refs"][2].parent == made, "the drive is shared beside the characters"
    assert job["raw"] == [False, False, True, True], "the videos go up with every frame"
    assert job["expect"] == 25 and job["extra"]["$length"] == 25
    assert job["extra"]["$width"] == 64 and job["seed"] == 1234
    assert job["timeout"] == state["machine"].video_timeout
    assert "velvet jacket" in job["prompt"] and "dancing in place" in job["prompt"]
    assert job["pending"] == job["out"] / "pending.json"
    assert json.loads((made / "drive.json").read_text())["method"] == "light-backdrop"

    assert len(restyle.calls) == 16
    for index, call in enumerate(restyle.calls):
        assert call["prompt"] == RESTYLE
        assert call["out"] == state["work_dir"] / "source" / "dance" / f"{index:02d}.png"
        assert call["refs"][0] == job["out"] / f"traced-{index:02d}.png", "the SCAIL frame is image 1"
        assert call["refs"][1] == state["key_art"], "the set reference is image 2"
        assert call["workflow"] == state["machine_graphs"]["restyle"]
        assert call["timeout"] == state["machine"].restyle_timeout and call["seed"] == 1234
        assert call["extra"] == {"$resolution": 1248, "$steps": 40}
    # The traced frame is the SCAIL frame at its traced time, at the reference's size.
    with Image.open(job["out"] / "traced-05.png") as picked, Image.open(job["out"] / f"{INDEX[5]:03d}.png") as frame:
        assert picked.size == (100, 150)
        assert picked.getpixel((50, 75)) == frame.resize((100, 150)).getpixel((50, 75))
    assert json.loads((job["out"] / "trace-index.json").read_text()) == INDEX

    assert world["snapped"] == [f"{n:02d}.png" for n in range(16)], "snapped once each"
    assert result["frame_sheets"] == [[n] for n in range(16)]
    assert result["shared_canvas"] and sorted(result["cells"]) == list(range(16))
    assert (state["work_dir"] / "source" / "dance" / "drawn.sha").read_text().startswith("video\t")
    assert not (state["work_dir"] / "motion" / "dance.note.txt").exists(), "no note is asked for"


def test_a_second_build_draws_nothing(world):
    world["build"]()
    _, restyle = world["build"]()
    assert len(world["scail"].calls) == 1 and restyle.calls == []


def test_a_deleted_frame_is_the_only_one_redrawn(world):
    world["build"]()
    (world["state"]["work_dir"] / "source" / "dance" / "03.png").unlink()
    _, restyle = world["build"]()
    assert len(world["scail"].calls) == 1, "the SCAIL video is reused"
    assert [call["out"].name for call in restyle.calls] == ["03.png"]


def test_a_changed_restyle_graph_supersedes_the_frames_but_keeps_the_video(world):
    world["build"]()
    graphs = dict(world["state"]["machine_graphs"])
    changed = json.loads(graphs["restyle"].read_text())
    changed["K"]["inputs"]["cfg"] = 2.0
    graphs["restyle"] = world["tmp"] / "restyle-cfg2.json"
    graphs["restyle"].write_text(json.dumps(changed))

    _, restyle = world["build"](machine_graphs=graphs)
    assert len(world["scail"].calls) == 1
    assert len(restyle.calls) == 16, "every frame redrawn under the new graph"
    source = world["state"]["work_dir"] / "source" / "dance"
    (kept,) = (source / "superseded").iterdir()
    assert kept.name.startswith("video-")
    assert sorted(p.name for p in kept.glob("[0-9][0-9].png")) == [f"{n:02d}.png" for n in range(16)]


def test_a_changed_set_reference_draws_a_new_video(world):
    world["build"]()
    Image.new("RGB", (100, 150), (250, 0, 250)).save(world["state"]["key_art"])
    world["build"]()
    assert len(world["scail"].calls) == 2


def test_a_failed_restyle_lets_the_rest_finish_then_names_it(world):
    with pytest.raises(DrawError, match=r"1 of 16 restyles failed: 07; rebuild"):
        world["build"](Restyle(fail={7}))
    source = world["state"]["work_dir"] / "source" / "dance"
    assert sorted(p.name for p in source.glob("[0-9][0-9].png")) == [
        f"{n:02d}.png" for n in range(16) if n != 7
    ]
    _, restyle = world["build"]()
    assert [call["out"].name for call in restyle.calls] == ["07.png"]


def test_a_traced_set_without_a_clip_says_how_to_get_one(world):
    motion = replace(world["state"]["motion"], clip=None)
    with pytest.raises(MotionError, match=r"sample carries no source clip.*cag clips sample"):
        world["build"](motion=motion)
    assert world["scail"].calls == []


def test_footage_with_no_light_backdrop_runs_the_mask_pass(world, monkeypatch):
    monkeypatch.setattr(drive, "decode", footage(light=False))
    scail = world["scail"]
    real = scail.__call__

    def masks_too(prompt, out_dir, references, workflow, timeout, **kwargs):
        if prompt != MASK_PASS_PROMPT:
            return real(prompt, out_dir, references, workflow, timeout, **kwargs)
        scail.calls.append({"prompt": prompt, "refs": list(references), "workflow": workflow, **kwargs})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        paths = []
        for k in range(kwargs["expect"]):
            image = Image.new("RGB", (64, 96))
            image.paste((0, 0, 255), (20 + k // 8, 20, 40 + k // 8, 70))
            paths.append(Path(out_dir) / f"{k:03d}.png")
            image.save(paths[-1])
        return paths

    monkeypatch.setattr(comfy, "render_frames", masks_too)
    world["build"]()
    masking, _ = scail.calls
    assert masking["refs"][0].name == "drive.png" and masking["raw"] == [True]
    assert masking["expect"] == 25 and masking["workflow"]["trk"]["class_type"] == "SAM3_VideoTrack"
    (made,) = (world["tmp"] / "work" / "drive").iterdir()
    assert json.loads((made / "drive.json").read_text())["method"] == "mask-pass"


def test_without_a_machine_a_traced_set_still_takes_the_pose_edit_path(world):
    state = {key: value for key, value in world["state"].items()
             if key not in ("machine", "machine_graphs", "local")}
    assert not animation.draws_by_video({**state, "photographic": True, "poses": {0: Path("x")}})
    assert animation.draws_by_video({**world["state"], "photographic": True, "poses": {0: Path("x")}})
