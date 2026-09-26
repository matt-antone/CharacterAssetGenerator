"""End to end through the CLI, with the model and the image tool faked out."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from cag.geometry import CELL_HEIGHT, CELL_WIDTH
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from PIL import Image, ImageSequence

from cag import animation, cli, mask, static_sheet
from cag.prompts import KEY_VIEW
from cag.motion import MotionError
from cag.sets import plan_for
from cag.spec import load_spec
from tests.test_animation import fake_draw
from tests.test_static_sheet import flat_cutout

SAMPLE = "motions/sample/motion.json"


@pytest.fixture
def built(tmp_path, monkeypatch):
    fake_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    monkeypatch.setattr(static_sheet, "draw", fake_draw)
    monkeypatch.setattr(animation, "draw", fake_draw)
    monkeypatch.setattr(
        cli,
        "text_model",
        lambda *a, **kw: FakeMessagesListChatModel(
            responses=[AIMessage("A lounge performer."), AIMessage("Mic in the character-right hand.")]
        ),
    )
    build_argv = [
            "build",
            "tests/fixtures/velvet-lou.json",
            "--draw-backend", "codex",
            "--set", "dance",
            "--motion", SAMPLE,
            "--per-frame",
            "--work", str(tmp_path / "work"),
            "--out", str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit):  # the key art gate
        cli.main(build_argv)
    cli.main(["approve", "tests/fixtures/velvet-lou.json", "--work", str(tmp_path / "work")])
    cli.main(build_argv)
    return tmp_path / "out" / "velvet-lou"


def test_the_gate_names_the_key_art_and_how_to_clear_it(tmp_path, monkeypatch):
    fake_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    monkeypatch.setattr(static_sheet, "draw", fake_draw)
    monkeypatch.setattr(
        cli,
        "text_model",
        lambda *a, **kw: FakeMessagesListChatModel(responses=[AIMessage("A lounge performer.")]),
    )
    with pytest.raises(SystemExit) as stop:
        cli.main(
            ["build", "tests/fixtures/velvet-lou.json", "--draw-backend", "codex",
             "--work", str(tmp_path / "w"),
             "--out", str(tmp_path / "o")]
        )
    assert "source/key.png" in str(stop.value)
    assert "uv run cag approve tests/fixtures/velvet-lou.json" in str(stop.value)
    assert not (tmp_path / "o").exists()  # nothing else was drawn or written


def test_writes_every_view_left_on(built):
    for view in (KEY_VIEW, *static_sheet.projection_views()):
        with Image.open(built / "views" / f"{view}.png") as cell:
            assert cell.size == (CELL_WIDTH, CELL_HEIGHT)


def test_writes_a_wrapped_sprite_sheet(built):
    with Image.open(built / "dance-sheet.png") as sheet:
        assert sheet.size == (CELL_WIDTH * 8, CELL_HEIGHT * 2)  # 16 frames, 8 to a row


def test_writes_a_looping_proof_at_the_declared_rate(built):
    with Image.open(built / "dance-proof.gif") as gif:
        assert len(list(ImageSequence.Iterator(gif))) == 16
        assert gif.info["duration"] == 250
        assert gif.info["loop"] == 0


def test_writes_a_gallery_that_points_at_the_deliverables(built):
    page = (built / "index.html").read_text()
    assert "Velvet Lou" in page
    for reference in ("views/key.png", "dance-sheet.png", "dance-proof.gif", "16 frames at 4 fps"):
        assert reference in page


def test_static_only_build_skips_the_animation(tmp_path, monkeypatch):
    fake_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    monkeypatch.setattr(static_sheet, "draw", fake_draw)
    monkeypatch.setattr(
        cli,
        "text_model",
        lambda *a, **kw: FakeMessagesListChatModel(responses=[AIMessage("A lounge performer.")]),
    )
    monkeypatch.setattr(cli, "write_motion", lambda *a, **kw: pytest.fail("no sheet needed"))
    argv = ["build", "tests/fixtures/no-animations.json", "--draw-backend", "codex",
            "--work", str(tmp_path / "w"),
            "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):  # the key art gate
        cli.main(argv)
    cli.main(["approve", "tests/fixtures/no-animations.json", "--work", str(tmp_path / "w")])
    cli.main(argv)
    out = tmp_path / "o" / "no-one"
    assert (out / "index.html").exists()
    assert not list(out.glob("*.gif"))
    assert len(fake_draw.calls) == 1 + len(static_sheet.projection_views()), "key art and the views left on"


def test_a_named_sheet_is_found_under_the_motion_root(tmp_path):
    """The brief names a choreography; the pipeline knows where sheets live."""
    root = tmp_path / "work"
    _bundle(root, "zs-loop")
    spec = replace(load_spec("tests/fixtures/velvet-lou.json"), motions={"dance": "zs-loop"})

    motion = cli.motion_for(spec, "dance", tmp_path / "work_dir", None, root)

    assert motion.frames, "the named sheet drives the set"


def test_the_brief_sets_the_playback_of_a_set_that_writes_its_own_sheet(tmp_path, monkeypatch):
    """The set plan is the default; this character's brief is the one that differs."""
    plans = []
    monkeypatch.setattr(cli, "write_motion", lambda *a, **kw: plans.append(a[4]))
    lou = load_spec("tests/fixtures/velvet-lou.json")

    cli.motion_for(replace(lou, playbacks={"dance": "pingpong"}), "dance",
                   tmp_path / "w", None, tmp_path / "gone")
    cli.motion_for(lou, "dance", tmp_path / "w", None, tmp_path / "gone")

    # Two characters, one set: the brief decides, and a silent brief takes the plan.
    assert [plan.playback for plan in plans] == ["pingpong", plan_for("dance").playback]


def test_a_named_sheet_that_is_not_there_says_so(tmp_path):
    spec = replace(load_spec("tests/fixtures/velvet-lou.json"), motions={"dance": "nope"})
    with pytest.raises(MotionError, match="nope"):
        cli.motion_for(spec, "dance", tmp_path / "work_dir", None, tmp_path / "work")


def test_the_motion_flag_wins_over_the_sheet_the_brief_names(tmp_path):
    """`--motion` is the operator pointing at one file for this run."""
    spec = replace(load_spec("tests/fixtures/velvet-lou.json"), motions={"dance": "nope"})

    motion = cli.motion_for(spec, "dance", tmp_path / "w", Path(SAMPLE), tmp_path / "gone")

    assert motion.frames, "the flag is used and the missing named sheet never looked for"


def test_the_motion_flag_brings_the_bundles_traced_frames(tmp_path):
    """Pointed at a bundle, `--motion` carries its photographs, not just its sheet."""
    spec = load_spec("tests/fixtures/velvet-lou.json")

    motion = cli.motion_for(spec, "dance", tmp_path / "w", Path(SAMPLE), tmp_path / "gone")

    assert motion.photos, "the traced frames are the pose reference; the flag must not drop them"


def _bundle(root, name):
    """A motion bundle is its manifest and the files the manifest names."""
    (root / name).mkdir(parents=True)
    shutil.copy(SAMPLE, root / name / "motion.json")
    sheet = json.loads(Path(SAMPLE).read_text())
    (root / name / "manifest.json").write_text(json.dumps({
        "bundle": "motion-source", "schema": "motion-artist/1",
        "name": name, "title": name.replace("-", " ").title(),
        "fps": sheet["fps"], "frame_count": len(sheet["frames"]),
        "playback": sheet["playback"], "view": sheet["view"],
        "seam": sheet.get("seam", ""), "files": {"motion.json": "unchecked"},
    }))
    return root / name


def _library(tmp_path, *names):
    root = tmp_path / "frame_sheets"
    for name in names:
        _bundle(root, name)
    return root


def test_auto_keeps_a_character_on_the_same_sheet_between_runs(tmp_path):
    root = _library(tmp_path, "a-loop", "b-loop", "c-loop")
    spec = replace(load_spec("tests/fixtures/velvet-lou.json"), motions={"dance": cli.AUTO})

    first = cli.auto_bundle(spec, "dance", root)[0]

    assert cli.auto_bundle(spec, "dance", root)[0] == first


def test_auto_spreads_the_library_across_a_roster(tmp_path):
    """Twelve characters must not all come back dancing the same dance."""
    root = _library(tmp_path, "a-loop", "b-loop", "c-loop")
    base = load_spec("tests/fixtures/velvet-lou.json")
    picks = {
        cli.auto_bundle(replace(base, name=f"Singer {n}"), "dance", root)[0] for n in range(12)
    }
    assert len(picks) > 1, f"every character landed on the same sheet: {picks}"


def test_auto_with_an_empty_library_says_so(tmp_path):
    spec = replace(load_spec("tests/fixtures/velvet-lou.json"), motions={"dance": cli.AUTO})
    with pytest.raises(MotionError, match="none under"):
        cli.motion_for(spec, "dance", tmp_path / "w", None, tmp_path / "empty")


def test_out_for_files_a_package_under_its_theme(tmp_path):
    specs, out = tmp_path / "specs", Path("outputs")
    (specs / "halloween").mkdir(parents=True)

    themed = cli.out_for(out, specs / "halloween" / "mort.json", "mort", specs)
    loose = cli.out_for(out, specs / "belter.json", "belter", specs)
    outside = cli.out_for(out, tmp_path / "elsewhere" / "oneoff.json", "oneoff", specs)

    assert themed == out / "halloween" / "mort"
    assert loose == out / "belter"  # no theme folder for a brief filed loose
    assert outside == out / "oneoff"  # a folder outside specs/ is not a theme
    assert out.resolve() in themed.resolve().parents  # never outside the output folder


def test_builds_the_location_at_2k_and_lists_it(built):
    from cag.assemble import LOCATION_SIZE

    with Image.open(built / "location.png") as background:
        assert background.size == LOCATION_SIZE
        assert background.mode == "RGB"  # a background is opaque; only cells carry alpha

    written = json.loads((built / "manifest.json").read_text())
    assert written["location"] == {"file": "location.png", "size": list(LOCATION_SIZE)}
    assert "location" not in written["views"]  # not a view: it is not a cell and is not cut out
    assert 'src="location.png"' in (built / "index.html").read_text()


def test_the_location_prompt_asks_for_an_empty_room_and_never_the_bible(built):
    from cag.prompts import location_prompt

    spec = load_spec("tests/fixtures/velvet-lou.json")
    prompt = location_prompt(spec)

    assert spec.location in prompt
    assert "no people, no characters, no figures" in prompt
    assert "cropped off when this is shown" in prompt  # the central 12:7 is the safe area
    # The costume must not reach a background prompt: a room in the character's
    # own colours is a room they disappear into.
    assert "velvet jacket" not in prompt
    assert spec.location not in "".join(call["prompt"] for call in fake_draw.calls
                                        if "location" not in call["out"].name)


def test_a_build_that_names_no_backend_stops_and_asks(tmp_path, monkeypatch):
    """Each backend is a different model, bill and licence: none is assumed."""
    monkeypatch.delenv("CAG_DRAW_BACKEND", raising=False)
    with pytest.raises(SystemExit, match="no draw backend.*codex,comfy,local"):
        cli.main(["build", "tests/fixtures/velvet-lou.json", "--work", str(tmp_path / "w"),
                  "--out", str(tmp_path / "o")])
    assert not (tmp_path / "w").exists(), "nothing is drawn before the backend is known"


# The video path's switches. `--machine` is checked before anything is drawn.

BUILD = ["build", "tests/fixtures/velvet-lou.json", "--set", "dance"]


@pytest.mark.parametrize(
    "backend, machine, says",
    [
        ("codex", "cloud", "--machine draws through ComfyUI graphs; --draw-backend codex"),
        ("local", "cloud", "machine 'cloud' runs on --draw-backend comfy, not local"),
        ("comfy", "smoke4", "machine 'smoke4' runs on --draw-backend local, not comfy"),
    ],
)
def test_a_machine_the_backend_cannot_draw_on_stops_the_build(tmp_path, backend, machine, says):
    with pytest.raises(SystemExit, match=says):
        cli.main([*BUILD, "--draw-backend", backend, "--machine", machine,
                  "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    assert not (tmp_path / "w" / "velvet-lou").exists(), "nothing is drawn"


def test_a_machine_does_not_draw_per_frame(tmp_path):
    with pytest.raises(SystemExit, match="drop --per-frame"):
        cli.main([*BUILD, "--draw-backend", "comfy", "--machine", "cloud", "--per-frame",
                  "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")])


def test_an_unknown_machine_lists_the_profiles(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main([*BUILD, "--draw-backend", "comfy", "--machine", "rtx5090"])
    assert "cloud" in capsys.readouterr().err
    # A name from $CAG_MACHINE skips argparse's check, so the profile loader says it.
    with pytest.raises(SystemExit, match="the profiles are: cloud, local16, smoke4"):
        cli.machine_with("comfy", "rtx5090", tmp_path)


def test_the_video_path_needs_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="ffmpeg"):
        cli.machine_with("comfy", "cloud", tmp_path)


def test_a_machine_writes_its_graphs_under_the_work_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    machine, graphs = cli.machine_with("comfy", "cloud", tmp_path)
    assert machine.name == "cloud"
    assert graphs == {stage: tmp_path / "comfy" / "cloud" / f"{stage}.json"
                      for stage in ("video", "restyle", "mask")}
    assert all(path.exists() for path in graphs.values())


def test_a_local_machine_lists_everything_its_server_lacks(tmp_path, monkeypatch):
    asked = []

    def preflight(client, workflows):
        names = [Path(w).name for w in workflows]
        asked.append(names)
        return ["video.json: UnetLoaderGGUF unet_name SCAIL-2-Q2_K.gguf is not installed",
                "restyle.json: node TextEncodeQwenImageEdit is not installed"] if "video.json" in names else []

    monkeypatch.setattr(cli.comfy, "preflight", preflight)
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    with pytest.raises(SystemExit, match=r"(?s)SCAIL-2-Q2_K\.gguf.*TextEncodeQwenImageEdit.*machines --check smoke4"):
        cli.machine_with("local", "smoke4", tmp_path, client=object())
    assert asked == [["video.json", "restyle.json"]]


def test_a_local_machine_without_the_mask_model_still_draws_and_says_so(tmp_path, monkeypatch, capsys):
    def preflight(client, workflows):
        names = [Path(w).name for w in workflows]
        return ["mask.json: CheckpointLoaderSimple ckpt_name sam3.1_multiplex_fp16.safetensors "
                "is not installed"] if names == ["mask.json"] else []

    monkeypatch.setattr(cli.comfy, "preflight", preflight)
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    machine, graphs = cli.machine_with("local", "smoke4", tmp_path, client=object())
    assert machine.name == "smoke4" and set(graphs) == {"video", "restyle", "mask"}
    assert "sam3.1_multiplex_fp16.safetensors is not installed; a set whose footage needs the mask pass" in (
        capsys.readouterr().err
    )


def test_no_machine_turns_the_video_path_off_whatever_cag_machine_says(tmp_path, monkeypatch):
    monkeypatch.setenv("CAG_MACHINE", "cloud")
    seen = {}
    monkeypatch.setattr(cli, "build", lambda *a, **kw: seen.update(kw))
    assert cli.main([*BUILD, "--draw-backend", "codex", "--no-machine",
                     "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")]) == 0
    assert seen["machine"] is None and seen["graphs"] is None
    with pytest.raises(SystemExit, match="--draw-backend codex has none"):
        cli.main([*BUILD, "--draw-backend", "codex", "--work", str(tmp_path / "w")])


def test_backend_state_carries_the_machine_to_the_animation_graph(tmp_path):
    from cag.machines import load_machine

    machine = load_machine("smoke4")
    graphs = {"video": tmp_path / "v.json"}
    state = cli.backend_state("local", None, machine, graphs)
    assert state["machine"] is machine and state["machine_graphs"] == graphs and state["local"]
    assert state["pose_workflow"], "a written set still has the pose-edit path to fall to"
    assert "machine" not in cli.backend_state("local")


def test_no_restyle_needs_a_machine(tmp_path, monkeypatch):
    monkeypatch.delenv("CAG_MACHINE", raising=False)
    with pytest.raises(SystemExit, match="--no-restyle skips the video path's restyle; it needs --machine"):
        cli.main([*BUILD, "--draw-backend", "comfy", "--no-restyle",
                  "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    assert not (tmp_path / "w").exists()


def test_no_restyle_reaches_the_build_and_the_machine_check(tmp_path, monkeypatch):
    seen, checked = {}, {}

    def machine_with(backend, name, work_root, given=None, client=None, restyle=True):
        checked["restyle"] = restyle
        return object(), {}

    monkeypatch.setattr(cli, "machine_with", machine_with)
    monkeypatch.setattr(cli, "build", lambda *a, **kw: seen.update(kw))
    args = [*BUILD, "--draw-backend", "comfy", "--machine", "cloud",
            "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")]
    assert cli.main([*args, "--no-restyle"]) == 0
    assert seen["scail_only"] is True and checked["restyle"] is False
    assert cli.main(args) == 0
    assert seen["scail_only"] is False and checked["restyle"] is True


def test_no_finish_needs_a_machine(tmp_path, monkeypatch):
    monkeypatch.delenv("CAG_MACHINE", raising=False)
    with pytest.raises(SystemExit, match="--no-finish turns off the video path's cell finish; it needs --machine"):
        cli.main([*BUILD, "--draw-backend", "comfy", "--no-finish",
                  "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    assert not (tmp_path / "w").exists()


def test_no_finish_reaches_the_build(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "machine_with", lambda *a, **kw: (object(), {}))
    monkeypatch.setattr(cli, "build", lambda *a, **kw: seen.update(kw))
    args = [*BUILD, "--draw-backend", "comfy", "--machine", "cloud",
            "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")]
    assert cli.main([*args, "--no-finish"]) == 0 and seen["finish"] is False
    assert cli.main(args) == 0 and seen["finish"] is True


def test_backend_state_carries_no_finish_only_with_a_machine():
    from cag.machines import load_machine

    machine = load_machine("smoke4")
    assert cli.backend_state("local", None, machine, {}, finish=False)["finish"] is False
    assert "finish" not in cli.backend_state("local", None, machine, {}), "on unless turned off"
    assert "finish" not in cli.backend_state("local", finish=False)


def test_backend_state_carries_scail_only_only_with_a_machine(tmp_path):
    from cag.machines import load_machine

    machine = load_machine("smoke4")
    assert cli.backend_state("local", None, machine, {}, scail_only=True)["scail_only"] is True
    assert "scail_only" not in cli.backend_state("local", None, machine, {})
    assert "scail_only" not in cli.backend_state("local", scail_only=True)


def test_a_local_scail_only_build_does_not_need_the_restyle_models(tmp_path, monkeypatch, capsys):
    asked = []

    def preflight(client, workflows):
        names = [Path(w).name for w in workflows]
        asked.append(names)
        return ["restyle.json: node TextEncodeQwenImageEdit is not installed"] if "restyle.json" in names else []

    monkeypatch.setattr(cli.comfy, "preflight", preflight)
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    machine, _ = cli.machine_with("local", "smoke4", tmp_path, client=object(), restyle=False)
    assert machine.name == "smoke4"
    assert asked[0] == ["video.json"] and ["restyle.json"] in asked
    err = capsys.readouterr().err
    assert "TextEncodeQwenImageEdit is not installed; not needed under --no-restyle" in err
    assert "SCAIL-2 only on smoke4" in err
    with pytest.raises(SystemExit, match="TextEncodeQwenImageEdit"):
        cli.machine_with("local", "smoke4", tmp_path, client=object())


def test_a_set_whose_drive_fails_is_logged_failed_and_the_next_set_draws(tmp_path, monkeypatch, capsys):
    from cag.drive import DriveError
    from cag.motion import load_motion
    from cag.static_sheet import projection_views

    cell = tmp_path / "cell.png"
    image = Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
    image.paste((200, 60, 60, 255), (100, 100, 200, 500))
    image.save(cell)
    static = {"bible": "A lounge performer.", "scale": 2.875, "location": None,
              "sources": {KEY_VIEW: cell}, "cells": {v: cell for v in [KEY_VIEW, *projection_views()]}}
    monkeypatch.setattr(cli, "build_static_graph",
                        lambda *a, **kw: type("G", (), {"invoke": lambda self, state: static})())
    drawn = []

    def render_set(spec, name, *args, **kwargs):
        if name == "dance":
            raise DriveError("club-01: clip pad too short")
        drawn.append(name)
        return {"set_name": name, "motion": load_motion(SAMPLE), "cells": {0: cell, 1: cell}, "last": cell}

    monkeypatch.setattr(cli, "render_set", render_set)
    cli.build(Path("tests/fixtures/velvet-lou.json"), None, ["dance", "sing"], tmp_path / "w",
              tmp_path / "o", backend="comfy")
    assert "[dance] FAILED: club-01: clip pad too short" in capsys.readouterr().err
    assert drawn == ["sing"]
    assert (tmp_path / "o" / "velvet-lou" / "sing-sheet.png").exists()


def test_clips_backfills_each_name_and_says_which_it_refused(tmp_path, monkeypatch, capsys):
    from cag import clips

    done = []

    def backfill(root, cache, dry_run=False):
        done.append((root.name, dry_run))
        if root.name == "sample":
            raise clips.ClipError("records no source url to cut a clip from")
        return "fine OK"

    monkeypatch.setattr(clips, "backfill", backfill)
    assert cli.main(["clips", "sample", "nobody", "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "sample REFUSED: records no source url" in out
    assert "nobody REFUSED: no bundle under motions calls itself that" in out
    assert done == [("sample", True)]


def test_one_bundles_fault_is_its_own_line_and_the_batch_goes_on(tmp_path, monkeypatch, capsys):
    from cag import clips

    for name in ("a", "b"):
        shutil.copytree("motions/sample", tmp_path / name)
        manifest = json.loads((tmp_path / name / "manifest.json").read_text())
        (tmp_path / name / "manifest.json").write_text(json.dumps({**manifest, "name": name}))
    done = []

    def backfill(root, cache, dry_run=False):
        if root.name == "a":
            raise FileNotFoundError(2, "No such file or directory", str(root / "thumbs" / "f07.jpg"))
        done.append(root.name)
        return f"{root.name} OK"

    monkeypatch.setattr(clips, "backfill", backfill)
    assert cli.main(["clips", "a", "b", "--motion-root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "a REFUSED: FileNotFoundError: [Errno 2] No such file or directory" in out
    assert "b OK" in out and done == ["b"]


def test_clips_missing_is_every_declared_clip_not_on_disk(tmp_path, monkeypatch, capsys):
    from cag import clips

    block = {"file": "clip.mp4", "start": 0.0, "fps": 24, "frame_count": 10, "size": [64, 48],
             "sha256": "ab" * 32}
    for name, clip in (("fresh", None), ("stale", b"another cut"), ("never", None)):
        shutil.copytree("motions/sample", tmp_path / name)
        manifest = json.loads((tmp_path / name / "manifest.json").read_text())
        manifest["name"] = name
        if name != "never":
            manifest["clip"] = block
        (tmp_path / name / "manifest.json").write_text(json.dumps(manifest))
        if clip:
            (tmp_path / name / "clip.mp4").write_bytes(clip)
    done = []
    monkeypatch.setattr(clips, "backfill", lambda root, cache, dry_run=False: done.append(root.name) or "OK")
    assert cli.main(["clips", "--missing", "--motion-root", str(tmp_path)]) == 0
    assert sorted(done) == ["fresh", "stale"]


def test_clips_check_reports_without_fetching(monkeypatch, capsys):
    from cag import clips

    monkeypatch.setattr(clips, "backfill", lambda *a, **kw: pytest.fail("--check fetches nothing"))
    assert cli.main(["clips", "sample", "--check"]) == 1
    assert capsys.readouterr().out.strip() == "sample clip=none"


def test_the_cast_is_every_bundle_a_brief_names(tmp_path):
    brief = json.loads(Path("tests/fixtures/velvet-lou.json").read_text())
    (tmp_path / "default").mkdir()
    for slug, motion in (("a", "club-01"), ("b", "club-01"), ("c", "auto")):
        (tmp_path / "default" / f"{slug}.json").write_text(
            json.dumps({**brief, "animations": {"dance": {"motion": motion}}})
        )
    (tmp_path / "character.schema.json").write_text("{}")
    assert cli.cast_bundles(tmp_path) == ["club-01"]


def test_motions_lists_what_each_bundle_holds_of_its_clip(capsys):
    assert cli.main(["motions"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "clip" in out[0].split()
    assert next(line for line in out if line.startswith("sample ")).split()[-2] == "none"


def test_machines_check_prints_the_download_checklist(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli.comfy, "Client", lambda local=False: object())
    monkeypatch.setattr(cli.comfy, "preflight", lambda client, graphs: ["x.json: lora_name y is not installed"])
    assert cli.main(["machines", "--check", "smoke4", "--work", str(tmp_path)]) == 1
    assert capsys.readouterr().out.strip() == "x.json: lora_name y is not installed"
    monkeypatch.setattr(cli.comfy, "preflight", lambda client, graphs: [])
    assert cli.main(["machines", "--check", "smoke4", "--work", str(tmp_path)]) == 0


@pytest.mark.parametrize("returncode", [0, 1])
def test_a_build_publishes_its_finished_package_even_with_a_failed_set(
    tmp_path, monkeypatch, capsys, returncode
):
    """A FAILED set still publishes what was written, and a failed publish is a
    warning: the build succeeds either way."""
    from cag import publish
    from tests.test_publish import REMOTE, Rclone, installed

    fake_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    monkeypatch.setattr(static_sheet, "draw", fake_draw)
    monkeypatch.setattr(
        cli, "text_model",
        lambda *a, **kw: FakeMessagesListChatModel(responses=[AIMessage("A lounge performer.")]),
    )
    run = Rclone(returncode=returncode, stderr="ERROR : network is unreachable")
    monkeypatch.setattr(publish.subprocess, "run", run)
    monkeypatch.setattr(publish.shutil, "which", installed)
    monkeypatch.setenv(publish.REMOTE_ENV, REMOTE)
    argv = ["build", "tests/fixtures/velvet-lou.json", "--draw-backend", "codex", "--set", "dance",
            "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):  # the key art gate: no package yet, nothing published
        cli.main(argv)
    assert run.calls == []
    cli.main(["approve", "tests/fixtures/velvet-lou.json", "--work", str(tmp_path / "w")])

    def fails(*a, **kw):
        raise RuntimeError("wrong figure count")

    monkeypatch.setattr(cli, "render_set", fails)
    assert cli.main(argv) == 0
    [(command, _)] = run.calls
    assert command[1:4] == ["copy", str(tmp_path / "o" / "velvet-lou"), f"{REMOTE}/velvet-lou"]
    assert (tmp_path / "o" / "velvet-lou" / "index.html").exists()
    err = capsys.readouterr().err
    assert "[dance] FAILED" in err
    assert ("network is unreachable" in err) == bool(returncode)

