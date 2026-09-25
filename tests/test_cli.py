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
            ["build", "tests/fixtures/velvet-lou.json", "--work", str(tmp_path / "w"),
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
    argv = ["build", "tests/fixtures/no-animations.json", "--work", str(tmp_path / "w"),
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
