"""End to end through the CLI, with the model and the image tool faked out."""

import pytest

from cag.geometry import CELL_HEIGHT, CELL_WIDTH
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from PIL import Image, ImageSequence

from cag import animation, cli, mask, static_sheet
from tests.test_animation import fake_draw
from tests.test_static_sheet import flat_cutout

SAMPLE = "/Users/matthewantone/Development/MotionArtist/work/sample/motion.json"


@pytest.fixture
def built(tmp_path, monkeypatch):
    fake_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    monkeypatch.setattr(static_sheet, "draw", fake_draw)
    monkeypatch.setattr(animation, "draw", fake_draw)
    monkeypatch.setattr(
        cli,
        "ChatCodex",
        lambda *a, **kw: FakeMessagesListChatModel(
            responses=[AIMessage("A lounge performer."), AIMessage("Mic in the character-right hand.")]
        ),
    )
    build_argv = [
            "build",
            "specs/velvet-lou.json",
            "--set", "dance",
            "--motion", SAMPLE,
            "--per-frame",
            "--work", str(tmp_path / "work"),
            "--out", str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit):  # the key art gate
        cli.main(build_argv)
    cli.main(["approve", "specs/velvet-lou.json", "--work", str(tmp_path / "work")])
    cli.main(build_argv)
    return tmp_path / "out" / "velvet-lou"


def test_the_gate_names_the_key_art_and_how_to_clear_it(tmp_path, monkeypatch):
    fake_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    monkeypatch.setattr(static_sheet, "draw", fake_draw)
    monkeypatch.setattr(
        cli,
        "ChatCodex",
        lambda *a, **kw: FakeMessagesListChatModel(responses=[AIMessage("A lounge performer.")]),
    )
    with pytest.raises(SystemExit) as stop:
        cli.main(
            ["build", "specs/velvet-lou.json", "--work", str(tmp_path / "w"),
             "--out", str(tmp_path / "o")]
        )
    assert "source/key.png" in str(stop.value)
    assert "cag approve specs/velvet-lou.json" in str(stop.value)
    assert not (tmp_path / "o").exists()  # nothing else was drawn or written


def test_writes_the_four_projection_views(built):
    for view in ("key", "front", "back", "profile"):
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
        "ChatCodex",
        lambda *a, **kw: FakeMessagesListChatModel(responses=[AIMessage("A lounge performer.")]),
    )
    monkeypatch.setattr(cli, "write_motion", lambda *a, **kw: pytest.fail("no sheet needed"))
    argv = ["build", "specs/no-animations.json", "--work", str(tmp_path / "w"),
            "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):  # the key art gate
        cli.main(argv)
    cli.main(["approve", "specs/no-animations.json", "--work", str(tmp_path / "w")])
    cli.main(argv)
    out = tmp_path / "o" / "no-one"
    assert (out / "index.html").exists()
    assert not list(out.glob("*.gif"))
    assert len(fake_draw.calls) == 4
